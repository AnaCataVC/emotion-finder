"""
Batch active learning and retraining pipeline with quality gates.

Reads verified user feedback from FeedbackStore, merges it with the canonical
training dataset (enforcing a strict 10% cap), retrains the classification
pipelines, and evaluates against mandatory regression probes.

If any regression probe fails or cross-validation F1 degrades below 0.95,
the script aborts and rejects the candidate model to prevent data poisoning.

Usage:
    python scripts/retrain_from_feedback.py [--dry-run] [--lang es|en|all]
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import joblib
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score

from feedback_store import FeedbackRecord, FeedbackStore, get_feedback_store
from train_model import (
    DATA_DIR,
    MODELS_DIR,
    RANDOM_STATE,
    REGRESSION_PROBES,
    _score_probes,
    build_pipeline,
    load_dataset,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("retrain_from_feedback")

# Maximum allowed ratio of user feedback samples relative to baseline dataset size (10% cap)
_MAX_FEEDBACK_RATIO = 0.10
_MIN_CV_F1_THRESHOLD = 0.95


def extract_training_samples(records: List[FeedbackRecord], lang: str) -> List[Tuple[str, str, str]]:
    """
    Extract clean (text, quadrant, record_id) tuples from verified feedback records.
    Deduplicates by normalized text so duplicate samples do not skew training.

    Filters:
    - Language matches target
    - Text length between 6 and 300 characters
    - Valid target quadrant exists (either corrected_quadrant, or predicted_quadrant if positive rating)
    """
    samples = []
    seen_texts = set()
    valid_quadrants = {"alta_positiva", "alta_negativa", "baja_positiva", "baja_negativa"}

    for r in records:
        if r.detected_lang != lang:
            continue

        text = r.user_text.strip()
        if len(text) < 6 or len(text) > 300:
            continue

        norm = r.normalized_text.strip()
        if norm in seen_texts:
            continue

        target_quadrant = None
        if r.rating == "positive" and r.predicted_quadrant in valid_quadrants:
            target_quadrant = r.predicted_quadrant
        elif r.rating == "negative" and r.corrected_quadrant in valid_quadrants:
            target_quadrant = r.corrected_quadrant

        if target_quadrant:
            samples.append((text, target_quadrant, r.id))
            seen_texts.add(norm)

    return samples


def retrain_language(lang: str, store: FeedbackStore, dry_run: bool = False) -> bool:
    """
    Execute protected batch retraining for a specific language.

    Returns:
        True if training succeeded and passed all quality gates, False otherwise.
    """
    logger.info("=== Starting Protected Retraining for Language: '%s' ===", lang)

    # 1. Load canonical dataset
    csv_name = f"emotions_{lang}_v2.csv"
    csv_path = DATA_DIR / csv_name
    if not csv_path.exists():
        logger.error("Canonical training dataset not found at %s", csv_path)
        return False

    base_texts, base_labels = load_dataset(csv_path)
    base_count = len(base_texts)
    logger.info("Loaded %d canonical samples from %s", base_count, csv_name)

    # 2. Ingest verified feedback
    verified_records = store.get_by_status("verified", limit=500)
    feedback_samples = extract_training_samples(verified_records, lang)
    logger.info("Found %d valid verified feedback samples for '%s'", len(feedback_samples), lang)

    # Enforce 10% sample cap
    max_allowed = int(base_count * _MAX_FEEDBACK_RATIO)
    if len(feedback_samples) > max_allowed:
        logger.warning(
            "Capping feedback samples from %d to %d (10%% safety cap)",
            len(feedback_samples), max_allowed
        )
        feedback_samples = feedback_samples[:max_allowed]

    # Combine datasets
    if feedback_samples:
        fb_texts, fb_labels, fb_ids = zip(*feedback_samples)
        combined_texts = list(base_texts) + list(fb_texts)
        combined_labels = list(base_labels) + list(fb_labels)
        incorporated_ids = set(fb_ids)
        logger.info("Total augmented dataset size: %d samples", len(combined_texts))
    else:
        combined_texts = list(base_texts)
        combined_labels = list(base_labels)
        incorporated_ids = set()
        logger.info("No feedback samples to incorporate. Running regression verification on baseline.")

    # 3. Stratified K-Fold Cross-Validation
    candidate_pipeline = build_pipeline(lang)
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(
        candidate_pipeline, combined_texts, combined_labels,
        cv=skf, scoring="f1_macro"
    )
    mean_f1 = float(np.mean(cv_scores))
    logger.info("5-Fold Cross-Validation Macro F1: %.4f (Scores: %s)", mean_f1, np.round(cv_scores, 4))

    if mean_f1 < _MIN_CV_F1_THRESHOLD:
        logger.error(
            "QUALITY GATE FAILED: Macro F1 (%.4f) fell below required threshold (%.4f). Aborting.",
            mean_f1, _MIN_CV_F1_THRESHOLD
        )
        return False

    # 4. Fit candidate pipeline
    candidate_pipeline.fit(combined_texts, combined_labels)

    # 5. Mandatory Regression Probes Gate (Must pass 100%)
    probes = REGRESSION_PROBES.get(lang, [])
    probe_score = _score_probes(candidate_pipeline, probes)
    logger.info("REGRESSION_PROBES accuracy: %.2f%% (%d probes tested)", probe_score * 100, len(probes))

    if probe_score < 1.0:
        logger.error(
            "QUALITY GATE FAILED: Candidate model failed dialectal REGRESSION_PROBES (Score: %.2f < 1.00). "
            "Rejecting model update to prevent dialectal regression / data poisoning.",
            probe_score
        )
        return False

    logger.info("ALL QUALITY GATES PASSED for '%s'!", lang)

    # 6. Save model or dry-run
    if dry_run:
        logger.info("[DRY-RUN] Model not written to disk. Pipeline verified successfully.")
    else:
        out_path = MODELS_DIR / f"model_{lang}.joblib"
        joblib.dump(candidate_pipeline, out_path, compress=3)
        logger.info("Saved updated model to %s", out_path)

        # Mark ONLY the records that were actually incorporated into this training run
        for rec_id in incorporated_ids:
            store.mark_status(rec_id, "incorporated")
        logger.info("Marked %d processed records as 'incorporated'.", len(incorporated_ids))

    return True


def main():
    parser = argparse.ArgumentParser(description="Active learning batch retrain pipeline.")
    parser.add_argument("--dry-run", action="store_true", help="Evaluate and validate without overwriting models.")
    parser.add_argument("--lang", choices=["es", "en", "all"], default="all", help="Target language to retrain.")
    args = parser.parse_args()

    store = get_feedback_store()
    languages = ["es", "en"] if args.lang == "all" else [args.lang]

    success = True
    for lang in languages:
        ok = retrain_language(lang, store, dry_run=args.dry_run)
        if not ok:
            success = False

    if not success:
        logger.error("Retraining failed one or more quality gates.")
        sys.exit(1)

    logger.info("Active learning retraining pipeline finished cleanly.")


if __name__ == "__main__":
    main()
