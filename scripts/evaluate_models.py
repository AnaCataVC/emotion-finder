"""
Model evaluation and benchmarking suite for Emotion Finder.

Evaluates candidate models across:
1. In-distribution 5-Fold Stratified Cross-Validation on synthetic affective datasets.
2. Out-of-distribution generalization on dialectal regression probes and held-out idioms.
3. Operational efficiency: model serialization size (KB), cold start latency (ms),
   and warm inference latency (ms).
4. Generates high-resolution confusion matrix figures for English and Spanish.

Usage:
    python scripts/evaluate_models.py
"""

import io
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except (AttributeError, io.UnsupportedOperation):
        pass

# Add project root to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import joblib
import matplotlib

matplotlib.use("Agg")  # Headless backend for server/CLI environments
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

import train_model
from preprocessing import tokenize_en, tokenize_es

RANDOM_STATE = 42
DOCS_EVAL_DIR = BASE_DIR / "docs" / "evaluation"


def get_candidate_models(lang: str) -> dict[str, Pipeline]:
    """Build candidate pipelines for comparison."""
    if lang == "es":
        tokenizer = tokenize_es
        stop_words = train_model._build_spanish_stopwords()
    else:
        tokenizer = tokenize_en
        stop_words = train_model._build_english_stopwords()

    tfidf_kwargs = {
        "tokenizer": tokenizer,
        "stop_words": stop_words,
        "ngram_range": (1, 2),
        "sublinear_tf": True,
        "max_features": 4000,
        "token_pattern": None,
    }

    return {
        "Dummy (Stratified)": Pipeline([
            ("tfidf", TfidfVectorizer(tokenizer=tokenizer, stop_words=stop_words, ngram_range=(1, 2), token_pattern=None)),
            ("clf", DummyClassifier(strategy="stratified", random_state=RANDOM_STATE)),
        ]),
        "Multinomial NB": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", MultinomialNB(alpha=1.0)),
        ]),
        "Linear SVM": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", LinearSVC(C=1.0, random_state=RANDOM_STATE)),
        ]),
        "Random Forest": Pipeline([
            ("tfidf", TfidfVectorizer(**tfidf_kwargs)),
            ("clf", RandomForestClassifier(n_estimators=100, max_depth=10, random_state=RANDOM_STATE)),
        ]),
        "Logistic Regression (Production)": train_model.build_pipeline(lang),
    }


def measure_latencies_and_size(pipeline: Pipeline, sample_query: str) -> tuple[float, float, float]:
    """
    Measure:
    - Serialized artifact size in KB (compressed with joblib level 3)
    - Cold load time in ms (from disk)
    - Warm inference latency in ms per single query (median over 300 runs)
    """
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = Path(tmp.name)

    try:
        # Save model
        joblib.dump(pipeline, tmp_path, compress=3)
        size_kb = tmp_path.stat().st_size / 1024.0

        # Cold load latency
        cold_times = []
        for _ in range(5):
            t0 = time.perf_counter()
            _ = joblib.load(tmp_path)
            cold_times.append((time.perf_counter() - t0) * 1000.0)
        cold_latency_ms = float(np.median(cold_times))

        # Warm inference latency
        warm_times = []
        # Warmup
        for _ in range(20):
            pipeline.predict([sample_query])
        # Timed runs
        for _ in range(300):
            t0 = time.perf_counter()
            pipeline.predict([sample_query])
            warm_times.append((time.perf_counter() - t0) * 1000.0)
        warm_latency_ms = float(np.median(warm_times))

        return size_kb, cold_latency_ms, warm_latency_ms
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def plot_confusion_matrix(
    cm: np.ndarray,
    labels: list[str],
    lang: str,
    output_path: Path,
) -> None:
    """Generate and save a publication-quality confusion matrix plot."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Human-readable quadrant labels
    label_map = {
        "alta_negativa": "Alta Negativa\n(High Negative)",
        "alta_positiva": "Alta Positiva\n(High Positive)",
        "baja_negativa": "Baja Negativa\n(Low Negative)",
        "baja_positiva": "Baja Positiva\n(Low Positive)",
    }
    display_labels = [label_map.get(lbl, lbl) for lbl in labels]

    cm_normalized = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]

    fig, ax = plt.subplots(figsize=(7.5, 6), dpi=300)

    # Annotations with count and percentage
    annot = np.empty_like(cm, dtype=object)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f"{cm[i, j]}\n({cm_normalized[i, j]:.1%})"

    sns.heatmap(
        cm_normalized,
        annot=annot,
        fmt="",
        cmap="Blues",
        xticklabels=display_labels,
        yticklabels=display_labels,
        cbar_kws={"label": "Normalized Recall"},
        ax=ax,
        vmin=0.0,
        vmax=1.0,
        linewidths=1.0,
        linecolor="#e2e8f0",
    )

    lang_title = "Spanish Pipeline (`model_es.joblib`)" if lang == "es" else "English Pipeline (`model_en.joblib`)"
    ax.set_title(f"Confusion Matrix (5-Fold Stratified CV) — {lang_title}", fontsize=12, pad=16, fontweight="bold")
    ax.set_xlabel("Predicted Russell Quadrant", fontsize=11, labelpad=10, fontweight="semibold")
    ax.set_ylabel("True Russell Quadrant", fontsize=11, labelpad=10, fontweight="semibold")
    plt.xticks(rotation=0, fontsize=9)
    plt.yticks(rotation=0, fontsize=9)

    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close(fig)
    print(f"  [OK] Saved confusion matrix: {output_path.relative_to(BASE_DIR)}")


def evaluate_language(lang: str) -> dict[str, Any]:
    """Run full benchmark for a single language."""
    lang_name = "Spanish" if lang == "es" else "English"
    data_file = BASE_DIR / "data" / f"emotions_{lang}_v2.csv"
    sample_query = (
        "siento el pecho apretado y me hierve la sangre"
        if lang == "es"
        else "my chest is tight and my blood is boiling"
    )

    print(f"\n{'='*70}")
    print(f"  Benchmarking {lang_name} ({data_file.name})")
    print(f"{'='*70}")

    texts, labels = train_model.load_dataset(data_file)
    unique_labels = sorted(set(labels))
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    regression_probes = train_model.REGRESSION_PROBES.get(lang, [])
    held_out_probes = train_model.HELD_OUT_IDIOM_PROBES.get(lang, [])

    candidates = get_candidate_models(lang)
    results = []

    prod_cm = None
    prod_report = None

    for name, pipeline in candidates.items():
        print(f"  Evaluating {name}...")

        # 5-Fold Stratified Cross-Validation on In-Distribution data
        y_pred = cross_val_predict(pipeline, texts, labels, cv=cv)
        acc = accuracy_score(labels, y_pred)
        prec, rec, f1, _ = precision_recall_fscore_support(labels, y_pred, average="macro", zero_division=0)

        # Train on full dataset to measure out-of-distribution probes, size and latency
        pipeline.fit(texts, labels)
        reg_acc = train_model._score_probes(pipeline, regression_probes)
        held_out_acc = train_model._score_probes(pipeline, held_out_probes)

        size_kb, cold_ms, warm_ms = measure_latencies_and_size(pipeline, sample_query)

        if name == "Logistic Regression (Production)":
            prod_cm = confusion_matrix(labels, y_pred, labels=unique_labels)
            prod_report = classification_report(labels, y_pred, target_names=unique_labels, output_dict=True, zero_division=0)

        results.append({
            "model": name,
            "cv_accuracy": acc,
            "cv_precision_macro": prec,
            "cv_recall_macro": rec,
            "cv_f1_macro": f1,
            "regression_probe_acc": reg_acc,
            "held_out_probe_acc": held_out_acc,
            "size_kb": size_kb,
            "cold_latency_ms": cold_ms,
            "warm_latency_ms": warm_ms,
        })

    # Plot confusion matrix for production model
    cm_path = DOCS_EVAL_DIR / f"confusion-matrix-{lang}.png"
    if prod_cm is not None:
        plot_confusion_matrix(prod_cm, unique_labels, lang, cm_path)

    return {
        "lang": lang,
        "lang_name": lang_name,
        "sample_count": len(texts),
        "results": results,
        "production_cm": prod_cm,
        "production_report": prod_report,
        "labels": unique_labels,
    }


def format_markdown_table(benchmark_data: list[dict[str, Any]]) -> str:
    """Format benchmark metrics into markdown tables."""
    buf = io.StringIO()
    buf.write("### Model Benchmark Comparison Table\n\n")

    for item in benchmark_data:
        lang_name = item["lang_name"]
        lang = item["lang"]
        buf.write(f"#### {lang_name} Models (`data/emotions_{lang}_v2.csv`, N={item['sample_count']})\n\n")
        buf.write("| Model | CV Macro F1 | CV Precision | CV Recall | In-Dist Acc | Known Idioms Acc | Held-Out Idioms Acc | Model Size | Cold Load | Warm Infer |\n")
        buf.write("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |\n")

        for r in item["results"]:
            bold = "**" if "Production" in r["model"] else ""
            name = f"{bold}{r['model']}{bold}"
            buf.write(
                f"| {name} | {r['cv_f1_macro']:.3f} | {r['cv_precision_macro']:.3f} | {r['cv_recall_macro']:.3f} | "
                f"{r['cv_accuracy']:.1%} | {r['regression_probe_acc']:.1%} | {r['held_out_probe_acc']:.1%} | "
                f"{r['size_kb']:.1f} KB | {r['cold_latency_ms']:.1f} ms | {r['warm_latency_ms']:.2f} ms |\n"
            )
        buf.write("\n")

        # Per-quadrant table for production model
        buf.write(f"##### Production Model Per-Quadrant Breakdown ({lang_name})\n\n")
        buf.write("| Russell Quadrant | Precision | Recall | F1-Score | Support |\n")
        buf.write("| :--- | :---: | :---: | :---: | :---: |\n")
        rep = item["production_report"]
        for lbl in item["labels"]:
            metrics = rep.get(lbl, {})
            buf.write(
                f"| `{lbl}` | {metrics.get('precision', 0.0):.3f} | {metrics.get('recall', 0.0):.3f} | "
                f"{metrics.get('f1-score', 0.0):.3f} | {int(metrics.get('support', 0))} |\n"
            )
        macro = rep.get("macro avg", {})
        buf.write(f"| **Macro Average** | **{macro.get('precision', 0.0):.3f}** | **{macro.get('recall', 0.0):.3f}** | **{macro.get('f1-score', 0.0):.3f}** | **{int(macro.get('support', 0))}** |\n\n")

    return buf.getvalue()


def main() -> None:
    print("Emotion Finder — Rigorous Model Evaluation & Benchmark Suite")
    print("=" * 70)
    DOCS_EVAL_DIR.mkdir(parents=True, exist_ok=True)

    benchmark_data = []
    for lang in ["es", "en"]:
        res = evaluate_language(lang)
        benchmark_data.append(res)

    md_output = format_markdown_table(benchmark_data)
    results_path = DOCS_EVAL_DIR / "_benchmark_tables.md"
    results_path.write_text(md_output, encoding="utf-8")
    print(f"\n[OK] Benchmark tables written to: {results_path.relative_to(BASE_DIR)}")
    print("\n" + md_output)


if __name__ == "__main__":
    main()
