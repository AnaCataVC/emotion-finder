"""
Emotion classification training script.

Trains TF-IDF + Logistic Regression pipelines for Spanish and English
emotion quadrant classification based on Russell's Circumplex Model of Affect.

Usage:
    python train_model.py

Output:
    models/model_es.joblib  - Spanish classification pipeline
    models/model_en.joblib  - English classification pipeline
"""

import csv
import sys
from pathlib import Path

# Ensure UTF-8 output on Windows consoles
if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import joblib
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import Pipeline


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RANDOM_STATE = 42

# Ensure NLTK data is available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords", quiet=True)

# ---------------------------------------------------------------------------
# Sentiment-aware stopword lists
# ---------------------------------------------------------------------------

# Words that MUST be preserved because they carry valence or arousal signals
_EXCLUDED_FROM_STOPWORDS = {
    # Negations (reverse valence)
    "no", "ni", "sin", "nada", "nunca", "jamas", "jamás", "tampoco", "contra",
    "not", "no", "never", "neither", "nor", "nothing", "without", "cannot",
    # Intensifiers / modulators
    "muy", "mas", "más", "mucho", "mucha", "muchos", "muchas",
    "poco", "poca", "pocos", "pocas", "demasiado", "demasiada",
    "tan", "tanto", "bastante",
    "very", "much", "more", "most", "really", "so", "too", "extremely",
    "quite", "rather", "fairly", "pretty", "little", "less", "least",
    # Core sentiment words that stop-word lists sometimes include
    "bien", "mal", "bueno", "malo", "sentir", "siento",
    "good", "bad", "well", "feel", "feeling",
}


from preprocessing import (
    strip_accents,
    tokenize_es,
    tokenize_en,
    stemmer_es,
    stemmer_en,
)


# Domain meta-words that don't indicate valence or arousal on their own
_DOMAIN_META_STOPWORDS_ES = {
    "emocion", "emociones", "emocional", "emocionales",
    "sentimiento", "sentimientos", "sensacion", "sensaciones",
}
_DOMAIN_META_STOPWORDS_EN = {
    "emotion", "emotions", "emotional",
    "feeling", "feelings", "sensation", "sensations",
}


def _build_spanish_stopwords() -> list[str]:
    """Build a sentiment-safe Spanish stopword list with stems."""
    base = set(stopwords.words("spanish"))
    safe = base - _EXCLUDED_FROM_STOPWORDS
    stemmed = {stemmer_es.stem(strip_accents(w)) for w in safe}
    stemmed.update(stemmer_es.stem(strip_accents(w)) for w in _DOMAIN_META_STOPWORDS_ES)
    return sorted(stemmed)


def _build_english_stopwords() -> list[str]:
    """Build a sentiment-safe English stopword list with stems."""
    base = set(stopwords.words("english"))
    safe = base - _EXCLUDED_FROM_STOPWORDS
    stemmed = {stemmer_en.stem(w.lower()) for w in safe}
    stemmed.update(stemmer_en.stem(w.lower()) for w in _DOMAIN_META_STOPWORDS_EN)
    return sorted(stemmed)




# ---------------------------------------------------------------------------
# Pipeline builders
# ---------------------------------------------------------------------------


def build_pipeline(lang: str) -> Pipeline:
    """
    Build a TF-IDF + Logistic Regression pipeline for the given language.

    Args:
        lang: 'es' for Spanish, 'en' for English.

    Returns:
        A scikit-learn Pipeline ready for .fit().
    """
    if lang == "es":
        tokenizer = tokenize_es
        stop_words = _build_spanish_stopwords()
    elif lang == "en":
        tokenizer = tokenize_en
        stop_words = _build_english_stopwords()
    else:
        raise ValueError(f"Unsupported language: {lang}")

    return Pipeline([
        ("tfidf", TfidfVectorizer(
            tokenizer=tokenizer,
            stop_words=stop_words,
            ngram_range=(1, 2),
            sublinear_tf=True,
            max_features=3000,
            min_df=1,
            token_pattern=None,  # Disable default pattern when custom tokenizer is used
        )),
        ("clf", LogisticRegression(
            C=1.0,
            class_weight="balanced",
            solver="lbfgs",
            max_iter=1000,
            random_state=RANDOM_STATE,
        )),
    ])



# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_dataset(filepath: Path) -> tuple[list[str], list[str]]:
    """
    Load a CSV dataset and return (texts, labels).

    Expects columns: text, activation, valence, quadrant
    Uses 'quadrant' as the target label.

    Args:
        filepath: Path to the CSV file.

    Returns:
        Tuple of (texts, quadrant_labels).
    """
    texts: list[str] = []
    labels: list[str] = []

    with open(filepath, encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = row["text"].strip()
            quadrant = row["quadrant"].strip()
            if text and quadrant:
                texts.append(text)
                labels.append(quadrant)

    return texts, labels


# ---------------------------------------------------------------------------
# Training & evaluation
# ---------------------------------------------------------------------------


def train_and_evaluate(lang: str, data_path: Path, model_path: Path) -> None:
    """
    Train a pipeline, evaluate with cross-validation, and save the model.

    Args:
        lang: Language code ('es' or 'en').
        data_path: Path to the training CSV.
        model_path: Path to save the trained model.
    """
    lang_name = "Spanish" if lang == "es" else "English"
    print(f"\n{'='*60}")
    print(f"  Training {lang_name} Pipeline")
    print(f"{'='*60}")

    # Load data
    texts, labels = load_dataset(data_path)
    print(f"  Loaded {len(texts)} samples from {data_path.name}")

    # Class distribution
    from collections import Counter
    dist = Counter(labels)
    print(f"  Class distribution:")
    for cls, count in sorted(dist.items()):
        print(f"    {cls}: {count} ({count/len(labels)*100:.1f}%)")

    # Cross-validation evaluation
    pipeline = build_pipeline(lang)
    n_splits = min(5, min(dist.values()))  # Adapt folds to smallest class
    if n_splits >= 2:
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
        y_pred = cross_val_predict(pipeline, texts, labels, cv=cv)

        print(f"\n  Cross-Validation Results ({n_splits}-fold):")
        print(classification_report(labels, y_pred, zero_division=0))

        # Confusion matrix
        unique_labels = sorted(set(labels))
        cm = confusion_matrix(labels, y_pred, labels=unique_labels)
        print("  Confusion Matrix:")
        header = "  " + " ".join(f"{lbl[:8]:>10}" for lbl in unique_labels)
        print(header)
        for i, row in enumerate(cm):
            row_str = " ".join(f"{val:>10}" for val in row)
            print(f"  {unique_labels[i][:8]:>10} {row_str}")
    else:
        print("  [WARN] Too few samples per class for cross-validation. Training on full dataset.")

    # Train final model on full dataset
    final_pipeline = build_pipeline(lang)
    final_pipeline.fit(texts, labels)

    # Save model
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(final_pipeline, model_path, compress=3)

    model_size_kb = model_path.stat().st_size / 1024
    print(f"\n  [OK] Model saved: {model_path.name} ({model_size_kb:.1f} KB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    """Train both Spanish and English emotion classification pipelines."""
    print("Emotion Finder -- Model Training Script")
    print("=" * 60)

    # Spanish pipeline
    es_data = DATA_DIR / "emotions_es_v2.csv"
    es_model = MODELS_DIR / "model_es.joblib"

    if es_data.exists():
        train_and_evaluate("es", es_data, es_model)
    else:
        print(f"  [WARN] Spanish dataset not found: {es_data}")

    # English pipeline
    en_data = DATA_DIR / "emotions_en_v2.csv"
    en_model = MODELS_DIR / "model_en.joblib"

    if en_data.exists():
        train_and_evaluate("en", en_data, en_model)
    else:
        print(f"  [WARN] English dataset not found: {en_data}")

    print(f"\n{'='*60}")
    print("Training complete!")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
