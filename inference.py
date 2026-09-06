"""
Inference module for emotion quadrant prediction.

Loads pre-trained scikit-learn pipelines at module level for fast
serverless invocation (global scope caching pattern).

Functions:
    predict_quadrant(text) - Classify text into a Russell quadrant.
    detect_language(text)  - Lightweight language detection heuristic.
"""

from pathlib import Path

import joblib

# Import preprocessing so unpickler resolves tokenize_es and tokenize_en
import preprocessing

# ---------------------------------------------------------------------------
# Global model loading (cold-start caching for serverless)
# ---------------------------------------------------------------------------

_BASE_DIR = Path(__file__).resolve().parent
_MODELS_DIR = _BASE_DIR / "models"

# Load models once at import time; persist across warm invocations
_model_es = None
_model_en = None


def _load_models() -> None:
    """Lazy-load models on first use."""
    global _model_es, _model_en

    es_path = _MODELS_DIR / "model_es.joblib"
    en_path = _MODELS_DIR / "model_en.joblib"

    if es_path.exists() and _model_es is None:
        _model_es = joblib.load(es_path)

    if en_path.exists() and _model_en is None:
        _model_en = joblib.load(en_path)


# ---------------------------------------------------------------------------
# Language detection (lightweight heuristic)
# ---------------------------------------------------------------------------

_ES_MARKERS = frozenset({
    "el", "la", "los", "las", "un", "una", "unos", "unas",
    "de", "del", "en", "que", "por", "para", "con", "como",
    "estoy", "tengo", "siento", "me", "mi", "muy", "pero",
    "es", "está", "son", "hay", "todo", "nada", "sin",
})

_EN_MARKERS = frozenset({
    "the", "a", "an", "in", "on", "at", "for", "with",
    "that", "this", "am", "is", "are", "was", "were",
    "have", "has", "feel", "feeling", "my", "i", "and",
    "but", "very", "really", "so", "just", "like", "can",
})


def detect_language(text: str) -> str:
    """
    Detect whether the input text is Spanish or English.

    Uses a fast stopword-frequency heuristic. Defaults to 'es'
    when the language cannot be determined.

    Args:
        text: Input text string.

    Returns:
        'es' for Spanish, 'en' for English.
    """
    tokens = set(text.lower().split())

    es_score = len(tokens & _ES_MARKERS)
    en_score = len(tokens & _EN_MARKERS)

    if en_score > es_score:
        return "en"
    # Default to Spanish (primary language of the app)
    return "es"


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

# Minimum confidence (max class probability) to accept a prediction, below
# which it's reported as "uncertain". 0.35 represents a clear margin over
# the 0.25 random 4-class uniform baseline while accepting nuanced phrasing.
_CONFIDENCE_THRESHOLD = {"es": 0.35, "en": 0.35}

# Top1-top2 probability gap below which the secondary-emotion-path affordance
# triggers (offering the runner-up quadrant's tree as an alternate reading).
_SECONDARY_GAP_THRESHOLD = {"es": 0.40, "en": 0.50}

# Confidence-to-intensity display bucketing
_INTENSITY_LOW_MAX = {"es": 0.60, "en": 0.60}
_INTENSITY_MED_MAX = {"es": 0.85, "en": 0.85}


def _bucket_intensity(confidence: float, lang: str) -> str:
    if confidence < _INTENSITY_LOW_MAX[lang]:
        return "baja"
    if confidence < _INTENSITY_MED_MAX[lang]:
        return "media"
    return "alta"


def _top2_quadrant(prob_dict: dict, primary: str) -> str | None:
    """Runner-up quadrant by raw probability, independent of any gap threshold.

    Used both by _find_secondary_quadrant() (automatic, gap-gated) and by the
    explicit quadrant-rejection flow in main.py (forced, gap-agnostic — the
    user already said the top prediction is wrong, so the gap no longer
    matters).
    """
    ranked = sorted(prob_dict.items(), key=lambda kv: kv[1], reverse=True)
    if len(ranked) < 2:
        return None
    (top1_label, _), (top2_label, _) = ranked[0], ranked[1]
    if top1_label != primary:
        return None  # defensive: should always match
    return top2_label


def _find_secondary_quadrant(prob_dict: dict, primary: str, lang: str) -> str | None:
    top2 = _top2_quadrant(prob_dict, primary)
    if top2 is None:
        return None
    if (prob_dict[primary] - prob_dict[top2]) < _SECONDARY_GAP_THRESHOLD[lang]:
        return top2
    return None


def predict_quadrant(text: str) -> dict:
    """
    Predict the emotional quadrant for a given text input.

    Automatically detects language and uses the appropriate model.
    Returns a confidence score and a low-confidence flag when the
    model is uncertain.

    Args:
        text: A short phrase describing how the user feels.

    Returns:
        A dict with keys:
            - quadrant: str (predicted quadrant or 'uncertain')
            - confidence: float (0.0 to 1.0)
            - lang: str ('es' or 'en')
            - low_confidence: bool (True if below threshold)
            - probabilities: dict[str, float] (per-class probabilities)
            - intensity: str | None ('baja'/'media'/'alta', None if uncertain)
            - secondary_quadrant: str | None (runner-up quadrant when the top-2
              probabilities are close; None if uncertain or clearly decided)
            - runner_up_quadrant: str | None (runner-up quadrant regardless of
              gap size; None only when uncertain. Used to offer an alternate
              quadrant when the user explicitly rejects the top prediction)
    """
    _load_models()

    # Sanitize input
    text = text.strip()
    if not text or len(text) < 2:
        return {
            "quadrant": "uncertain",
            "confidence": 0.0,
            "lang": "es",
            "low_confidence": True,
            "probabilities": {},
            "intensity": None,
            "secondary_quadrant": None,
            "runner_up_quadrant": None,
        }

    # Truncate very long inputs
    if len(text) > 500:
        text = text[:500]

    # Detect language and select model
    lang = detect_language(text)
    model = _model_es if lang == "es" else _model_en

    # Fallback: if target-language model is missing, try the other one
    if model is None:
        model = _model_en if lang == "es" else _model_es
        lang = "en" if lang == "es" else "es"

    if model is None:
        return {
            "quadrant": "uncertain",
            "confidence": 0.0,
            "lang": lang,
            "low_confidence": True,
            "probabilities": {},
            "intensity": None,
            "secondary_quadrant": None,
            "runner_up_quadrant": None,
        }

    # Predict
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    prob_dict = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
    confidence = float(max(probabilities))

    low_confidence = bool(confidence < _CONFIDENCE_THRESHOLD[lang])
    primary = str(prediction)

    return {
        "quadrant": primary if not low_confidence else "uncertain",
        "confidence": round(confidence, 4),
        "lang": lang,
        "low_confidence": low_confidence,
        "probabilities": prob_dict,
        "intensity": None if low_confidence else _bucket_intensity(confidence, lang),
        "secondary_quadrant": (
            None if low_confidence else _find_secondary_quadrant(prob_dict, primary, lang)
        ),
        "runner_up_quadrant": (
            None if low_confidence else _top2_quadrant(prob_dict, primary)
        ),
    }

