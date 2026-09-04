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

# Minimum confidence threshold to accept a prediction
_CONFIDENCE_THRESHOLD = 0.35

# Valid quadrant values
_VALID_QUADRANTS = frozenset({
    "alta_positiva", "alta_negativa",
    "baja_positiva", "baja_negativa",
})


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
        }

    # Predict
    prediction = model.predict([text])[0]
    probabilities = model.predict_proba([text])[0]
    classes = model.classes_

    prob_dict = {str(cls): round(float(prob), 4) for cls, prob in zip(classes, probabilities)}
    confidence = float(max(probabilities))

    low_confidence = bool(confidence < _CONFIDENCE_THRESHOLD)

    return {
        "quadrant": str(prediction) if not low_confidence else "uncertain",
        "confidence": round(confidence, 4),
        "lang": lang,
        "low_confidence": low_confidence,
        "probabilities": prob_dict,
    }

