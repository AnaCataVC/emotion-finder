"""
Direct emotion matching within a quadrant.

Ranks a quadrant's 16 emotions by TF-IDF cosine similarity between the user's
raw text and each emotion's name + description (in the detected language).
When a confident direct match is found (score >= _MATCH_CONFIDENCE_THRESHOLD),
the app jumps straight to that emotion, while keeping the 4-step somatic decision
tree available as an exploratory fallback.
"""

from typing import Dict, Tuple, List, Any
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.metrics.pairwise import cosine_similarity

from decision_tree import QUADRANTS, get_quadrant_emotions

# Ensure NLTK stopwords are available
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    try:
        nltk.download("stopwords", quiet=True)
    except Exception:
        pass

# Minimum cosine similarity threshold to skip the manual tree.
# Calibrated with stopword filtering so neutral/out-of-domain sentences
# produce 0.0 and only genuine emotional lexical overlap triggers direct match.
_MATCH_CONFIDENCE_THRESHOLD = 0.35

try:
    _es_stopwords = set(stopwords.words("spanish"))
except Exception:
    _es_stopwords = {
        "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por",
        "un", "para", "con", "no", "una", "su", "al", "lo", "como", "mas", "pero",
        "sus", "le", "ya", "o", "este", "si", "porque", "esta", "entre", "cuando"
    }

try:
    _en_stopwords = set(stopwords.words("english"))
except Exception:
    _en_stopwords = set(ENGLISH_STOP_WORDS)

_STOPWORDS: Dict[str, List[str]] = {
    "es": sorted(_es_stopwords | {"como", "hacia", "durante", "mediante", "tras"}),
    "en": sorted(_en_stopwords | set(ENGLISH_STOP_WORDS)),
}

# Precompute vectorizers and document matrices per (quadrant, lang) on module load
# to eliminate runtime re-vectorization overhead on every HTTP request.
_CACHED_PROFILES: Dict[Tuple[str, str], Tuple[TfidfVectorizer, Any, List[Dict[str, Any]]]] = {}


def _init_cache() -> None:
    quadrants = QUADRANTS
    langs = ["es", "en"]

    for quad in quadrants:
        emotions = get_quadrant_emotions(quad)
        if not emotions:
            continue
        for lang in langs:
            name_key, desc_key = f"emotion_{lang}", f"description_{lang}"
            profiles = [f"{e.get(name_key, '')} {e.get(desc_key, '')}" for e in emotions]
            
            vec = TfidfVectorizer(
                stop_words=_STOPWORDS.get(lang, "english"),
                sublinear_tf=True,
            )
            matrix = vec.fit_transform(profiles)
            _CACHED_PROFILES[(quad, lang)] = (vec, matrix, emotions)


_init_cache()


def match_emotion(text: str, quadrant: str, lang: str) -> tuple[dict, float] | None:
    """
    Rank a quadrant's emotions by similarity to `text` and return the best match.

    Args:
        text: The user's raw input describing how they feel.
        quadrant: One of the four Russell quadrants.
        lang: 'es' or 'en' -- selects which name/description fields to compare against.

    Returns:
        (best_matching_emotion_node, similarity_score), or None if the quadrant
        has no emotions. A score below _MATCH_CONFIDENCE_THRESHOLD means the
        caller should fall back to the manual yes/no tree instead.
    """
    cached = _CACHED_PROFILES.get((quadrant, lang))
    if cached is None:
        return None

    vectorizer, matrix, emotions = cached

    query_vec = vectorizer.transform([text])
    # Zero overlapping content words with any emotion in this quadrant
    if query_vec.nnz == 0:
        return emotions[0], 0.0

    scores = cosine_similarity(query_vec, matrix)[0]
    best_idx = int(scores.argmax())
    best_score = float(scores[best_idx])

    return emotions[best_idx], best_score


def is_confident_match(score: float) -> bool:
    """Whether a match_emotion() score is strong enough to skip the manual tree."""
    return score >= _MATCH_CONFIDENCE_THRESHOLD
