"""
Text preprocessing and tokenization for Emotion Finder NLP pipelines.

Provides stemming tokenizers and accent stripping for Spanish and English
text classification. Kept in a standalone module so that serialized
scikit-learn pipelines can reliably resolve tokenizer functions across
training and inference environments.
"""

import unicodedata
from nltk.stem import PorterStemmer
from nltk.stem.snowball import SnowballStemmer

_stemmer_es = SnowballStemmer("spanish")
_stemmer_en = PorterStemmer()

# Public references
stemmer_es = _stemmer_es
stemmer_en = _stemmer_en



def strip_accents(text: str) -> str:
    """Normalize unicode and strip diacritical marks."""
    nfkd = unicodedata.normalize("NFKD", text)
    return nfkd.encode("ascii", "ignore").decode("utf-8")


def tokenize_es(text: str) -> list[str]:
    """Tokenize and stem Spanish text with accent normalization."""
    text = strip_accents(text.lower())
    tokens = text.split()
    return [_stemmer_es.stem(tok) for tok in tokens if tok.isalnum()]


def tokenize_en(text: str) -> list[str]:
    """Tokenize and stem English text."""
    text = text.lower()
    tokens = text.split()
    return [_stemmer_en.stem(tok) for tok in tokens if tok.isalnum()]
