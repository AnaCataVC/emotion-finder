> **Created:** 2026-09-04
> **Last Updated:** 2026-09-04
> **Topic:** ML Emotion Pipeline Architecture, Feature Engineering & Benchmarks

# ML Emotion Pipeline Architecture & Research

## 1. Executive Summary & Key Findings
- **Architecture**: Dual-pipeline TF-IDF Vectorizer + L2-regularized Logistic Regression with lightweight stopword-frequency language router (`inference.py`).
- **Footprint & Cold Starts**: Models compress to **~27 KB** via joblib (compression level 3), delivering **<1.5s cold-start** and **<5ms warm inference** within Vercel's Serverless Python runtime.
- **Stemming vs. Lemmatization**: SnowballStemmer (`'spanish'`) and PorterStemmer (`'english'`) provide near-instant deterministic tokenization, eliminating heavyweight spaCy dependencies.
- **Sentiment-Aware Stopwords**: Negations (`no`, `sin`, `nunca`, `not`, `without`) and intensifiers (`muy`, `demasiado`, `very`, `so`) are whitelisted from standard stopword pruning to avoid valence inversion.
- **Domain Meta-Word Neutralization**: Generic affective nouns (`emocion`, `sentimiento`, `sensacion`, `emotion`, `feeling`) are pruned to prevent spurious correlation shortcuts in idiomatic phrases.
- **Dialectal Idiom Integration**: Functional affective mapping integrates Chilean Spanish and British English collocations directly into Russell's 4 quadrants.
- **Dataset Scale**: ~650 curated, balanced samples per language in `data/emotions_es_v2.csv` and `data/emotions_en_v2.csv`.

## 2. Theoretical Grounding
- **Russell's Circumplex Model (1980)**: Emotional states decomposed into orthogonal dimensions of Valence ($\pm$) and Arousal/Activation ($\pm$).
- **Affective Dimensions**:
  - `alta_positiva`: High Activation $+$ Positive Valence
  - `alta_negativa`: High Activation $+$ Negative Valence
  - `baja_negativa`: Low Activation $+$ Negative Valence
  - `baja_positiva`: Low Activation $+$ Positive Valence

## 3. Serialization & Decoupled Preprocessing Invariant
- To prevent `AttributeError` or missing symbol references during joblib unpickling in serverless microVMs, custom tokenizers (`tokenize_es`, `tokenize_en`, `strip_accents`) are isolated in `preprocessing.py` and imported by `inference.py` prior to model loading.

## 4. Key Sources
- Russell, J. A. (1980). *A circumplex model of affect*. Journal of Personality and Social Psychology.
- Redondo, J., et al. (2007). *Spanish adaptation of the Affective Norms for English Words (ANEW)*.
- Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. ACL.
- scikit-learn & NLTK documentation.
