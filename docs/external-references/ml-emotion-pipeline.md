> **Created:** 2026-09-04
> **Last Updated:** 2026-09-05
> **Topic:** ML Emotion Pipeline Architecture, Feature Engineering & Benchmarks

# ML Emotion Pipeline Architecture & Research

## 1. Executive Summary & Key Findings
- **Architecture**: Dual-pipeline TF-IDF Vectorizer (ngram_range=(1, 2)) + L2-regularized Logistic Regression with lightweight stopword-frequency language router (`inference.py`), followed by a precomputed, stopword-filtered TF-IDF cosine-similarity matcher (`emotion_matcher.py`) that ranks the 16 emotions within the predicted quadrant.
- **Footprint & Cold Starts**: Models compress to **~27 KB** via joblib (compression level 3), delivering **<1.5s cold-start** and **<5ms warm inference** within Vercel's Serverless Python runtime.
- **Stemming vs. Lemmatization**: SnowballStemmer (`'spanish'`) and PorterStemmer (`'english'`) provide near-instant deterministic tokenization, eliminating heavyweight spaCy dependencies.
- **Sentiment-Aware Stopwords**: Negations (`no`, `sin`, `nunca`, `not`, `without`) and intensifiers (`muy`, `demasiado`, `very`, `so`) are whitelisted from standard stopword pruning to avoid valence inversion, while intensifiers are uniformly distributed across all quadrants to prevent false valence attraction.
- **Domain Meta-Word Neutralization**: Generic affective nouns (`emocion`, `sentimiento`, `sensacion`, `emotion`, `feeling`) are pruned to prevent spurious correlation shortcuts in idiomatic phrases.
- **Dialectal Idiom Integration**: Functional affective mapping integrates Chilean Spanish and British English collocations directly into Russell's 4 quadrants.
- **Dataset Scale**: 700 curated, balanced samples per language (175/quadrant) in `data/emotions_es_v2.csv` and `data/emotions_en_v2.csv`, generated deterministically by `data/generate_datasets.py`.

## 2. Theoretical Grounding
- **Russell's Circumplex Model (1980)**: Emotional states decomposed into orthogonal dimensions of Valence ($\pm$) and Arousal/Activation ($\pm$).
- **Affective Dimensions**:
  - `alta_positiva`: High Activation $+$ Positive Valence
  - `alta_negativa`: High Activation $+$ Negative Valence
  - `baja_negativa`: Low Activation $+$ Negative Valence
  - `baja_positiva`: Low Activation $+$ Positive Valence

## 3. Serialization & Decoupled Preprocessing Invariant
- To prevent `AttributeError` or missing symbol references during joblib unpickling in serverless microVMs, custom tokenizers (`tokenize_es`, `tokenize_en`, `strip_accents`) are isolated in `preprocessing.py` and imported by `inference.py` prior to model loading.

## 4. Semantic Emotion Matcher (`emotion_matcher.py`)
Instead of always walking the user through four generic yes/no somatic questions to reach 1 of 16 emotions per quadrant, `/predict` first ranks the quadrant's 16 emotions by TF-IDF cosine similarity between the user's own words and each emotion's name + description. To eliminate per-request overhead and prevent false-positive hijacking by neutral phrases (*"computadora de escritorio"*, *"please review PR"*), the matcher precomputes static TF-IDF document matrices on module initialization, enforces comprehensive stopword filtering, and requires both content-word overlap (`nnz > 0`) and a calibrated similarity threshold of `_MATCH_CONFIDENCE_THRESHOLD = 0.35`. Confident direct matches jump straight to that emotion, while weak or neutral inputs cleanly fall back to the interactive binary tree in `decision_tree.py`. The tree also remains accessible from the direct-match result card via an "explore manually" button.

## 5. Dataset Generation Determinism
`data/generate_datasets.py` collects generated phrases in a `set()` before shuffling with a fixed `random.seed(42)`. A Python `set`'s iteration order depends on the process's hash seed, which is randomized by default and is *not* controlled by `random.seed` — so two runs of the "reproducible" generator produced different CSVs despite the fixed seed (verified: ~65% of rows differed between consecutive runs). Fixed by sorting the set (`sorted(phrases)`) before shuffling, restoring true run-to-run reproducibility. This was a pre-existing bug (present since the initial commit), not something introduced by a later change — it explains why re-running the pipeline could silently swap out the specific idioms and phrasings a trained model happened to cover.

## 6. Regression vs. Held-Out Idiom Probes (`train_model.py`)
CV `f1_macro` on the templated synthetic data saturates near 1.0 for almost any hyperparameter choice, so it cannot tell a genuinely-generalizing config apart from one that overfits the combinatorial template structure. `train_model.py` keeps two small probe sets to compensate:
- **`REGRESSION_PROBES`**: dialectal idioms whose key vocabulary was deliberately added to `generate_datasets.py`'s templates after they kept failing during evaluation. Used to pick between `GridSearchCV`'s best hyperparameters and the previous defaults (only switches if the tuned config doesn't regress here). No longer a generalization test in the strict sense — its vocabulary is now in-distribution — but still catches an overfit hyperparameter choice.
- **`HELD_OUT_IDIOM_PROBES`**: a second, disjoint set of dialectal idioms whose vocabulary is deliberately kept out of the training templates. Printed as a training-time diagnostic only (never used to pick hyperparameters — every configuration scores similarly low, so the comparison carries no signal). Measured accuracy: **~29% (ES) / ~40% (EN)** on phrasing the model has truly never seen. This is treated as an honest ceiling of a bag-of-words TF-IDF classifier on figurative language, not a bug to chase by adding each new failing idiom's vocabulary to training one at a time — that pattern (memorize the specific failing phrase, watch the next one fail) is exactly what this split is meant to make visible instead of hiding.

## 7. Related References
- [Dialectal Idioms & Affective Mapping](dialectal-idioms-affective-mapping.md)
- [FastHTML Stack & Serverless Deployment](fasthtml-stack.md)
- [Adversarial Audit & Robustness Learnings](../learning/adversarial-audit-lessons.md)

## 8. Key Sources
- Russell, J. A. (1980). *A circumplex model of affect*. Journal of Personality and Social Psychology.
- Redondo, J., et al. (2007). *Spanish adaptation of the Affective Norms for English Words (ANEW)*.
- Demszky, D., et al. (2020). *GoEmotions: A Dataset of Fine-Grained Emotions*. ACL.
- scikit-learn & NLTK documentation.
