> **Created:** 2026-09-05  
> **Topic:** Architectural Learnings: Adversarial Audit & Robustness Engineering  
> **Status:** Active / Long-Term Knowledge  

# Architectural Learnings: Adversarial Audit & Robustness Engineering

## 1. Executive Summary

During the hardening and adversarial verification of Emotion Finder's hybrid classification pipeline, three distinct NLP failure modes were identified and resolved:
1. **Stopword collision in TF-IDF direct emotion matching** allowing neutral out-of-domain sentences to trigger emotion matches.
2. **Synthetic dataset shortcut learning** where combinatorial templates saturated cross-validation metrics while masking severe brittleness on held-out dialectal idioms.
3. **Intensifier valence inversion** where unbalanced intensifiers (*"muy"*, *"very"*) overpowered negative valence keywords (*"triste"*, *"sad"*), flipping predictions across the circumplex axis.

This document formalizes the mechanisms, failure modes, and permanent architectural defenses implemented across the codebase.

---

## 2. Stopword Collision & Zero-Overlap Gating

### Vulnerability Analysis
In `emotion_matcher.py`, direct matching within a quadrant calculates TF-IDF cosine similarity between the user's raw text and the 16 emotion descriptions. If stopwords are retained, common function words (*"el"*, *"la"*, *"de"*, *"en"*, *"the"*, *"at"*, *"with"*) overlap between neutral input and emotion definitions:
- Query: *"computadora de escritorio"* (Neutral)
- Target: *"Ira: Sentimiento de enojo e indignación violenta..."*
- Result: Overlap on *"de"* yielded a similarity score of ~0.15–0.22.

With an uncalibrated low threshold (e.g. `0.10`), completely irrelevant text bypassed the 4-step somatic decision tree and falsely triggered a direct emotion hit.

### Architectural Defense
1. **Comprehensive Bilingual Stopwords (`_STOPWORDS`):** Combines NLTK base stopwords with domain function prepositions (`"como"`, `"hacia"`, `"durante"`, `"mediante"`, `"tras"`).
2. **Zero-Overlap Gating:**
   ```python
   query_vec = vectorizer.transform([text])
   if query_vec.nnz == 0:
       return emotions[0], 0.0
   ```
   If the preprocessed query contains no overlapping content words with the quadrant's vocabulary, the similarity is forced to `0.0`.
3. **Calibrated Threshold (`_MATCH_CONFIDENCE_THRESHOLD = 0.35`):** Calibrated against stopword-filtered profiles so that only inputs with authentic semantic overlap trigger direct leaf rendering.

---

## 3. Synthetic Dataset Shortcut Learning & Dual-Probe Splitting

### Vulnerability Analysis
Generating datasets from slot-filling templates (`[prefix] + [action] + [qualifier]`) creates two major risks:
1. **Domain Meta-Word Shortcuts:** Generic words like *"emociones"*, *"sentimientos"*, or *"feelings"* clustered disproportionately in certain quadrants. Logistic regression assigned heavy weights to the meta-word rather than affective indicators.
2. **CV F1 Saturation & False Optimism:** 5-fold Stratified CV on templated data yielded $F_1 = 1.00$ for nearly all hyperparameter combinations, rendering grid search unable to distinguish robust models from those memorizing template artifacts. An initial grid selected `ngram_range=(1, 3)` and `C=0.1`, which subsequently failed 7 out of 12 real colloquial idioms.
3. **Set Iteration Non-Determinism:** Storing template outputs in an un-ordered `set` caused `random.seed(42)` to shuffle different lists across runs due to Python's randomized hash seed, varying ~65% of rows between generator executions.

### Architectural Defense
1. **Meta-Word Pruning:** Explicit stopword removal for domain meta-words (`_DOMAIN_META_STOPWORDS_ES`, `_DOMAIN_META_STOPWORDS_EN`).
2. **Deterministic Sampling:** Enforcing `phrases_list = sorted(phrases)` in `data/generate_datasets.py` prior to shuffling ensures identical 700-row datasets across all environments.
3. **Dual-Probe Evaluation Split (`train_model.py`):**
   - **`REGRESSION_PROBES`:** Known idioms incorporated into training templates used strictly to prevent hyperparameter regressions.
   - **`HELD_OUT_IDIOM_PROBES`:** Fully disjoint, unlearned idioms used to track the genuine generalization ceiling (~29% ES / ~40% EN) of bag-of-words classifiers on novel figurative speech without falling into endless template patch churn.

---

## 4. Intensifier Valence Inversion & Negation Integrity

### Vulnerability Analysis
Standard stopword pruning removes negations (`no`, `not`, `without`) and intensifiers (`muy`, `demasiado`, `very`, `extremely`). This produces two critical failures:
1. **Negation Flipping:** *"no me siento bien"* is stripped to *"bien"*, classifying a cry of distress as `alta_positiva` or `baja_positiva`.
2. **Intensifier Skew:** In templated datasets, if *"muy"* or *"very"* appears more frequently in positive or high-arousal sentences, the linear classifier assigns a positive coefficient to the unigram. In user inputs like *"estoy muy triste"*, the positive weight of *"muy"* overwhelmed the negative weight of *"triste"*, misclassifying the expression.

### Architectural Defense
1. **Sentiment-Aware Whitelist (`_EXCLUDED_FROM_STOPWORDS`):** Preserves all negation particles and intensifiers during stopword generation.
2. **Mandatory Bigrams (`ngram_range=(1, 2)`):** Binds intensifiers and negations directly to affective stems (`"no bien"`, `"muy trist"`), generating distinct feature columns.
3. **Balanced Generator Distribution:** In `data/generate_datasets.py`, intensifiers and padding prefixes are uniformly distributed across all 4 quadrants (high/low arousal, positive/negative valence).

---

## 5. Verification Test Suite Matrix

These invariants are enforced by automated test suites in `tests/test_pipeline.py`:

| Test Suite | Target Invariant | Assertion |
| :--- | :--- | :--- |
| `test_adversarial_neutral_sentences_rejected_by_matcher` | Stopword collision & zero overlap | Score `< 0.35` or `match is None` for neutral queries. |
| `test_adversarial_core_emotion_vocabulary` | Intensifier balance | *"estoy muy triste"*, *"I am very angry"* map strictly to correct negative quadrants. |
| `test_adversarial_negation_handling` | Negation preservation | *"no me siento bien"*, *"I am not happy"* classify strictly into negative quadrants. |
| `test_adversarial_high_confidence_spanish` | Low-confidence calibration | High-arousal colloquial input does not trigger false `uncertain` state. |
| `test_adversarial_secondary_emotion_sanity` | Secondary gap threshold | Unambiguous queries do not spuriously trigger the secondary quadrant affordance. |

---

## 6. Related Documentation
- [ML Emotion Pipeline Architecture](../external-references/ml-emotion-pipeline.md)
- [Dialectal Idioms & Affective Mapping](../external-references/dialectal-idioms-affective-mapping.md)
- [FastHTML Stack & Serverless Deployment](../external-references/fasthtml-stack.md)
