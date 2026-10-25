# Model & Data Card — Emotion Finder Affective Classifier

This card documents the architectural properties, training data provenance, operational parameters, ethical boundaries, and evaluation metrics for the Emotion Finder classification system.

---

## 1. Model Details

- **Model Name:** Emotion Finder Dual-Pipeline Affective Classifier
- **Model Type:** Supervised Natural Language Processing (NLP) text classification
- **Architecture:** Sublinear TF-IDF Vectorizer with unigram/bigram features (`ngram_range=(1, 2)`) + L2-regularized Multinomial Logistic Regression (`class_weight="balanced"`, L-BFGS solver)
- **Languages:** Spanish (`model_es.joblib`) and English (`model_en.joblib`)
- **Model Size:** $\approx 27\text{ KB}$ per language (serialized with `joblib` level 3 compression)
- **License:** MIT License
- **Target Platform:** Serverless Python ASGI runtime on Vercel (AWS Lambda microVMs)

---

## 2. Intended Use & Non-Goals

### Intended Use
- **Affective Self-Awareness & Introspection:** Translating vague bodily/cognitive sensations into Russell's 4 affective quadrants to facilitate structured somatic inquiry.
- **Educational Exploration:** Demonstrating how human-in-the-loop hybrid architectures combine lightweight machine learning with deterministic decision trees to overcome statistical NLP limitations.
- **Low-Latency Serverless Applications:** Delivering instantaneous ($<1\text{ ms}$) client-side transitions without relying on heavyweight container infrastructure or GPU acceleration.

### Explicit Non-Goals (Out-of-Scope)
- ❌ **Psychiatric or Psychological Diagnosis:** Emotion Finder is **not** a clinical diagnostic instrument, medical device, or therapeutic diagnostic tool. It cannot diagnose anxiety disorders, clinical depression, bipolar disorder, or other psychological conditions.
- ❌ **Crisis Intervention & Emergency Triage:** The system is **not** equipped for suicide prevention, crisis hotline triage, or acute mental health emergency management.
- ❌ **Forensic or Legal Sentiment Analysis:** The model must **not** be used for employee surveillance, workplace sentiment monitoring, legal interrogation analysis, or credit scoring.

---

## 3. Data Card & Dataset Provenance

### Dataset Summary
- **Primary Corpora:** `data/emotions_es_v2.csv` (Spanish) and `data/emotions_en_v2.csv` (English).
- **Volume:** 700 balanced rows per language (exactly 175 samples per quadrant: `alta_negativa`, `alta_positiva`, `baja_negativa`, `baja_positiva`).
- **Generation Strategy:** Deterministic combinatorial slot-filling generator (`data/generate_datasets.py`) combining somatic sensations, visceral cues, cognitive patterns, and dialectal collocations.
- **Sorting Invariant:** Sets are explicitly sorted (`phrases_list = sorted(phrases)`) prior to seed-based shuffling (`seed=42`) to guarantee byte-for-byte reproducibility across runs and environments.

### Feature Engineering & Sentiment Safeguards
1. **Sentiment-Aware Stopword Preservation:** Standard NLTK stopword pruning strips negations (`no`, `sin`, `nunca`, `not`, `without`, `never`) and intensifiers (`muy`, `demasiado`, `very`, `extremely`). In Emotion Finder, a custom whitelist preserves these tokens to prevent valence inversion (e.g., *"no me siento bien"* $\to$ inverted to positive).
2. **Uniform Intensifier Distribution:** Intensifiers and modifier prefixes are uniformly distributed across all four quadrants to prevent spurious correlation shortcuts where the model learns that *"muy"* implies high activation regardless of affective context.
3. **Domain Meta-Word Pruning:** Non-discriminative generic affective nouns (`emoción`, `sentimiento`, `sensación`, `emotion`, `feeling`) are pruned to avoid shortcut learning on collocations containing the word *"emoción"*.

---

## 4. Evaluation Splits & Generalization Bounds

| Evaluation Split | Volume | Spanish Model Performance | English Model Performance | Role & Interpretation |
| :--- | :---: | :---: | :---: | :--- |
| **In-Distribution Synthetic (5-Fold CV)** | 700 | Macro $F_1 = 0.991$ | Macro $F_1 = 0.984$ | Measures internal separability on template patterns. |
| **Dialectal Regression Probes** | 7 (ES) / 5 (EN) | $100\%$ Accuracy | $100\%$ Accuracy | Mandatory CI/CD regression gate on known idioms. |
| **Held-Out Unlearned Probes** | 7 (ES) / 5 (EN) | $14.3\%$ Accuracy | $40.0\%$ Accuracy | Honest measurement of bag-of-words ceiling on unseen figurative language. |

---

## 5. Known Biases & Model Limitations

1. **Bag-of-Words Vocabulary Ceiling:** Linear models operating on TF-IDF n-grams cannot perform zero-shot semantic reasoning on figurative phrases or cultural metaphors whose tokens do not appear in the vocabulary (e.g., *"estar chata"*, *"con los cables cruzados"*, *"seeing red"*).
2. **Dialectal Scope:** While specific Chilean Spanish and British English collocations are integrated, the model has limited coverage of regional vernaculars across other Spanish-speaking regions (e.g., Mexican, Argentine, Caribbean) and other English dialects (e.g., Australian, South African).
3. **Complex Syntactic Negation:** The model relies on bigram features (`ngram_range=(1, 2)`) to capture negation. Sentences with long-range dependency negations (e.g., *"no es que realmente sienta que esté completamente mal"*) may bypass bigram windows and cause misclassification.

---

## 6. When Predictions Should NOT Be Trusted

Predictions should be treated with skepticism or reviewed manually under the following conditions:
- **Low Confidence Flag (`low_confidence: true`):** When the top-1 posterior probability is $< 0.44$.
- **Small Quadrant Margin (`gap < 0.15`):** When the top-1 and top-2 probabilities are close, indicating mixed emotional activation (e.g., bittersweet sensations, agitated exhaustion).
- **Novel Idiomatic Speech:** When the input consists primarily of unseen cultural slang, internet colloquialisms, or abstract poetry without explicit somatic or physiological markers.
- **Mixed-Valence States:** Sarcasm, irony, or simultaneous positive and negative feelings (e.g., *"nervioso pero emocionado"*).

---

## 7. Human-in-the-Loop Safeguards & Anti-Poisoning Architecture

To defend against public manipulation, troll attacks, and data poisoning while learning from real user corrections, the system enforces a strict four-layer defense:

```
  User Correction Submission (/feedback or /tree)
                         │
                         ▼
  [Layer 1: Network-Anchored Rate Limiting]
  Max 5 submissions/hour per IP + User-Agent session hash (FeedbackStore.count_since)
                         │
                         ▼
  [Layer 2: Hybrid Curation Lifecycle]
  Thumbs-Up Validations ──> status = 'verified' (auto-promoted for retrain)
  Corrections / Dislikes ──> status = 'pending' (requires CLI curate_feedback.py review)
                         │
                         ▼
  [Layer 3: Volume & Deduplication Cap]
  10% maximum dataset expansion cap (max 70 feedback rows on 700 baseline rows)
  Deduplication by normalized string; length gating (6–300 characters)
                         │
                         ▼
  [Layer 4: Automated CI/CD Regression Quality Gate]
  Weekly retraining (.github/workflows/retrain.yml) runs in dry-run first
  Requires 5-Fold Stratified CV Macro F1 >= 0.95 AND 100% pass on REGRESSION_PROBES
```
