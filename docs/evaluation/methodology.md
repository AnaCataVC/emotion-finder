# Evaluation Methodology & Benchmark Protocol

This document formalizes the experimental design, evaluation splits, benchmark metrics, and operational constraints used to validate the Emotion Finder affective classification pipelines.

---

## 1. Theoretical Grounding & Target Variables

Emotion Finder grounds affective categorization in **Russell's Circumplex Model of Affect (1980)**, which posits that all emotional states arise from two independent, orthogonal neurophysiological dimensions:
1. **Valence** ($\pm$): Cognitive evaluation of pleasantness vs. unpleasantness.
2. **Arousal / Physiological Activation** ($\pm$): Autonomic sympathetic nervous system energy (high motor activation vs. low motor deactivation).

The NLP classifier models this space as a 4-class discrete classification task corresponding to the four Cartesian quadrants:

| Label | Russell Quadrant | Valence | Activation | Core Somatic Manifestations |
| :--- | :--- | :---: | :---: | :--- |
| `alta_negativa` | High Negative | $-$ | $+$ | Tight chest, accelerated pulse, muscle tension, rage, panic, anxiety |
| `alta_positiva` | High Positive | $+$ | $+$ | Surge of warmth, elevated heart rate, urge to move/jump, euphoria, joy |
| `baja_negativa` | Low Negative | $-$ | $-$ | Heavy limbs, somatic slump, hollow stomach, desolation, sadness, apathy |
| `baja_positiva` | Low Positive | $+$ | $-$ | Slow deep breathing, muscle release, grounded stillness, calm, serenity |

---

## 2. Dataset Partitioning & The Dual-Probe Methodology

Affective state datasets generated via combinatorial slot-filling templates (`data/generate_datasets.py`) exhibit high internal cohesion. While templates guarantee balanced quadrant coverage (175 samples per quadrant; $N = 700$ total per language), they introduce a known evaluation risk: **cross-validation metric saturation**.

To prevent misleading claims of near-perfect accuracy and to expose the true limits of n-gram linear models, the evaluation protocol establishes three distinct tiers:

```
                               Evaluation Tiers
  ┌────────────────────────────────────────────────────────────────────────┐
  │ 1. In-Distribution Synthetic (N=700)                                  │
  │    5-Fold Stratified Cross-Validation (emotions_es_v2 / emotions_en_v2)│
  │    Measures internal separability and template representation          │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────┴────────────────────────────────────┐
  │ 2. Dialectal Regression Probes (In-Distribution Known Collocations)    │
  │    Known Chilean and British idioms incorporated into templates        │
  │    Enforced as non-negotiable CI/CD gate (100% pass required)          │
  └───────────────────────────────────┬────────────────────────────────────┘
                                      │
  ┌───────────────────────────────────┴────────────────────────────────────┐
  │ 3. Held-Out Generalization Probes (Out-of-Distribution Figurative)     │
  │    Zero vocabulary overlap with training templates                     │
  │    Monitors the real bag-of-words inductive ceiling (~14%–40%)         │
  └────────────────────────────────────────────────────────────────────────┘
```

### Tier 1: In-Distribution Synthetic Corpora ($N = 700$)
- **Spanish Corpus (`data/emotions_es_v2.csv`)**: 700 samples (175/quadrant), sorted deterministically before shuffling to ensure exact environment reproducibility.
- **English Corpus (`data/emotions_en_v2.csv`)**: 700 samples (175/quadrant), structured with identical syntactic variety and uniform intensifier distribution.
- **Validation Scheme**: 5-Fold Stratified Cross-Validation (`StratifiedKFold`, $k=5$, `shuffle=True`, `random_state=42`), ensuring proportional quadrant distribution across each fold.

### Tier 2: Dialectal Regression Probes (`REGRESSION_PROBES`)
Dialectal collocations from Chilean Spanish and British English (e.g., *"estar con las emociones a flor de piel"*, *"proper wound up"*). These idioms previously failed early model iterations and were integrated into the template vocabulary. They serve strictly as a **regression guard**: any candidate hyperparameter configuration or retrained model that drops below 100% accuracy on this set is rejected automatically.

### Tier 3: Genuinely Held-Out Probes (`HELD_OUT_IDIOM_PROBES`)
A permanently disjoint set of colloquial and figurative expressions (e.g., *"estar chata"*, *"con los cables cruzados"*, *"proud as punch"*, *"seeing red"*) whose core tokens **never appear** in `generate_datasets.py`.
- **Purpose**: Diagnostic only. It measures the true out-of-distribution performance of bag-of-words models on unobserved figurative speech without masking failure modes.

---

## 3. Candidate Baseline Architectures

To substantiate the architectural choice of L2-regularized Logistic Regression, five models were benchmarked under identical tokenization and stopword conditions:

1. **Dummy Baseline (`DummyClassifier`)**:
   - Random stratified guessing based on empirical class distributions. Serves as the statistical baseline floor ($Macro\ F_1 \approx 0.25$).
2. **Multinomial Naive Bayes (`MultinomialNB`)**:
   - Standard generative NLP baseline with Laplace smoothing ($\alpha = 1.0$) over TF-IDF features.
3. **Linear Support Vector Classifier (`LinearSVC`)**:
   - Maximum-margin linear separator with L2 penalty ($C = 1.0$).
4. **Random Forest (`RandomForestClassifier`)**:
   - Non-linear ensemble baseline ($n=100$ estimators, `max_depth=10`).
5. **Logistic Regression — Production (`LogisticRegression`)**:
   - Multinomial Logistic Regression with L2 regularization ($C = 1.0$), `class_weight="balanced"`, and L-BFGS solver.

---

## 4. Operational & Serverless Constraints

Because Emotion Finder deploys as a serverless ASGI application on Vercel (AWS Lambda microVMs), offline classification metrics alone are insufficient. Models must satisfy strict operational thresholds:

- **Serialized Artifact Size**: Must remain $< 50\text{ KB}$ (achieved $\approx 27\text{ KB}$ via `joblib` compression level 3) to fit into edge deployment constraints.
- **Cold-Start Deserialization Latency**: Must load from disk in $< 15\text{ ms}$ during new container initialization.
- **Warm Inference Latency**: Must execute prediction on single queries in $< 2\text{ ms}$ ($O(1)$ inference).
- **Probability Calibration**: Must produce well-calibrated class probabilities via `predict_proba` to power the low-confidence cutoff threshold (`_CONFIDENCE_THRESHOLD = 0.44`) and runner-up quadrant detection (`_SECONDARY_GAP_THRESHOLD`).

---

## 5. Metric Definitions

- **Macro-Averaged Precision, Recall, and $F_1$**:
  $$\text{Macro } F_1 = \frac{1}{|C|} \sum_{c \in C} \frac{2 \cdot P_c \cdot R_c}{P_c + R_c}$$
  Enforcing equal weight across all four quadrants regardless of sample balance.
- **Dialectal Probe Accuracy**:
  $$\text{Accuracy}_{\text{probes}} = \frac{1}{|P|} \sum_{(x, y) \in P} \mathbb{I}(\hat{y} = y)$$
- **Confusion Matrix Normalization**:
  Row-normalized confusion matrices measuring class-specific recall rates ($CM_{i,j} = \frac{N_{i,j}}{\sum_k N_{i,k}}$).
