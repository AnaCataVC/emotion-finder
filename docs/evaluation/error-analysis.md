# Systematic Error Analysis & Failure Mode Taxonomy

This document presents a rigorous qualitative and quantitative error analysis of Emotion Finder's classification pipelines, focusing on out-of-distribution figurative language, colloquial idioms, and inductive limitations of bag-of-words representations.

---

## 1. Executive Summary & Observed Generalization Ceiling

While 5-fold cross-validation on synthetic in-distribution templates achieves near-perfect metrics ($Macro\ F_1 \ge 0.984$), the classifier encounters an inductive ceiling when evaluated against held-out dialectal expressions:
- **Spanish Held-Out Probes (`HELD_OUT_IDIOM_PROBES`):** $14.3\%$ raw pipeline accuracy ($1 / 7$).
- **English Held-Out Probes (`HELD_OUT_IDIOM_PROBES`):** $40.0\%$ raw pipeline accuracy ($2 / 5$).

Rather than treating this ceiling as a defect to be artificially concealed by memorizing specific expressions, Emotion Finder treats this as an expected boundary of linear n-gram models. The system relies on **calibrated confidence gating** and **multi-stage architectural fallbacks** to prevent catastrophic user-facing misclassifications.

---

## 2. Failure Mode Taxonomy on Figurative Language

Empirical inspection of out-of-distribution probe predictions reveals four primary failure modes:

### Failure Mode A: Zero-Overlap Figurative Tokens (OOV Ingestion)
- **Manifestation:** Phrases where the core affective signal resides entirely in colloquial terms that never appeared in training templates (e.g., Chilean Spanish *"achacada"*, *"prendida"*, *"en llamas"*; British English *"knackered"*, *"at my wits' end"*).
- **Mechanism:** The TF-IDF vectorizer extracts zero non-zero features (`query_vec.nnz == 0`) or only matches ubiquitous auxiliary verbs (*"estoy"*, *"ando"*, *"I'm"*). The linear model assigns near-uniform posterior probabilities ($\approx 0.25\text{--}0.30$) across all four quadrants.
- **Architectural Safeguard:** Because posterior probabilities hover around $0.28\text{--}0.31$, they fall strictly below the production confidence threshold (`_CONFIDENCE_THRESHOLD = 0.44`). The runtime marks the state as `uncertain`, deliberately warning the user and defaulting to manual somatic exploration.

### Failure Mode B: Metaphorical Somatic Collisions
- **Manifestation:** Expressions that employ anatomical or visceral metaphors whose literal terms carry an opposing affective valence in training templates.
  - *Example (ES):* `"siento el corazón en la mano de la angustia"`
  - *Expected:* `alta_negativa` (High Arousal $\cdot$ Negative Valence: acute anxiety/dread)
  - *Raw Prediction:* `alta_positiva` ($P = 0.477$, runner-up `alta_negativa` $P = 0.201$)
- **Root Cause Analysis:** In the synthetic training corpus, terms derived from the stem `"corazón"` predominantly occur in expressions of affection, enthusiasm, and joy (*"mi corazón late de emoción"*, *"siento el corazón lleno de gratitud"*). The strong positive unigram coefficient of `"corazon"` overpowered the negative signal of `"angustia"`.

### Failure Mode C: Negation Boundary & Collocation Mismatch
- **Manifestation:** Complex colloquial negations or idiomatic modifier chains.
  - *Example (ES):* `"ando livianita de sangre y sin ninguna preocupación"`
  - *Expected:* `baja_positiva` (Low Arousal $\cdot$ Positive Valence: quiet peace/relief)
  - *Raw Prediction:* `alta_negativa` ($P = 0.562$)
- **Root Cause Analysis:** The phrase contains the negative indicator token `"preocupación"`. While `"sin"` is preserved in stopword whitelisting, the bigram `"sin preocup"` in this specific construction collided with synthetic templates where worries were described with high arousal. Furthermore, colloquial Chilean expressions like `"livianita de sangre"` were unobserved, yielding zero positive counterweight.

### Failure Mode D: Arousal vs. Valence Granularity Confusion
- **Manifestation:** Correct identification of valence dimension (pleasant vs. unpleasant) but boundary confusion along the physiological arousal axis.
  - *Example (ES):* `"estoy chata y ya no puedo con la presión"`
  - *Expected:* `alta_negativa` (High Arousal $\cdot$ Negative Valence: overwhelmed, reactive saturation)
  - *Predicted:* `baja_negativa` ($P = 0.516$), with `alta_negativa` as runner-up ($P = 0.380$)
- **Root Cause Analysis:** In Chilean vernacular, *"estar chato/chata"* signifies feeling saturated, exhausted, and burdened by pressure. The semantic boundary between *exhaustion/helplessness* (`baja_negativa`) and *reactive agitation/overwhelm* (`alta_negativa`) is genuinely ambiguous even for human evaluators.
- **Architectural Safeguard:** The probability margin between top-1 ($0.516$) and top-2 ($0.380$) is only $0.136$, cleanly triggering `_find_secondary_quadrant()` (`gap < 0.15`). The UI explicitly surfaces `alta_negativa` as an alternative path.

---

## 3. The "Whack-a-Mole" Anti-Pattern in Affective NLP

When an NLP model misclassifies a colloquial phrase like *"estar con la mecha corta"* or *"estar chata"*, the naive response is to copy that exact sentence into the training CSV.

```text
The Whack-a-Mole Cycle:
  1. Identify misclassified colloquial idiom (e.g. "achacada")
  2. Add "achacada" to generate_datasets.py templates
  3. Re-train pipeline -> CV F1 increases, probe passes 100%
  4. Real user inputs novel phrasing (e.g. "cabreada", "tirada en el suelo")
  5. Pipeline fails again -> Repeat step 1
```

### Why This Fails:
1. **False Sense of Generalization:** Moving an idiom into training templates converts an out-of-distribution evaluation into an in-distribution check. It inflates metrics without improving the model's capacity to handle the long tail of unobserved dialectal expressions.
2. **Combinatorial Noise:** Injecting one-off vernacular terms into slot-filling generators can distort global class priors and introduce spurious unigram shortcuts.
3. **The Inductive Reality:** Bag-of-words linear models simply lack the pre-trained semantic embedding space (found in massive transformer models) required to perform zero-shot metaphorical inference.

---

## 4. Architectural Defense-in-Depth Solution

Rather than relying on brittle lexical memorization, Emotion Finder adopts a multi-tiered engineering architecture that absorbs classification errors gracefully:

```
  User Freeform Text Input
             │
             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 1. Confidence-Gated Quadrant Prediction                │
  │    If P(top1) < 0.44 -> Flag low_confidence: true      │
  │    If P(top1) - P(top2) < 0.15 -> Offer secondary path │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 2. Plain-Language User Confirmation Step               │
  │    "Does this feel right?"                             │
  │    - YES -> Continue to semantic/somatic tree          │
  │    - NO  -> Instantly pivots to runner-up quadrant tree│
  │             and logs negative feedback sample          │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 3. Deterministic Somatic Decision Tree (16 Leaves)     │
  │    4 binary yes/no physiological questions             │
  │    Guarantees 100% topological determinism             │
  │    Completely bypasses statistical classifier          │
  └──────────────────────────┬─────────────────────────────┘
                             │
                             ▼
  ┌────────────────────────────────────────────────────────┐
  │ 4. Active Learning Curation Pipeline                   │
  │    Positive user ratings -> status = 'verified'        │
  │    User corrections -> status = 'pending' (CLI review) │
  │    Weekly automated retraining under strict CI/CD gates│
  └────────────────────────────────────────────────────────┘
```

By decoupling affective routing into **coarse statistical inference** followed by **deterministic somatic navigation** and **human-in-the-loop verification**, the application ensures that an out-of-distribution NLP failure never traps the user in an erroneous emotional outcome.
