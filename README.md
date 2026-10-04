<p align="center">
  <img src="icon.png" alt="emotion-finder Logo" width="120" />
</p>

# Emotion Finder — Interactive Emotion Detector
> **Interactive Emotion Detector based on Russell's Circumplex Model of Affect**

[English](README.md) | [Español](README.es.md)

[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastHTML](https://img.shields.io/badge/FastHTML-0.14.x-FF6F00.svg?logo=fastapi&logoColor=white)](https://docs.fastht.ml/)
[![HTMX](https://img.shields.io/badge/HTMX-2.0-336699.svg?logo=htmx&logoColor=white)](https://htmx.org/)
[![PicoCSS](https://img.shields.io/badge/PicoCSS-v2-1095c1.svg?logo=css3&logoColor=white)](https://picocss.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Active Learning](https://img.shields.io/badge/Active%20Learning-HITL%20Loop-8A2BE2.svg)](#)
[![Status](https://img.shields.io/badge/Status-Early%20Stage-orange.svg)](#)
[![Turso LibSQL](https://img.shields.io/badge/Database-Turso%20LibSQL-00E599.svg?logo=sqlite&logoColor=white)](https://turso.tech/)
[![Automated Retraining](https://img.shields.io/badge/CI%2FCD-Weekly%20Retrain-2088FF.svg?logo=githubactions&logoColor=white)](https://github.com/AnaCataVC/emotion-finder/actions)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Serverless-black.svg?logo=vercel&logoColor=white)](https://vercel.com/)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)](https://github.com/AnaCataVC/emotion-finder)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

### 📌 Project Overview
**Emotion Finder** is an interactive, bilingual web application engineered to bridge the gap between vague somatic/mental sensations and emotional self-awareness. 

> ⚠️ **Status: early-stage / experimental.** Quadrant and emotion detection are not yet reliable — the classifier's honest ceiling on figurative language it has never seen is ~29% (ES) / ~40% (EN) (see `HELD_OUT_IDIOM_PROBES` below). Accuracy improves over time through the active learning feedback loop described in point 6.

Instead of forcing users to guess abstract psychological labels from an overwhelming drop-down list, Emotion Finder employs a multi-stage hybrid architecture:
1. **Affective NLP Classifier**: Maps freeform natural language descriptions of physical and cognitive sensations into one of the 4 quadrants of **Russell's Circumplex Model of Affect** (Activation/Arousal $\times$ Valence). Before walking the user through the 4-question tree, it asks them in plain language to confirm the detected quadrant feels right — a "no" jumps straight to the runner-up quadrant's tree and records the rejection as immediate training feedback.
2. **Semantic Emotion Matcher**: Ranks the quadrant's 16 emotions by TF-IDF cosine similarity between the user's own words and each emotion's description, jumping straight to a confident match instead of always asking generic sensation questions.
3. **Binary Somatic Decision Tree (fallback)**: When no match is confident enough, guides the user through an interactive 4-step sequence of body-focused Yes/No questions to pinpoint **1 of 64 precise emotions** (16 per quadrant) — always reachable manually from a direct match too, for users who want to refine it.
4. **Empathetic Emotional Clarity**: Concludes with a focused card presenting exclusively the identified emotion, its visual archetype, and an empathetic, introspective definition to facilitate emotional clarity.
5. **Hypermedia-Driven Architecture (FastHTML + HTMX)**: Delivers smooth, SPA-like partial DOM transitions rendered entirely in server-side Python with zero client-side JavaScript build steps.
6. **Human-in-the-Loop Feedback & Active Learning Loop**: Empowers users to validate or correct detected emotions — both at the quadrant-confirmation step and on the final leaf card. High-signal user corrections are safely persisted across serverless environments and incorporated into a quality-gated batch retraining pipeline.

---

### 🏛️ System Architecture & Dataflow

```
   User Freeform Input ("siento el pecho apretado y me hierve la sangre")
                                     │
                                     ▼
                   Language Router (ES / EN Heuristic)
                                     │
                                     ▼
                   Text Preprocessing & Normalization
           (Accent stripping, Snowball/Porter stemming, domain stopwords)
                                     │
                                     ▼
                TF-IDF + Logistic Regression Classifier
       (Trained on 700 balanced samples, 175/quadrant; ngram_range=(1,2); ~27 KB)
                                     │
                                     ▼
                     Affective Quadrant Prediction
            (e.g., High Arousal · Negative Valence / alta_negativa)
                                     │
                                     ▼
           Precomputed TF-IDF Semantic Matcher (emotion_matcher.py)
        (Stopword-filtered, zero-overlap gating nnz==0, score >= 0.35)
                                     │
                confident match ────┴──── weak match
                       │                       │
                       │                       ▼
                       │          HTMX Partial DOM Replacement
                       │                       │
                       │                       ▼
                       │       Plain-Language Quadrant Confirmation
                       │      ("does this feel right?" — a "no" jumps to the
                       │       runner-up quadrant's tree & records feedback)
                       │                       │
                       │                       ▼
                       │          Binary Somatic Decision Tree (4 Steps)
                       │        [Q1: Body Tension] ──Yes/No──> [Q2: Heart Rate]
                       │                                       │
                       │                                     Yes/No
                       │                                       │
                       │                                       ▼
                       │                            [Q3: Cognitive Focus]
                       │                                       │
                       │                                     Yes/No
                       │                                       │
                       │                                       ▼
                       │                             [Q4: Action Impulse]
                       │                                       │
                       └───────────────────┬───────────────────┘
                                            ▼
                       Final Emotion Result Card
       (Identified Emotion + Representative Emoji + Empathetic Definition)
     (a confident match also offers a manual "explore with questions" escape hatch)
                                            │
                                            ▼
                    Interactive Feedback Collection (HTMX)
                         [👍 Exactly]      [👎 Not quite]
                                                │
                                                ▼
                                  Corrective Form (Taxonomy Select)
                                                │
                                                ▼
                        Decoupled Storage Layer (feedback_store.py)
                  ┌─────────────────────────────┼─────────────────────────────┐
                  ▼                             ▼                             ▼
        Local SQLite DB                Turso LibSQL (HTTP)            NullStore (Fallback)
        (data/feedback.db)            (Vercel Serverless)            (Read-only fail-open)
                  │                             │
                  └─────────────────────────────┘
                                │
                                ▼
                Batch Active Learning Pipeline (scripts/retrain_from_feedback.py)
                ├─ Deduplication & Text Quality Filters (6–300 chars)
                ├─ 10% Maximum Sample Cap (Data poisoning defense)
                ├─ 5-Fold Stratified Cross-Validation (Macro F1 >= 0.95)
                └─ Mandatory Dialectal Regression Probes Gate (100% Pass)
                                │
                                ▼
                    Updated Model Artifacts (models/model_{lang}.joblib)
```

---

### 🧠 Russell's Circumplex Model & 64-Emotion Decision Tree

Russell's Circumplex Model posits that emotional states lie on a continuous two-dimensional space defined by **Valence** (pleasant vs. unpleasant) and **Arousal/Activation** (high energy vs. low energy). Emotion Finder structures this continuum into 4 quadrants, each branching into a balanced binary decision tree of depth 4 ($2^4 = 16$ distinct leaves per quadrant, totaling **64 granular emotions**):

| Quadrant | Valence | Activation | Core Somatic Indicators | Total Leaves | Representative Emotions |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **High Positive (`alta_positiva`)** | $+$ | $+$ | Accelerated heart rate, surge of warmth, urge to jump, expansive posture | 16 | Ecstasy, Euphoria, Passion, Enthusiasm, Joy, Triumph, Inspiration, Pride... |
| **High Negative (`alta_negativa`)** | $-$ | $+$ | Tight chest, clenched jaw, racing pulse, trembling, reactive agitation | 16 | Rage, Anger, Panic, Terror, Anxiety, Frustration, Hostility, Overwhelm... |
| **Low Negative (`baja_negativa`)** | $-$ | $-$ | Heavy limbs, slump, low motor energy, hollow stomach, withdrawal | 16 | Deep Sadness, Melancholy, Desolation, Grief, Apathy, Fatigue, Emptiness... |
| **Low Positive (`baja_positiva`)** | $+$ | $-$ | Slow deep breathing, muscle release, gentle warmth, grounded stillness | 16 | Serenity, Inner Peace, Calm, Relief, Contentment, Harmony, Trust... |

```
Binary Tree Traversal (Depth 4 = 16 Leaves per Quadrant):
Step 1: Root Question (Somatic baseline)
 ├── Yes ──> Step 2: High intensity physiological discriminator
 │            ├── Yes ──> Step 3: Specific focus discriminator
 │            │            ├── Yes ──> Step 4 ──> [Leaf A] or [Leaf B]
 │            │            └── No  ──> Step 4 ──> [Leaf C] or [Leaf D]
 │            └── No  ──> Step 3: Secondary visceral discriminator
 │                         ├── Yes ──> Step 4 ──> [Leaf E] or [Leaf F]
 │                         └── No  ──> Step 4 ──> [Leaf G] or [Leaf H]
 └── No  ──> Step 2: Moderate/diffuse discriminator
              ├── Yes ──> Step 3: Cognitive/relational discriminator
              │            ├── Yes ──> Step 4 ──> [Leaf I] or [Leaf J]
              │            └── No  ──> Step 4 ──> [Leaf K] or [Leaf L]
              └── No  ──> Step 3: Attenuated impulse discriminator
                           ├── Yes ──> Step 4 ──> [Leaf M] or [Leaf N]
                           └── No  ──> Step 4 ──> [Leaf O] or [Leaf P]
```

---

### 🗣️ Dialectal Idioms Architecture (Chilean Spanish & British English)

Emotional idioms are heavily culturally bound and cannot be translated literally without collapsing affective semantics (e.g., *"estar con las emociones a flor de piel"* literally translates to *"to have emotions at flower of skin"*, which is nonsensical in English).

Emotion Finder addresses this through **affective functional mapping** rather than lexical translation, mapping dialectal collocations from **Chilean Spanish** (*Español Chileno*) and **British English** directly into their corresponding Russell quadrant:

| Russell Quadrant | Chilean Spanish Idiom | Functional Affective Meaning | British English Functional Equivalent | Core Emotion Target |
| :--- | :--- | :--- | :--- | :--- |
| **Alta Negativa** | *"Estar con las emociones a flor de piel"* | Extreme affective reactivity, raw nerves | *"My nerves are on edge"* / *"Feeling raw"* | Agobio / Ansiedad |
| **Alta Negativa** | *"Estar con la mecha corta"* | Highly irritable, reactive, zero tolerance | *"Proper wound up"* / *"Ready to blow my top"* | Irritabilidad / Ira |
| **Alta Negativa** | *"Estar chato / chata"* | Saturated, overwhelmed by pressure | *"At the end of my tether"* / *"At my wits' end"* | Frustración / Agobio |
| **Alta Negativa** | *"Estar con la pera"* | Acute trepidation, terror, dread | *"Bricking it"* / *"Shaking like a leaf"* | Miedo / Pánico |
| **Baja Negativa** | *"Estar con el bajón"* / *"Bajoneado"* | Depressive slump, heavy despondency | *"Down in the dumps"* / *"Feeling blue"* | Tristeza / Melancolía |
| **Baja Negativa** | *"Me da lata todo"* | Complete apathy, zero motivation | *"Can't be bothered with anything"* | Apatía / Desgana |
| **Baja Negativa** | *"Estar achacado / achacada"* | Mournful brooding, sorrowful slump | *"Gutted"* / *"Feeling down on myself"* | Duelo / Decepción |
| **Alta Positiva** | *"Estar que salto en una pata"* | Thrilled, jumping with uncontainable joy | *"Over the moon"* / *"Chuffed to bits"* | Alegría / Euforia |
| **Alta Positiva** | *"Estar prendido / prendida"* | High energy excitement, pumped | *"Buzzing"* / *"Full of beans"* | Entusiasmo / Excitación |
| **Alta Positiva** | *"Estar en llamas"* | Unstoppable passionate flow | *"On fire"* / *"Having a flyer"* | Pasión / Determinación |
| **Baja Positiva** | *"Estar piola y relajado"* | Undisturbed, peaceful tranquility | *"Chill as anything"* / *"At ease"* | Calma / Serenidad |
| **Baja Positiva** | *"Estar tranqui"* | Quiet contentment, unbothered ease | *"Right as rain"* / *"Ticking along nicely"* | Relajación / Paz interior |

> Not every idiom in this table actually made it into the training data — some are deliberately kept out as a held-out generalization check. See [docs/external-references/dialectal-idioms-affective-mapping.md](docs/external-references/dialectal-idioms-affective-mapping.md) §4 for the exact split and measured accuracy on the ones that aren't.

---

### 💡 Key Technical Learnings & Engineering Decisions

#### 1. Architectural Decisions
- **Hypermedia over Heavy SPAs**: FastHTML paired with HTMX replaces multi-megabyte JavaScript bundles with pure server-rendered Python components. All UI transitions (loading spinners, question progressions, leaf cards) perform partial DOM swaps via `hx-post` and `hx-swap="innerHTML transition:true"`, resulting in a client bundle size of **0 KB**.
- **Decoupled Somatic Navigation**: Classifying 64 distinct emotions purely via NLP would require massive multi-class training data, resulting in low classification confidence and boundary confusion between adjacent affective states (e.g., *Ira* vs. *Furia*). By decoupling the architecture into **coarse NLP classification (4 quadrants)** followed by **fine-grained somatic decision trees (16 leaves)**, the system guarantees $100\%$ topological determinism while requiring only 4 binary user decisions ($O(1)$ latency).
- **Stateless Serverless ASGI**: The application is packaged for serverless ASGI deployment on Vercel (`api/index.py`), eliminating container orchestration overhead and operating entirely within zero-cost serverless tiers.

#### 2. NLP Feature Bias Debugging
- **The Domain Meta-Word Shortcut**: Generic meta-words (*"emociones"*, *"sentimientos"*, *"sensaciones"*) can act as spurious correlation shortcuts if unevenly distributed across dataset quadrants — the model learns to key off the word *"emociones"* rather than the affective signal.
- **The Engineering Fix**: Domain meta-stopwords (`_DOMAIN_META_STOPWORDS_ES` and `_DOMAIN_META_STOPWORDS_EN`) in `train_model.py` systematically neutralize non-discriminative terms (`emocion`, `sentimiento`, `sensacion`, `emotion`, `feeling`), while `ngram_range=(1, 2)` preserves discriminative bigrams like `"flor piel"` or `"mecha corta"`.
- **Sentiment-Aware Stopword Preservation**: Standard NLTK stopword lists remove essential negation markers (`no`, `sin`, `nunca`, `jamás`, `not`, `without`) and intensifiers (`muy`, `demasiado`, `very`, `extremely`). In affective NLP, naive stopword removal inverts valence (e.g., *"no me siento bien"* becomes *"bien"*). A custom whitelist explicitly preserves these crucial valence and arousal modulators.

#### 3. Dataset Generation Determinism
- **The Bug**: `data/generate_datasets.py` collects generated phrases in a `set()` before shuffling with a fixed `random.seed(42)`. A `set`'s iteration order depends on the process's hash seed, which is randomized by default and unaffected by `random.seed` — so two runs of the "reproducible" generator silently produced different training CSVs (~65% of rows differed between consecutive runs, verified empirically). A model trained on one run's data could pass an idiom test that the *next* run's data would fail, with no code change in between.
- **The Fix**: `phrases_list = sorted(phrases)` before shuffling, in both the Spanish and English generators, restoring true run-to-run reproducibility.
- **Consequence for thresholds**: `inference.py`'s per-language `_CONFIDENCE_THRESHOLD` / `_INTENSITY_LOW_MAX` / `_SECONDARY_GAP_THRESHOLD` are derived from `train_model.py`'s printed report and must be resynced after every retrain — they're calibrated to one specific model, not a fixed constant.

#### 4. Regression vs. Held-Out Idiom Probes
- **The Problem**: CV `f1_macro` on the templated synthetic data saturates near 1.0 for almost any hyperparameter choice, so it can't tell a genuinely-generalizing config from one that overfits the combinatorial template structure.
- **`REGRESSION_PROBES`** (`train_model.py`): dialectal idioms that kept misclassifying during evaluation, whose key vocabulary was then added to the training templates. Used to pick between tuned and default hyperparameters — no longer a generalization test (the vocabulary is now in-distribution), but still catches an overfit hyperparameter choice.
- **`HELD_OUT_IDIOM_PROBES`** (`train_model.py`): a second, disjoint set of idioms whose vocabulary is deliberately kept **out** of the training templates. Printed as a diagnostic only, never used to pick hyperparameters. Measured accuracy: **~29% (ES) / ~40% (EN)** — the honest ceiling of a bag-of-words TF-IDF classifier on figurative language it has never seen. See `docs/external-references/ml-emotion-pipeline.md` §6 for the full writeup.

#### 5. Semantic Emotion Matcher (`emotion_matcher.py`)
- **Precomputed TF-IDF Matching**: Instead of always asking four generic yes/no somatic questions to pick 1 of 16 emotions, the app precomputes TF-IDF vector representations of the 16 emotion descriptions per quadrant on module initialization.
- **Stopword Gating & Calibrated Threshold**: By enforcing strict stopword filtering (ignoring generic function words like *"de"*, *"la"*, *"the"*, *"with"*) and a calibrated threshold (`score >= 0.35`), neutral or ambiguous sentences cleanly fall through to the 4-step somatic decision tree, while genuine semantic matches skip straight to the identified emotion card (with manual tree exploration still available as a fallback).

#### 6. FastHTML on Vercel Serverless
- **Cold-Start & Memory Constraints**: Pre-trained Transformer models (BERT, RoBERTa) exceed Vercel's 250 MB / 500 MB serverless limits and introduce 5–12s cold starts. By pairing TF-IDF with L2-regularized Logistic Regression and Snowball/Porter stemming, each model compresses to **~27 KB** with **<1.5s cold starts** and **<5ms warm inference**.
- **Global Scope Singleton Pattern**: Models are loaded lazily into module globals (`inference.py`), persisting across warm AWS Lambda / Vercel microVM invocations.
- **Serialization Isolation**: Tokenizer functions (`tokenize_es`, `tokenize_en`, `strip_accents`) are isolated into an independent `preprocessing.py` module, ensuring that joblib unpickling succeeds seamlessly across training, testing, and Vercel ASGI execution contexts without namespace shadowing.

#### 7. Human-in-the-Loop Feedback & Serverless Active Learning Loop
- **Zero-Dependency Serverless Persistence (`feedback_store.py`)**: Because Emotion Finder runs on Vercel Serverless with an ephemeral, read-only filesystem (`/var/task`), writing directly to local SQLite files in production triggers runtime exceptions or loses state on cold restart. We implemented a decoupled storage repository pattern supporting three distinct environments:
  1. `LocalSQLiteFeedbackStore`: Manages local development and CI testing (`data/feedback.db` or `:memory:`) with indexed normalized text and status lookups.
  2. `TursoHttpFeedbackStore`: Serverless production client targeting Turso LibSQL Pipeline API (`/v2/pipeline`), strictly implemented using Python's standard library `urllib.request` (zero extra dependencies in `requirements.txt`, <200ms cold start overhead) with a strict 1.8-second fail-open timeout.
  3. `NullFeedbackStore`: Fail-open graceful degradation fallback that safely logs feedback events in memory when remote database credentials are omitted, preventing unhandled 500 errors.
- **Resilience Against Data Poisoning & Drift (`scripts/retrain_from_feedback.py`)**: Naive online learning on public endpoints exposes models to adversarial attacks (e.g., trolls submitting inverted labels) and catastrophic forgetting. The batch retraining pipeline enforces four strict defense layers:
  1. **Strict 10% Safety Cap**: Feedback samples are capped at a maximum of 10% of the baseline training dataset volume (max 70 samples against 700 baseline rows).
  2. **Deduplication & Length Gating**: User texts under 6 characters or over 300 characters are rejected, and inputs are deduplicated by lowercased normalized string.
  3. **Cross-Validation Quality Gate**: Candidate models must sustain Stratified 5-Fold Cross-Validation Macro $F_1 \ge 0.95$.
  4. **Non-Negotiable Dialectal Regression Probes Gate**: Retrained candidate pipelines must achieve 100% accuracy on the dialectal `REGRESSION_PROBES` suite (Chilean and British idioms). If a single probe fails, the candidate artifact is rejected immediately.

#### 8. Hybrid Curation Model, Anti-Bot Hardening & FastHTML Resolution
- **Hybrid Curation Model (Option 3)**: To balance zero-friction automation with strict protection against data poisoning, user feedback follows a hybrid lifecycle:
  - **Positive Validations (`verified`)**: Submissions where the user confirms the model's prediction ("thumbs-up") on a valid quadrant with sufficient text length ($\ge 6$ chars) are automatically promoted to `status = 'verified'`, continuously nourishing the weekly retraining pipeline without requiring manual intervention.
  - **Corrective Feedback (`pending`)**: User corrections and negative ratings are held in `status = 'pending'` to prevent trolls or mislabeled suggestions from poisoning linear classification boundaries.
  - **Interactive CLI Review Tool (`scripts/curate_feedback.py`)**: Maintainers can review pending submissions with single-keystroke speed (`[v]erify`, `[r]eject`, `[s]kip`, `[q]uit`).
- **Network-Anchored Rate Limiting**: `/feedback` and somatic rejection events on `/tree` enforce a strict rate limit of `FEEDBACK_RATE_LIMIT_MAX` (5 requests/hour) evaluated by `FeedbackStore.count_since()`. The anonymous `session_hash` is anchored to client network identity (IP + User-Agent), neutralizing bot scripts that attempt to bypass limits by rotating or omitting session cookies.
- **Tree-Route Validation & Quota Guard**: `POST /tree` validates that suggested and rejected quadrants strictly belong to Russell's 4-quadrant schema before persistence, preventing injection of arbitrary quadrant tags.
- **Automated CI/CD Retraining Guarantee**: Scheduled GitHub Actions executions (`.github/workflows/retrain.yml`) ingest strictly `verified` samples, ensuring zero uncurated or malicious inputs alter production weights unattended.

---

### 🚀 Local Setup & Installation

Follow these step-by-step instructions to run Emotion Finder locally:

#### Prerequisites
- Python 3.11 or higher
- Git

#### 1. Clone the repository
```bash
git clone https://github.com/AnaCataVC/emotion-finder.git
cd emotion-finder
```

#### 2. Create and activate a virtual environment
- **Windows (PowerShell):**
```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
```

- **Linux / macOS:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

#### 3. Install dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

#### 4. Run the automated test suite
Verify ML pipelines, dialectal mappings, decision trees, HTMX endpoints, and the feedback subsystem (36 automated tests):
```bash
# Run the complete test suite (100% passing)
pytest tests/ -v

# Or execute individual test modules
python tests/test_pipeline.py
pytest tests/test_feedback.py
```

#### 5. Interactive feedback curation tool
```bash
python scripts/curate_feedback.py
```

#### 6. Active learning batch retraining & CI/CD automation
Evaluate and retrain models incorporating verified user feedback under strict quality gates:
```bash
# Validate candidate retrain in dry-run mode (evaluates gates without overwriting models)
python scripts/retrain_from_feedback.py --dry-run

# Execute retrain locally with verified data
python scripts/retrain_from_feedback.py --lang all
```

> **Automated Weekly Retraining**: A GitHub Actions workflow (`.github/workflows/retrain.yml`) runs automatically every Sunday at 03:00 UTC. It securely connects to Turso using environment secrets (`TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN`), retrains the models using verified feedback, passes all 36 tests, and commits the updated weights to `main` without manual intervention. You can also trigger it manually from the **Actions** tab on GitHub with optional pending inclusion.

#### 7. (Optional) Production Vercel Turso configuration
To enable remote feedback persistence on Vercel Serverless, configure Turso LibSQL environment variables (if omitted, the system gracefully defaults to `NullFeedbackStore` fail-open mode):
```bash
export TURSO_DATABASE_URL="libsql://your-database.turso.io"
export TURSO_AUTH_TOKEN="your-turso-auth-token"
```

#### 8. Start the development server
```bash
python main.py
```
Open your browser and navigate to **[http://localhost:5001](http://localhost:5001)**.

---


---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

