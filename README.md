# 🧠 Emotion Finder — Interactive Emotion Detector
> **Detector Interactivo de Emociones basado en el Modelo Circunflejo del Afecto**

[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastHTML](https://img.shields.io/badge/FastHTML-0.14.x-FF6F00.svg?logo=fastapi&logoColor=white)](https://docs.fastht.ml/)
[![HTMX](https://img.shields.io/badge/HTMX-2.0-336699.svg?logo=htmx&logoColor=white)](https://htmx.org/)
[![PicoCSS](https://img.shields.io/badge/PicoCSS-v2-1095c1.svg?logo=css3&logoColor=white)](https://picocss.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
[![Active Learning](https://img.shields.io/badge/Active%20Learning-HITL%20Loop-8A2BE2.svg)](#)
[![Turso LibSQL](https://img.shields.io/badge/Database-Turso%20LibSQL-00E599.svg?logo=sqlite&logoColor=white)](https://turso.tech/)
[![Automated Retraining](https://img.shields.io/badge/CI%2FCD-Weekly%20Retrain-2088FF.svg?logo=githubactions&logoColor=white)](https://github.com/AnaCataVC/emotion-finder/actions)
[![Vercel Ready](https://img.shields.io/badge/Vercel-Serverless-black.svg?logo=vercel&logoColor=white)](https://vercel.com/)
[![Tests](https://img.shields.io/badge/Tests-100%25%20Passing-brightgreen.svg)](https://github.com/AnaCataVC/emotion-finder)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🌐 Language / Idioma
- [English Documentation](#english)
- [Documentación en Español](#español)

---

<a name="english"></a>
## English

### 📌 Project Overview
**Emotion Finder** is an interactive, bilingual web application engineered to bridge the gap between vague somatic/mental sensations and emotional self-awareness. 

Instead of forcing users to guess abstract psychological labels from an overwhelming drop-down list, Emotion Finder employs a multi-stage hybrid architecture:
1. **Affective NLP Classifier**: Maps freeform natural language descriptions of physical and cognitive sensations into one of the 4 quadrants of **Russell's Circumplex Model of Affect** (Activation/Arousal $\times$ Valence).
2. **Semantic Emotion Matcher**: Ranks the quadrant's 16 emotions by TF-IDF cosine similarity between the user's own words and each emotion's description, jumping straight to a confident match instead of always asking generic sensation questions.
3. **Binary Somatic Decision Tree (fallback)**: When no match is confident enough, guides the user through an interactive 4-step sequence of body-focused Yes/No questions to pinpoint **1 of 64 precise emotions** (16 per quadrant) — always reachable manually from a direct match too, for users who want to refine it.
4. **Empathetic Emotional Clarity**: Concludes with a focused card presenting exclusively the identified emotion, its visual archetype, and an empathetic, introspective definition to facilitate emotional clarity.
5. **Hypermedia-Driven Architecture (FastHTML + HTMX)**: Delivers smooth, SPA-like partial DOM transitions rendered entirely in server-side Python with zero client-side JavaScript build steps.
6. **Human-in-the-Loop Feedback & Active Learning Loop**: Empowers users to validate or correct detected emotions directly on the leaf card. High-signal user corrections are safely persisted across serverless environments and incorporated into a quality-gated batch retraining pipeline.

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
- **Anti-Bot & UI Honeypot Defense**: An invisible honeypot input (`hp_confirm`) silently intercepts automated crawlers without persisting spam, while preserving full user input text across multi-step somatic navigation.
- **Automated CI/CD Retraining Pipeline (`.github/workflows/retrain.yml`)**: A scheduled GitHub Actions workflow executes autonomously every Sunday at 03:00 UTC (and on-demand via `workflow_dispatch`). It pulls feedback directly from Turso LibSQL, enforces the 10% safety cap, evaluates candidate pipelines against 5-fold cross-validation ($F_1 \ge 0.95$) and dialectal regression probes (100%), and pushes the updated `.joblib` model weights back to `main` with `[skip ci]`, triggering a live zero-downtime redeploy on Vercel without manual intervention.

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

- **Linux / macOS (Bash/Zsh):**
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
Verify ML pipelines, dialectal mappings, decision trees, HTMX endpoints, and the feedback subsystem (30 automated tests):
```bash
# Run the complete test suite (100% passing)
pytest tests/ -v

# Or execute individual test modules
python tests/test_pipeline.py
pytest tests/test_feedback.py
```

#### 5. Active learning batch retraining & CI/CD automation
Evaluate and retrain models incorporating user feedback under strict quality gates:
```bash
# Validate candidate retrain in dry-run mode (evaluates gates without overwriting models)
python scripts/retrain_from_feedback.py --dry-run --include-pending

# Execute retrain locally and update models when quality gates pass
python scripts/retrain_from_feedback.py --lang all --include-pending
```

> **Automated Weekly Retraining**: A GitHub Actions workflow (`.github/workflows/retrain.yml`) runs automatically every Sunday at 03:00 UTC. It securely connects to Turso using environment secrets (`TURSO_DATABASE_URL` and `TURSO_AUTH_TOKEN`), retrains the models, passes all 30 tests, and commits the updated weights to `main` without manual intervention. You can also trigger it manually from the **Actions** tab on GitHub.

#### 6. (Optional) Production Vercel Turso configuration
To enable remote feedback persistence on Vercel Serverless, configure Turso LibSQL environment variables (if omitted, the system gracefully defaults to `NullFeedbackStore` fail-open mode):
```bash
export TURSO_DATABASE_URL="libsql://your-database.turso.io"
export TURSO_AUTH_TOKEN="your-turso-auth-token"
```

#### 7. Start the development server
```bash
python main.py
```
Open your browser and navigate to **[http://localhost:5001](http://localhost:5001)**.

---

<a name="español"></a>
## Español

### 📌 Descripción del Proyecto
**Emotion Finder** es una aplicación web interactiva y bilingüe diseñada para transformar sensaciones físicas y estados mentales difusos en autoconocimiento emocional preciso.

En lugar de obligar al usuario a elegir etiquetas psicológicas abstractas de una lista abrumadora, Emotion Finder implementa una arquitectura híbrida de varias etapas:
1. **Clasificador NLP de Afecto**: Mapea descripciones en lenguaje natural sobre sensaciones físicas y cognitivas en uno de los 4 cuadrantes del **Modelo Circunflejo del Afecto de Russell** (Activación $\times$ Valencia).
2. **Matcher Semántico de Emoción**: Ordena las 16 emociones del cuadrante por similitud coseno TF-IDF entre las propias palabras del usuario y la descripción de cada emoción, saltando directo a un match confiable en vez de preguntar siempre lo mismo.
3. **Árbol de Decisión Somático Binario (respaldo)**: Cuando ningún match es suficientemente confiable, guía al usuario a través de una secuencia interactiva de 4 preguntas corporales de Sí/No para identificar **1 de 64 emociones precisas** (16 por cuadrante) — también accesible manualmente desde un match directo, para quien prefiera refinarlo.
4. **Claridad Emocional Empática**: Concluye en una tarjeta enfocada que presenta exclusivamente la emoción identificada, su emoji representativo y una definición empática e introspectiva orientada a facilitar la comprensión emocional.
5. **Arquitectura Hypermedia (FastHTML + HTMX)**: Brinda transiciones de página suaves y reactivas tipo SPA renderizadas 100% en Python del lado del servidor, sin dependencias de compilación ni frameworks pesados de JavaScript.
6. **Bucle de Feedback y Aprendizaje Activo (HITL)**: Permite a los usuarios validar o corregir la emoción detectada directamente en la tarjeta de resultado. Las correcciones se persisten de forma desacoplada y alimentan un pipeline de reentrenamiento por lotes con compuertas estrictas de calidad.

---

### 🏛️ Arquitectura del Sistema y Flujo de Datos

```
   Entrada Libre del Usuario ("siento el pecho apretado y me hierve la sangre")
                                     │
                                     ▼
                   Enrutador de Idioma (Heurística ES / EN)
                                     │
                                     ▼
                 Preprocesamiento y Normalización de Texto
       (Eliminación de tildes, stemming Snowball/Porter, stopwords de dominio)
                                     │
                                     ▼
                 Clasificador TF-IDF + Regresión Logística
     (Entrenado con 700 muestras balanceadas, 175/cuadrante; n-gramas (1,2); ~27 KB)
                                     │
                                     ▼
                     Predicción del Cuadrante Afectivo
           (ej., Alta Activación · Valencia Negativa / alta_negativa)
                                     │
                                     ▼
           Matcher Semántico TF-IDF Precalculado (emotion_matcher.py)
        (Filtrado de stopwords, compuerta de solapamiento nnz==0, score >= 0.35)
                                     │
                match confiable ────┴──── match débil
                       │                       │
                       │                       ▼
                       │          Reemplazo Parcial de DOM vía HTMX
                       │                       │
                       │                       ▼
                       │         Árbol de Decisión Somático Binario (4 Pasos)
                       │        [P1: Tensión Corporal] ──Sí/No──> [P2: Ritmo Cardíaco]
                       │                                          │
                       │                                        Sí/No
                       │                                          │
                       │                                          ▼
                       │                              [P3: Enfoque Cognitivo]
                       │                                          │
                       │                                        Sí/No
                       │                                          │
                       │                                          ▼
                       │                             [P4: Impulso de Acción]
                       │                                          │
                       └───────────────────┬──────────────────────┘
                                            ▼
                      Tarjeta de Resultado Emocional
        (Emoción Identificada + Emoji Representativo + Definición Empática)
   (un match confiable también ofrece explorar manualmente con preguntas)
                                            │
                                            ▼
                   Recolección Interactiva de Feedback (HTMX)
                         [👍 Acertaron]    [👎 No del todo]
                                                │
                                                ▼
                              Formulario de Corrección (Taxonomía 64)
                                                │
                                                ▼
                       Capa de Persistencia Desacoplada (feedback_store.py)
                 ┌─────────────────────────────┼─────────────────────────────┐
                 ▼                             ▼                             ▼
        SQLite Local (Dev)            Turso LibSQL (HTTP)           NullStore (Fallback)
        (data/feedback.db)            (Vercel Serverless)           (Fail-open seguro)
                 │                             │
                 └─────────────────────────────┘
                               │
                               ▼
               Pipeline de Aprendizaje Activo (scripts/retrain_from_feedback.py)
               ├─ Filtros de Calidad y Deduplicación (6–300 caracteres)
               ├─ Límite de Ingesta del 10% (Defensa contra envenenamiento)
               ├─ Validación Cruzada Estratificada 5-Fold (Macro F1 >= 0.95)
               └─ Compuerta Obligatoria de Modismos (REGRESSION_PROBES 100%)
                               │
                               ▼
                   Artefactos de Modelo Actualizados (models/model_{lang}.joblib)
```

---

### 🧠 Modelo Circunflejo de Russell y Árbol de 64 Emociones

El Modelo Circunflejo de Russell postula que las emociones se ubican en un espacio bidimensional continuo definido por la **Valencia** (placentera vs. displacentera) y la **Activación** (alta energía vs. baja energía). Emotion Finder divide este espacio en 4 cuadrantes balanceados con árboles binarios de profundidad 4 ($2^4 = 16$ hojas por cuadrante, sumando **64 emociones precisas**):

| Cuadrante | Valencia | Activación | Indicadores Somáticos Principales | Hojas Totales | Emociones Representativas |
| :--- | :---: | :---: | :--- | :---: | :--- |
| **Alta Positiva (`alta_positiva`)** | $+$ | $+$ | Pulso acelerado, calor corporal expansivo, impulso de saltar o moverse | 16 | Éxtasis, Euforia, Pasión, Entusiasmo, Alegría, Triunfo, Inspiración, Orgullo... |
| **Alta Negativa (`alta_negativa`)** | $-$ | $+$ | Pecho ocre, mandíbula tensa, respiración agitada, temblor, reactividad | 16 | Furia, Ira, Pánico, Terror, Ansiedad, Frustración, Hostilidad, Agobio... |
| **Baja Negativa (`baja_negativa`)** | $-$ | $-$ | Pesadez muscular, postura encorvada, lentitud motora, vacío en el estómago | 16 | Tristeza profunda, Melancolía, Desolación, Duelo, Apatía, Fatiga, Vacío... |
| **Baja Positiva (`baja_positiva`)** | $+$ | $-$ | Respiración lenta y profunda, distensión muscular, calma sosegada | 16 | Serenidad, Paz interior, Calma, Alivio, Satisfacción, Armonía, Confianza... |

```
Recorrido en Árbol Binario (Profundidad 4 = 16 Hojas por Cuadrante):
Paso 1: Pregunta Raíz (Línea base somática)
 ├── Sí ──> Paso 2: Discriminador fisiológico de alta intensidad
 │           ├── Sí ──> Paso 3: Discriminador de foco específico
 │           │           ├── Sí ──> Paso 4 ──> [Hoja A] o [Hoja B]
 │           │           └── No  ──> Paso 4 ──> [Hoja C] o [Hoja D]
 │           └── No  ──> Paso 3: Discriminador visceral secundario
 │                        ├── Sí ──> Paso 4 ──> [Hoja E] o [Hoja F]
 │                        └── No  ──> Paso 4 ──> [Hoja G] o [Hoja H]
 └── No  ──> Paso 2: Discriminador difuso o moderado
              ├── Sí ──> Paso 3: Discriminador cognitivo / relacional
              │           ├── Sí ──> Paso 4 ──> [Hoja I] o [Hoja J]
              │           └── No  ──> Paso 4 ──> [Hoja K] o [Hoja L]
              └── No  ──> Paso 3: Discriminador de impulso atenuado
                           ├── Sí ──> Paso 4 ──> [Hoja M] o [Hoja N]
                           └── No  ──> Paso 4 ──> [Hoja O] o [Hoja P]
```

---

### 🗣️ Arquitectura de Modismos Dialectales (Español Chileno e Inglés Británico)

Las expresiones coloquiales sobre estados afectivos no pueden traducirse literalmente sin perder por completo su significado (por ejemplo, *"estar con las emociones a flor de piel"* traducido literalmente al inglés carece de todo sentido).

Emotion Finder resuelve esto implementando un **mapeo funcional afectivo** en lugar de una traducción léxica, asignando modismos típicos del **Español Chileno** y del **Inglés Británico** a su respectivo cuadrante de Russell:

| Cuadrante de Russell | Modismo (Español Chileno) | Significado Funcional Afectivo | Equivalente Funcional (Inglés Británico) | Emoción Objetivo |
| :--- | :--- | :--- | :--- | :--- |
| **Alta Negativa** | *"Estar con las emociones a flor de piel"* | Hipersensibilidad reactiva, nerviosismo al límite | *"My nerves are on edge"* / *"Feeling raw"* | Agobio / Ansiedad |
| **Alta Negativa** | *"Estar con la mecha corta"* | Irritabilidad aguda, baja tolerancia, explosividad | *"Proper wound up"* / *"Ready to blow my top"* | Irritabilidad / Ira |
| **Alta Negativa** | *"Estar chato / chata"* | Saturación extrema, agotamiento por presión | *"At the end of my tether"* / *"At my wits' end"* | Frustración / Agobio |
| **Alta Negativa** | *"Estar con la pera"* | Temor agudo, miedo físico, susto repentino | *"Bricking it"* / *"Shaking like a leaf"* | Miedo / Pánico |
| **Baja Negativa** | *"Estar con el bajón"* / *"Bajoneado"* | Decaimiento anímico, desánimo profundo | *"Down in the dumps"* / *"Feeling blue"* | Tristeza / Melancolía |
| **Baja Negativa** | *"Me da lata todo"* | Falta total de motivación, desgana, aburrimiento | *"Can't be bothered with anything"* | Apatía / Desgana |
| **Baja Negativa** | *"Estar achacado / achacada"* | Rumiación triste, pesadumbre, congoja | *"Gutted"* / *"Feeling down on myself"* | Duelo / Decepción |
| **Alta Positiva** | *"Estar que salto en una pata"* | Alegría desbordante, euforia contagiosa | *"Over the moon"* / *"Chuffed to bits"* | Alegría / Euforia |
| **Alta Positiva** | *"Estar prendido / prendida"* | Alta motivación, entusiasmo festivo | *"Buzzing"* / *"Full of beans"* | Entusiasmo / Excitación |
| **Alta Positiva** | *"Estar en llamas"* | Flujo productivo y apasionado imparable | *"On fire"* / *"Having a flyer"* | Pasión / Determinación |
| **Baja Positiva** | *"Estar piola y relajado"* | Sosiego no perturbado, tranquilidad íntima | *"Chill as anything"* / *"At ease"* | Calma / Serenidad |
| **Baja Positiva** | *"Estar tranqui"* | Calma y descanso sin preocupaciones | *"Right as rain"* / *"Ticking along nicely"* | Relajación / Paz interior |

> No todos los modismos de esta tabla llegaron al dataset de entrenamiento — algunos se mantienen deliberadamente fuera como chequeo de generalización held-out. Ver [docs/external-references/dialectal-idioms-affective-mapping.md](docs/external-references/dialectal-idioms-affective-mapping.md) §4 para el detalle exacto y la precisión medida en los que quedaron fuera.

---

### 💡 Aprendizajes Clave de Ingeniería

#### 1. Decisiones Arquitectónicas
- **Hypermedia frente a SPAs Complejas**: FastHTML junto a HTMX permite construir interfaces reactivas sin descargar paquetes pesados de JavaScript. La interacción del árbol se maneja mediante reemplazos parciales de DOM (`hx-post="/tree"`), reduciendo el bundle de cliente a **0 KB**.
- **Desacoplamiento Somático**: Clasificar directamente 64 emociones con NLP demandaría un volumen masivo de datos de entrenamiento y generaría confusión en fronteras difusas (como *Ira* frente a *Furia*). Desacoplar la clasificación general (NLP en 4 cuadrantes) de la indagación somática interactiva (árbol binario de 4 pasos) garantiza exactitud matemática y una latencia predecible $O(1)$.
- **Microarquitectura Serverless ASGI**: El proyecto se ejecuta en Vercel mediante un punto de entrada ASGI nativo (`api/index.py`), eliminando la necesidad de gestionar contenedores o servidores dedicados.

#### 2. Depuración de Sesgos en NLP
- **El Atajo de las Meta-Palabras de Dominio**: Palabras genéricas (*"emociones"*, *"sentimientos"*, *"sensaciones"*) pueden actuar como atajo de correlación espuria si están desbalanceadas entre cuadrantes — el modelo aprende a fijarse en la palabra *"emociones"* en vez de la señal afectiva real.
- **Solución Implementada**: Listas de stopwords de dominio (`_DOMAIN_META_STOPWORDS_ES` y `_DOMAIN_META_STOPWORDS_EN`) en `train_model.py` neutralizan términos genéricos (`emocion`, `sentimiento`, `sensacion`), mientras `ngram_range=(1, 2)` preserva bigramas discriminativos como `"flor piel"` o `"mecha corta"`.
- **Conservación de Stopwords Críticas**: Las listas estándar de stopwords en español remueven negaciones (`no`, `sin`, `nunca`, `jamás`) e intensificadores (`muy`, `demasiado`). Una frase como *"no me siento bien"* se convertía erróneamente en *"bien"*, invirtiendo la valencia afectiva. Diseñar una lista de exclusión que preserve estas partículas fue indispensable.

#### 3. Determinismo en la Generación del Dataset
- **El Bug**: `data/generate_datasets.py` acumula las frases generadas en un `set()` antes de mezclarlas con `random.seed(42)` fijo. El orden de iteración de un `set` depende del hash-seed del proceso, que se aleatoriza por defecto y `random.seed` no lo controla — así que dos corridas del generador "reproducible" producían CSVs de entrenamiento distintos (~65% de filas diferían entre corridas consecutivas, verificado empíricamente). Un modelo entrenado en una corrida podía pasar un test de un modismo que la *siguiente* corrida haría fallar, sin ningún cambio de código de por medio.
- **La Solución**: `phrases_list = sorted(phrases)` antes de mezclar, en ambos generadores (español e inglés), restaurando la reproducibilidad real entre corridas.
- **Consecuencia para los umbrales**: los `_CONFIDENCE_THRESHOLD` / `_INTENSITY_LOW_MAX` / `_SECONDARY_GAP_THRESHOLD` por idioma en `inference.py` se derivan del reporte que imprime `train_model.py` y deben resincronizarse después de cada reentrenamiento — están calibrados para un modelo específico, no son una constante fija.

#### 4. Probes de Regresión vs. Probes Genuinamente Held-Out
- **El Problema**: El `f1_macro` de CV sobre los datos sintéticos templados satura cerca de 1.0 con casi cualquier hiperparámetro, así que no distingue una configuración que generaliza de verdad de una que sobreajusta la estructura combinatoria de las plantillas.
- **`REGRESSION_PROBES`** (`train_model.py`): modismos dialectales que seguían clasificándose mal en la evaluación, cuyo vocabulario clave se agregó después a las plantillas de entrenamiento. Se usan para elegir entre hiperparámetros ajustados y los por defecto — ya no es un test de generalización (el vocabulario ya está en la distribución de entrenamiento), pero sigue detectando una elección de hiperparámetros que sobreajusta.
- **`HELD_OUT_IDIOM_PROBES`** (`train_model.py`): un segundo conjunto, disjunto, de modismos cuyo vocabulario se mantiene deliberadamente **fuera** de las plantillas de entrenamiento. Se imprime solo como diagnóstico, nunca se usa para elegir hiperparámetros. Precisión medida: **~29% (ES) / ~40% (EN)** — el techo honesto de un clasificador TF-IDF de bolsa de palabras frente a lenguaje figurado que nunca vio. Ver `docs/external-references/ml-emotion-pipeline.md` §6 para el detalle completo.

#### 5. Matcher Semántico de Emoción (`emotion_matcher.py`)
- **Matching TF-IDF Precalculado**: En vez de preguntar siempre cuatro preguntas somáticas genéricas de sí/no para elegir 1 de 16 emociones, la aplicación precalcula representaciones vectoriales TF-IDF para las descripciones de las 16 emociones por cuadrante durante la inicialización del módulo.
- **Compuerta de Stopwords y Umbral Calibrado**: Al incorporar filtrado estricto de stopwords (ignorando palabras funcionales genéricas) y un umbral calibrado (`score >= 0.35`) junto con validación de solapamiento léxico no nulo (`nnz == 0 -> 0.0`), las frases neutras o ambiguas caen limpiamente al árbol somático de 4 pasos, mientras que las coincidencias semánticas genuinas acceden directamente a la tarjeta de emoción identificada (manteniendo el árbol manual como opción de exploración).

#### 6. FastHTML en Vercel Serverless
- **Optimización de Arranque en Frío y Memoria**: Redes neuronales masivas exceden los límites de Vercel y generan arranques en frío de 5 a 12 segundos. Un pipeline TF-IDF + Regresión Logística optimizado con n-gramas (1, 2) pesa **~27 KB** y responde en **<5ms** en caliente con arranques en frío de **<1.5s**.
- **Patrón Singleton a Nivel de Módulo**: Los modelos se cargan de forma perezosa en variables globales del módulo (`inference.py`), manteniéndose en memoria durante invocaciones calientes sucesivas.
- **Aislamiento de Serialización**: Las funciones de tokenización se ubicaron en `preprocessing.py`, garantizando que joblib pueda deserializar los pipelines sin fallos de importación cruzada en el entorno serverless de Vercel.

#### 7. Feedback Human-in-the-Loop y Aprendizaje Activo Serverless
- **Persistencia Serverless con Cero Dependencias (`feedback_store.py`)**: El entorno serverless de Vercel opera con un sistema de archivos efímero y de solo lectura (`/var/task`). Intentar escribir directamente en un archivo SQLite local en producción causa excepciones en tiempo de ejecución o pérdida de datos tras el reciclaje de microVMs. Diseñamos un patrón de repositorio desacoplado con tres adaptadores:
  1. `LocalSQLiteFeedbackStore`: Para desarrollo local y pruebas (`data/feedback.db` o `:memory:`) con índices en texto normalizado y estado.
  2. `TursoHttpFeedbackStore`: Conector ligero para Turso LibSQL Pipeline API (`/v2/pipeline`), implementado exclusivamente mediante la biblioteca estándar de Python (`urllib.request`, sin inflar `requirements.txt` y con arranques en frío <200ms) y un timeout de 1.8 segundos con comportamiento fail-open.
  3. `NullFeedbackStore`: Fallback de degradación elegante que registra los eventos de feedback en memoria cuando no hay credenciales remotas, asegurando que la UI responda con éxito y jamás arroje un error 500.
- **Protección contra Envenenamiento de Datos y Deriva (`scripts/retrain_from_feedback.py`)**: El aprendizaje continuo en tiempo real sobre endpoints públicos expone el modelo a ataques de envenenamiento y olvido catastrófico. El pipeline de reentrenamiento por lotes implementa cuatro barreras defensivas:
  1. **Tope de Seguridad del 10%**: Las muestras de feedback no pueden superar el 10% del tamaño del dataset base canónico (máximo 70 muestras sobre las 700 sintéticas).
  2. **Deduplicación y Filtrado de Longitud**: Se descartan textos con longitud menor a 6 o mayor a 300 caracteres, unificando duplicados por hash normalizado.
  3. **Compuerta de Validación Cruzada**: El modelo candidato debe alcanzar un Macro $F_1 \ge 0.95$ en 5-Fold Stratified CV.
  4. **Compuerta Inviolable de Modismos (`REGRESSION_PROBES`)**: El modelo candidato debe superar con un 100% de precisión la suite de pruebas de modismos chilenos y británicos. Si un solo modismo falla, el modelo es rechazado automáticamente.
- **Defensa Anti-Bot mediante Honeypot**: Un campo oculto invisible (`hp_confirm`) intercepta scripts automatizados y descarta el spam silenciosamente sin penalizar la experiencia de usuario.
- **Pipeline de Reentrenamiento Automatizado en CI/CD (`.github/workflows/retrain.yml`)**: Un flujo de trabajo programado en GitHub Actions se ejecuta de forma autónoma cada domingo a las 03:00 UTC (y a demanda mediante `workflow_dispatch`). Descarga el feedback directamente desde Turso LibSQL, aplica el tope de seguridad del 10%, evalúa los modelos candidatos bajo validación cruzada ($F_1 \ge 0.95$) y compuertas de modismos (100%), y publica los nuevos pesos compilados `.joblib` en `main` con `[skip ci]`, disparando la actualización en producción en Vercel sin intervención manual.

---

### 🚀 Instalación y Uso Local

```bash
# 1. Clonar el repositorio
git clone https://github.com/AnaCataVC/emotion-finder.git
cd emotion-finder

# 2. Crear entorno virtual
# En Windows (PowerShell):
python -m venv .venv
.venv\Scripts\Activate.ps1

# En Linux / macOS:
python3 -m venv .venv
source .venv/bin/activate

# 3. Instalar librerías
pip install --upgrade pip
pip install -r requirements.txt

# 4. Ejecutar pruebas automatizadas (100% pasando, 30 tests)
pytest tests/ -v

# O ejecutar suites individuales
python tests/test_pipeline.py
pytest tests/test_feedback.py

# 5. Reentrenamiento por lotes con aprendizaje activo y CI/CD
# Modo dry-run (evalúa compuertas sin sobreescribir modelos en disco)
python scripts/retrain_from_feedback.py --dry-run --include-pending

# Reentrenar localmente e incorporar feedback si pasa todas las compuertas
python scripts/retrain_from_feedback.py --lang all --include-pending
```

> **Reentrenamiento Semanal Automatizado**: Un workflow de GitHub Actions (`.github/workflows/retrain.yml`) se ejecuta de forma autónoma cada domingo a las 03:00 UTC. Se conecta a Turso usando los secretos del entorno (`TURSO_DATABASE_URL` y `TURSO_AUTH_TOKEN`), reentrena los clasificadores, verifica los 30 tests y actualiza los pesos en `main` de manera 100% desatendida. También puede dispararse manualmente desde la pestaña **Actions** de GitHub.

```bash
# 6. (Opcional) Configuración de Turso en Vercel
export TURSO_DATABASE_URL="libsql://your-database.turso.io"
export TURSO_AUTH_TOKEN="your-turso-auth-token"

# 7. Iniciar servidor de desarrollo
python main.py
```
Abre tu navegador en **[http://localhost:5001](http://localhost:5001)**.

---

## 📂 Repository Structure / Estructura del Repositorio

```
emotion-finder/
├── .github/
│   └── workflows/
│       └── retrain.yml      # Automated weekly active learning retraining workflow (Cron / Dispatch)
├── main.py                  # FastHTML web app (routes, UI components, HTMX feedback widget)
├── inference.py             # ML inference module & language routing heuristic
├── emotion_matcher.py       # TF-IDF cosine-similarity direct emotion matcher
├── feedback_store.py        # Decoupled feedback persistence (SQLite, Turso HTTP, NullStore)
├── preprocessing.py         # Stemming tokenizers & unicode accent normalization
├── decision_tree.py         # 64-emotion bilingual binary decision tree structure
├── train_model.py           # scikit-learn training script (TF-IDF + LogisticRegression)
├── scripts/
│   └── retrain_from_feedback.py # Batch active learning retraining pipeline with quality gates
├── api/
│   └── index.py             # Serverless ASGI entrypoint for Vercel deployment
├── data/
│   ├── emotions_es_v2.csv   # Balanced Spanish affective dataset (700 samples)
│   ├── emotions_en_v2.csv   # Balanced English affective dataset (700 samples)
│   ├── generate_datasets.py # Deterministic synthetic training dataset generator
│   └── archive/             # Historical baseline seed datasets
├── models/
│   ├── model_es.joblib      # Compressed Spanish pipeline (~27 KB)
│   └── model_en.joblib      # Compressed English pipeline (~27 KB)
├── tests/
│   ├── test_pipeline.py     # NLP classifications, dialectal idioms, tree topology, web app
│   └── test_feedback.py     # Feedback storage, HTMX routes, honeypot, active learning gates
├── docs/
│   ├── external-references/ # Technical research & architecture references
│   │   ├── active-learning-feedback-loop.md          # HITL active learning & serverless notes
│   │   ├── dialectal-idioms-affective-mapping.md     # Chilean & British idioms mapping
│   │   ├── fasthtml-stack.md                         # FastHTML & Vercel serverless notes
│   │   ├── feedback-active-learning-stress-test.md   # Adversarial stress-test report
│   │   └── ml-emotion-pipeline.md                    # Pipeline architecture & benchmarks
│   └── learning/            # Post-audit architectural lessons & learnings
│       └── adversarial-audit-lessons.md              # Adversarial review learnings & insights
├── requirements.txt         # Production runtime dependencies
├── requirements-dev.txt     # Development & training dependencies
├── vercel.json              # Vercel serverless build & routing configuration
└── README.md                # Project documentation (Bilingual EN / ES)
```

---

## 📄 License / Licencia
This project is open source and available under the [MIT License](LICENSE).
