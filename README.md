# 🧠 Emotion Finder — Interactive Emotion Detector
> **Detector Interactivo de Emociones basado en el Modelo Circunflejo del Afecto**

[![Python 3.11](https://img.shields.io/badge/python-3.11+-blue.svg?logo=python&logoColor=white)](https://www.python.org/)
[![FastHTML](https://img.shields.io/badge/FastHTML-0.14.x-FF6F00.svg?logo=fastapi&logoColor=white)](https://docs.fastht.ml/)
[![HTMX](https://img.shields.io/badge/HTMX-2.0-336699.svg?logo=htmx&logoColor=white)](https://htmx.org/)
[![PicoCSS](https://img.shields.io/badge/PicoCSS-v2-1095c1.svg?logo=css3&logoColor=white)](https://picocss.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4+-F7931E.svg?logo=scikit-learn&logoColor=white)](https://scikit-learn.org/)
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

Instead of forcing users to guess abstract psychological labels from an overwhelming drop-down list, Emotion Finder employs a two-stage hybrid architecture:
1. **Affective NLP Classifier**: Maps freeform natural language descriptions of physical and cognitive sensations into one of the 4 quadrants of **Russell's Circumplex Model of Affect** (Activation/Arousal $\times$ Valence).
2. **Binary Somatic Decision Tree**: Guides the user through an interactive 4-step sequence of body-focused Yes/No questions to pinpoint **1 of 64 precise emotions** (16 per quadrant).
3. **Empathetic Emotional Clarity**: Concludes with a focused card presenting exclusively the identified emotion, its visual archetype, and an empathetic, introspective definition to facilitate emotional clarity.
4. **Hypermedia-Driven Architecture (FastHTML + HTMX)**: Delivers smooth, SPA-like partial DOM transitions rendered entirely in server-side Python with zero client-side JavaScript build steps.

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
                                     │
                                     ▼
                     Affective Quadrant Prediction
            (e.g., High Arousal · Negative Valence / alta_negativa)
                                     │
                                     ▼
                       HTMX Partial DOM Replacement
                                     │
                                     ▼
                  Binary Somatic Decision Tree (4 Steps)
            [Q1: Body Tension] ──Yes/No──> [Q2: Heart Rate]
                                           │
                                         Yes/No
                                           │
                                           ▼
                                [Q3: Cognitive Focus]
                                           │
                                         Yes/No
                                           │
                                           ▼
                                 [Q4: Action Impulse]
                                     │
                                     ▼
                       Final Emotion Result Card
       (Identified Emotion + Representative Emoji + Empathetic Definition)
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

> For comprehensive documentation on corpus curation and linguistic validation, see [docs/external-references/dialectal-idioms-affective-mapping.md](docs/external-references/dialectal-idioms-affective-mapping.md).

---

### 💡 Key Technical Learnings & Engineering Decisions

#### 1. Architectural Decisions
- **Hypermedia over Heavy SPAs**: FastHTML paired with HTMX replaces multi-megabyte JavaScript bundles with pure server-rendered Python components. All UI transitions (loading spinners, question progressions, leaf cards) perform partial DOM swaps via `hx-post` and `hx-swap="innerHTML transition:true"`, resulting in a client bundle size of **0 KB**.
- **Decoupled Somatic Navigation**: Classifying 64 distinct emotions purely via NLP would require massive multi-class training data, resulting in low classification confidence and boundary confusion between adjacent affective states (e.g., *Ira* vs. *Furia*). By decoupling the architecture into **coarse NLP classification (4 quadrants)** followed by **fine-grained somatic decision trees (16 leaves)**, the system guarantees $100\%$ topological determinism while requiring only 4 binary user decisions ($O(1)$ latency).
- **Stateless Serverless ASGI**: The application is packaged for serverless ASGI deployment on Vercel (`api/index.py`), eliminating container orchestration overhead and operating entirely within zero-cost serverless tiers.

#### 2. NLP Feature Bias Debugging
- **The Domain Meta-Word Shortcut**: During evaluation with idiomatic phrases like *"estoy con las emociones a flor de piel"*, the classifier initially misclassified the text. Investigation revealed a **feature bias**: generic meta-words (*"emociones"*, *"sentimientos"*, *"sensaciones"*) were unevenly distributed across dataset quadrants, acting as spurious correlation shortcuts. The model learned to classify the presence of the word *"emociones"* rather than the affective signal.
- **The Engineering Fix**:
  1. Implemented domain meta-stopwords (`_DOMAIN_META_STOPWORDS_ES` and `_DOMAIN_META_STOPWORDS_EN`) in `train_model.py` to systematically neutralize non-discriminative terms (`emocion`, `sentimiento`, `sensacion`, `emotion`, `feeling`).
  2. Balanced all meta-words uniformly across the 4 quadrants in the synthetic training datasets (`data/emotions_es_v2.csv` and `data/emotions_en_v2.csv`, ~650 rows each).
  3. Preserved critical bigrams with `ngram_range=(1, 2)`, enabling collocations such as `"flor piel"`, `"mecha corta"`, `"on edge"`, and `"proper wound"` to retain their discriminative weight.
- **Sentiment-Aware Stopword Preservation**: Standard NLTK stopword lists remove essential negation markers (`no`, `sin`, `nunca`, `jamás`, `not`, `without`) and intensifiers (`muy`, `demasiado`, `very`, `extremely`). In affective NLP, naive stopword removal inverts valence (e.g., *"no me siento bien"* becomes *"bien"*). A custom whitelist explicitly preserves these crucial valence and arousal modulators.

#### 3. FastHTML on Vercel Serverless
- **Cold-Start & Memory Constraints**: Pre-trained Transformer models (BERT, RoBERTa) exceed Vercel's 250 MB / 500 MB serverless limits and introduce 5–12s cold starts. By pairing TF-IDF with L2-regularized Logistic Regression and Snowball/Porter stemming, each model compresses to **~27 KB** with **<1.5s cold starts** and **<5ms warm inference**.
- **Global Scope Singleton Pattern**: Models are loaded lazily into module globals (`inference.py`), persisting across warm AWS Lambda / Vercel microVM invocations.
- **Serialization Isolation**: Tokenizer functions (`tokenize_es`, `tokenize_en`, `strip_accents`) are isolated into an independent `preprocessing.py` module, ensuring that joblib unpickling succeeds seamlessly across training, testing, and Vercel ASGI execution contexts without namespace shadowing.

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
Verify that ML pipelines, dialectal mappings, edge cases, decision trees, and routes pass:
```bash
python tests/test_pipeline.py
```
*(All assertions should pass with 100% success).*

#### 5. (Optional) Re-train the classification models
```bash
python train_model.py
```
This trains both `models/model_es.joblib` and `models/model_en.joblib` using stratified cross-validation.

#### 6. Start the development server
```bash
python main.py
```
Open your browser and navigate to **[http://localhost:5001](http://localhost:5001)**.

---

<a name="español"></a>
## Español

### 📌 Descripción del Proyecto
**Emotion Finder** es una aplicación web interactiva y bilingüe diseñada para transformar sensaciones físicas y estados mentales difusos en autoconocimiento emocional preciso.

En lugar de obligar al usuario a elegir etiquetas psicológicas abstractas de una lista abrumadora, Emotion Finder implementa una arquitectura híbrida de dos etapas:
1. **Clasificador NLP de Afecto**: Mapea descripciones en lenguaje natural sobre sensaciones físicas y cognitivas en uno de los 4 cuadrantes del **Modelo Circunflejo del Afecto de Russell** (Activación $\times$ Valencia).
2. **Árbol de Decisión Somático Binario**: Guía al usuario a través de una secuencia interactiva de 4 preguntas corporales de Sí/No para identificar exactamente **1 de 64 emociones precisas** (16 por cuadrante).
3. **Claridad Emocional Empática**: Concluye en una tarjeta enfocada que presenta exclusivamente la emoción identificada, su emoji representativo y una definición empática e introspectiva orientada a facilitar la comprensión emocional.
4. **Arquitectura Hypermedia (FastHTML + HTMX)**: Brinda transiciones de página suaves y reactivas tipo SPA renderizadas 100% en Python del lado del servidor, sin dependencias de compilación ni frameworks pesados de JavaScript.

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
                                     │
                                     ▼
                     Predicción del Cuadrante Afectivo
           (ej., Alta Activación · Valencia Negativa / alta_negativa)
                                     │
                                     ▼
                        Reemplazo Parcial de DOM vía HTMX
                                     │
                                     ▼
                 Árbol de Decisión Somático Binario (4 Pasos)
           [P1: Tensión Corporal] ──Sí/No──> [P2: Ritmo Cardíaco]
                                             │
                                           Sí/No
                                             │
                                             ▼
                                   [P3: Enfoque Cognitivo]
                                             │
                                           Sí/No
                                             │
                                             ▼
                                  [P4: Impulso de Acción]
                                     │
                                     ▼
                      Tarjeta de Resultado Emocional
        (Emoción Identificada + Emoji Representativo + Definición Empática)
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

---

### 💡 Aprendizajes Clave de Ingeniería

#### 1. Decisiones Arquitectónicas
- **Hypermedia frente a SPAs Complejas**: FastHTML junto a HTMX permite construir interfaces reactivas sin descargar paquetes pesados de JavaScript. La interacción del árbol se maneja mediante reemplazos parciales de DOM (`hx-post="/tree"`), reduciendo el bundle de cliente a **0 KB**.
- **Desacoplamiento Somático**: Clasificar directamente 64 emociones con NLP demandaría un volumen masivo de datos de entrenamiento y generaría confusión en fronteras difusas (como *Ira* frente a *Furia*). Desacoplar la clasificación general (NLP en 4 cuadrantes) de la indagación somática interactiva (árbol binario de 4 pasos) garantiza exactitud matemática y una latencia predecible $O(1)$.
- **Microarquitectura Serverless ASGI**: El proyecto se ejecuta en Vercel mediante un punto de entrada ASGI nativo (`api/index.py`), eliminando la necesidad de gestionar contenedores o servidores dedicados.

#### 2. Depuración de Sesgos en NLP
- **El Atajo de las Meta-Palabras de Dominio**: Al analizar frases como *"estoy con las emociones a flor de piel"*, el clasificador fallaba inicialmente debido a un **sesgo de características**: palabras genéricas (*"emociones"*, *"sentimientos"*, *"sensaciones"*) estaban desbalanceadas en los datos sintéticos iniciales, provocando que el modelo aprendiera que la mera presencia de *"emociones"* indicaba un cuadrante determinado.
- **Solución Implementada**:
  1. Se crearon listas de stopwords de dominio (`_DOMAIN_META_STOPWORDS_ES` y `_DOMAIN_META_STOPWORDS_EN`) en `train_model.py` para neutralizar términos genéricos.
  2. Se equilibró la presencia de estas palabras de forma uniforme en los 4 cuadrantes de los datasets de entrenamiento (`data/emotions_es_v2.csv` y `data/emotions_en_v2.csv`).
  3. Se preservaron los bigramas mediante `ngram_range=(1, 2)` para que expresiones como `"flor piel"` o `"mecha corta"` conserven su peso contextual íntegro.
- **Conservación de Stopwords Críticas**: Las listas estándar de stopwords en español remueven negaciones (`no`, `sin`, `nunca`, `jamás`) e intensificadores (`muy`, `demasiado`). Una frase como *"no me siento bien"* se convertía erróneamente en *"bien"*, invirtiendo la valencia afectiva. Diseñar una lista de exclusión que preserve estas partículas fue indispensable.

#### 3. FastHTML en Vercel Serverless
- **Optimización de Arranque en Frío y Memoria**: Redes neuronales masivas exceden los límites de Vercel y generan arranques en frío de 5 a 12 segundos. Un pipeline TF-IDF + Regresión Logística optimizado pesa **~27 KB** y responde en **<5ms**.
- **Patrón Singleton a Nivel de Módulo**: Los modelos se cargan de forma perezosa en variables globales del módulo (`inference.py`), manteniéndose en memoria durante invocaciones calientes sucesivas.
- **Aislamiento de Serialización**: Las funciones de tokenización se ubicaron en `preprocessing.py`, garantizando que joblib pueda deserializar los pipelines sin fallos de importación cruzada en el entorno serverless de Vercel.

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

# 4. Ejecutar pruebas automatizadas
python tests/test_pipeline.py

# 5. (Opcional) Reentrenar modelos
python train_model.py

# 6. Iniciar servidor de desarrollo
python main.py
```
Abre tu navegador en **[http://localhost:5001](http://localhost:5001)**.

---

## 📂 Repository Structure / Estructura del Repositorio

```
emotion-finder/
├── main.py                  # FastHTML web app (routes, UI components, HTMX swaps)
├── inference.py             # ML inference module & language routing heuristic
├── preprocessing.py         # Stemming tokenizers & unicode accent normalization
├── decision_tree.py         # 64-emotion bilingual binary decision tree structure
├── train_model.py           # scikit-learn training script (TF-IDF + LogisticRegression)
├── api/
│   └── index.py             # Serverless ASGI entrypoint for Vercel deployment
├── data/
│   ├── emotions_es_v2.csv   # Balanced Spanish affective dataset (~650 samples)
│   ├── emotions_en_v2.csv   # Balanced English affective dataset (~650 samples)
│   ├── generate_datasets.py # Script for synthetic training dataset expansion
│   └── archive/             # Historical baseline seed datasets
├── models/
│   ├── model_es.joblib      # Compressed Spanish pipeline (~27 KB)
│   └── model_en.joblib      # Compressed English pipeline (~27 KB)
├── tests/
│   └── test_pipeline.py     # Automated test suite (NLP, edge cases, tree topology, web app)
├── docs/
│   └── external-references/ # Technical research & architecture references
│       ├── dialectal-idioms-affective-mapping.md # Chilean & British idioms mapping
│       ├── fasthtml-stack.md                 # FastHTML & Vercel serverless notes
│       └── ml-emotion-pipeline.md            # Pipeline architecture & benchmarks
├── requirements.txt         # Production runtime dependencies
├── requirements-dev.txt     # Development & training dependencies
├── vercel.json              # Vercel serverless build & routing configuration
└── README.md                # Project documentation (Bilingual EN / ES)
```

---

## 📄 License / Licencia
This project is open source and available under the [MIT License](LICENSE).
