<p align="center">
  <img src="icon.png" alt="emotion-finder Logo" width="120" />
</p>

# Emotion Finder — Detector Interactivo de Emociones
> **Detector Interactivo de Emociones basado en el Modelo Circunflejo del Afecto**

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


### 📌 Descripción del Proyecto
**Emotion Finder** es una aplicación web interactiva y bilingüe diseñada para transformar sensaciones físicas y estados mentales difusos en autoconocimiento emocional preciso.

> ⚠️ **Estado: etapa temprana / experimental.** La identificación de cuadrante y emoción todavía no es confiable — el techo honesto del clasificador frente a lenguaje figurado nunca visto es ~29% (ES) / ~40% (EN) (ver `HELD_OUT_IDIOM_PROBES` más abajo). La precisión mejora con el tiempo gracias al bucle de aprendizaje activo descrito en el punto 6.

En lugar de obligar al usuario a elegir etiquetas psicológicas abstractas de una lista abrumadora, Emotion Finder implementa una arquitectura híbrida de varias etapas:
1. **Clasificador NLP de Afecto**: Mapea descripciones en lenguaje natural sobre sensaciones físicas y cognitivas en uno de los 4 cuadrantes del **Modelo Circunflejo del Afecto de Russell** (Activación $\times$ Valencia). Antes de guiar al usuario por el árbol de 4 preguntas, le confirma en lenguaje natural si el cuadrante detectado tiene sentido — un "no" salta directo al árbol del cuadrante alternativo y registra el rechazo como feedback inmediato para el reentrenamiento.
2. **Matcher Semántico de Emoción**: Ordena las 16 emociones del cuadrante por similitud coseno TF-IDF entre las propias palabras del usuario y la descripción de cada emoción, saltando directo a un match confiable en vez de preguntar siempre lo mismo.
3. **Árbol de Decisión Somático Binario (respaldo)**: Cuando ningún match es suficientemente confiable, guía al usuario a través de una secuencia interactiva de 4 preguntas corporales de Sí/No para identificar **1 de 64 emociones precisas** (16 por cuadrante) — también accesible manualmente desde un match directo, para quien prefiera refinarlo.
4. **Claridad Emocional Empática**: Concluye en una tarjeta enfocada que presenta exclusivamente la emoción identificada, su emoji representativo y una definición empática e introspectiva orientada a facilitar la comprensión emocional.
5. **Arquitectura Hypermedia (FastHTML + HTMX)**: Brinda transiciones de página suaves y reactivas tipo SPA renderizadas 100% en Python del lado del servidor, sin dependencias de compilación ni frameworks pesados de JavaScript.
6. **Bucle de Feedback y Aprendizaje Activo (HITL)**: Permite a los usuarios validar o corregir la emoción detectada — tanto en el paso de confirmación de cuadrante como en la tarjeta de resultado final. Las correcciones se persisten de forma desacoplada y alimentan un pipeline de reentrenamiento por lotes con compuertas estrictas de calidad.

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
                       │     Confirmación de Cuadrante en Lenguaje Natural
                       │    ("¿tiene sentido?" — un "no" salta al árbol del
                       │     cuadrante alternativo y registra feedback)
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

#### 8. Modelo Híbrido de Curaduría, Blindaje Anti-Bot y Resolución en FastHTML
- **Modelo Híbrido de Curaduría (Opción 3)**: Para equilibrar la automatización continua con una rigurosa protección contra envenenamiento de datos, el feedback sigue un ciclo de vida híbrido:
  - **Confirmaciones Positivas (`verified`)**: Los envíos donde el usuario confirma la predicción ("pulgar arriba") sobre un cuadrante válido con texto suficiente ($\ge 6$ caracteres) se promueven automáticamente a `status = 'verified'`, nutriendo el reentrenamiento semanal de forma continua y autónoma.
  - **Correcciones de Usuario (`pending`)**: Las correcciones manuales y calificaciones negativas se mantienen en `status = 'pending'` para evitar que usuarios maliciosos o confundidos alteren los límites de decisión.
  - **Herramienta CLI Interactiva (`scripts/curate_feedback.py`)**: Los mantenedores pueden revisar las correcciones pendientes a máxima velocidad con una sola tecla (`[v]erify`, `[r]eject`, `[s]kip`, `[q]uit`).
- **Rate Limiting Anclado a Red**: `/feedback` y los eventos de rechazo en `/tree` aplican un tope estricto de `FEEDBACK_RATE_LIMIT_MAX` (5 envíos/hora) respaldado por `FeedbackStore.count_since()`. El `session_hash` anónimo se ancla a la identidad de red (IP + User-Agent), neutralizando bots que intenten evadir el límite rotando u omitiendo cookies de sesión.
- **Validación de Cuadrantes en `/tree`**: `POST /tree` valida que los cuadrantes sugeridos y rechazados pertenezcan estrictamente al esquema de 4 cuadrantes de Russell antes de persistirlos, previniendo la inyección de etiquetas arbitrarias.
- **Garantía de Reentrenamiento en CI/CD**: El flujo programado en GitHub Actions (`.github/workflows/retrain.yml`) ingiere exclusivamente muestras `verified`, asegurando que ningún dato sin curar o malicioso modifique los modelos de producción de manera desatendida.

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

# 4. Ejecutar pruebas automatizadas (100% pasando, 36 tests)
pytest tests/ -v

# O ejecutar suites individuales
python tests/test_pipeline.py
pytest tests/test_feedback.py

# 5. Herramienta interactiva de curaduría de feedback
python scripts/curate_feedback.py

# 6. Reentrenamiento por lotes con aprendizaje activo y CI/CD
# Modo dry-run (evalúa compuertas sin sobreescribir modelos en disco)
python scripts/retrain_from_feedback.py --dry-run

# Reentrenar localmente con datos verificados
python scripts/retrain_from_feedback.py --lang all
```

> **Reentrenamiento Semanal Automatizado**: Un workflow de GitHub Actions (`.github/workflows/retrain.yml`) se ejecuta de forma autónoma cada domingo a las 03:00 UTC. Se conecta a Turso usando los secretos del entorno (`TURSO_DATABASE_URL` y `TURSO_AUTH_TOKEN`), reentrena los clasificadores utilizando únicamente feedback `verified`, verifica los 36 tests y actualiza los pesos en `main` de manera 100% desatendida. También puede dispararse manualmente desde la pestaña **Actions** de GitHub con opción de incluir registros pendientes si se desea.

```bash
# 7. (Opcional) Configuración de Turso en Vercel
export TURSO_DATABASE_URL="libsql://your-database.turso.io"
export TURSO_AUTH_TOKEN="your-turso-auth-token"

# 8. Iniciar servidor de desarrollo
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
│   └── test_feedback.py     # Feedback storage, HTMX routes, honeypot, rate limiting, active learning gates
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

---

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

