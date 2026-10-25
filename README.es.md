<p align="center">
  <img src="icon.png" alt="Logo de Emotion Finder" width="120" />
</p>

# Emotion Finder — Detector Interactivo de Emociones
> **Clasificador NLP Afectivo Bilingüe y Árbol de Decisión Somático basado en el Modelo Circunflejo de Russell**

[English](README.md) | [Español](README.es.md)

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

## Descripción del Proyecto

**Emotion Finder** es una aplicación web interactiva y bilingüe (español/inglés) diseñada para transformar sensaciones físicas y estados mentales difusos en autoconocimiento emocional preciso.

En lugar de forzar a los usuarios a adivinar etiquetas psicológicas abstractas de una lista desplegable abrumadora, Emotion Finder implementa una **arquitectura híbrida**:
1. **Clasificador NLP de Afecto**: Mapea descripciones en lenguaje natural sobre sensaciones físicas y cognitivas en uno de los 4 cuadrantes del **Modelo Circunflejo del Afecto de Russell** (Activación/Arousal $\times$ Valencia).
2. **Matcher Semántico de Emoción**: Ordena las 16 emociones del cuadrante por similitud coseno TF-IDF entre las palabras del usuario y la definición de cada emoción, saltando directo a un resultado certero cuando la entrada es clara.
3. **Árbol de Decisión Somático Binario (Respaldo)**: Cuando ningún match directo supera el umbral de confianza, guía al usuario a través de una secuencia interactiva de 4 preguntas corporales de Sí/No para identificar **1 de 64 emociones precisas** (16 por cuadrante) con $100\%$ de determinismo topológico.
4. **Bucle de Active Learning Serverless (HITL)**: Captura validaciones y correcciones de los usuarios bajo estrictas defensas anti-envenenamiento (rate limiting por identidad de red, promoción híbrida de estado, tope de volumen del 10% y reentrenamiento semanal automatizado en CI/CD).

---

## Cómo Funciona

```
  1. Entrada Libre ("siento el pecho apretado y me hierve la sangre")
                             │
                             ▼
  2. Enrutador de Idioma y Preprocesamiento Seguro (Snowball/Porter + Whitelist Personalizada)
                             │
                             ▼
  3. TF-IDF + Regresión Logística (27 KB · Inferencia 0.8 ms · Vercel Serverless)
                             │
                             ▼
  4. Predicción del Cuadrante Afectivo (ej. Alta Activación · Negativa)
                             │
         ┌───────────────────┴───────────────────┐
         ▼                                       ▼
  [Match Semántico Directo]               [Match Débil / Ambiguo]
  Similitud coseno >= 0.35                       │
         │                                       ▼
         │                        [Confirmación de Cuadrante en Lenguaje Natural]
         │                        ("¿Tiene sentido este cuadrante?" — No salta al alternativo)
         │                                       │
         │                                       ▼
         │                        [Árbol Somático Binario de 4 Pasos]
         │                        (Tensión Corporal -> Pulso -> Enfoque -> Impulso de Acción)
         │                                       │
         └───────────────────┬───────────────────┘
                             ▼
  5. Tarjeta de Resultado Empática (Emoción Identificada + Definición + Formulario de Feedback)
                             │
                             ▼
  6. Almacenamiento Desacoplado de Feedback (SQLite Local / Pipeline HTTP Turso LibSQL / NullStore)
```

---

## Evaluación de ML y Benchmarks Empíricos

Para proporcionar evidencia experimental rigurosa y reproducible en lugar de limitarse a métricas sintéticas in-distribution, los modelos candidatos fueron evaluados mediante Validación Cruzada Estratificada de 5 Folds ($N=700$ muestras balanceadas por idioma) y un protocolo de doble sonda (dual-probe) fuera de distribución:

### Tabla Comparativa de Arquitecturas Candidatas

| Arquitectura | CV Macro $F_1$ (ES) | CV Macro $F_1$ (EN) | Acierto en Modismos Conocidos | Generalización No Vista (Held-Out) | Tamaño de Modelo | Carga en Frío | Inferencia en Caliente | Probabilidades Calibradas |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **Dummy (Estratificado)** | $0.227$ | $0.227$ | $14\%\text{--}40\%$ | $20\%\text{--}28\%$ | $7.6\text{ KB}$ | $2.0\text{ ms}$ | $0.94\text{ ms}$ | No |
| **Multinomial NB** | $0.987$ | $0.979$ | $100\%$ | $28\%\text{--}40\%$ | $21.0\text{ KB}$ | $4.5\text{ ms}$ | $0.73\text{ ms}$ | Dirichlet no calibrada |
| **SVM Lineal (LinearSVC)** | $\mathbf{0.996}$ | $\mathbf{0.987}$ | $100\%$ | $\mathbf{40\%\text{--}43\%}$ | $26.8\text{ KB}$ | $4.8\text{ ms}$ | $0.77\text{ ms}$ | No (Hinge loss) |
| **Random Forest** | $0.870$ | $0.860$ | $57\%\text{--}60\%$ | $28\%\text{--}40\%$ | $91.4\text{ KB}$ | $25.3\text{ ms}$ | $6.97\text{ ms}$ | Deficiente (Votos de árboles) |
| **Regresión Logística (Producción)** | **$0.991$** | **$0.984$** | **$100\%$** | **$14\%\text{--}40\%$** | **$27.3\text{ KB}$** | **$4.3\text{ ms}$** | **$0.84\text{ ms}$** | **Sí (Softmax L-BFGS)** |

### Matrices de Confusión (CV Estratificada de 5 Folds)

<p align="center">
  <img src="docs/evaluation/confusion-matrix-es.png" alt="Matriz de Confusión en Español" width="420" />
  <img src="docs/evaluation/confusion-matrix-en.png" alt="Matriz de Confusión en Inglés" width="420" />
</p>

### Documentación Detallada de Evaluación
- 📖 [Metodología y Protocolo de Benchmark](docs/evaluation/methodology.md): Diseño experimental, particiones y perfilado de latencias.
- 📊 [Resultados Completos de Benchmark](docs/evaluation/benchmark-results.md): Tablas completas por modelo, métricas por cuadrante y análisis de trade-offs.
- 📇 [Model & Data Card](docs/evaluation/model-card.md): Ficha de procedencia de datos, uso previsto, límites éticos y salvaguardas de feedback.
- 🔍 [Análisis Sistemático de Errores](docs/evaluation/error-analysis.md): Modos de falla cualitativos en lenguaje figurado y el antipatrón "whack-a-mole".

---

## Arquitectura del Sistema y Responsabilidad de Componentes

| Componente | Archivo | Responsabilidad |
| :--- | :--- | :--- |
| **Enrutador de Idioma e Inferencia** | `inference.py` | Detecta el idioma, ejecuta el pipeline TF-IDF, evalúa el umbral de baja confianza (`_CONFIDENCE_THRESHOLD = 0.44`) y detecta cuadrantes alternativos. |
| **Preprocesamiento y Tokenización** | `preprocessing.py` | Eliminación de tildes, stemming Snowball/Porter y tokenizadores aislados para evitar sombreado de nombres al deserializar con `joblib`. |
| **Matcher Semántico de Emoción** | `emotion_matcher.py` | Similitud coseno TF-IDF precomputada contra 16 definiciones de emoción por cuadrante; filtro de solapamiento nulo (`nnz == 0`) y umbral (`>= 0.35`). |
| **Árbol de Decisión Somático** | `decision_tree.py` | Árbol binario balanceado de profundidad 4 ($2^4 = 16$ hojas por cuadrante, 64 emociones en total) que mapea sensaciones viscerales con determinismo. |
| **Capa de Almacenamiento de Feedback** | `feedback_store.py` | Repositorio desacoplado que soporta `LocalSQLiteFeedbackStore`, `TursoHttpFeedbackStore` (pipeline HTTP sin dependencias adicionales) y `NullFeedbackStore` (fail-open). |
| **Pipeline de Active Learning** | `scripts/retrain_from_feedback.py` | Reentrenamiento por lotes con tope de seguridad del 10%, filtros de longitud de texto, compuerta de CV 5-Fold ($F_1 \ge 0.95$) y 100% en sondas de regresión. |
| **Herramienta CLI de Curación** | `scripts/curate_feedback.py` | Utilidad de terminal para que los mantenedores auditen sugerencias pendientes con una sola tecla (`[v]erificar`, `[r]echazar`, `[s]altar`). |
| **UI Hypermedia y Servidor ASGI** | `main.py` | Componentes reactivos renderizados en el servidor con FastHTML + HTMX; rate limiting por identidad de red en endpoints de feedback. |

---

## Trade-Offs de Ingeniería y Limitaciones Honestas

1. **Modelos Lineales Ligeros vs. Transformers Pesados**:
   - Modelos preentrenados tipo BERT o RoBERTa superan el límite de 250 MB de Vercel e introducen arranques en frío de 5 a 12 segundos.
   - El pipeline dual TF-IDF + Regresión Logística compila a **$\approx 27\text{ KB}$**, logrando **arranque en frío $<5\text{ ms}$** e **inferencia $<1\text{ ms}$**, ejecutándose en capas gratuitas serverless.
2. **El Techo en Lenguaje Figurado (~14%–40%)**:
   - Los modelos de bolsa de palabras no pueden realizar razonamiento metafórico zero-shot ante modismos coloquiales no observados (ej. *"estar chata"*, *"con los cables cruzados"*, *"seeing red"*).
   - En lugar del antipatrón "whack-a-mole" (copiar frases infinitamente a los templates de entrenamiento), Emotion Finder mantiene estas sondas aisladas y amortigua errores mediante la **confirmación de cuadrante**, **rutas alternativas** y el **árbol somático determinista**.
3. **Defensa en Profundidad contra Envenenamiento de Datos**:
   - Los endpoints públicos están protegidos por rate limiting por identidad de red (máx. 5 envíos/hora por hash IP+UA), ciclo de vida híbrido (validaciones positivas autoverificadas, correcciones en espera) y un tope del 10% de volumen en reentrenamiento.

---

## Instalación y Ejecución Local

### Prerrequisitos
- Python 3.11 o superior
- Git

### 1. Clonar y Configurar el Entorno
```powershell
git clone https://github.com/AnaCataVC/emotion-finder.git
cd emotion-finder
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements-dev.txt
```

### 2. Ejecutar la Suite de Pruebas Automatizadas
Verificar pipelines de ML, mapeos dialectales, árboles de decisión, endpoints HTMX e integridad de evaluación:
```powershell
pytest -v
```
*(Verificar que la suite completa de pruebas automatizadas pase con 100% de éxito / cero regresiones).*

### 3. Ejecutar la Suite de Benchmark y Evaluación
Reproducir las tablas comparativas y generar las figuras de matriz de confusión:
```powershell
python scripts/evaluate_models.py
```

### 4. Curación Interactiva de Feedback y Reentrenamiento
```powershell
# Revisar feedback de usuarios en espera
python scripts/curate_feedback.py

# Evaluar reentrenamiento candidato en modo dry-run
python scripts/retrain_from_feedback.py --dry-run
```

### 5. Iniciar Servidor de Desarrollo
```powershell
python main.py
```
Abrir el navegador en **[http://localhost:5001](http://localhost:5001)**.

---

## Decisiones de Arquitectura y Aprendizajes Clave

1. **Neutralización de Atajos por Palabras Meta-Afectivas**:
   - Términos genéricos como *"emociones"*, *"sentimientos"* o *"sensaciones"* se concentraban desproporcionadamente en ciertos templates sintéticos, provocando que el clasificador aprendiera a depender de la palabra meta en lugar de las señales viscerales.
   - Neutralizado mediante `_DOMAIN_META_STOPWORDS_ES` y `_DOMAIN_META_STOPWORDS_EN` en `train_model.py`.
2. **Preservación de Stopwords Críticas para el Sentimiento**:
   - Las listas de stopwords estándar eliminan negaciones (`no`, `sin`, `nunca`, `not`, `without`) e intensificadores (`muy`, `demasiado`, `very`). En NLP afectivo, esto produce inversión de valencia (*"no me siento bien"* $\to$ *"bien"*).
   - Resuelto mediante una lista de exclusión personalizada (`_EXCLUDED_FROM_STOPWORDS`) y captura obligatoria de bigramas (`ngram_range=(1, 2)`).
3. **Generación Determinista de Datasets**:
   - El uso de colecciones `set` en Python antes de mezclar introdujo aleatoriedad por el hash del proceso, alterando ~65% de las filas entre ejecuciones del generador a pesar de usar una semilla fija.
   - Corregido ordenando explícitamente las frases (`phrases_list = sorted(phrases)`) antes del shuffle.
4. **Navegación Somática Desacoplada sobre Clasificación Plana de 64 Clases**:
   - Clasificar 64 emociones mediante NLP puro requeriría corpus masivos y sufriría colapsos de frontera entre estados similares (ej. *Ira* vs. *Furia*).
   - Desacoplar la clasificación gruesa en 4 cuadrantes del recorrido determinista de 4 pasos garantiza $100\%$ de determinismo topológico con latencia $O(1)$.
5. **Persistencia Serverless con Cero Dependencias Adicionales**:
   - Escribir en SQLite local en entornos serverless con sistemas de archivos efímeros (`/var/task`) causa pérdida de datos y fallos en caliente.
   - Diseñado con un patrón repositorio desacoplado mediante `TursoHttpFeedbackStore` utilizando la librería estándar `urllib.request` (cero dependencias en producción) y un `NullFeedbackStore` de degradación fail-open.

---

## Índice de Documentación Técnica

- 🗂️ **Evaluación y Benchmarks**:
  - [Metodología y Protocolo Experimental](docs/evaluation/methodology.md)
  - [Resultados Empíricos de Benchmark y Matrices de Confusión](docs/evaluation/benchmark-results.md)
  - [Model & Data Card](docs/evaluation/model-card.md)
  - [Análisis Sistemático de Errores y Taxonomía de Fallas](docs/evaluation/error-analysis.md)
- 🧠 **Arquitectura e Investigación**:
  - [Arquitectura del Pipeline de Emociones ML](docs/external-references/ml-emotion-pipeline.md)
  - [Modismos Dialectales y Mapeo Afectivo Funcional](docs/external-references/dialectal-idioms-affective-mapping.md)
  - [Bucle de Feedback y Aprendizaje Activo](docs/external-references/active-learning-feedback-loop.md)
  - [Blindaje y Pruebas de Estrés en Feedback Activo](docs/external-references/feedback-active-learning-stress-test.md)
  - [Stack FastHTML y Despliegue Serverless](docs/external-references/fasthtml-stack.md)
  - [Lecciones de Auditoría Adversarial y Robustez](docs/learning/adversarial-audit-lessons.md)

---

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.
