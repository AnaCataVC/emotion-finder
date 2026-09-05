> **Created:** 2026-09-05
> **Last Updated:** 2026-09-05
> **Status:** Active
> **Scope:** Feedback Loop, Serverless Persistence & Active Learning Pipeline

# Adversarial Stress-Test & Premortem: Feedback & Active Learning System

## 1. Executive Summary & Premortem Verdict

**Premortem Simulation (6 Months in Production):**  
> *"El sistema de feedback fue explotado por un script automatizado que envió 5,000 correcciones invertidas ('estoy feliz' -> 'alta_negativa'). Un script de reentrenamiento automático absorbió estos datos sin curaduría, destruyendo la precisión de la regresión logística para toda la base de usuarios. Paralelamente, la promesa del 'Fast Path en memoria' nunca funcionó en Vercel porque las microVMs no comparten memoria y mueren a los 5 minutos, mientras que un fallo de red hacia Turso provocó timeouts de 15 segundos que congelaron la UI con errores 504 Gateway Timeout."*

Este análisis de Red Team audita el diseño propuesto y formula salvaguardas no negociables para blindar la arquitectura antes de escribir una sola línea de código de producción.

---

## 2. Los 5 Vectores de Ataque y Vulnerabilidades Identificadas

### 🔴 Vector 1: La Ilusión del "Fast Path" en Serverless y Deriva de Estado
* **Severidad:** **[Critical / Blocker]**
* **Vulnerabilidad:** Vercel Serverless ejecuta instancias aisladas y efímeras (microVMs de AWS Firecracker). Proponer una variable global `_FAST_PATH_CACHE` en `inference.py` que se actualice en caliente asume erróneamente un servidor monolítico persistente. Si un usuario en la instancia $A$ envía una corrección, las instancias $B$, $C$ y $D$ jamás se enteran. Cuando la instancia $A$ se recicla por inactividad (después de 5-15 min), el cache desaparece por completo.
* **Intento de Bypass Inviable:** Consultar la base de datos remota (Turso/Supabase) en cada llamada a `/predict` para chequear sobreescrituras añade de 30 a 70 ms de latencia por request, destruyendo el invariante de arquitectura del proyecto de **inferencia en <5ms**.
* **Mitigación Mandatoria:**
  1. **Honestidad Arquitectónica:** Descartar la promesa de "aprendizaje en tiempo real por request". El pipeline de inferencia debe mantenerse 100% estático y ultra-veloz (<5ms).
  2. Las sobreescrituras solo se consolidan en el modelo mediante la compilación periódica por lotes (*Batch Retraining Artifact*), o mediante una tabla estática compacta de excepciones empaquetada junto al artefacto `.joblib` en despliegues.

---

### 🔴 Vector 2: Envenenamiento de Datos (Data Poisoning) & Ataques Sybil
* **Severidad:** **[Critical / Blocker]**
* **Vulnerabilidad:** El endpoint `/feedback` es público. Un actor malicioso o un usuario frustrado puede enviar clasificaciones invertidas intencionalmente:
  - `"estoy rebosante de alegría y paz"` $\rightarrow$ Corregido a: `alta_negativa` (Ira / Pánico).
  - `"me quiero morir de tristeza y dolor"` $\rightarrow$ Corregido a: `alta_positiva` (Éxtasis).
  Dado que el modelo actual utiliza $N$-gramas TF-IDF con un dataset compacto de 700 filas (175 por cuadrante), **apenas 15-20 muestras envenenadas** son suficientes para alterar los coeficientes lineales de palabras clave (*"alegría"*, *"tristeza"*, *"paz"*), invirtiendo el plano de Russell para todos los usuarios.
* **Mitigación Mandatoria:**
  1. **Compuerta de Estado (`status = 'pending'`):** Ningún feedback de usuario entra directamente al conjunto de entrenamiento de forma automática.
  2. **Regla de Consenso Multi-Sesión:** Una corrección solo pasa a estado elegible si al menos 3 sesiones/IPs distintas concuerdan en la misma corrección para la misma frase semántica, o tras aprobación explícita en un proceso de curaduría.
  3. **Cuota Máxima de Ingesta (10% Cap):** El dataset de reentrenamiento jamás puede tener más de un 10% de datos provenientes de feedback frente al baseline sintético canónico (máximo 70 filas de feedback sobre las 700 filas base).
  4. **Quality Gate Inviolable (`REGRESSION_PROBES` 100%):** Si el modelo candidato tras reentrenar falla aunque sea un solo caso de la suite de modismos chilenos y británicos en `train_model.py`, el reentrenamiento se aborta inmediatamente y el archivo `.joblib` es rechazado.

---

### 🟠 Vector 3: Agotamiento de Cuota y DoS en Endpoint Público
* **Severidad:** **[Major / Hardening Required]**
* **Vulnerabilidad:** `/feedback` carece de autenticación. Un bot en bucle `while true; do curl ...; done` puede saturar las 500,000 escrituras mensuales del free-tier de Turso o generar tablas gigantescas sin control.
* **Mitigación Mandatoria:**
  1. **Honeypot Sintáctico en FastHTML:** Campo oculto `<input type="text" name="hp_confirm" style="display:none" tabindex="-1">`. Si el campo viene con cualquier valor, el bot es detectado y la petición se descarta silenciosamente (`return HTTP 200` simulado).
  2. **Límites de Payload Estrictos:**
     - `len(user_text)` debe estar estrictamente entre 6 y 300 caracteres.
     - `len(comments)` máximo 150 caracteres.
     - Sanitización de caracteres de control y tags HTML para prevenir inyecciones.
  3. **Idempotencia por Hash:** Clave de unicidad `SHA-256(normalized_text + session_hash + str(today))`. La misma sesión no puede enviar más de 1 feedback por minuto para el mismo texto.

---

### 🟠 Vector 4: Timeouts de Red y Caídas en Cascada (Vercel Serverless)
* **Severidad:** **[Major / Hardening Required]**
* **Vulnerabilidad:** Si Turso experimenta degradación o la conexión desde AWS/Vercel sufre packet-loss, una llamada síncrona `urllib.request` sin timeout estricto puede esperar hasta el límite del runtime de Vercel (10-15s), arrojando un error `504 GATEWAY_TIMEOUT` visible al usuario.
* **Mitigación Mandatoria:**
  1. **Timeout Ultracorto:** `urllib.request.urlopen(..., timeout=1.8)`.
  2. **Fail-Open Elegante (Circuit Breaker Local):** Si la llamada a Turso falla por socket timeout, DNS o error HTTP, se captura la excepción silenciosamente, se loguea en `stdout` (Vercel Logs) y la UI responde de inmediato con el mensaje optimista: `"¡Gracias por tu aporte!"`. El usuario **jamás** debe ver un crash 500/504 por un fallo en el sistema de feedback.

---

### 🟡 Vector 5: Fricción de Onboarding y Desacople de Dependencias
* **Severidad:** **[Minor / Observability]**
* **Vulnerabilidad:** Forzar configuración de credenciales de Turso rompería `pytest` en local o en CI/CD (GitHub Actions) si no hay variables de entorno.
* **Mitigación Mandatoria:**
  1. `get_feedback_store()` debe detectar automáticamente el entorno:
     - En tests (`pytest`): usa SQLite en memoria `:memory:`. Cero llamadas de red.
     - En local dev (`TURSO_DATABASE_URL` ausente): usa `LocalSQLiteFeedbackStore` en `data/feedback.db`.
     - En producción (Vercel): usa `TursoHttpFeedbackStore` si hay token, o `NullFeedbackStore` seguro con log si no está configurado.

---

## 3. Matriz de Mitigaciones y Decisiones de Hardening

| Vulnerabilidad Auditada | Severidad | Decisión de Hardening Aprobada |
| :--- | :---: | :--- |
| **Deriva de caché en Serverless** | 🔴 Crítico | Descartar Fast-Path en caliente; la inferencia sigue siendo estática e inmutable en <5ms. |
| **Data Poisoning & Sybil Attacks** | 🔴 Crítico | Estado `pending` obligatorio, cap de 10% en el dataset, y bloqueo absoluto si falla `REGRESSION_PROBES`. |
| **DoS / Quota Flooding** | 🟠 Mayor | Honeypot invisible en el form de FastHTML + validación de longitud (6-300 chars) + deduplicación diaria. |
| **Timeouts hacia Turso en Vercel** | 🟠 Mayor | Timeout estricto de 1.8s + captura total de excepciones con respuesta optimista garantizada. |
| **Fricción en desarrollo local** | 🟡 Menor | Cero credenciales requeridas en local (SQLite nativo) y SQLite en memoria para `pytest`. |

---

## 4. Referencias y Lecciones Relacionadas
- [Adversarial Audit Lessons & Robustness Engineering](../learning/adversarial-audit-lessons.md)
- [Active Learning Feedback Loop Research](active-learning-feedback-loop.md)
- [ML Emotion Pipeline Architecture](ml-emotion-pipeline.md)
