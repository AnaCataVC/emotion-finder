> **Created:** 2026-09-05
> **Last Updated:** 2026-09-05
> **Topic:** Human-in-the-Loop Active Learning & Serverless Feedback Architecture

# Human-in-the-Loop Active Learning & Serverless Feedback Architecture

## 1. Executive Summary & Key Insights
Adding an interactive feedback mechanism for continuous or active learning in Emotion Finder involves three architectural layers:
1. **Frontend / UX Feedback Capture**: Non-intrusive interactive widgets rendered via FastHTML + HTMX on the final emotion result card. Allows binary agreement ("👍 Right on target" / "👎 Not quite right") and corrective labeling (selecting actual quadrant or specific emotion from the 64-leaf taxonomy).
2. **Serverless Persistence Layer**: Because Emotion Finder runs on Vercel Serverless with an ephemeral, read-only filesystem (`/var/task`), feedback cannot be appended to local CSV or SQLite files in production without failing or vanishing on cold restarts. A decoupled storage abstraction is mandatory:
   - **Local Development**: Append to local SQLite (`data/feedback.db`) or JSONL (`data/feedback.jsonl`).
   - **Production (Vercel)**: Pluggable remote persistence using lightweight serverless stores (Turso LibSQL, Supabase Postgres, or GitHub Repository Dispatch / Issue Webhook).
3. **Machine Learning Continuous Adaptation (HITL Active Learning)**:
   - **Online/Real-time Weight Updates**: Strictly avoided due to catastrophic forgetting, latency spikes, and vulnerability to adversarial data poisoning.
   - **Immediate In-Memory Override (Fast Path)**: Exact-phrase hash lookup or high-confidence nearest neighbor adjustment to provide immediate gratification to user corrections without risking global model instability.
   - **Batch Active Learning Pipeline (Slow Path)**: Offline or scheduled retraining script (`train_model.py` / `retrain_from_feedback.py`) that filters noisy feedback, merges validated samples with synthetic datasets, runs stratified 5-fold cross-validation, enforces regression probe thresholds (`REGRESSION_PROBES` and `HELD_OUT_IDIOM_PROBES`), and outputs new compressed joblib models.

## 2. Serverless Storage Trade-Offs (Vercel Environment)

| Storage Strategy | Read Latency | Write Latency | Cold Start Impact | Vercel Serverless Feasibility | Cost / Setup Overhead |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Local SQLite / JSONL** | <0.1 ms | <0.5 ms | Zero | **Fails in Prod** (Read-only filesystem; wiped on cold restart) | Zero (Dev only) |
| **Turso (LibSQL HTTP)** | ~15-30 ms | ~30-50 ms | Negligible (`libsql-experimental` or HTTP REST) | **High** (Native SQLite semantics over HTTP) | Free tier generous |
| **Supabase (PostgreSQL)** | ~25-45 ms | ~40-70 ms | Low (`supabase-py` or `httpx` REST) | **High** (Standard REST/PostgREST) | Free tier available |
| **GitHub Webhook / Dispatch** | N/A (fire-and-forget) | ~150-300 ms | Zero (async background task) | **High** (Pushes issue/commit triggers for retraining) | Free (GitHub Actions) |

## 3. Active Learning Loop & Anti-Poisoning Defenses
To prevent adversarial data poisoning or quality degradation when incorporating user feedback:
1. **Heuristic Sanitization**:
   - Minimum character length ($\ge 6$ chars) and maximum length ($\le 500$ chars).
   - Rate limiting per IP/session to prevent automated spam flooding.
   - Profanity and spam pattern filtering.
2. **Consensus & Confidence Gating**:
   - Only samples with verified user consensus or high model disagreement (uncertainty sampling where top-2 probability gap is small) are flagged for retraining pool.
3. **Regression Probes as Mandatory Gate**:
   - Any retraining batch MUST validate against existing `REGRESSION_PROBES` (dialectal idioms) and maintain $F_1 \ge 0.95$ across the 700 synthetic baseline rows before model artifacts are promoted to production.

## 4. References & Sources
- Settles, B. (2009). *Active Learning Literature Survey*. University of Wisconsin-Madison.
- modAL: Modular Active Learning framework for Python (https://github.com/modAL-python/modAL)
- FastHTML Documentation (https://docs.fastht.ml/)
- Vercel Serverless Python Environment & Storage Guidelines (https://vercel.com/docs/storage)
