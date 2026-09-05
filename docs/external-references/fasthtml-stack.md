> **Created:** 2026-09-04
> **Last Updated:** 2026-09-04

# FastHTML Stack Research

## Key Findings
- FastHTML v0.14.x (pre-1.0, pin exact versions)
- Built on Starlette/ASGI, native HTMX integration, PicoCSS default
- **Vercel: Officially supported** (ASGI auto-detection, templates exist)
- Vercel limits: 500MB uncompressed, 300s timeout, 1024MB memory
- Single-file main.py is idiomatic for small apps
- Model loading: global scope singleton caching pattern in `inference.py` for persistent execution across warm microVMs.
- **Benchmarked Cold Starts**: **<1.5s cold starts** and **<5ms warm inference** using lightweight, joblib-compressed scikit-learn pipelines (~27 KB).

## Related References
- [ML Emotion Pipeline Architecture](ml-emotion-pipeline.md)
- [Dialectal Idioms & Affective Mapping](dialectal-idioms-affective-mapping.md)
- [Adversarial Audit & Robustness Learnings](../learning/adversarial-audit-lessons.md)

## Sources
- https://docs.fastht.ml/
- https://github.com/AnswerDotAI/fasthtml
- https://vercel.com/templates/python/fasthtml-boilerplate
