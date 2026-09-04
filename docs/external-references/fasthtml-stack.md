> **Created:** 2026-09-04
> **Last Updated:** 2026-09-04

# FastHTML Stack Research

## Key Findings
- FastHTML v0.14.x (pre-1.0, pin exact versions)
- Built on Starlette/ASGI, native HTMX integration, PicoCSS default
- **Vercel: Officially supported** (ASGI auto-detection, templates exist)
- Vercel limits: 500MB uncompressed, 300s timeout, 1024MB memory
- Single-file main.py is idiomatic for small apps
- Model loading: global scope caching pattern for serverless
- Cold starts: 1.2-3.5s with ML model

## Sources
- https://docs.fastht.ml/
- https://github.com/AnswerDotAI/fasthtml
- https://vercel.com/templates/python/fasthtml-boilerplate
