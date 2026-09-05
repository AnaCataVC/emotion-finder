"""
Emotion Finder — Interactive Emotion Detector.

A FastHTML web application that helps users identify their emotions
by mapping physical and mental sensations using Russell's Circumplex
Model of Affect (Activation × Valence).

Usage:
    python main.py
"""

import os
from fasthtml.common import *

from decision_tree import DECISION_TREES, get_all_emotions, get_node, get_quadrant_emotions, get_tree
from emotion_matcher import is_confident_match, match_emotion
from feedback_store import create_feedback_record, get_feedback_store
from inference import detect_language, predict_quadrant

# ---------------------------------------------------------------------------
# Custom Brand CSS (Matches App Icon: Fuchsia -> Violet -> Indigo Gradient)
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
:root {
    --brand-magenta: #e84393;
    --brand-violet: #7b40d4;
    --brand-indigo: #3b38c2;
    --brand-gradient: linear-gradient(135deg, #e84393 0%, #7b40d4 50%, #3b38c2 100%);
    --brand-gradient-hover: linear-gradient(135deg, #f054a5 0%, #8c52e6 50%, #4b48d6 100%);
    --brand-glow: rgba(232, 67, 147, 0.25);
    --card-radius: 22px;
}

/* Base Body & Container Adjustments */
body > header.container {
    padding-top: 1.25rem;
    padding-bottom: 0.75rem;
}

/* Glassmorphic Brand Cards */
article.brand-card {
    border-radius: var(--card-radius);
    border: 1px solid rgba(123, 64, 212, 0.18);
    box-shadow: 0 16px 40px -12px rgba(59, 56, 194, 0.12), 0 4px 16px -2px rgba(232, 67, 147, 0.08);
    backdrop-filter: blur(12px);
    transition: transform 0.25s ease, box-shadow 0.25s ease;
    margin-bottom: 2rem;
}

article.brand-card:hover {
    box-shadow: 0 20px 48px -10px rgba(59, 56, 194, 0.18), 0 6px 20px -2px rgba(232, 67, 147, 0.12);
}

/* Navigation & Brand Logo */
header.container nav {
    display: flex !important;
    justify-content: space-between !important;
    align-items: center !important;
    min-height: 52px;
    padding: 0.5rem 0;
}

header.container nav ul {
    display: flex !important;
    align-items: center !important;
    margin: 0 !important;
    padding: 0 !important;
    list-style: none !important;
}

header.container nav ul li {
    display: flex !important;
    align-items: center !important;
    padding: 0 !important;
    margin: 0 !important;
}

.nav-brand {
    display: inline-flex;
    align-items: center;
    gap: 0.75rem;
    text-decoration: none;
    font-weight: 700;
    font-size: 1.25rem;
    color: var(--pico-color);
    padding: 0 !important;
    margin: 0 !important;
}

.brand-icon-sm {
    width: 36px;
    height: 36px;
    border-radius: 10px;
    box-shadow: 0 4px 12px rgba(123, 64, 212, 0.3);
    vertical-align: middle;
}

/* Segmented Pill Language Switcher (Strict Zero Overlap) */
.lang-switcher {
    display: inline-flex !important;
    flex-direction: row !important;
    align-items: center !important;
    justify-content: center !important;
    background: rgba(123, 64, 212, 0.08) !important;
    padding: 3px !important;
    border-radius: 9999px !important;
    border: 1px solid rgba(123, 64, 212, 0.22) !important;
    gap: 2px !important;
    margin: 0 !important;
    box-sizing: border-box !important;
    height: 36px !important;
}

.lang-switcher a.lang-pill,
nav ul li .lang-switcher a.lang-pill {
    display: inline-flex !important;
    align-items: center !important;
    justify-content: center !important;
    width: 44px !important;
    min-width: 44px !important;
    height: 28px !important;
    padding: 0 !important;
    margin: 0 !important;
    border-radius: 9999px !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.05em !important;
    text-decoration: none !important;
    line-height: 1 !important;
    box-sizing: border-box !important;
    border: none !important;
    color: var(--pico-muted-color) !important;
    transition: all 0.2s cubic-bezier(0.4, 0, 0.2, 1) !important;
}

.lang-switcher a.lang-pill:hover:not(.active),
nav ul li .lang-switcher a.lang-pill:hover:not(.active) {
    color: var(--pico-color) !important;
    background: rgba(123, 64, 212, 0.12) !important;
}

.lang-switcher a.lang-pill.active,
nav ul li .lang-switcher a.lang-pill.active {
    background: var(--brand-gradient) !important;
    color: #ffffff !important;
    box-shadow: 0 2px 10px rgba(232, 67, 147, 0.4) !important;
}


/* Hero Section */
.hero-wrapper {
    text-align: center;
    padding: 1.75rem 1rem 1.25rem;
}


.gradient-title {
    font-size: 2.4rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    margin-bottom: 0.5rem;
    background: var(--brand-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline-block;
}

.hero-subtitle {
    font-size: 1.15rem;
    color: var(--pico-muted-color);
    max-width: 540px;
    margin: 0 auto 1.5rem;
    line-height: 1.5;
}

/* Primary Gradient Buttons */
button.btn-gradient,
a[role="button"].btn-gradient {
    background: var(--brand-gradient);
    color: #ffffff !important;
    border: none;
    font-weight: 600;
    border-radius: 9999px;
    padding: 0.75rem 1.75rem;
    box-shadow: 0 6px 20px rgba(232, 67, 147, 0.35);
    transition: all 0.25s ease;
    cursor: pointer;
}

button.btn-gradient:hover,
a[role="button"].btn-gradient:hover {
    background: var(--brand-gradient-hover);
    transform: translateY(-2px);
    box-shadow: 0 8px 26px rgba(232, 67, 147, 0.45);
}

button.btn-outline,
a[role="button"].btn-outline {
    background: transparent;
    border: 2px solid rgba(123, 64, 212, 0.4);
    color: var(--pico-color);
    font-weight: 600;
    border-radius: 9999px;
    padding: 0.75rem 1.75rem;
    transition: all 0.25s ease;
}

button.btn-outline:hover,
a[role="button"].btn-outline:hover {
    border-color: var(--brand-violet);
    background: rgba(123, 64, 212, 0.08);
    transform: translateY(-2px);
}

/* Smooth HTMX Swaps */
.htmx-swapping { opacity: 0; transform: scale(0.98); transition: all 0.25s ease-out; }
.htmx-settling { opacity: 1; transform: scale(1); transition: all 0.25s ease-in; }

/* Progress Bar */
.progress-bar {
    display: flex;
    gap: 0.6rem;
    margin: 1rem 0 2rem;
    justify-content: center;
}

.progress-step {
    width: 3.5rem;
    height: 0.45rem;
    border-radius: 9999px;
    background: rgba(123, 64, 212, 0.15);
    transition: background 0.4s ease, transform 0.3s ease;
}

.progress-step.active {
    background: var(--brand-gradient);
    transform: scaleY(1.3);
    box-shadow: 0 0 10px rgba(232, 67, 147, 0.5);
}

.progress-step.done {
    background: var(--brand-violet);
    opacity: 0.7;
}

/* Question & Result Presentation */
.question-card {
    text-align: center;
    padding: 2.2rem 1.5rem;
}

.question-prompt {
    font-size: 1.35rem;
    font-weight: 600;
    line-height: 1.45;
    margin-bottom: 2rem;
    color: var(--pico-color);
}

.answer-buttons {
    display: flex;
    gap: 1.25rem;
    justify-content: center;
    flex-wrap: wrap;
}

.answer-buttons button {
    min-width: 9.5rem;
}

/* Quadrant Pill Badges */
.quadrant-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    padding: 0.4rem 1.1rem;
    border-radius: 9999px;
    font-size: 0.85rem;
    font-weight: 700;
    margin-bottom: 1.25rem;
    text-transform: uppercase;
    letter-spacing: 0.04em;
}

.intensity-badge {
    display: inline-flex;
    align-items: center;
    padding: 0.3rem 0.9rem;
    border-radius: 9999px;
    font-size: 0.75rem;
    font-weight: 600;
    margin: 0 0 1.25rem 0.5rem;
    color: var(--pico-muted-color);
    border: 1px solid var(--pico-muted-border-color);
}

.q-alta_positiva {
    background: linear-gradient(135deg, rgba(255, 154, 68, 0.2), rgba(232, 67, 147, 0.2));
    color: #e84393;
    border: 1px solid rgba(232, 67, 147, 0.35);
}

.q-alta_negativa {
    background: linear-gradient(135deg, rgba(255, 65, 108, 0.2), rgba(138, 35, 135, 0.2));
    color: #ff416c;
    border: 1px solid rgba(255, 65, 108, 0.35);
}

.q-baja_positiva {
    background: linear-gradient(135deg, rgba(0, 180, 219, 0.2), rgba(108, 92, 231, 0.2));
    color: #00b4db;
    border: 1px solid rgba(0, 180, 219, 0.35);
}

.q-baja_negativa {
    background: linear-gradient(135deg, rgba(83, 105, 118, 0.2), rgba(41, 46, 73, 0.2));
    color: #7d8fa9;
    border: 1px solid rgba(125, 143, 169, 0.35);
}

/* Final Emotion Result */
.emotion-result-box {
    text-align: center;
    padding: 2rem 1.5rem 1rem;
}

.emotion-emoji-giant {
    font-size: 5.5rem;
    line-height: 1;
    margin-bottom: 1.25rem;
    display: inline-block;
    animation: pulseEmoji 3s ease-in-out infinite;
}

@keyframes pulseEmoji {
    0%, 100% { transform: scale(1); }
    50% { transform: scale(1.08); }
}

.emotion-title-final {
    font-size: 2.2rem;
    font-weight: 800;
    margin-bottom: 0.75rem;
    background: var(--brand-gradient);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}

.emotion-desc-final {
    font-size: 1.15rem;
    color: var(--pico-muted-color);
    max-width: 580px;
    margin: 0 auto 1.5rem;
    line-height: 1.6;
}

/* Spinner */
.spinner-box {
    text-align: center;
    padding: 1.5rem;
    color: var(--brand-violet);
    font-weight: 600;
}
"""

# ---------------------------------------------------------------------------
# App configuration (Explicit secret_key avoids read-only filesystem crash on Vercel)
# ---------------------------------------------------------------------------

_SESSION_SECRET = os.environ.get(
    "SESSION_SECRET",
    "emotion-finder-production-secret-key-32bytes-min-security"
)

app, rt = fast_app(
    pico=True,
    static_path="public",
    secret_key=_SESSION_SECRET,
    hdrs=(
        Meta(charset="utf-8"),
        Meta(name="viewport", content="width=device-width, initial-scale=1"),
        Meta(name="description", content="Interactive emotion detector using Russell's Circumplex Model of Affect"),
        Link(rel="icon", type="image/png", href="/favicon.png"),
        Style(_CUSTOM_CSS),
    ),
)

# Explicit top-level aliases for Vercel ASGI serverless function AST detection
app = app
application = app
handler = app

# ---------------------------------------------------------------------------
# Internationalization strings
# ---------------------------------------------------------------------------

_I18N = {
    "es": {
        "title": "Emotion Finder",
        "hero_title": "¿Qué estás sintiendo en este momento?",
        "subtitle": "Mapea tus sensaciones físicas y mentales para descubrir tu emoción exacta",
        "placeholder": "Describe brevemente cómo te sientes... (ej: siento un nudo en la garganta y me hierve la sangre de la rabia)",
        "submit": "Analizar mi emoción",
        "analyzing": "Analizando sensaciones...",
        "uncertain_title": "Necesitamos más detalles",
        "uncertain_msg": "No pude identificar con certeza tus sensaciones. Intenta describir qué pasa en tu cuerpo (palpitaciones, respiración, músculos, estómago...).",
        "try_again": "Intentar de nuevo",
        "yes": "Sí",
        "no": "No",
        "your_emotion": "Tu emoción identificada es:",
        "your_secondary_emotion": "Esta es tu emoción alternativa:",
        "secondary_prompt": "🌗 También podrías estar sintiendo algo distinto. ¿Quieres explorar ese otro camino?",
        "explore_secondary": "Explorar esa otra posibilidad",
        "explore_manually": "¿No es exacto? Explorar con preguntas",
        "step": "Paso",
        "of": "de",
        "footer_text": "Emotion Finder · Basado en el Modelo Circunflejo del Afecto de Russell",
        "feedback_prompt": "¿Acertamos con tu emoción?",
        "feedback_thumbs_up": "👍 Sí, acertaron",
        "feedback_thumbs_down": "👎 No del todo",
        "feedback_correct_prompt": "¿Cuál emoción describe mejor lo que sientes?",
        "feedback_select_emotion": "Selecciona una emoción...",
        "feedback_or_quadrant": "O selecciona solo el cuadrante:",
        "feedback_comments_placeholder": "Comentario opcional (máx 150 caracteres)...",
        "feedback_submit": "Enviar corrección",
        "feedback_cancel": "Cancelar",
        "feedback_thanks": "¡Gracias por tu aporte! Tu feedback ayuda a entrenar el algoritmo.",
    },
    "en": {
        "title": "Emotion Finder",
        "hero_title": "What are you feeling right now?",
        "subtitle": "Map your physical and mental sensations to uncover your exact emotion",
        "placeholder": "Briefly describe how you feel... (e.g., my heart is racing and I feel overwhelmed by stress)",
        "submit": "Analyze my emotion",
        "analyzing": "Analyzing sensations...",
        "uncertain_title": "We need a bit more detail",
        "uncertain_msg": "I couldn't identify your sensations with certainty. Try describing physical signals in your body (heart rate, breathing, muscle tension, stomach...).",
        "try_again": "Try again",
        "yes": "Yes",
        "no": "No",
        "your_emotion": "Your identified emotion is:",
        "your_secondary_emotion": "Here's your alternate emotion:",
        "secondary_prompt": "🌗 You might also be feeling something different. Want to explore that other path?",
        "explore_secondary": "Explore that other possibility",
        "explore_manually": "Not quite right? Explore with questions",
        "step": "Step",
        "of": "of",
        "footer_text": "Emotion Finder · Grounded in Russell's Circumplex Model of Affect",
        "feedback_prompt": "Did we get your emotion right?",
        "feedback_thumbs_up": "👍 Yes, exactly",
        "feedback_thumbs_down": "👎 Not quite",
        "feedback_correct_prompt": "Which emotion best describes how you feel?",
        "feedback_select_emotion": "Select an emotion...",
        "feedback_or_quadrant": "Or select just the quadrant:",
        "feedback_comments_placeholder": "Optional comments (max 150 characters)...",
        "feedback_submit": "Submit correction",
        "feedback_cancel": "Cancel",
        "feedback_thanks": "Thank you! Your feedback helps train the algorithm.",
    },
}


_QUADRANT_LABELS = {
    "es": {
        "alta_positiva": "⚡ Alta Activación · Positiva",
        "alta_negativa": "🔥 Alta Activación · Negativa",
        "baja_positiva": "🌿 Baja Activación · Positiva",
        "baja_negativa": "🌧️ Baja Activación · Negativa",
    },
    "en": {
        "alta_positiva": "⚡ High Arousal · Positive",
        "alta_negativa": "🔥 High Arousal · Negative",
        "baja_positiva": "🌿 Low Arousal · Positive",
        "baja_negativa": "🌧️ Low Arousal · Negative",
    },
}

_INTENSITY_LABELS = {
    "es": {"baja": "Intensidad baja", "media": "Intensidad media", "alta": "Intensidad alta"},
    "en": {"baja": "Low intensity", "media": "Medium intensity", "alta": "High intensity"},
}


def t(key: str, lang: str = "es") -> str:
    """Get translated string for the given key and language."""
    return _I18N.get(lang, _I18N["es"]).get(key, key)


# ---------------------------------------------------------------------------
# UI Components
# ---------------------------------------------------------------------------


def _page_shell(content, lang: str = "es"):
    """Wrap content in modern responsive layout with brand navigation and footer."""
    return (
        Title(f"{t('title', lang)} — {t('subtitle', lang)}"),
        Header(
            Nav(
                Ul(
                    Li(
                        A(
                            Img(src="/icon.png", alt="Emotion Finder Logo", cls="brand-icon-sm"),
                            Span("Emotion Finder", style="margin-left: 8px;"),
                            href=f"/?lang={lang}",
                            cls="nav-brand",
                        )
                    )
                ),
                Ul(
                    Li(
                        Div(
                            A(
                                "ES",
                                href="/?lang=es",
                                cls=f"lang-pill {'active' if lang == 'es' else ''}",
                            ),
                            A(
                                "EN",
                                href="/?lang=en",
                                cls=f"lang-pill {'active' if lang == 'en' else ''}",
                            ),
                            cls="lang-switcher",
                        )

                    ),
                ),
            ),
            cls="container",
        ),
        Main(content, cls="container"),
        Footer(
            P(Small(t("footer_text", lang)), style="text-align: center; color: var(--pico-muted-color);"),
            cls="container",
            style="margin-top: 3rem; padding-bottom: 2rem;",
        ),
    )


def _progress_bar(current_step: int, total_steps: int = 5):
    """Render a visual progress bar with brand gradient styling."""
    steps = []
    for i in range(1, total_steps + 1):
        if i < current_step:
            step_cls = "progress-step done"
        elif i == current_step:
            step_cls = "progress-step active"
        else:
            step_cls = "progress-step"
        steps.append(Div(cls=step_cls))
    return Div(*steps, cls="progress-bar")


def _render_question(question_text: str, quadrant: str, path: str, lang: str, step: int,
                      intensity: str = "", secondary: str = "0", user_text: str = ""):
    """Helper to render a binary question card with Yes/No HTMX buttons (DRY)."""
    next_yes = f"{path}.yes" if path else "yes"
    next_no = f"{path}.no" if path else "no"
    next_step = str(step + 1)

    return Article(
        _progress_bar(current_step=step, total_steps=5),
        Div(
            H3(question_text, cls="question-prompt"),
            Div(
                Button(
                    t("yes", lang),
                    hx_post="/tree",
                    hx_target="#result-area",
                    hx_swap="innerHTML transition:true",
                    hx_vals={
                        "quadrant": quadrant,
                        "path": next_yes,
                        "lang": lang,
                        "step": next_step,
                        "intensity": intensity,
                        "secondary": secondary,
                        "user_text": user_text,
                    },
                    cls="btn-gradient",
                ),
                Button(
                    t("no", lang),
                    hx_post="/tree",
                    hx_target="#result-area",
                    hx_swap="innerHTML transition:true",
                    hx_vals={
                        "quadrant": quadrant,
                        "path": next_no,
                        "lang": lang,
                        "step": next_step,
                        "intensity": intensity,
                        "secondary": secondary,
                        "user_text": user_text,
                    },
                    cls="btn-outline",
                ),
                cls="answer-buttons",
            ),
            cls="question-card",
        ),
        cls="brand-card",
    )


# ---------------------------------------------------------------------------
# Routes (Disambiguated names eliminate module shadowing)
# ---------------------------------------------------------------------------


@rt("/")
def get(lang: str = "es"):
    """Render the home page with brand hero section and input form."""
    if lang not in ("es", "en"):
        lang = "es"

    content = (
        Div(
            H1(t("hero_title", lang), cls="gradient-title"),
            P(t("subtitle", lang), cls="hero-subtitle"),
            cls="hero-wrapper",
        ),


        Article(
            Form(
                Textarea(
                    name="text",
                    placeholder=t("placeholder", lang),
                    required=True,
                    rows=3,
                    style="border-radius: 16px; font-size: 1.05rem; padding: 1rem;",
                    onkeydown="if(event.key === 'Enter' && !event.shiftKey) { event.preventDefault(); this.form.requestSubmit(); }",
                ),
                Input(type="hidden", name="lang", value=lang),
                Div(
                    Button(
                        t("submit", lang),
                        type="submit",
                        cls="btn-gradient",
                        style="width: 100%; font-size: 1.1rem;",
                    ),
                    style="margin-top: 1rem;",
                ),
                Div(
                    Span(aria_busy="true"),
                    Small(f" {t('analyzing', lang)}"),
                    id="spinner",
                    cls="htmx-indicator spinner-box",
                ),
                hx_post="/predict",
                hx_target="#result-area",
                hx_swap="innerHTML transition:true",
                hx_indicator="#spinner",
            ),
            cls="brand-card",
            style="padding: 2rem;",
        ),
        Div(id="result-area"),
    )
    return _page_shell(content, lang)


def _render_feedback_widget(user_text: str, lang: str, quadrant: str, emotion: str):
    """Render the initial inline thumbs-up / thumbs-down feedback buttons."""
    if not user_text or len(user_text.strip()) < 2:
        return None

    return Div(
        P(
            t("feedback_prompt", lang),
            style="font-size: 0.95rem; font-weight: 600; color: var(--pico-muted-color); margin-bottom: 0.6rem;",
        ),
        Div(
            Button(
                t("feedback_thumbs_up", lang),
                hx_post="/feedback",
                hx_target="#feedback-container",
                hx_swap="innerHTML transition:true",
                hx_vals={
                    "user_text": user_text,
                    "lang": lang,
                    "quadrant": quadrant,
                    "emotion": emotion,
                    "rating": "positive",
                },
                cls="btn-outline",
                style="padding: 0.45rem 1.1rem; font-size: 0.9rem;",
            ),
            Button(
                t("feedback_thumbs_down", lang),
                hx_post="/feedback-form",
                hx_target="#feedback-container",
                hx_swap="innerHTML transition:true",
                hx_vals={
                    "user_text": user_text,
                    "lang": lang,
                    "quadrant": quadrant,
                    "emotion": emotion,
                },
                cls="btn-outline",
                style="padding: 0.45rem 1.1rem; font-size: 0.9rem;",
            ),
            style="display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap;",
        ),
        id="feedback-container",
        style="margin-top: 2rem; padding-top: 1.25rem; border-top: 1px solid rgba(123, 64, 212, 0.15);",
    )


def _render_emotion_result(node: dict, quadrant: str, lang: str,
                            intensity: str = "", secondary: str = "0",
                            manual_explore_path: str | None = None,
                            user_text: str = ""):
    """Render a leaf emotion node (name, emoji, description) as the final result card."""
    emotion_key = f"emotion_{lang}"
    desc_key = f"description_{lang}"

    emotion_name = node.get(emotion_key, node.get("emotion_es", ""))
    emoji = node.get("emoji", "🎯")
    description = node.get(desc_key, node.get("description_es", ""))
    quadrant_label = _QUADRANT_LABELS.get(lang, _QUADRANT_LABELS["es"]).get(quadrant, quadrant)
    heading_key = "your_secondary_emotion" if secondary == "1" else "your_emotion"

    intensity_label = _INTENSITY_LABELS.get(lang, {}).get(intensity)
    intensity_badge = (
        Span(intensity_label, cls="intensity-badge") if intensity_label else None
    )

    manual_explore_button = None
    if manual_explore_path is not None:
        manual_explore_button = Button(
            t("explore_manually", lang),
            hx_post="/tree",
            hx_target="#result-area",
            hx_swap="innerHTML transition:true",
            hx_vals={
                "quadrant": quadrant,
                "path": manual_explore_path,
                "lang": lang,
                "step": "2",
                "intensity": intensity,
                "secondary": secondary,
                "user_text": user_text,
            },
            cls="btn-outline",
        )

    feedback_block = _render_feedback_widget(user_text, lang, quadrant, emotion_name)

    return Article(
        _progress_bar(current_step=5, total_steps=5),
        Div(
            Span(quadrant_label, cls=f"quadrant-badge q-{quadrant}"),
            *((intensity_badge,) if intensity_badge else ()),
            Div(emoji, cls="emotion-emoji-giant"),
            P(t(heading_key, lang), style="color: var(--pico-muted-color); font-size: 0.95rem; margin-bottom: 0.25rem;"),
            H2(emotion_name, cls="emotion-title-final"),
            P(description, cls="emotion-desc-final"),
            Div(
                A(
                    f"🔄 {t('try_again', lang)}",
                    href=f"/?lang={lang}",
                    role="button",
                    cls="btn-gradient",
                ),
                *((manual_explore_button,) if manual_explore_button else ()),
                style="margin-top: 1.5rem; display: flex; gap: 0.75rem; justify-content: center; flex-wrap: wrap;",
            ),
            *((feedback_block,) if feedback_block else ()),
            cls="emotion-result-box",
        ),
        cls="brand-card",
    )


@app.post("/predict")
def predict_route(text: str, lang: str = "es"):
    """Receive user text, predict Russell quadrant, return first question."""
    if lang not in ("es", "en"):
        lang = detect_language(text)

    result = predict_quadrant(text)

    if result["low_confidence"]:
        return Article(
            Div(
                H3(f"🤔 {t('uncertain_title', lang)}", cls="question-prompt"),
                P(t("uncertain_msg", lang), style="color: var(--pico-muted-color); font-size: 1.05rem; margin-bottom: 1.5rem;"),
                A(
                    t("try_again", lang),
                    href=f"/?lang={lang}",
                    role="button",
                    cls="btn-outline",
                ),
                cls="question-card",
            ),
            cls="brand-card",
        )

    quadrant = result["quadrant"]
    quadrant_label = _QUADRANT_LABELS.get(lang, _QUADRANT_LABELS["es"]).get(quadrant, quadrant)
    intensity = result.get("intensity") or ""
    secondary_quadrant = result.get("secondary_quadrant")

    tree_root = get_tree(quadrant)
    if tree_root is None:
        return Div(P("Error: Decision tree not found for this quadrant."))

    extra_blocks = []
    if secondary_quadrant:
        extra_blocks.append(Div(
            P(t("secondary_prompt", lang), style="color: var(--pico-muted-color); font-size: 0.95rem; margin: 0.75rem 0 0.25rem;"),
            Button(
                t("explore_secondary", lang),
                hx_post="/tree",
                hx_target="#result-area",
                hx_swap="innerHTML transition:true",
                hx_vals={
                    "quadrant": secondary_quadrant,
                    "path": "",
                    "lang": lang,
                    "step": "2",
                    "intensity": intensity,
                    "secondary": "1",
                    "user_text": text,
                },
                cls="btn-outline",
            ),
            style="text-align: center;",
        ))

    # Try to jump straight to the closest of the quadrant's 16 emotions by
    # comparing the user's own words against each one's description, instead
    # of always asking four generic yes/no sensation questions. Falls back to
    # the manual tree below when nothing matches confidently.
    match = match_emotion(text, quadrant, lang)
    if match is not None:
        node, score = match
        if is_confident_match(score):
            return Div(
                _render_emotion_result(
                    node, quadrant, lang, intensity=intensity,
                    manual_explore_path="", user_text=text,
                ),
                *extra_blocks,
            )

    question_key = f"question_{lang}"
    question_text = tree_root.get(question_key, tree_root.get("question_es", ""))

    return Div(
        Div(
            Span(quadrant_label, cls=f"quadrant-badge q-{quadrant}"),
            style="text-align: center; margin-bottom: 0.5rem;",
        ),
        _render_question(
            question_text=question_text,
            quadrant=quadrant,
            path="",
            lang=lang,
            step=2,
            intensity=intensity,
            user_text=text,
        ),
        *extra_blocks,
    )


@app.post("/tree")
def tree_route(quadrant: str, path: str, lang: str = "es", step: str = "3",
               intensity: str = "", secondary: str = "0", user_text: str = ""):
    """Navigate binary decision tree and render next question or final emotion."""
    if lang not in ("es", "en"):
        lang = "es"

    current_step = int(step) if step.isdigit() else 3
    node = get_node(quadrant, path)

    if node is None:
        return Article(
            P("Error: Could not navigate the decision tree.", style="text-align:center;"),
            A(t("try_again", lang), href=f"/?lang={lang}", role="button", cls="btn-outline"),
            cls="brand-card",
        )

    # Leaf node reached (Emotion detected)
    if "emotion_es" in node:
        return _render_emotion_result(
            node, quadrant, lang, intensity=intensity, secondary=secondary,
            user_text=user_text
        )

    # Intermediate question node
    question_key = f"question_{lang}"
    question_text = node.get(question_key, node.get("question_es", ""))

    return _render_question(
        question_text=question_text,
        quadrant=quadrant,
        path=path,
        lang=lang,
        step=current_step,
        intensity=intensity,
        secondary=secondary,
        user_text=user_text,
    )


# ---------------------------------------------------------------------------
# Feedback & Active Learning Routes
# ---------------------------------------------------------------------------


@app.post("/feedback")
def feedback_route(
    user_text: str = "",
    lang: str = "es",
    quadrant: str = "",
    emotion: str = "",
    rating: str = "positive",
    corrected_quadrant: str = "",
    corrected_emotion: str = "",
    comments: str = "",
    hp_confirm: str = "",
):
    """Receive and securely persist user feedback with anti-bot and fail-open defenses."""
    if lang not in ("es", "en"):
        lang = "es"

    # Anti-bot honeypot defense: silent fake OK
    if hp_confirm:
        return Div(
            P(f"✨ {t('feedback_thanks', lang)}",
              style="color: var(--brand-violet); font-weight: 600; font-size: 0.95rem; margin: 0.5rem 0;"),
            style="text-align: center;",
        )

    clean_text = (user_text or "").strip()
    if not clean_text or len(clean_text) < 2:
        return Div(
            P(f"✨ {t('feedback_thanks', lang)}",
              style="color: var(--brand-violet); font-weight: 600; font-size: 0.95rem; margin: 0.5rem 0;"),
            style="text-align: center;",
        )

    clean_text = clean_text[:300]
    clean_comments = (comments or "").strip()[:150] if comments else None

    # Resolve canonical quadrant: specific emotion takes precedence over radio button
    final_corrected_quad = None
    if corrected_emotion:
        for q_key in ["alta_positiva", "alta_negativa", "baja_positiva", "baja_negativa"]:
            for e in get_quadrant_emotions(q_key):
                if e.get("emotion_es") == corrected_emotion or e.get("emotion_en") == corrected_emotion:
                    final_corrected_quad = q_key
                    break
            if final_corrected_quad:
                break
    if not final_corrected_quad and corrected_quadrant in ["alta_positiva", "alta_negativa", "baja_positiva", "baja_negativa"]:
        final_corrected_quad = corrected_quadrant

    record = create_feedback_record(
        user_text=clean_text,
        detected_lang=lang,
        predicted_quadrant=quadrant,
        predicted_emotion=emotion,
        model_confidence=1.0,
        rating=rating,
        corrected_quadrant=final_corrected_quad,
        corrected_emotion=corrected_emotion if corrected_emotion else None,
        comments=clean_comments,
    )

    store = get_feedback_store()
    store.save(record)

    return Div(
        P(
            f"✨ {t('feedback_thanks', lang)}",
            style="color: var(--brand-violet); font-weight: 600; font-size: 0.95rem; margin: 0.5rem 0;",
        ),
        style="text-align: center;",
    )


@app.post("/feedback-form")
def feedback_form_route(user_text: str, lang: str = "es", quadrant: str = "", emotion: str = ""):
    """Render the expanded corrective feedback form via HTMX."""
    if lang not in ("es", "en"):
        lang = "es"

    optgroups = []
    for q_key in ["alta_positiva", "alta_negativa", "baja_positiva", "baja_negativa"]:
        q_label = _QUADRANT_LABELS.get(lang, {}).get(q_key, q_key)
        emotions = get_quadrant_emotions(q_key)
        options = [
            Option(
                f"{e.get('emoji', '')} {e.get(f'emotion_{lang}', e.get('emotion_es', ''))}",
                value=e.get(f"emotion_{lang}", e.get("emotion_es", "")),
            )
            for e in emotions
        ]
        optgroups.append(Optgroup(*options, label=q_label))

    emotion_select = Select(
        Option(t("feedback_select_emotion", lang), value="", selected=True, disabled=True),
        *optgroups,
        name="corrected_emotion",
        id="corrected-emotion-select",
        style="border-radius: 12px; margin-bottom: 0.75rem;",
    )

    quad_radios = [
        Label(
            Input(type="radio", name="corrected_quadrant", value=q_key),
            f" {_QUADRANT_LABELS.get(lang, {}).get(q_key, q_key)}",
            style="display: inline-flex; align-items: center; margin-right: 1rem; font-size: 0.85rem;",
        )
        for q_key in ["alta_positiva", "alta_negativa", "baja_positiva", "baja_negativa"]
    ]

    return Div(
        Form(
            # Honeypot field (anti-bot)
            Input(type="text", name="hp_confirm", style="display:none !important;", tabindex="-1", autocomplete="off"),
            # Context fields
            Input(type="hidden", name="user_text", value=user_text),
            Input(type="hidden", name="lang", value=lang),
            Input(type="hidden", name="quadrant", value=quadrant),
            Input(type="hidden", name="emotion", value=emotion),
            Input(type="hidden", name="rating", value="negative"),

            P(t("feedback_correct_prompt", lang), style="font-weight: 600; font-size: 0.95rem; margin-bottom: 0.5rem; text-align: left;"),
            emotion_select,

            P(Small(t("feedback_or_quadrant", lang)), style="color: var(--pico-muted-color); margin-bottom: 0.35rem; text-align: left;"),
            Div(*quad_radios, style="display: flex; flex-wrap: wrap; gap: 0.5rem; margin-bottom: 0.75rem; justify-content: flex-start;"),

            Input(
                type="text",
                name="comments",
                placeholder=t("feedback_comments_placeholder", lang),
                maxlength="150",
                style="border-radius: 12px; font-size: 0.9rem; margin-bottom: 1rem;",
            ),

            Div(
                Button(
                    t("feedback_submit", lang),
                    type="submit",
                    cls="btn-gradient",
                    style="padding: 0.5rem 1.25rem; font-size: 0.9rem;",
                ),
                Button(
                    t("feedback_cancel", lang),
                    hx_post="/feedback-reset",
                    hx_target="#feedback-container",
                    hx_swap="innerHTML transition:true",
                    hx_vals={
                        "user_text": user_text,
                        "lang": lang,
                        "quadrant": quadrant,
                        "emotion": emotion,
                    },
                    cls="btn-outline",
                    style="padding: 0.5rem 1.25rem; font-size: 0.9rem;",
                ),
                style="display: flex; gap: 0.75rem; justify-content: center;",
            ),
            hx_post="/feedback",
            hx_target="#feedback-container",
            hx_swap="innerHTML transition:true",
            style="max-width: 520px; margin: 0 auto; text-align: center;",
        ),
        id="feedback-container",
        style="margin-top: 2rem; padding-top: 1.25rem; border-top: 1px solid rgba(123, 64, 212, 0.15);",
    )


@app.post("/feedback-reset")
def feedback_reset_route(user_text: str, lang: str = "es", quadrant: str = "", emotion: str = ""):
    """Reset back to initial thumbs-up / thumbs-down buttons if user cancels."""
    if lang not in ("es", "en"):
        lang = "es"
    widget = _render_feedback_widget(user_text, lang, quadrant, emotion)
    return widget or Div(id="feedback-container")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    serve()
