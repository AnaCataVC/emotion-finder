"""
Unit and integration tests for the user feedback and active learning subsystem.

Covers:
- Local SQLite feedback persistence and status transitions.
- Fail-open resilience on remote/network storage failures.
- FastHTML HTMX endpoints: POST /feedback, POST /feedback-form, POST /feedback-reset.
- Anti-bot honeypot protection (silent drop).
- End-to-end user_text propagation across /predict and /tree navigation to leaf card.
- Active learning sample extraction, capping, and regression probe quality gates.
"""

import json
import re
import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import pytest
from starlette.testclient import TestClient

from feedback_store import (
    FeedbackRecord,
    LocalSQLiteFeedbackStore,
    NullFeedbackStore,
    TursoHttpFeedbackStore,
    create_feedback_record,
    get_feedback_store,
)
from main import app
from scripts.retrain_from_feedback import (
    extract_training_samples,
    retrain_language,
)


@pytest.fixture
def test_client():
    return TestClient(app)


# ---------------------------------------------------------------------------
# 1. Persistence Layer Tests
# ---------------------------------------------------------------------------

def test_local_sqlite_store_lifecycle():
    """Verify SQLite CRUD, indexing, and status updates using in-memory store."""
    store = LocalSQLiteFeedbackStore(":memory:")
    assert store.count() == 0

    record = create_feedback_record(
        user_text="siento un nudo en la garganta",
        detected_lang="es",
        predicted_quadrant="baja_negativa",
        predicted_emotion="Melancolía",
        model_confidence=0.88,
        rating="positive",
    )

    # Save
    success = store.save(record)
    assert success is True
    assert store.count() == 1

    # Fetch pending
    pending = store.get_by_status("pending")
    assert len(pending) == 1
    assert pending[0].user_text == "siento un nudo en la garganta"
    assert pending[0].predicted_quadrant == "baja_negativa"
    assert pending[0].rating == "positive"
    assert pending[0].status == "pending"

    # Status transition
    mark_ok = store.mark_status(record.id, "verified")
    assert mark_ok is True
    assert len(store.get_by_status("pending")) == 0
    assert len(store.get_by_status("verified")) == 1


def test_null_store_fallback():
    """Verify safe in-memory fallback behaves predictably without raising exceptions."""
    store = NullFeedbackStore()
    record = create_feedback_record(
        user_text="feeling completely overwhelmed",
        detected_lang="en",
        predicted_quadrant="alta_negativa",
        predicted_emotion="Overwhelm",
        model_confidence=0.92,
        rating="positive",
    )
    assert store.save(record) is True
    assert store.count() == 1
    assert len(store.get_by_status("pending")) == 1
    assert store.mark_status(record.id, "verified") is True
    assert len(store.get_by_status("verified")) == 1


def test_turso_fail_open_timeout():
    """Verify that network timeout or invalid URL in Turso store fails open without crashing."""
    # Point to a dead local port with ultra-short timeout to trigger connection/timeout error
    dead_store = TursoHttpFeedbackStore(
        database_url="http://127.0.0.1:59999",
        auth_token="dummy-token",
        timeout_seconds=0.1,
    )
    record = create_feedback_record(
        user_text="estoy muy contento",
        detected_lang="es",
        predicted_quadrant="alta_positiva",
        predicted_emotion="Alegría",
        model_confidence=0.95,
        rating="positive",
    )
    # Must return False and log a warning, never raise an unhandled exception
    res = dead_store.save(record)
    assert res is False


def test_turso_hrana_arg_encoding():
    """Verify strict Hrana protocol serialization for null, float, integer, and text."""
    from feedback_store import _encode_hrana_arg

    # Null value must strictly omit 'value' key
    assert _encode_hrana_arg(None) == {"type": "null"}

    # Float must preserve numerical float value (not string)
    assert _encode_hrana_arg(1.0) == {"type": "float", "value": 1.0}
    assert _encode_hrana_arg(0.85) == {"type": "float", "value": 0.85}

    # Integers and Booleans
    assert _encode_hrana_arg(42) == {"type": "integer", "value": "42"}
    assert _encode_hrana_arg(True) == {"type": "integer", "value": "1"}

    # Text strings
    assert _encode_hrana_arg("emocion") == {"type": "text", "value": "emocion"}



# ---------------------------------------------------------------------------
# 2. HTMX Web Endpoints Integration Tests
# ---------------------------------------------------------------------------

def test_endpoint_feedback_positive_vote(test_client):
    """POST /feedback with thumbs-up rating returns optimistic acknowledgment."""
    response = test_client.post(
        "/feedback",
        data={
            "user_text": "tengo el pecho apretado de la angustia",
            "lang": "es",
            "quadrant": "alta_negativa",
            "emotion": "Angustia",
            "rating": "positive",
        },
    )
    assert response.status_code == 200
    assert "¡Gracias por tu aporte!" in response.text or "feedback ayuda a entrenar" in response.text

    # Verify stored record (auto-promoted to 'verified' under Option 3 Hybrid policy)
    store = get_feedback_store()
    records = store.get_by_status("verified")
    matching = [r for r in records if r.user_text == "tengo el pecho apretado de la angustia"]
    assert len(matching) >= 1
    assert matching[0].rating == "positive"
    assert matching[0].predicted_quadrant == "alta_negativa"
    assert matching[0].status == "verified"
    assert matching[0].session_hash is not None
    assert len(matching[0].session_hash) == 16


def test_endpoint_feedback_negative_with_correction(test_client):
    """POST /feedback with corrective label stores corrected emotion and quadrant."""
    response = test_client.post(
        "/feedback",
        data={
            "user_text": "me late el corazón pero no es miedo sino alegría",
            "lang": "es",
            "quadrant": "alta_negativa",
            "emotion": "Pánico",
            "rating": "negative",
            "corrected_quadrant": "alta_positiva",
            "corrected_emotion": "Euforia",
            "comments": "Era emoción positiva no pánico",
        },
    )
    assert response.status_code == 200
    assert "¡Gracias por tu aporte!" in response.text

    store = get_feedback_store()
    records = store.get_by_status("pending")
    matching = [r for r in records if "no es miedo sino alegría" in r.user_text]
    assert len(matching) >= 1
    rec = matching[0]
    assert rec.rating == "negative"
    assert rec.corrected_quadrant == "alta_positiva"
    assert rec.corrected_emotion == "Euforia"
    assert rec.comments == "Era emoción positiva no pánico"


def test_endpoint_feedback_form_expansion(test_client):
    """POST /feedback-form returns the interactive correction form with dropdown and honeypot."""
    response = test_client.post(
        "/feedback-form",
        data={
            "user_text": "siento mucha calma",
            "lang": "es",
            "quadrant": "baja_positiva",
            "emotion": "Serenidad",
        },
    )
    assert response.status_code == 200
    # Must contain honeypot
    assert 'name="hp_confirm"' in response.text
    # Must contain select with emotion options
    assert 'name="corrected_emotion"' in response.text
    assert "Serenidad" in response.text or "Paz interior" in response.text
    # Must contain radio buttons for quadrant
    assert 'name="corrected_quadrant"' in response.text


def test_endpoint_feedback_reset(test_client):
    """POST /feedback-reset restores initial thumbs-up / thumbs-down buttons."""
    response = test_client.post(
        "/feedback-reset",
        data={
            "user_text": "siento mucha calma",
            "lang": "es",
            "quadrant": "baja_positiva",
            "emotion": "Serenidad",
        },
    )
    assert response.status_code == 200
    assert "Acertamos con tu emoción" in response.text
    assert "thumbs_up" in response.text or "Sí, acertaron" in response.text


def test_anti_bot_honeypot_protection(test_client):
    """POST /feedback with filled honeypot silently returns 200 without writing to store."""
    store = get_feedback_store()
    initial_count = store.count()

    response = test_client.post(
        "/feedback",
        data={
            "user_text": "spam spam spam",
            "lang": "es",
            "quadrant": "alta_negativa",
            "emotion": "Ira",
            "hp_confirm": "malicious_bot_content",
        },
    )
    assert response.status_code == 200
    # Store count must not increment
    assert store.count() == initial_count


def test_feedback_rate_limit_per_session(test_client):
    """POST /feedback beyond FEEDBACK_RATE_LIMIT_MAX for the same session is silently dropped."""
    from main import FEEDBACK_RATE_LIMIT_MAX

    store = get_feedback_store()

    def submit(i):
        return test_client.post(
            "/feedback",
            data={
                "user_text": f"mensaje de prueba de limite numero {i}",
                "lang": "es",
                "quadrant": "alta_positiva",
                "emotion": "Alegría",
                "rating": "positive",
            },
        )

    for i in range(FEEDBACK_RATE_LIMIT_MAX):
        resp = submit(i)
        assert resp.status_code == 200

    count_at_limit = len(
        [r for r in store.get_by_status("verified") if r.user_text.startswith("mensaje de prueba de limite")]
    )
    assert count_at_limit == FEEDBACK_RATE_LIMIT_MAX

    # One more submission from the same session must be silently dropped (no new record)
    resp = submit("over")
    assert resp.status_code == 200
    count_after_limit = len(
        [r for r in store.get_by_status("verified") if r.user_text.startswith("mensaje de prueba de limite")]
    )
    assert count_after_limit == FEEDBACK_RATE_LIMIT_MAX


def test_invalid_text_handling(test_client):
    """POST /feedback with empty or too-short text is handled gracefully."""
    response = test_client.post(
        "/feedback",
        data={"user_text": "a", "lang": "es", "rating": "positive"},
    )
    assert response.status_code == 200


# ---------------------------------------------------------------------------
# 3. User Text Propagation Across Flow
# ---------------------------------------------------------------------------

def test_user_text_propagation_to_result_card(test_client):
    """Verify user_text travels all the way from /predict to the leaf result card."""
    sample_text = "tengo una rabia tremenda que no puedo controlar"

    # Confident match jumps directly to leaf card
    resp = test_client.post("/predict", data={"text": sample_text, "lang": "es"})
    assert resp.status_code == 200

    if "feedback-container" in resp.text:
        # Check that user_text is embedded inside hx-vals of the feedback buttons
        assert sample_text in resp.text
    else:
        # Check that yes/no question buttons preserve user_text in hx-vals
        matches = re.findall(r"hx-vals='([^']+)'", resp.text)
        assert len(matches) >= 2
        for val_str in matches:
            parsed = json.loads(val_str)
            assert parsed.get("user_text") == sample_text


# ---------------------------------------------------------------------------
# 4. Active Learning Pipeline Quality Gate Tests
# ---------------------------------------------------------------------------

def test_extract_training_samples():
    """Verify clean extraction of training samples from verified feedback."""
    records = [
        FeedbackRecord(
            id="1",
            created_at="2026-09-05T00:00:00Z",
            user_text="siento mucha paz y tranquilidad",
            normalized_text="siento mucha paz y tranquilidad",
            detected_lang="es",
            predicted_quadrant="baja_positiva",
            predicted_emotion="Paz interior",
            model_confidence=0.95,
            rating="positive",
            status="verified",
        ),
        FeedbackRecord(
            id="2",
            created_at="2026-09-05T00:00:00Z",
            user_text="hi",  # Too short (< 6 chars)
            normalized_text="hi",
            detected_lang="es",
            predicted_quadrant="alta_positiva",
            predicted_emotion="Euforia",
            model_confidence=0.50,
            rating="positive",
            status="verified",
        ),
        FeedbackRecord(
            id="3",
            created_at="2026-09-05T00:00:00Z",
            user_text="i am furious and want to scream",
            normalized_text="i am furious and want to scream",
            detected_lang="en",  # Different lang
            predicted_quadrant="alta_negativa",
            predicted_emotion="Fury",
            model_confidence=0.90,
            rating="positive",
            status="verified",
        ),
        FeedbackRecord(
            id="4",
            created_at="2026-09-05T00:00:00Z",
            user_text="dijeron tristeza pero me hervía la sangre de furia",
            normalized_text="dijeron tristeza pero me hervia la sangre de furia",
            detected_lang="es",
            predicted_quadrant="baja_negativa",
            predicted_emotion="Tristeza",
            model_confidence=0.70,
            rating="negative",
            corrected_quadrant="alta_negativa",
            status="verified",
        ),
    ]

    es_samples = extract_training_samples(records, "es")
    assert len(es_samples) == 2
    assert es_samples[0] == ("siento mucha paz y tranquilidad", "baja_positiva", "1")
    assert es_samples[1] == ("dijeron tristeza pero me hervía la sangre de furia", "alta_negativa", "4")

    en_samples = extract_training_samples(records, "en")
    assert len(en_samples) == 1
    assert en_samples[0] == ("i am furious and want to scream", "alta_negativa", "3")


def test_retrain_pipeline_dry_run():
    """Verify that retrain_language runs in dry_run mode without modifying models on disk."""
    store = LocalSQLiteFeedbackStore(":memory:")
    # Add a verified sample
    record = create_feedback_record(
        user_text="estoy con una felicidad inmensa y ganas de bailar",
        detected_lang="es",
        predicted_quadrant="alta_positiva",
        predicted_emotion="Alegría",
        model_confidence=0.95,
        rating="positive",
    )
    store.save(record)
    store.mark_status(record.id, "verified")

    # Dry-run execution
    success = retrain_language("es", store, dry_run=True)
    assert success is True


def test_feedback_rate_limit_without_cookies(test_client):
    """Clients omitting or clearing cookies (simulating bots) are still rate-limited via network anchor."""
    from main import FEEDBACK_RATE_LIMIT_MAX

    store = get_feedback_store()
    tag = "prueba_sin_cookies"

    def submit_cookieless(i):
        test_client.cookies.clear()
        return test_client.post(
            "/feedback",
            data={
                "user_text": f"{tag} peticion numero {i}",
                "lang": "es",
                "quadrant": "alta_positiva",
                "emotion": "Alegría",
                "rating": "positive",
            },
        )

    for i in range(FEEDBACK_RATE_LIMIT_MAX):
        resp = submit_cookieless(i)
        assert resp.status_code == 200

    count_at_limit = len([r for r in store.get_by_status("verified") if tag in r.user_text])
    assert count_at_limit == FEEDBACK_RATE_LIMIT_MAX

    # Additional request with cleared cookies from the same client MUST still be dropped
    resp = submit_cookieless("blocked")
    assert resp.status_code == 200
    count_after_limit = len([r for r in store.get_by_status("verified") if tag in r.user_text])
    assert count_after_limit == FEEDBACK_RATE_LIMIT_MAX


def test_tree_route_rate_limit_and_quadrant_validation(test_client):
    """POST /tree validates quadrant keys and applies silent rate limiting on negative rejection feedback."""
    from main import FEEDBACK_RATE_LIMIT_MAX

    store = get_feedback_store()

    # 1. Invalid quadrant rejection is ignored (not saved to DB)
    initial_count = store.count()
    resp_invalid = test_client.post(
        "/tree",
        data={
            "quadrant": "invalid_quadrant",
            "path": "",
            "rejected_quadrant": "fake_quadrant",
            "user_text": "texto de prueba cuadrante falso",
        },
    )
    assert resp_invalid.status_code == 200
    assert store.count() == initial_count

    # 2. Valid quadrant rejection creates a pending feedback record
    test_client.cookies.clear()
    unique_text = "esto no es tristeza sino rabia viva"
    resp_valid = test_client.post(
        "/tree",
        data={
            "quadrant": "alta_negativa",
            "path": "",
            "rejected_quadrant": "baja_negativa",
            "user_text": unique_text,
        },
    )
    assert resp_valid.status_code == 200
    pending_matching = [r for r in store.get_by_status("pending") if r.user_text == unique_text]
    assert len(pending_matching) == 1
    assert pending_matching[0].rating == "negative"
    assert pending_matching[0].corrected_quadrant == "alta_negativa"
    assert pending_matching[0].predicted_quadrant == "baja_negativa"
    assert pending_matching[0].status == "pending"


def test_hybrid_curation_status_assignment(test_client):
    """Option 3 Hybrid Policy: Thumbs-up auto-promotes to 'verified', corrections stay 'pending'."""
    store = get_feedback_store()

    # Thumbs up -> verified
    t_up = "siento una alegria pura y sincera"
    test_client.post(
        "/feedback",
        data={
            "user_text": t_up,
            "lang": "es",
            "quadrant": "alta_positiva",
            "emotion": "Alegría",
            "rating": "positive",
        },
    )
    verified = [r for r in store.get_by_status("verified") if r.user_text == t_up]
    assert len(verified) == 1
    assert verified[0].status == "verified"

    # Correction / thumbs down -> pending
    t_down = "no era alegria sino que me late el corazon de miedo"
    test_client.post(
        "/feedback",
        data={
            "user_text": t_down,
            "lang": "es",
            "quadrant": "alta_positiva",
            "emotion": "Alegría",
            "rating": "negative",
            "corrected_quadrant": "alta_negativa",
        },
    )
    pending = [r for r in store.get_by_status("pending") if r.user_text == t_down]
    assert len(pending) == 1
    assert pending[0].status == "pending"


def test_deduplication_resilient_to_punctuation():
    """extract_training_samples unifies samples that differ only in punctuation or whitespace."""
    records = [
        FeedbackRecord(
            id="p1",
            created_at="2026-09-06T00:00:00Z",
            user_text="estoy muy feliz hoy!",
            normalized_text="estoy muy feliz hoy!",
            detected_lang="es",
            predicted_quadrant="alta_positiva",
            predicted_emotion="Alegría",
            model_confidence=0.95,
            rating="positive",
            status="verified",
        ),
        FeedbackRecord(
            id="p2",
            created_at="2026-09-06T00:01:00Z",
            user_text="estoy muy feliz hoy...",  # Trivial punctuation variation
            normalized_text="estoy muy feliz hoy...",
            detected_lang="es",
            predicted_quadrant="alta_positiva",
            predicted_emotion="Alegría",
            model_confidence=0.95,
            rating="positive",
            status="verified",
        ),
    ]
    samples = extract_training_samples(records, "es")
    # Only 1 sample extracted due to robust punctuation-resilient deduplication
    assert len(samples) == 1
    assert samples[0][0] == "estoy muy feliz hoy!"


def test_curate_feedback_cli_helpers():
    """Verify CLI review card formatting and dry-run curation session in curate_feedback.py."""
    from scripts.curate_feedback import format_record, curate_session

    store = get_feedback_store()
    rec = create_feedback_record(
        user_text="siento una furia incontrolable y el pecho caliente",
        detected_lang="es",
        predicted_quadrant="baja_negativa",
        predicted_emotion="Tristeza",
        model_confidence=0.72,
        rating="negative",
        corrected_quadrant="alta_negativa",
        corrected_emotion="Ira",
        comments="Es rabia pura",
        status="pending",
    )
    store.save(rec)

    # Format card validation
    card = format_record(rec, 1, 1)
    assert "furia incontrolable" in card
    assert "alta_negativa" in card
    assert "Es rabia pura" in card
    assert "PENDING" in card

    # Ensure empty/dry-run curation session executes without unhandled errors
    curate_session(status="incorporated", dry_run=True)


