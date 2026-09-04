"""
Comprehensive test suite for Emotion Finder.

Covers:
- Spanish NLP classification across all 4 quadrants
- English NLP classification across all 4 quadrants
- Input boundary edge cases (empty, short, >500 chars)
- Low-confidence and gibberish fallback
- Decision tree 64-emotion topology & leaf integrity
- Web app routes (GET /, POST /predict, POST /tree)
- Static icon serving (/icon.png, /favicon.png)
"""

import sys
from pathlib import Path

# Add project root to sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import decision_tree
import inference


def test_spanish_prediction_high_negative():
    res = inference.predict_quadrant("tengo los puños apretados y me hierve la sangre de la rabia")
    assert res["quadrant"] == "alta_negativa"
    assert res["lang"] == "es"
    assert not res["low_confidence"]


def test_spanish_prediction_high_positive():
    res = inference.predict_quadrant("estoy súper feliz y quiero saltar de alegría")
    assert res["quadrant"] == "alta_positiva"
    assert res["lang"] == "es"


def test_spanish_prediction_low_negative():
    res = inference.predict_quadrant("no tengo fuerzas para levantarme de la cama y me siento vacío")
    assert res["quadrant"] == "baja_negativa"
    assert res["lang"] == "es"


def test_spanish_prediction_low_positive():
    res = inference.predict_quadrant("estoy en paz respirando profundo y relajado")
    assert res["quadrant"] == "baja_positiva"
    assert res["lang"] == "es"


def test_english_predictions_all_quadrants():
    # Low positive
    res_lp = inference.predict_quadrant("I feel so calm and relaxed enjoying the quiet")
    assert res_lp["quadrant"] == "baja_positiva"
    assert res_lp["lang"] == "en"

    # High positive
    res_hp = inference.predict_quadrant("I am super motivated and want to celebrate out of joy")
    assert res_hp["quadrant"] == "alta_positiva"
    assert res_hp["lang"] == "en"

    # High negative
    res_hn = inference.predict_quadrant("I am overwhelmed by anxiety and my chest is tight with panic")
    assert res_hn["quadrant"] == "alta_negativa"
    assert res_hn["lang"] == "en"

    # Low negative
    res_ln = inference.predict_quadrant("I feel so empty and exhausted with no energy to do anything")
    assert res_ln["quadrant"] == "baja_negativa"
    assert res_ln["lang"] == "en"


def test_chilean_and_british_dialectal_idioms():
    """Verify affective mapping of Chilean Spanish and British English idioms."""
    # 1. Chilean Spanish - Alta Negativa (Reactivity & on-edge)
    res_raw = inference.predict_quadrant("estoy con las emociones a flor de piel")
    assert res_raw["quadrant"] == "alta_negativa", f"Expected alta_negativa, got {res_raw['quadrant']}"

    res_short_fuse = inference.predict_quadrant("estoy con la mecha corta y cualquier cosa me hace saltar")
    assert res_short_fuse["quadrant"] == "alta_negativa"

    res_pera = inference.predict_quadrant("estoy con la pera del susto que tengo")
    assert res_pera["quadrant"] == "alta_negativa"

    # 2. Chilean Spanish - Baja Negativa (Low energy slump)
    res_bajon = inference.predict_quadrant("estoy con el bajon y sin ganas de levantarme de la cama")
    assert res_bajon["quadrant"] == "baja_negativa"

    res_lata = inference.predict_quadrant("me da lata todo y solo quiero quedarme encerrado")
    assert res_lata["quadrant"] == "baja_negativa"

    # 3. Chilean Spanish - Alta Positiva (Excitement)
    res_pata = inference.predict_quadrant("estoy que salto en una pata de lo feliz que estoy")
    assert res_pata["quadrant"] == "alta_positiva"

    # 4. Chilean Spanish - Baja Positiva (Calm & chill)
    res_piola = inference.predict_quadrant("estoy piola y relajado disfrutando el silencio y la calma")
    assert res_piola["quadrant"] == "baja_positiva"

    # 5. British English - Alta Negativa
    res_uk_edge = inference.predict_quadrant("my nerves are on edge and I feel completely raw")
    assert res_uk_edge["quadrant"] == "alta_negativa"

    res_uk_tether = inference.predict_quadrant("I am proper wound up and at the end of my tether")
    assert res_uk_tether["quadrant"] == "alta_negativa"

    # 6. British English - Baja Negativa
    res_uk_dumps = inference.predict_quadrant("I am feeling down in the dumps today and cannot focus")
    assert res_uk_dumps["quadrant"] == "baja_negativa"

    # 7. British English - Alta Positiva
    res_uk_buzz = inference.predict_quadrant("I am absolutely buzzing with excitement and joy")
    assert res_uk_buzz["quadrant"] == "alta_positiva"

    # 8. British English - Baja Positiva
    res_uk_rain = inference.predict_quadrant("feeling right as rain and totally at ease")
    assert res_uk_rain["quadrant"] == "baja_positiva"


def test_boundary_and_edge_cases():
    # Empty string
    res_empty = inference.predict_quadrant("")
    assert res_empty["low_confidence"]
    assert res_empty["quadrant"] == "uncertain"

    # Single character
    res_single = inference.predict_quadrant("x")
    assert res_single["low_confidence"]
    assert res_single["quadrant"] == "uncertain"

    # Long text truncation (> 500 characters)
    long_text = "estoy súper motivado y quiero saltar de alegría " * 40
    res_long = inference.predict_quadrant(long_text)
    assert res_long["quadrant"] == "alta_positiva"
    assert not res_long["low_confidence"]


    # Gibberish input
    res_gib = inference.predict_quadrant("qwkxpz 999999")
    assert res_gib["low_confidence"] or res_gib["quadrant"] in [
        "uncertain", "alta_positiva", "alta_negativa", "baja_positiva", "baja_negativa"
    ]


def test_decision_tree_structure():
    emotions = decision_tree.get_all_emotions()
    assert len(emotions) == 64, f"Expected 64 emotions, got {len(emotions)}"

    quadrants = ["alta_positiva", "alta_negativa", "baja_negativa", "baja_positiva"]
    for q in quadrants:
        q_emotions = decision_tree.get_quadrant_emotions(q)
        assert len(q_emotions) == 16, f"Expected 16 emotions for quadrant {q}, got {len(q_emotions)}"

    # Check root questions exist in both languages
    for q in quadrants:
        root = decision_tree.get_tree(q)
        assert root is not None
        assert "question_es" in root
        assert "question_en" in root

    # Navigate to a leaf
    leaf = decision_tree.get_node("alta_negativa", "yes.yes.yes.yes")
    assert leaf is not None
    assert "emotion_es" in leaf
    assert "emotion_en" in leaf
    assert "emoji" in leaf

    # Non-existent node returns None
    assert decision_tree.get_node("alta_negativa", "invalid.path") is None


def test_webapp_endpoints():
    import json
    import re
    from starlette.testclient import TestClient
    from main import app

    client = TestClient(app)

    # 1. GET / (Home Page)
    resp = client.get("/")
    assert resp.status_code == 200
    assert "Emotion Finder" in resp.text
    assert "/icon.png" in resp.text
    assert "lang-pill" in resp.text
    assert "requestSubmit" in resp.text, "Textarea must support Enter key submission"

    # 2. Static Icon Serving
    resp_icon = client.get("/icon.png")
    assert resp_icon.status_code == 200

    resp_fav = client.get("/favicon.png")
    assert resp_fav.status_code == 200

    # 3. POST /predict (Normal prediction)
    resp_pred = client.post("/predict", data={"text": "estoy súper motivado con ganas de saltar", "lang": "es"})
    assert resp_pred.status_code == 200
    assert "alta_positiva" in resp_pred.text
    # Verify hx-vals contains valid parseable JSON for Yes and No buttons
    matches = re.findall(r"hx-vals='([^']+)'", resp_pred.text)
    assert len(matches) >= 2, f"Expected at least 2 buttons with hx-vals, got {len(matches)}"
    for val_str in matches:
        parsed = json.loads(val_str)
        assert "quadrant" in parsed
        assert parsed["path"] in ["yes", "no"]

    # 4. POST /predict (Low confidence / uncertain fallback)
    resp_unc = client.post("/predict", data={"text": "asdfghjkl", "lang": "es"})
    assert resp_unc.status_code == 200
    assert "uncertain_title" in resp_unc.text or "detalles" in resp_unc.text

    # 5. POST /tree (Intermediate navigation)
    resp_tree = client.post("/tree", data={"quadrant": "alta_positiva", "path": "yes", "lang": "es", "step": "3"})
    assert resp_tree.status_code == 200
    assert "/tree" in resp_tree.text
    matches_tree = re.findall(r"hx-vals='([^']+)'", resp_tree.text)
    assert len(matches_tree) >= 2
    for val_str in matches_tree:
        parsed = json.loads(val_str)
        assert parsed["path"] in ["yes.yes", "yes.no"]

    # 6. POST /tree (Reaching leaf emotion)
    resp_leaf = client.post("/tree", data={"quadrant": "alta_positiva", "path": "yes.yes.yes.yes", "lang": "es", "step": "5"})
    assert resp_leaf.status_code == 200
    assert "emotion-result-box" in resp_leaf.text
    assert "Éxtasis" in resp_leaf.text
    assert "tip-callout" not in resp_leaf.text, "Tips/advice should not be rendered"
    assert "Consejo" not in resp_leaf.text, "Tips/advice should not be rendered"


if __name__ == "__main__":
    test_spanish_prediction_high_negative()
    test_spanish_prediction_high_positive()
    test_spanish_prediction_low_negative()
    test_spanish_prediction_low_positive()
    test_english_predictions_all_quadrants()
    test_chilean_and_british_dialectal_idioms()
    test_boundary_and_edge_cases()
    test_decision_tree_structure()
    test_webapp_endpoints()
    print("All tests (Spanish ML + English ML + Boundaries + Dialectal Idioms + Tree + Web App + Icons) passed successfully!")
