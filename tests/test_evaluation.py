"""
Automated tests for evaluation artifacts and benchmarking pipeline.

Validates:
- Benchmark candidate model construction for ES and EN.
- Existence, integrity, and non-empty status of generated evaluation documents.
- Generated confusion matrix image artifacts (PNG magic header and size checks).
"""

from pathlib import Path

from scripts.evaluate_models import get_candidate_models

BASE_DIR = Path(__file__).resolve().parent.parent
DOCS_EVAL_DIR = BASE_DIR / "docs" / "evaluation"


def test_candidate_models_structure():
    """Verify all 5 candidate models are correctly defined with valid pipelines."""
    for lang in ["es", "en"]:
        candidates = get_candidate_models(lang)
        expected_models = [
            "Dummy (Stratified)",
            "Multinomial NB",
            "Linear SVM",
            "Random Forest",
            "Logistic Regression (Production)",
        ]
        for name in expected_models:
            assert name in candidates, f"Missing candidate model: {name} for {lang}"
            pipeline = candidates[name]
            assert hasattr(pipeline, "fit"), f"{name} must have a .fit() method"
            assert "tfidf" in pipeline.named_steps
            assert "clf" in pipeline.named_steps


def test_evaluation_documentation_artifacts_exist():
    """Verify that all formal evaluation docs exist and contain substantive content."""
    required_docs = [
        "methodology.md",
        "benchmark-results.md",
        "model-card.md",
        "error-analysis.md",
    ]
    for doc_name in required_docs:
        doc_path = DOCS_EVAL_DIR / doc_name
        assert doc_path.exists(), f"Missing required evaluation document: {doc_name}"
        content = doc_path.read_text(encoding="utf-8")
        assert len(content) > 300, f"Evaluation doc {doc_name} is unexpectedly short ({len(content)} chars)"


def test_confusion_matrix_images_exist_and_valid():
    """Verify that confusion matrix PNG images exist, have valid PNG magic bytes, and are not empty."""
    png_magic = b"\x89PNG\r\n\x1a\n"
    for lang in ["es", "en"]:
        img_path = DOCS_EVAL_DIR / f"confusion-matrix-{lang}.png"
        assert img_path.exists(), f"Confusion matrix for {lang} does not exist: {img_path}"
        assert img_path.stat().st_size > 20_000, f"Confusion matrix for {lang} is too small"
        with open(img_path, "rb") as f:
            header = f.read(8)
            assert header == png_magic, f"File {img_path} is not a valid PNG image"
