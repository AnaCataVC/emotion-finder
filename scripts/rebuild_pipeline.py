"""
Unified ML pipeline rebuild and quality verification task runner.

Automates the end-to-end developer workflow:
1. Regenerate deterministic synthetic datasets (data/generate_datasets.py)
2. Train and serialize baseline NLP models (train_model.py)
3. Run empirical benchmark evaluation & confusion matrix plots (scripts/evaluate_models.py)
4. Execute Ruff static analysis (ruff check .)
5. Execute automated test suite (pytest -v)

Usage:
    python scripts/rebuild_pipeline.py [--skip-eval] [--check-only]
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent


def run_step(label: str, cmd: list[str], env: dict[str, str]) -> None:
    """Execute a pipeline step and fail fast on non-zero exit code."""
    print(f"\n==> [{label}] Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=ROOT_DIR, env=env, check=False)
    if result.returncode != 0:
        print(f"\n[ERROR] Step '{label}' failed with exit code {result.returncode}.", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Unified task runner for rebuilding the ML pipeline and running quality gates."
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip the multi-architecture benchmark & plot generation step (scripts/evaluate_models.py).",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Run only Ruff linting and Pytest suite without regenerating datasets or retraining models.",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env["PYTHONHASHSEED"] = "0"
    py = sys.executable

    if not args.check_only:
        run_step("1/5 Generate Datasets", [py, "data/generate_datasets.py"], env)
        run_step("2/5 Train Baseline Models", [py, "train_model.py"], env)
        if not args.skip_eval:
            run_step("3/5 Evaluate & Benchmark Models", [py, "scripts/evaluate_models.py"], env)
        else:
            print("\n==> [3/5 Evaluate & Benchmark Models] Skipped (--skip-eval)")

    run_step("Ruff Linter", [py, "-m", "ruff", "check", "."], env)
    run_step("Pytest Suite", [py, "-m", "pytest", "-v"], env)
    print("\nAll pipeline steps completed successfully!")


if __name__ == "__main__":
    main()
