"""
Global test configuration and fixtures for Emotion Finder.
Ensures hermetic test execution with isolated in-memory feedback persistence per test.
"""

import sys
from pathlib import Path
import pytest

# Ensure project root is in sys.path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import feedback_store
from feedback_store import LocalSQLiteFeedbackStore


@pytest.fixture(autouse=True)
def reset_global_feedback_store():
    """Reset the global feedback store to a fresh in-memory SQLite database before each test.
    Prevents cross-test pollution and rate-limit interference.
    """
    feedback_store._GLOBAL_STORE = LocalSQLiteFeedbackStore(":memory:")
    yield feedback_store._GLOBAL_STORE
