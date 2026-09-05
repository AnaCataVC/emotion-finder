"""
Feedback persistence module for Emotion Finder.

Provides a decoupled storage repository supporting:
- Local SQLite database (for local development and testing)
- Lightweight HTTP REST client for Turso / LibSQL (for Vercel serverless)
- Null / logging fallback for misconfigured environments (fail-open invariant)

Invariants:
- Zero heavy dependencies in requirements.txt (uses standard library urllib.request).
- Fail-open resilience: errors in external persistence log to stderr but never raise unhandled 500s.
"""

import json
import logging
import os
import sqlite3
import urllib.request
import urllib.error
from abc import ABC, abstractmethod
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional
import uuid

logger = logging.getLogger("emotion_finder.feedback")

# ---------------------------------------------------------------------------
# Data Contract
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class FeedbackRecord:
    """Immutable feedback record contract."""
    id: str
    created_at: str
    user_text: str
    normalized_text: str
    detected_lang: str
    predicted_quadrant: str
    predicted_emotion: str
    model_confidence: float
    rating: str  # 'positive' | 'negative'
    corrected_quadrant: Optional[str] = None
    corrected_emotion: Optional[str] = None
    comments: Optional[str] = None
    session_hash: Optional[str] = None
    status: str = "pending"  # 'pending' | 'verified' | 'rejected' | 'incorporated'

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


# ---------------------------------------------------------------------------
# Storage Abstraction Interface
# ---------------------------------------------------------------------------

class FeedbackStore(ABC):
    """Abstract interface for feedback event persistence."""

    @abstractmethod
    def save(self, record: FeedbackRecord) -> bool:
        """Persist a feedback record. Returns True on success, False on failure."""
        pass

    @abstractmethod
    def get_by_status(self, status: str = "pending", limit: int = 100) -> List[FeedbackRecord]:
        """Retrieve feedback records filtered by status."""
        pass

    @abstractmethod
    def mark_status(self, record_id: str, new_status: str) -> bool:
        """Update status of a feedback record."""
        pass

    @abstractmethod
    def count(self) -> int:
        """Return total number of feedback records stored."""
        pass


# ---------------------------------------------------------------------------
# Local SQLite Implementation (Dev & Test)
# ---------------------------------------------------------------------------

_SQLITE_INIT_SCHEMA = """
CREATE TABLE IF NOT EXISTS feedback_events (
    id TEXT PRIMARY KEY,
    created_at TEXT NOT NULL,
    user_text TEXT NOT NULL,
    normalized_text TEXT NOT NULL,
    detected_lang TEXT NOT NULL,
    predicted_quadrant TEXT NOT NULL,
    predicted_emotion TEXT NOT NULL,
    model_confidence REAL NOT NULL,
    rating TEXT NOT NULL,
    corrected_quadrant TEXT,
    corrected_emotion TEXT,
    comments TEXT,
    session_hash TEXT,
    status TEXT DEFAULT 'pending'
);
CREATE INDEX IF NOT EXISTS idx_feedback_normalized ON feedback_events(normalized_text);
CREATE INDEX IF NOT EXISTS idx_feedback_status ON feedback_events(status);
"""


class LocalSQLiteFeedbackStore(FeedbackStore):
    """SQLite-backed feedback store for local development and in-memory tests."""

    def __init__(self, db_path: str | Path = "data/feedback.db"):
        self.db_path = str(db_path)
        self._is_memory = (self.db_path == ":memory:")
        self._mem_conn = None
        if not self._is_memory:
            Path(self.db_path).parent.mkdir(parents=True, exist_ok=True)
        else:
            self._mem_conn = sqlite3.connect(":memory:", check_same_thread=False)
            self._mem_conn.row_factory = sqlite3.Row
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        if self._is_memory and self._mem_conn is not None:
            return self._mem_conn
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        try:
            conn = self._get_connection()
            conn.executescript(_SQLITE_INIT_SCHEMA)
            if not self._is_memory:
                conn.close()
        except Exception as exc:
            logger.error("Failed to initialize SQLite feedback table: %s", exc)

    def save(self, record: FeedbackRecord) -> bool:
        query = """
        INSERT INTO feedback_events (
            id, created_at, user_text, normalized_text, detected_lang,
            predicted_quadrant, predicted_emotion, model_confidence,
            rating, corrected_quadrant, corrected_emotion, comments,
            session_hash, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        try:
            with self._get_connection() as conn:
                conn.execute(query, (
                    record.id, record.created_at, record.user_text,
                    record.normalized_text, record.detected_lang,
                    record.predicted_quadrant, record.predicted_emotion,
                    record.model_confidence, record.rating,
                    record.corrected_quadrant, record.corrected_emotion,
                    record.comments, record.session_hash, record.status
                ))
            return True
        except Exception as exc:
            logger.error("Failed to insert feedback record into SQLite: %s", exc)
            return False

    def get_by_status(self, status: str = "pending", limit: int = 100) -> List[FeedbackRecord]:
        query = "SELECT * FROM feedback_events WHERE status = ? ORDER BY created_at DESC LIMIT ?"
        try:
            with self._get_connection() as conn:
                rows = conn.execute(query, (status, limit)).fetchall()
                return [
                    FeedbackRecord(
                        id=row["id"],
                        created_at=row["created_at"],
                        user_text=row["user_text"],
                        normalized_text=row["normalized_text"],
                        detected_lang=row["detected_lang"],
                        predicted_quadrant=row["predicted_quadrant"],
                        predicted_emotion=row["predicted_emotion"],
                        model_confidence=row["model_confidence"],
                        rating=row["rating"],
                        corrected_quadrant=row["corrected_quadrant"],
                        corrected_emotion=row["corrected_emotion"],
                        comments=row["comments"],
                        session_hash=row["session_hash"],
                        status=row["status"],
                    )
                    for row in rows
                ]
        except Exception as exc:
            logger.error("Failed to fetch feedback records from SQLite: %s", exc)
            return []

    def mark_status(self, record_id: str, new_status: str) -> bool:
        query = "UPDATE feedback_events SET status = ? WHERE id = ?"
        try:
            with self._get_connection() as conn:
                cursor = conn.execute(query, (new_status, record_id))
                return cursor.rowcount > 0
        except Exception as exc:
            logger.error("Failed to update feedback status in SQLite: %s", exc)
            return False

    def count(self) -> int:
        try:
            with self._get_connection() as conn:
                row = conn.execute("SELECT COUNT(*) AS total FROM feedback_events").fetchone()
                return int(row["total"]) if row else 0
        except Exception:
            return 0


# ---------------------------------------------------------------------------
# Turso / LibSQL HTTP REST Implementation (Vercel Serverless)
# ---------------------------------------------------------------------------

def _encode_hrana_arg(arg: Any) -> Dict[str, Any]:
    """Encode a Python value into strict Hrana protocol Value JSON."""
    if arg is None:
        return {"type": "null"}
    elif isinstance(arg, bool):
        return {"type": "integer", "value": "1" if arg else "0"}
    elif isinstance(arg, int):
        return {"type": "integer", "value": str(arg)}
    elif isinstance(arg, float):
        return {"type": "float", "value": float(arg)}
    else:
        return {"type": "text", "value": str(arg)}


class TursoHttpFeedbackStore(FeedbackStore):
    """
    Lightweight HTTP client for Turso LibSQL Pipeline API.
    
    Zero third-party library dependencies (uses Python's standard urllib.request).
    Enforces a strict 1.8-second socket timeout to prevent serverless execution freezes.
    """

    def __init__(self, database_url: str, auth_token: str, timeout_seconds: float = 1.8):
        # Convert libsql:// url to https://
        clean_url = database_url.replace("libsql://", "https://").rstrip("/")
        if not clean_url.startswith("http"):
            clean_url = f"https://{clean_url}"
        self.endpoint = f"{clean_url}/v2/pipeline"
        self.auth_token = auth_token
        self.timeout = timeout_seconds
        self._schema_checked = False

    def _execute_sql(self, statement: str, args: List[Any]) -> Optional[Dict[str, Any]]:
        payload = {
            "requests": [
                {
                    "type": "execute",
                    "stmt": {
                        "sql": statement,
                        "args": [_encode_hrana_arg(a) for a in args],
                    }
                },
                {"type": "close"}
            ]
        }
        data = json.dumps(payload).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self.auth_token}",
            "Content-Type": "application/json",
        }
        req = urllib.request.Request(self.endpoint, data=data, headers=headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as response:
                result = json.loads(response.read().decode("utf-8"))
                results = result.get("results", [])
                if results and results[0].get("type") == "error":
                    logger.warning("Turso statement error: %s", results[0].get("error"))
                    return None
                return result
        except urllib.error.HTTPError as exc:
            err_body = ""
            try:
                err_body = exc.read().decode("utf-8")
            except Exception:
                pass
            logger.warning("Turso HTTP error %s: %s (body: %s)", exc.code, exc.reason, err_body)
            return None
        except (urllib.error.URLError, TimeoutError, Exception) as exc:
            logger.warning("Turso HTTP pipeline request failed (fail-open triggered): %s", exc)
            return None

    def _ensure_schema(self) -> None:
        """Automatically initialize the feedback table in Turso if not present."""
        if self._schema_checked:
            return
        schema_sql = """
        CREATE TABLE IF NOT EXISTS feedback_events (
            id TEXT PRIMARY KEY,
            created_at TEXT NOT NULL,
            user_text TEXT NOT NULL,
            normalized_text TEXT NOT NULL,
            detected_lang TEXT NOT NULL,
            predicted_quadrant TEXT NOT NULL,
            predicted_emotion TEXT NOT NULL,
            model_confidence REAL NOT NULL,
            rating TEXT NOT NULL,
            corrected_quadrant TEXT,
            corrected_emotion TEXT,
            comments TEXT,
            session_hash TEXT,
            status TEXT DEFAULT 'pending'
        );
        """
        self._execute_sql(schema_sql, [])
        self._schema_checked = True

    def save(self, record: FeedbackRecord) -> bool:
        if not self._schema_checked:
            self._ensure_schema()

        stmt = """
        INSERT INTO feedback_events (
            id, created_at, user_text, normalized_text, detected_lang,
            predicted_quadrant, predicted_emotion, model_confidence,
            rating, corrected_quadrant, corrected_emotion, comments,
            session_hash, status
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """
        args = [
            record.id, record.created_at, record.user_text,
            record.normalized_text, record.detected_lang,
            record.predicted_quadrant, record.predicted_emotion,
            record.model_confidence, record.rating,
            record.corrected_quadrant, record.corrected_emotion,
            record.comments, record.session_hash, record.status
        ]
        res = self._execute_sql(stmt, args)
        return res is not None

    def get_by_status(self, status: str = "pending", limit: int = 100) -> List[FeedbackRecord]:
        # Batch querying for active learning scripts
        stmt = "SELECT * FROM feedback_events WHERE status = ? ORDER BY created_at DESC LIMIT ?"
        # Lightweight fallback: retrain scripts running locally pull via local SQLite or admin export
        return []

    def mark_status(self, record_id: str, new_status: str) -> bool:
        stmt = "UPDATE feedback_events SET status = ? WHERE id = ?"
        res = self._execute_sql(stmt, [new_status, record_id])
        return res is not None

    def count(self) -> int:
        return 0


# ---------------------------------------------------------------------------
# Fallback Null Implementation (Fail-Open / Memory)
# ---------------------------------------------------------------------------

class NullFeedbackStore(FeedbackStore):
    """Safe fallback store that logs records without crashing when storage is unconfigured."""

    def __init__(self):
        self._memory_log: List[FeedbackRecord] = []

    def save(self, record: FeedbackRecord) -> bool:
        logger.info("[NullFeedbackStore] Received feedback: %s (%s)", record.id, record.rating)
        self._memory_log.append(record)
        return True

    def get_by_status(self, status: str = "pending", limit: int = 100) -> List[FeedbackRecord]:
        return [r for r in self._memory_log if r.status == status][:limit]

    def mark_status(self, record_id: str, new_status: str) -> bool:
        for i, r in enumerate(self._memory_log):
            if r.id == record_id:
                # Replace with new status
                d = r.to_dict()
                d["status"] = new_status
                self._memory_log[i] = FeedbackRecord(**d)
                return True
        return False

    def count(self) -> int:
        return len(self._memory_log)


# ---------------------------------------------------------------------------
# Store Factory
# ---------------------------------------------------------------------------

_GLOBAL_STORE: Optional[FeedbackStore] = None


def get_feedback_store() -> FeedbackStore:
    """
    Factory resolving the appropriate persistence adapter based on environment variables.

    - If PYTEST_CURRENT_TEST is set: In-memory SQLite store.
    - If TURSO_DATABASE_URL and TURSO_AUTH_TOKEN are set: TursoHttpFeedbackStore.
    - If running on Vercel without Turso credentials: NullFeedbackStore (read-only filesystem protection).
    - Otherwise (local development): LocalSQLiteFeedbackStore (data/feedback.db).
    """
    global _GLOBAL_STORE
    if _GLOBAL_STORE is not None:
        return _GLOBAL_STORE

    # Testing environment invariant
    if "PYTEST_CURRENT_TEST" in os.environ:
        _GLOBAL_STORE = LocalSQLiteFeedbackStore(":memory:")
        return _GLOBAL_STORE

    turso_url = os.environ.get("TURSO_DATABASE_URL")
    turso_token = os.environ.get("TURSO_AUTH_TOKEN")

    if turso_url and turso_token:
        _GLOBAL_STORE = TursoHttpFeedbackStore(turso_url, turso_token)
        return _GLOBAL_STORE

    # Vercel Serverless environment detection (read-only filesystem guard)
    if os.environ.get("VERCEL"):
        logger.warning("Running on Vercel without Turso credentials; activating NullFeedbackStore.")
        _GLOBAL_STORE = NullFeedbackStore()
        return _GLOBAL_STORE

    # Default: Local SQLite for development
    _GLOBAL_STORE = LocalSQLiteFeedbackStore()
    return _GLOBAL_STORE


def create_feedback_record(
    user_text: str,
    detected_lang: str,
    predicted_quadrant: str,
    predicted_emotion: str,
    model_confidence: float,
    rating: str,
    corrected_quadrant: Optional[str] = None,
    corrected_emotion: Optional[str] = None,
    comments: Optional[str] = None,
    session_hash: Optional[str] = None,
) -> FeedbackRecord:
    """Helper to construct a validated, timestamped FeedbackRecord."""
    # Preprocessing text normalization for clean deduplication
    clean_text = user_text.strip()
    norm_text = clean_text.lower()

    return FeedbackRecord(
        id=str(uuid.uuid4()),
        created_at=datetime.now(timezone.utc).isoformat(),
        user_text=clean_text,
        normalized_text=norm_text,
        detected_lang=detected_lang,
        predicted_quadrant=predicted_quadrant,
        predicted_emotion=predicted_emotion,
        model_confidence=round(float(model_confidence), 4),
        rating="positive" if rating in ("positive", "1", "up") else "negative",
        corrected_quadrant=corrected_quadrant if corrected_quadrant else None,
        corrected_emotion=corrected_emotion if corrected_emotion else None,
        comments=comments.strip()[:150] if comments else None,
        session_hash=session_hash,
        status="pending",
    )
