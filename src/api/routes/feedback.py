"""Feedback route for the CatanRL inference API."""

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone
from pathlib import Path

import structlog
from fastapi import APIRouter

from ..schemas import FeedbackRequest, FeedbackResponse

logger = structlog.get_logger()

router = APIRouter()

_DB_PATH = Path(__file__).resolve().parent.parent.parent.parent / "data" / "feedback.db"
_DB_INITIALIZED = False


def _ensure_db() -> sqlite3.Connection:
    """Create the feedback database and table if they don't exist, then return a connection."""
    global _DB_INITIALIZED

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(_DB_PATH))

    if not _DB_INITIALIZED:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS feedback (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                request_id TEXT NOT NULL,
                was_move_good INTEGER NOT NULL,
                comment TEXT,
                created_at TEXT NOT NULL
            )
            """
        )
        conn.commit()
        _DB_INITIALIZED = True

    return conn


@router.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(body: FeedbackRequest) -> FeedbackResponse:
    """Record user feedback on a move recommendation."""
    conn = _ensure_db()
    try:
        conn.execute(
            "INSERT INTO feedback (request_id, was_move_good, comment, created_at) "
            "VALUES (?, ?, ?, ?)",
            (
                body.request_id,
                int(body.was_move_good),
                body.comment,
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
        logger.info("feedback_stored", request_id=body.request_id, good=body.was_move_good)
    finally:
        conn.close()

    return FeedbackResponse(status="stored", request_id=body.request_id)
