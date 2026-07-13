"""Transactional event-ID deduplication outside bounded graph checkpoints."""

from __future__ import annotations

import os
import sqlite3
from pathlib import Path

from orchestrator.events import InterviewEvent


class EventLedgerConflictError(ValueError):
    """A previously recorded event ID was reused with different content."""


class EventLedger:
    """Claim event IDs atomically across graph restarts and ingress workers."""

    def __init__(self, path: Path) -> None:
        self._path = path

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)
        connection = sqlite3.connect(self._path)
        os.chmod(self._path, 0o600)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS event_ledger (
                session_id TEXT NOT NULL,
                event_id TEXT NOT NULL,
                event_json TEXT NOT NULL,
                PRIMARY KEY (session_id, event_id)
            )
            """
        )
        return connection

    def append(self, event: InterviewEvent) -> bool:
        """Atomically record an event; return false for an exact duplicate."""

        event_json = event.model_dump_json()
        with self._connect() as connection:
            try:
                connection.execute(
                    """
                    INSERT INTO event_ledger (session_id, event_id, event_json)
                    VALUES (?, ?, ?)
                    """,
                    (event.session_id, event.event_id, event_json),
                )
            except sqlite3.IntegrityError:
                stored = connection.execute(
                    """
                    SELECT event_json FROM event_ledger
                    WHERE session_id = ? AND event_id = ?
                    """,
                    (event.session_id, event.event_id),
                ).fetchone()
                assert stored is not None
                if InterviewEvent.model_validate_json(stored[0]) != event:
                    raise EventLedgerConflictError(
                        f"event_id {event.event_id!r} was reused with different content"
                    )
                return False
        return True


__all__ = ["EventLedger", "EventLedgerConflictError"]
