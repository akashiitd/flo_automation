"""Durable, append-only event-ID deduplication outside graph checkpoints."""

from __future__ import annotations

import os
from pathlib import Path

from orchestrator.events import InterviewEvent


class EventLedgerConflictError(ValueError):
    """A previously recorded event ID was reused with different content."""


class EventLedger:
    """Persist full event identities while graph state retains only a recent window."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._events_by_id: dict[str, InterviewEvent] | None = None

    def _load(self) -> dict[str, InterviewEvent]:
        if self._events_by_id is not None:
            return self._events_by_id
        events_by_id: dict[str, InterviewEvent] = {}
        if self._path.exists():
            for line_number, line in enumerate(
                self._path.read_text(encoding="utf-8").splitlines(), start=1
            ):
                if not line.strip():
                    continue
                event = InterviewEvent.model_validate_json(line)
                prior = events_by_id.get(event.event_id)
                if prior is not None and prior != event:
                    raise EventLedgerConflictError(
                        f"event_id {event.event_id!r} conflicts at line {line_number}"
                    )
                events_by_id[event.event_id] = event
        self._events_by_id = events_by_id
        return events_by_id

    def append(self, event: InterviewEvent) -> bool:
        """Record an event once; return false for an exact durable duplicate."""

        events_by_id = self._load()
        prior = events_by_id.get(event.event_id)
        if prior is not None:
            if prior != event:
                raise EventLedgerConflictError(
                    f"event_id {event.event_id!r} was reused with different content"
                )
            return False

        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        descriptor = os.open(
            self._path,
            os.O_APPEND | os.O_CREAT | os.O_WRONLY,
            0o600,
        )
        try:
            with os.fdopen(descriptor, "a", encoding="utf-8") as stream:
                stream.write(event.model_dump_json() + "\n")
                stream.flush()
                os.fsync(stream.fileno())
        finally:
            os.chmod(self._path, 0o600)
        events_by_id[event.event_id] = event
        return True


__all__ = ["EventLedger", "EventLedgerConflictError"]
