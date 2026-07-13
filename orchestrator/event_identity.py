"""Stable, non-PII identifiers for deduplicable interview events."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from orchestrator.events import EventSource, EventType


def stable_event_id(
    prefix: str,
    *,
    session_id: str,
    event_type: EventType,
    source: EventSource,
    question_id: int | None,
    identity: dict[str, Any],
) -> str:
    """Hash stable observation identity without including receipt time or raw PII."""

    canonical = json.dumps(
        {
            "event_type": event_type,
            "identity": identity,
            "question_id": question_id,
            "session_id": session_id,
            "source": source,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return f"{prefix}-{hashlib.sha256(canonical.encode()).hexdigest()[:24]}"


__all__ = ["stable_event_id"]
