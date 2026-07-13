"""Explicit append-only reducers for checkpointed interview state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from orchestrator.events import InterviewEvent

RECENT_EVENT_LIMIT = 50


class DynamicStateConflictError(ValueError):
    """A supposedly immutable state record was reused with changed content."""


def _append_immutable_by_identifier(
    existing: Sequence[Any],
    incoming: Sequence[Any],
    *,
    identifier_attribute: str,
    limit: int | None = None,
) -> list[Any]:
    """Append immutable records, rejecting changed content under one identifier."""

    result = list(existing)
    seen = {getattr(record, identifier_attribute): record for record in result}
    for record in incoming:
        identifier = getattr(record, identifier_attribute)
        prior = seen.get(identifier)
        if prior is None:
            result.append(record)
            seen[identifier] = record
        elif prior != record:
            raise DynamicStateConflictError(
                f"{identifier_attribute} {identifier!r} was reused with different content"
            )
    return result if limit is None else result[-limit:]


def append_interview_events(
    existing: Sequence[InterviewEvent], incoming: Sequence[InterviewEvent]
) -> list[InterviewEvent]:
    """Maintain a bounded recent-event projection for graph checkpoints.

    The complete event/audit ledger is intentionally external to checkpointed
    graph state. Event ingress performs durable full-history deduplication.
    """

    return _append_immutable_by_identifier(
        existing,
        incoming,
        identifier_attribute="event_id",
        limit=RECENT_EVENT_LIMIT,
    )


def append_skill_evidence(
    existing: Sequence[Any], incoming: Sequence[Any]
) -> list[Any]:
    """Append unseen evidence while rejecting conflicting evidence identifiers."""

    return _append_immutable_by_identifier(
        existing,
        incoming,
        identifier_attribute="evidence_id",
    )


__all__ = [
    "DynamicStateConflictError",
    "RECENT_EVENT_LIMIT",
    "append_interview_events",
    "append_skill_evidence",
]
