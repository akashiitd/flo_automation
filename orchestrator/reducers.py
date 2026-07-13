"""Explicit append-only reducers for checkpointed interview state."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from orchestrator.events import InterviewEvent


class DynamicStateConflictError(ValueError):
    """A supposedly immutable state record was reused with changed content."""


def append_interview_events(
    existing: Sequence[InterviewEvent], incoming: Sequence[InterviewEvent]
) -> list[InterviewEvent]:
    """Append unseen events while rejecting conflicting duplicate event IDs."""

    result = list(existing)
    seen = {event.event_id: event for event in result}
    for event in incoming:
        prior = seen.get(event.event_id)
        if prior is None:
            result.append(event)
            seen[event.event_id] = event
        elif prior != event:
            raise DynamicStateConflictError(
                f"event_id {event.event_id!r} was reused with different content"
            )
    return result


def append_skill_evidence(
    existing: Sequence[Any], incoming: Sequence[Any]
) -> list[Any]:
    """Append unseen evidence while rejecting conflicting evidence identifiers."""

    result = list(existing)
    seen = {evidence.evidence_id: evidence for evidence in result}
    for evidence in incoming:
        prior = seen.get(evidence.evidence_id)
        if prior is None:
            result.append(evidence)
            seen[evidence.evidence_id] = evidence
        elif prior != evidence:
            raise DynamicStateConflictError(
                f"evidence_id {evidence.evidence_id!r} was reused with different content"
            )
    return result


__all__ = [
    "DynamicStateConflictError",
    "append_interview_events",
    "append_skill_evidence",
]
