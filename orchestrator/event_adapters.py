"""Offline-safe normalization and ingress for external interview observations."""

from __future__ import annotations

from collections.abc import Callable
from datetime import UTC, datetime
from typing import Any, Literal

from orchestrator.event_ledger import EventLedger
from orchestrator.event_identity import stable_event_id
from orchestrator.events import EventSource, EventType, InterviewEvent
from orchestrator.reducers import append_interview_events

_TIMER_EVENT_TYPES = {
    "FIFTEEN_MINUTES_REMAINING": EventType.TIMER_WARNING,
    "TEN_MINUTES_REMAINING": EventType.TIMER_WARNING,
    "FIVE_MINUTES_REMAINING": EventType.TIMER_WARNING,
    "TWO_MINUTES_REMAINING": EventType.TIMER_WARNING,
    "ONE_MINUTE_REMAINING": EventType.TIMER_WARNING,
    "TIME_LIMIT_REACHED": EventType.TIME_LIMIT_REACHED,
}


class EventNormalizer:
    """Convert trusted callbacks into deduplicable, typed interview events."""

    def __init__(
        self,
        *,
        session_id: str,
        now: Callable[[], datetime] | None = None,
    ) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        self._session_id = session_id
        self._now = now or (lambda: datetime.now(UTC))
        self._candidate_has_connected = False
        self._room_events_by_transition_id: dict[str, InterviewEvent | None] = {}
        self._events_by_id: dict[str, InterviewEvent] = {}

    def apple_speech_callback(
        self,
        segment: object,
        *,
        question_id: int | None,
        is_final: bool,
    ) -> InterviewEvent | None:
        """Normalize candidate-only ASR text only while a question is active."""

        text = str(getattr(segment, "text", "")).strip()
        source = str(getattr(segment, "source", "")).strip()
        if question_id is None or source != "system" or not text:
            return None
        occurred_at = self._callback_timestamp(segment)
        payload: dict[str, Any] = {
            "text": text,
            "start_time": float(getattr(segment, "start_time", 0.0)),
            "end_time": float(getattr(segment, "end_time", 0.0)),
            "confidence": float(getattr(segment, "confidence", 0.0)),
        }
        event_type = (
            EventType.TRANSCRIPT_FINAL if is_final else EventType.TRANSCRIPT_PARTIAL
        )
        segment_identity = {
            "callback_segment_id": str(
                getattr(segment, "segment_id", getattr(segment, "id", ""))
            ),
            "confidence": payload["confidence"],
            "end_time": payload["end_time"],
            "start_time": payload["start_time"],
            "text": text,
        }
        payload["segment_id"] = stable_event_id(
            "segment",
            session_id=self._session_id,
            event_type=event_type,
            source=EventSource.CANDIDATE_ASR,
            question_id=question_id,
            identity=segment_identity,
        )
        return self._event(
            event_type=event_type,
            source=EventSource.CANDIDATE_ASR,
            occurred_at=occurred_at,
            question_id=question_id,
            payload=payload,
            identity={"segment_id": payload["segment_id"]},
        )

    def timer_threshold(
        self, threshold: str, *, remaining_seconds: float
    ) -> InterviewEvent:
        """Normalize a crossed timer threshold without waiting on wall-clock I/O."""

        normalized_threshold = threshold.strip()
        if not normalized_threshold:
            raise ValueError("timer threshold must not be empty")
        if remaining_seconds < 0:
            raise ValueError("remaining_seconds cannot be negative")
        try:
            event_type = _TIMER_EVENT_TYPES[normalized_threshold]
        except KeyError as error:
            raise ValueError(
                f"unsupported timer threshold: {normalized_threshold}"
            ) from error
        return self._event(
            event_type=event_type,
            source=EventSource.TIMER,
            occurred_at=self._now(),
            question_id=None,
            payload={
                "threshold": normalized_threshold,
                "remaining_seconds": float(remaining_seconds),
            },
            identity={"threshold": normalized_threshold},
        )

    def room_state_changed(
        self, *, transition_id: str, previous: object | None, current: object
    ) -> InterviewEvent | None:
        """Normalize observed room transitions, retaining reconnect semantics."""

        if not transition_id.strip():
            raise ValueError("room transition_id must not be empty")
        if transition_id in self._room_events_by_transition_id:
            return self._room_events_by_transition_id[transition_id]
        previous_name = self._room_state_name(previous)
        current_name = self._room_state_name(current)
        if current_name == "INTERVIEWER_IN_ROOM" and previous_name in {
            None,
            "LAUNCHED",
        }:
            event_type = EventType.JOINED
        elif current_name == "CANDIDATE_CONNECTED":
            event_type = (
                EventType.CANDIDATE_RECONNECTED
                if self._candidate_has_connected
                else EventType.CANDIDATE_CONNECTED
            )
            self._candidate_has_connected = True
        elif previous_name == "CANDIDATE_CONNECTED":
            event_type = EventType.CANDIDATE_DISCONNECTED
        else:
            self._room_events_by_transition_id[transition_id] = None
            return None
        event = self._event(
            event_type=event_type,
            source=EventSource.BROWSER,
            occurred_at=self._now(),
            question_id=None,
            payload={"previous_state": previous_name, "current_state": current_name},
            identity={"transition_id": transition_id},
        )
        self._room_events_by_transition_id[transition_id] = event
        return event

    def tts_result(
        self,
        *,
        effect_id: str,
        outcome: Literal["started", "completed", "cancelled", "failed"],
        question_id: int | None,
        result_summary: str | None = None,
        result_status: str | None = None,
    ) -> InterviewEvent:
        """Normalize a playback lifecycle callback into the closed event type."""

        if not effect_id.strip():
            raise ValueError("effect_id must not be empty")
        event_type = {
            "started": EventType.TTS_STARTED,
            "completed": EventType.TTS_COMPLETED,
            "cancelled": EventType.TTS_CANCELLED,
            "failed": EventType.TTS_FAILED,
        }[outcome]
        return self._event(
            event_type=event_type,
            source=EventSource.TTS,
            occurred_at=self._now(),
            question_id=question_id,
            payload={
                "effect_id": effect_id,
                "outcome": outcome,
                **(
                    {"result_summary": result_summary}
                    if result_summary is not None
                    else {}
                ),
                **(
                    {"result_status": result_status}
                    if result_status is not None
                    else {}
                ),
            },
            identity={
                "effect_id": effect_id,
                "outcome": outcome,
                "result_status": result_status,
            },
        )

    def audio_route_result(
        self,
        *,
        effect_id: str,
        outcome: Literal["completed", "failed"],
        question_id: int | None,
        result_summary: str,
    ) -> InterviewEvent:
        """Normalize the supervised output-route check that precedes an audio retry."""

        if not effect_id.strip():
            raise ValueError("effect_id must not be empty")
        event_type = {
            "completed": EventType.AUDIO_ROUTE_COMPLETED,
            "failed": EventType.AUDIO_ROUTE_FAILED,
        }[outcome]
        return self._event(
            event_type=event_type,
            source=EventSource.TTS,
            occurred_at=self._now(),
            question_id=question_id,
            payload={
                "effect_id": effect_id,
                "outcome": outcome,
                "result_summary": result_summary,
            },
            identity={"effect_id": effect_id, "outcome": outcome},
        )

    def operator_action(
        self,
        action: Literal["pause", "takeover", "resume", "stop", "complete_turn"],
        *,
        question_id: int | None = None,
    ) -> InterviewEvent:
        """Normalize a supervised operator control without granting an effect."""

        event_type = {
            "pause": EventType.OPERATOR_PAUSE,
            "takeover": EventType.OPERATOR_TAKEOVER,
            "resume": EventType.OPERATOR_RESUME,
            "stop": EventType.OPERATOR_STOP,
            "complete_turn": EventType.TURN_COMPLETE,
        }[action]
        if action == "complete_turn" and question_id is None:
            raise ValueError("complete_turn requires an active question_id")
        return self._event(
            event_type=event_type,
            source=EventSource.OPERATOR,
            occurred_at=self._now(),
            question_id=question_id,
            payload={"action": action},
            identity={"action": action, "question_id": question_id},
        )

    def _callback_timestamp(self, segment: object) -> datetime:
        timestamp = getattr(segment, "timestamp", None)
        if isinstance(timestamp, datetime) and timestamp.tzinfo is not None:
            return timestamp
        return self._now()

    @staticmethod
    def _room_state_name(state: object | None) -> str | None:
        if state is None:
            return None
        value = getattr(state, "value", state)
        return str(value).strip()

    def _event(
        self,
        *,
        event_type: EventType,
        source: EventSource,
        occurred_at: datetime,
        question_id: int | None,
        payload: dict[str, Any],
        identity: dict[str, Any],
    ) -> InterviewEvent:
        event_id = stable_event_id(
            "event",
            session_id=self._session_id,
            event_type=event_type,
            source=source,
            question_id=question_id,
            identity=identity,
        )
        prior_event = self._events_by_id.get(event_id)
        if prior_event is not None:
            return prior_event
        event = InterviewEvent(
            event_id=event_id,
            event_type=event_type,
            occurred_at=occurred_at,
            source=source,
            session_id=self._session_id,
            question_id=question_id,
            payload=payload,
        )
        self._events_by_id[event_id] = event
        return event


class EventIngress:
    """Deduplicate incoming events and retain only the bounded recent projection."""

    def __init__(self, *, session_id: str, ledger: EventLedger) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        self._session_id = session_id
        self._ledger = ledger
        self._recent_events: list[InterviewEvent] = []

    @property
    def recent_events(self) -> tuple[InterviewEvent, ...]:
        """Return the bounded ingress projection in arrival order."""

        return tuple(self._recent_events)

    def ingest(self, event: InterviewEvent) -> InterviewEvent | None:
        """Claim an event once; exact duplicates do not reach the graph."""

        if event.session_id != self._session_id:
            raise ValueError("event session_id does not match this ingress")
        if not self._ledger.append(event):
            return None
        self._recent_events = append_interview_events(self._recent_events, [event])
        return event


__all__ = ["EventIngress", "EventNormalizer"]
