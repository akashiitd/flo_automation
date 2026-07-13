"""Pure, question-bound automatic end-of-answer detection."""

from __future__ import annotations

from datetime import datetime, timedelta

from orchestrator.event_identity import stable_event_id
from orchestrator.events import EventSource, EventType, InterviewEvent
from orchestrator.state import TurnState

_SHORT_ANSWER_WORDS = 25
_SHORT_ANSWER_SILENCE_SECONDS = 3.5
_NORMAL_ANSWER_SILENCE_SECONDS = 2.5
_NORMAL_ANSWER_CONFIRMATION_SECONDS = 1.0
_THINKING_EXTENSION_SECONDS = 7.0
_COMPLETION_DEBOUNCE_SECONDS = 0.75
_COMPLETION_UTTERANCES = frozenset({"that is all", "that's all", "thats all"})


class TurnDetector:
    """Produce turn-bound timing events without evaluating candidate content."""

    def __init__(self, *, session_id: str, question_id: int) -> None:
        if not session_id.strip():
            raise ValueError("session_id must not be empty")
        if question_id < 1:
            raise ValueError("question_id must be positive")
        self._session_id = session_id
        self._question_id = question_id
        self._answer_segments: list[str] = []
        self._control_utterances: list[str] = []
        self._seen_segment_ids: set[str] = set()
        self._answer_started_at: datetime | None = None
        self._last_answer_at: datetime | None = None
        self._thinking_until: datetime | None = None
        self._explicit_completion_at: datetime | None = None
        self._silence_started = False
        self._completed = False

    @property
    def answer_segments(self) -> tuple[str, ...]:
        """Return only content recorded under this explicit question boundary."""

        return tuple(self._answer_segments)

    @property
    def answer_text(self) -> str:
        """Return the accumulated answer content without control utterances."""

        return " ".join(self._answer_segments)

    @property
    def control_utterances(self) -> tuple[str, ...]:
        """Keep deterministic control phrases separate from answer content."""

        return tuple(self._control_utterances)

    @property
    def turn_state(self) -> TurnState:
        """Expose the bounded, question-bound state ready for a graph update."""

        return TurnState(
            question_id=self._question_id,
            answer_segments=list(self._answer_segments),
            control_utterances=list(self._control_utterances),
            answer_started_at=self._answer_started_at,
        )

    def observe(self, event: InterviewEvent) -> tuple[InterviewEvent, ...]:
        """Accept one final candidate segment only when it belongs to this question."""

        if (
            event.session_id != self._session_id
            or event.question_id != self._question_id
        ):
            return ()
        if self._completed:
            return ()
        if (
            event.event_type is EventType.TURN_COMPLETE
            and event.source is EventSource.OPERATOR
        ):
            return self._complete(
                event.occurred_at,
                reason="operator_override",
                source=EventSource.OPERATOR,
            )
        if event.source is not EventSource.CANDIDATE_ASR:
            return ()
        if event.event_type is EventType.SPEECH_STARTED:
            self._silence_started = False
            return ()
        if event.event_type is EventType.SILENCE_STARTED:
            if self._last_answer_at is not None:
                self._silence_started = True
            return ()
        if event.event_type is EventType.SILENCE_TIMEOUT:
            if self._last_answer_at is None:
                return ()
            return self._complete(
                event.occurred_at,
                reason="external_silence_timeout",
                source=EventSource.CANDIDATE_ASR,
            )
        if event.event_type is not EventType.TRANSCRIPT_FINAL:
            return ()
        segment_id = event.payload.get("segment_id")
        text = event.payload.get("text")
        if not isinstance(segment_id, str) or not isinstance(text, str):
            return ()
        normalized = text.strip()
        if not normalized or segment_id in self._seen_segment_ids:
            return ()
        self._seen_segment_ids.add(segment_id)
        normalized_control = " ".join(normalized.casefold().strip(".,!?").split())
        if normalized_control.startswith("let me think"):
            self._control_utterances.append(normalized)
            self._thinking_until = event.occurred_at + timedelta(
                seconds=_THINKING_EXTENSION_SECONDS
            )
            return ()
        if normalized_control in _COMPLETION_UTTERANCES:
            self._control_utterances.append(normalized)
            self._explicit_completion_at = event.occurred_at
            return ()
        answer_started = self._last_answer_at is None
        self._answer_segments.append(normalized)
        if self._answer_started_at is None:
            self._answer_started_at = event.occurred_at
        self._last_answer_at = event.occurred_at
        self._thinking_until = None
        self._explicit_completion_at = None
        self._silence_started = False
        if not answer_started:
            return ()
        return (self._event(EventType.SPEECH_STARTED, event.occurred_at, {}),)

    def poll(self, now: datetime) -> tuple[InterviewEvent, ...]:
        """Emit silence and completion events only after answer content exists."""

        if self._completed or self._last_answer_at is None:
            return ()
        if now < self._last_answer_at:
            raise ValueError("poll time cannot precede the latest answer segment")
        if self._explicit_completion_at is not None:
            if now < self._explicit_completion_at + timedelta(
                seconds=_COMPLETION_DEBOUNCE_SECONDS
            ):
                return ()
            return self._complete(
                now,
                reason="explicit_completion",
                source=EventSource.CANDIDATE_ASR,
            )

        silence_base = max(
            self._last_answer_at,
            self._thinking_until or self._last_answer_at,
        )
        if now < silence_base:
            return ()
        silence_seconds = (now - silence_base).total_seconds()
        if len(self.answer_text.split()) < _SHORT_ANSWER_WORDS:
            if silence_seconds < _SHORT_ANSWER_SILENCE_SECONDS:
                return ()
            return self._complete(
                now,
                reason="short_answer_silence",
                source=EventSource.CANDIDATE_ASR,
            )

        events: list[InterviewEvent] = []
        if (
            silence_seconds >= _NORMAL_ANSWER_SILENCE_SECONDS
            and not self._silence_started
        ):
            self._silence_started = True
            events.append(
                self._event(
                    EventType.SILENCE_STARTED,
                    now,
                    {"silence_seconds": round(silence_seconds, 3)},
                )
            )
        if silence_seconds >= (
            _NORMAL_ANSWER_SILENCE_SECONDS + _NORMAL_ANSWER_CONFIRMATION_SECONDS
        ):
            events.extend(
                self._complete(
                    now,
                    reason="normal_answer_silence",
                    source=EventSource.CANDIDATE_ASR,
                )
            )
        return tuple(events)

    def operator_complete(self, occurred_at: datetime) -> tuple[InterviewEvent, ...]:
        """Provide the Phase 6 operator override without synthesizing content."""

        if self._completed or self._last_answer_at is None:
            return ()
        return self._complete(
            occurred_at,
            reason="operator_override",
            source=EventSource.OPERATOR,
        )

    def _complete(
        self, occurred_at: datetime, *, reason: str, source: EventSource
    ) -> tuple[InterviewEvent, ...]:
        self._completed = True
        return (
            self._event(
                EventType.TURN_COMPLETE,
                occurred_at,
                {
                    "answer_character_count": len(self.answer_text),
                    "answer_word_count": len(self.answer_text.split()),
                    "reason": reason,
                },
                source=source,
            ),
        )

    def _event(
        self,
        event_type: EventType,
        occurred_at: datetime,
        payload: dict[str, object],
        *,
        source: EventSource = EventSource.CANDIDATE_ASR,
    ) -> InterviewEvent:
        return InterviewEvent(
            event_id=stable_event_id(
                "turn",
                session_id=self._session_id,
                event_type=event_type,
                source=source,
                question_id=self._question_id,
                identity={"occurred_at": occurred_at.isoformat(), "payload": payload},
            ),
            event_type=event_type,
            occurred_at=occurred_at,
            source=source,
            session_id=self._session_id,
            question_id=self._question_id,
            payload=payload,
        )


__all__ = ["TurnDetector"]
