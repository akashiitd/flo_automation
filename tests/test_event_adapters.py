"""Public offline seams for Phase 6 event normalization and ingress."""

from __future__ import annotations

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from orchestrator.event_adapters import EventIngress, EventNormalizer
from orchestrator.event_ledger import EventLedger
from orchestrator.events import EventSource, EventType, InterviewEvent
from orchestrator.reducers import RECENT_EVENT_LIMIT


def test_apple_speech_callback_is_question_bound_and_deduplicated(tmp_path) -> None:
    normalizer = EventNormalizer(
        session_id="phase6-session",
        now=lambda: datetime(2026, 7, 13, 10, 0, tzinfo=UTC),
    )
    candidate_segment = SimpleNamespace(
        text="I would use exponential backoff.",
        source="system",
        start_time=1.25,
        end_time=3.5,
        confidence=0.94,
        timestamp=datetime(2026, 7, 13, 10, 0, tzinfo=UTC),
    )

    event = normalizer.apple_speech_callback(
        candidate_segment, question_id=7, is_final=True
    )
    redelivered_without_timestamp = normalizer.apple_speech_callback(
        SimpleNamespace(
            text=candidate_segment.text,
            source=candidate_segment.source,
            start_time=candidate_segment.start_time,
            end_time=candidate_segment.end_time,
            confidence=candidate_segment.confidence,
        ),
        question_id=7,
        is_final=True,
    )

    assert event is not None
    assert event.event_type is EventType.TRANSCRIPT_FINAL
    assert event.source is EventSource.CANDIDATE_ASR
    assert event.question_id == 7
    assert event.payload["text"] == "I would use exponential backoff."
    assert redelivered_without_timestamp is not None
    assert redelivered_without_timestamp.event_id == event.event_id
    receipt_times = iter(
        (
            datetime(2026, 7, 13, 10, 1, tzinfo=UTC),
            datetime(2026, 7, 13, 10, 2, tzinfo=UTC),
        )
    )
    redelivery_normalizer = EventNormalizer(
        session_id="phase6-session", now=lambda: next(receipt_times)
    )
    first_without_timestamp = redelivery_normalizer.apple_speech_callback(
        SimpleNamespace(
            text=candidate_segment.text,
            source=candidate_segment.source,
            start_time=candidate_segment.start_time,
            end_time=candidate_segment.end_time,
            confidence=candidate_segment.confidence,
        ),
        question_id=7,
        is_final=True,
    )
    second_without_timestamp = redelivery_normalizer.apple_speech_callback(
        SimpleNamespace(
            text=candidate_segment.text,
            source=candidate_segment.source,
            start_time=candidate_segment.start_time,
            end_time=candidate_segment.end_time,
            confidence=candidate_segment.confidence,
        ),
        question_id=7,
        is_final=True,
    )
    assert first_without_timestamp is not None
    assert second_without_timestamp is not None
    assert second_without_timestamp.event_id == first_without_timestamp.event_id
    assert second_without_timestamp == first_without_timestamp
    redelivery_ingress = EventIngress(
        session_id="phase6-session",
        ledger=EventLedger(tmp_path / "redelivery_ledger.sqlite3"),
    )
    assert redelivery_ingress.ingest(first_without_timestamp) == first_without_timestamp
    assert redelivery_ingress.ingest(second_without_timestamp) is None
    reopened_normalizer = EventNormalizer(
        session_id="phase6-session",
        now=lambda: datetime(2026, 7, 13, 10, 3, tzinfo=UTC),
    )
    reopened_redelivery = reopened_normalizer.apple_speech_callback(
        SimpleNamespace(
            text=candidate_segment.text,
            source=candidate_segment.source,
            start_time=candidate_segment.start_time,
            end_time=candidate_segment.end_time,
            confidence=candidate_segment.confidence,
        ),
        question_id=7,
        is_final=True,
    )
    assert reopened_redelivery is not None
    assert reopened_redelivery.event_id == first_without_timestamp.event_id
    assert reopened_redelivery.occurred_at != first_without_timestamp.occurred_at
    assert reopened_redelivery != first_without_timestamp
    assert redelivery_ingress.ingest(reopened_redelivery) is None
    assert (
        normalizer.apple_speech_callback(
            candidate_segment, question_id=None, is_final=True
        )
        is None
    )
    assert (
        normalizer.apple_speech_callback(
            SimpleNamespace(**{**candidate_segment.__dict__, "source": "microphone"}),
            question_id=7,
            is_final=True,
        )
        is None
    )

    ingress = EventIngress(
        session_id="phase6-session",
        ledger=EventLedger(tmp_path / "event_ledger.sqlite3"),
    )

    assert ingress.ingest(event) == event
    assert ingress.ingest(event) is None
    assert ingress.recent_events == (event,)


def test_other_callbacks_normalize_to_the_closed_event_vocabulary() -> None:
    normalizer = EventNormalizer(
        session_id="phase6-session",
        now=lambda: datetime(2026, 7, 13, 10, 0, tzinfo=UTC),
    )

    warning = normalizer.timer_threshold(
        "FIVE_MINUTES_REMAINING", remaining_seconds=300
    )
    expiry = normalizer.timer_threshold("TIME_LIMIT_REACHED", remaining_seconds=0)
    joined = normalizer.room_state_changed(
        transition_id="room-joined",
        previous="LAUNCHED",
        current="INTERVIEWER_IN_ROOM",
    )
    waiting = normalizer.room_state_changed(
        transition_id="room-waiting",
        previous="INTERVIEWER_IN_ROOM",
        current="WAITING_FOR_CANDIDATE",
    )
    connected = normalizer.room_state_changed(
        transition_id="room-connected",
        previous="WAITING_FOR_CANDIDATE",
        current="CANDIDATE_CONNECTED",
    )
    disconnected = normalizer.room_state_changed(
        transition_id="room-disconnected",
        previous="CANDIDATE_CONNECTED",
        current="WAITING_FOR_CANDIDATE",
    )
    reconnected = normalizer.room_state_changed(
        transition_id="room-reconnected",
        previous="WAITING_FOR_CANDIDATE",
        current="CANDIDATE_CONNECTED",
    )
    completed = normalizer.tts_result(
        effect_id="effect-1", outcome="completed", question_id=7
    )
    paused = normalizer.operator_action("pause")
    complete_turn = normalizer.operator_action("complete_turn", question_id=7)

    assert warning.event_type is EventType.TIMER_WARNING
    assert warning.source is EventSource.TIMER
    assert warning.payload == {
        "remaining_seconds": 300.0,
        "threshold": "FIVE_MINUTES_REMAINING",
    }
    assert expiry.event_type is EventType.TIME_LIMIT_REACHED
    assert joined.event_type is EventType.JOINED
    assert waiting is None
    assert (
        normalizer.room_state_changed(
            transition_id="room-connected",
            previous="WAITING_FOR_CANDIDATE",
            current="CANDIDATE_CONNECTED",
        )
        == connected
    )
    assert [event.event_type for event in (connected, disconnected, reconnected)] == [
        EventType.CANDIDATE_CONNECTED,
        EventType.CANDIDATE_DISCONNECTED,
        EventType.CANDIDATE_RECONNECTED,
    ]
    assert completed.event_type is EventType.TTS_COMPLETED
    assert completed.source is EventSource.TTS
    assert completed.question_id == 7
    assert paused.event_type is EventType.OPERATOR_PAUSE
    assert paused.source is EventSource.OPERATOR
    assert complete_turn.event_type is EventType.TURN_COMPLETE
    assert complete_turn.source is EventSource.OPERATOR
    assert complete_turn.question_id == 7
    with pytest.raises(ValueError, match="unsupported timer threshold"):
        normalizer.timer_threshold("almost-time", remaining_seconds=1)


def test_event_ingress_keeps_only_the_bounded_recent_event_window(tmp_path) -> None:
    ingress = EventIngress(
        session_id="phase6-session",
        ledger=EventLedger(tmp_path / "event_ledger.sqlite3"),
    )
    occurred_at = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)

    for index in range(RECENT_EVENT_LIMIT + 2):
        event = InterviewEvent(
            event_id=f"event-{index}",
            event_type=EventType.TIMER_WARNING,
            occurred_at=occurred_at,
            source=EventSource.TIMER,
            session_id="phase6-session",
            payload={"threshold": f"threshold-{index}"},
        )
        assert ingress.ingest(event) == event

    assert len(ingress.recent_events) == RECENT_EVENT_LIMIT
    assert ingress.recent_events[0].event_id == "event-2"
    assert ingress.recent_events[-1].event_id == f"event-{RECENT_EVENT_LIMIT + 1}"
    with pytest.raises(ValueError, match="session_id"):
        ingress.ingest(
            InterviewEvent(
                event_id="other-session",
                event_type=EventType.TIMER_WARNING,
                occurred_at=occurred_at,
                source=EventSource.TIMER,
                session_id="another-session",
                payload={"threshold": "other"},
            )
        )
