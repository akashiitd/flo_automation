"""Offline behavioral seams for automatic candidate-turn detection."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from orchestrator.events import EventSource, EventType, InterviewEvent
from orchestrator.state import TurnState
from orchestrator.turn_detection import TurnDetector


def _final_segment(
    *, question_id: int, text: str, at: datetime, segment_id: str = "segment-1"
) -> InterviewEvent:
    return InterviewEvent(
        event_id=f"segment-{question_id}-{at.timestamp()}",
        event_type=EventType.TRANSCRIPT_FINAL,
        occurred_at=at,
        source=EventSource.CANDIDATE_ASR,
        session_id="phase6-session",
        question_id=question_id,
        payload={"segment_id": segment_id, "text": text},
    )


def test_turn_detector_completes_a_normal_question_bound_answer_after_silence() -> None:
    started_at = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    detector = TurnDetector(session_id="phase6-session", question_id=7)
    answer = " ".join(f"word{index}" for index in range(25))

    assert [
        event.event_type
        for event in detector.observe(
            _final_segment(question_id=7, text=answer, at=started_at)
        )
    ] == [EventType.SPEECH_STARTED]
    assert detector.answer_segments == (answer,)
    assert (
        detector.observe(
            _final_segment(
                question_id=7,
                text="foreign audio",
                at=started_at,
                segment_id="foreign",
            ).model_copy(update={"session_id": "other-session"})
        )
        == ()
    )
    assert (
        detector.observe(
            _final_segment(question_id=8, text="late audio", at=started_at)
        )
        == ()
    )
    assert detector.answer_segments == (answer,)

    assert [
        event.event_type for event in detector.poll(started_at + timedelta(seconds=2.5))
    ] == [EventType.SILENCE_STARTED]
    assert detector.poll(started_at + timedelta(seconds=3.49)) == ()
    assert [
        event.event_type for event in detector.poll(started_at + timedelta(seconds=3.5))
    ] == [EventType.TURN_COMPLETE]
    assert detector.answer_text == answer


def test_turn_detector_keeps_control_speech_out_of_answer_content() -> None:
    started_at = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    detector = TurnDetector(session_id="phase6-session", question_id=7)

    detector.observe(
        _final_segment(
            question_id=7,
            text="I would inspect the retry budget.",
            at=started_at,
            segment_id="answer",
        )
    )
    detector.observe(
        _final_segment(
            question_id=7,
            text="Let me think about that.",
            at=started_at + timedelta(milliseconds=100),
            segment_id="thinking",
        )
    )

    assert detector.answer_text == "I would inspect the retry budget."
    assert detector.control_utterances == ("Let me think about that.",)
    assert detector.poll(started_at + timedelta(seconds=4)) == ()

    explicit = TurnDetector(session_id="phase6-session", question_id=7)
    explicit.observe(
        _final_segment(
            question_id=7,
            text="I would inspect the retry budget.",
            at=started_at,
            segment_id="answer",
        )
    )
    explicit.observe(
        _final_segment(
            question_id=7,
            text="That's all.",
            at=started_at + timedelta(milliseconds=100),
            segment_id="complete",
        )
    )

    assert explicit.answer_text == "I would inspect the retry budget."
    assert explicit.poll(started_at + timedelta(milliseconds=849)) == ()
    assert [
        event.event_type
        for event in explicit.poll(started_at + timedelta(milliseconds=850))
    ] == [EventType.TURN_COMPLETE]


def test_turn_detector_accepts_typed_operator_and_external_silence_observations() -> (
    None
):
    started_at = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    detector = TurnDetector(session_id="phase6-session", question_id=7)
    detector.observe(
        _final_segment(
            question_id=7,
            text="I would inspect the retry budget.",
            at=started_at,
        )
    )
    operator_completion = InterviewEvent(
        event_id="operator-complete",
        event_type=EventType.TURN_COMPLETE,
        occurred_at=started_at + timedelta(seconds=1),
        source=EventSource.OPERATOR,
        session_id="phase6-session",
        question_id=7,
        payload={"action": "complete_turn"},
    )

    completion = detector.observe(operator_completion)

    assert [event.event_type for event in completion] == [EventType.TURN_COMPLETE]
    assert completion[0].source is EventSource.OPERATOR
    assert completion[0].payload["reason"] == "operator_override"

    silence_detector = TurnDetector(session_id="phase6-session", question_id=7)
    silence_detector.observe(
        _final_segment(
            question_id=7,
            text="I would inspect the retry budget.",
            at=started_at,
        )
    )
    external_timeout = InterviewEvent(
        event_id="silence-timeout",
        event_type=EventType.SILENCE_TIMEOUT,
        occurred_at=started_at + timedelta(seconds=2),
        source=EventSource.CANDIDATE_ASR,
        session_id="phase6-session",
        question_id=7,
        payload={},
    )

    timed_out = silence_detector.observe(external_timeout)

    assert [event.event_type for event in timed_out] == [EventType.TURN_COMPLETE]
    assert timed_out[0].payload["reason"] == "external_silence_timeout"


def test_turn_detector_starts_speech_once_and_exposes_a_question_bound_snapshot() -> (
    None
):
    started_at = datetime(2026, 7, 13, 10, 0, tzinfo=UTC)
    detector = TurnDetector(session_id="phase6-session", question_id=7)

    first = detector.observe(
        _final_segment(
            question_id=7,
            text="First answer part.",
            at=started_at,
            segment_id="first",
        )
    )
    second = detector.observe(
        _final_segment(
            question_id=7,
            text="Second answer part.",
            at=started_at + timedelta(seconds=1),
            segment_id="second",
        )
    )

    assert [event.event_type for event in first] == [EventType.SPEECH_STARTED]
    assert second == ()
    assert detector.turn_state == TurnState(
        question_id=7,
        answer_segments=["First answer part.", "Second answer part."],
        answer_started_at=started_at,
    )
