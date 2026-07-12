from __future__ import annotations

from types import SimpleNamespace

from app.questions import InterviewQuestion
from orchestrator.graph import InterviewController
from orchestrator.live_loop import CandidateTurnRouter
from tts.audio_output import PCMPlaybackSession, PlaybackBargeInController


class FakeOutput:
    def write(self, pcm: bytes) -> None:
        pass

    def close(self) -> None:
        pass


def test_candidate_only_system_segment_cancels_current_playback() -> None:
    playback = PCMPlaybackSession(FakeOutput())
    barge_in = PlaybackBargeInController()
    barge_in.register(playback)

    cancelled = barge_in.on_transcript_segment(
        SimpleNamespace(text="Please wait", source="system")
    )

    assert cancelled is True
    assert playback.cancelled is True


def test_empty_or_non_candidate_segment_never_cancels_playback() -> None:
    playback = PCMPlaybackSession(FakeOutput())
    barge_in = PlaybackBargeInController()
    barge_in.register(playback)

    assert (
        barge_in.on_transcript_segment(SimpleNamespace(text="", source="system"))
        is False
    )
    assert (
        barge_in.on_transcript_segment(
            SimpleNamespace(text="Hello", source="microphone")
        )
        is False
    )
    assert playback.cancelled is False


def test_candidate_turn_router_cancels_playback_and_records_question_bound_answer() -> (
    None
):
    controller = InterviewController(
        candidate_name="Candidate Alpha",
        questions=(
            InterviewQuestion(
                id=7,
                question_text="How do retries work?",
                ideal_answer="Use bounded exponential backoff.",
            ),
        ),
    )
    controller.start()
    controller.approve_candidate_prompt()
    controller.approve_candidate_prompt()
    playback = PCMPlaybackSession(FakeOutput())
    barge_in = PlaybackBargeInController()
    barge_in.register(playback)
    router = CandidateTurnRouter(controller, barge_in)

    outcome = router.on_transcript_segment(
        SimpleNamespace(text="I would use capped backoff.", source="system")
    )

    assert outcome.cancelled_playback is True
    assert outcome.recorded_answer is True
    assert outcome.question_id == 7
    assert playback.cancelled is True
    assert controller.complete_answer() == "I would use capped backoff."


def test_candidate_turn_router_does_not_score_audio_outside_an_active_answer_turn() -> (
    None
):
    controller = InterviewController(
        candidate_name="Candidate Alpha",
        questions=(
            InterviewQuestion(id=7, question_text="Question?", ideal_answer="Answer."),
        ),
    )
    router = CandidateTurnRouter(controller, PlaybackBargeInController())

    outcome = router.on_transcript_segment(
        SimpleNamespace(text="Hello", source="system")
    )

    assert outcome.cancelled_playback is False
    assert outcome.recorded_answer is False
    assert outcome.question_id is None
    assert router.active_question_id is None
