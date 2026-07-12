"""Thread-safe boundary between candidate capture, playback, and turn state.

This module deliberately owns no browser or audio-device setup.  The caller
must explicitly attach its ``on_transcript_segment`` callback to the selected
candidate-only Apple Speech adapter and register live playback through the
shared barge-in controller.
"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock

from orchestrator.graph import InterviewController
from orchestrator.state import InterviewPhase
from tts.audio_output import PlaybackBargeInController


@dataclass(frozen=True, slots=True)
class CandidateSegmentOutcome:
    """The safe, observable result of one transcript callback."""

    cancelled_playback: bool
    recorded_answer: bool
    question_id: int | None


class CandidateTurnRouter:
    """Route only candidate-only answer segments into the active question turn."""

    def __init__(
        self,
        controller: InterviewController,
        barge_in: PlaybackBargeInController,
    ) -> None:
        self._controller = controller
        self._barge_in = barge_in
        self._lock = RLock()
        self._capture_enabled = True

    @property
    def active_question_id(self) -> int | None:
        """Return a boundary only while the controller accepts an answer."""

        with self._lock:
            if (
                not self._capture_enabled
                or self._controller.state.phase is not InterviewPhase.LISTENING
            ):
                return None
            return self._controller.state.current_question_id

    def begin_question_repeat(self) -> str:
        """Atomically discard the repeat request and pause answer capture."""

        with self._lock:
            self._capture_enabled = False
            return self._controller.repeat_current_question()

    def resume_answer_capture(self) -> None:
        """Resume capture only after the repeated question has finished speaking."""

        with self._lock:
            if self._controller.state.phase is not InterviewPhase.LISTENING:
                raise RuntimeError("answer capture can resume only while listening")
            self._capture_enabled = True

    def on_transcript_segment(self, segment: object) -> CandidateSegmentOutcome:
        """Cancel speech promptly, then persist only an active candidate answer."""

        cancelled = self._barge_in.on_transcript_segment(segment)
        if str(getattr(segment, "source", "")).strip() != "system":
            return CandidateSegmentOutcome(cancelled, False, None)
        text = str(getattr(segment, "text", "")).strip()
        if not text:
            return CandidateSegmentOutcome(cancelled, False, None)

        with self._lock:
            question_id = self.active_question_id
            if question_id is None:
                return CandidateSegmentOutcome(cancelled, False, None)
            self._controller.record_candidate_segment(text)
            return CandidateSegmentOutcome(cancelled, True, question_id)


__all__ = ["CandidateSegmentOutcome", "CandidateTurnRouter"]
