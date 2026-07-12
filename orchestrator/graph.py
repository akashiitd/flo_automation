"""Explicit, human-gated interview turn state machine.

This module intentionally has no browser or TTS dependency. Candidate-visible
speech is represented as a pending prompt and must be approved by its caller.
"""

from __future__ import annotations

from dataclasses import dataclass

from app.questions import InterviewQuestion
from llm.prompts import IDENTITY_DISCLOSURE
from orchestrator.state import InterviewPhase, InterviewState, PendingPromptKind


@dataclass(frozen=True, slots=True)
class Transition:
    from_phase: InterviewPhase
    to_phase: InterviewPhase
    reason: str


class InterviewController:
    """Own a turn sequence while leaving all candidate-visible actions to a human."""

    def __init__(
        self, *, candidate_name: str, questions: tuple[InterviewQuestion, ...]
    ):
        if not candidate_name.strip():
            raise ValueError("candidate_name must not be empty")
        if not questions:
            raise ValueError("at least one question is required")
        self.state = InterviewState(candidate_name=candidate_name, questions=questions)
        self.state.phase = InterviewPhase.START
        self.transitions: list[Transition] = []

    def _transition(self, target: InterviewPhase, reason: str) -> None:
        current = self.state.phase
        self.transitions.append(Transition(current, target, reason))
        self.state.phase = target

    def _current_question(self) -> InterviewQuestion:
        return self.state.questions[self.state.current_question_index]

    def start(self) -> str:
        if self.state.phase is not InterviewPhase.START:
            raise RuntimeError("interview controller has already started")
        self.state.pending_candidate_prompt = (
            f"Hello. {IDENTITY_DISCLOSURE} Please introduce yourself briefly."
        )
        self.state.pending_prompt_kind = PendingPromptKind.INTRODUCTION
        self._transition(InterviewPhase.HUMAN_APPROVAL, "introduction_pending")
        return self.state.pending_candidate_prompt

    def approve_candidate_prompt(self) -> str:
        """Record a human approval and move to the next listening boundary."""

        if self.state.phase is not InterviewPhase.HUMAN_APPROVAL:
            raise RuntimeError("no candidate-visible prompt is awaiting approval")
        prompt = self.state.pending_candidate_prompt
        kind = self.state.pending_prompt_kind
        if prompt is None or kind is None:
            raise RuntimeError("the pending prompt is incomplete")
        if kind is PendingPromptKind.INTRODUCTION:
            self.state.pending_candidate_prompt = self._current_question().question_text
            self.state.pending_prompt_kind = PendingPromptKind.QUESTION
            self._transition(InterviewPhase.HUMAN_APPROVAL, "question_pending_approval")
            return prompt
        self.state.pending_candidate_prompt = None
        self.state.pending_prompt_kind = None
        if kind is PendingPromptKind.FOLLOW_UP:
            self.state.pending_follow_up = None
            self.state.candidate_answer_segments.clear()
            self._transition(InterviewPhase.LISTENING, "follow_up_approved")
            return prompt
        question = self._current_question()
        self.state.current_question_id = question.id
        self.state.candidate_answer_segments.clear()
        self._transition(InterviewPhase.LISTENING, "question_approved")
        return question.question_text

    def record_candidate_segment(self, text: str) -> None:
        if self.state.phase is not InterviewPhase.LISTENING:
            raise RuntimeError("candidate segments are accepted only while listening")
        normalized = text.strip()
        if normalized:
            self.state.candidate_answer_segments.append(normalized)

    def complete_answer(self) -> str:
        if self.state.phase is not InterviewPhase.LISTENING:
            raise RuntimeError("an answer can be completed only while listening")
        answer = " ".join(self.state.candidate_answer_segments).strip()
        if not answer:
            raise RuntimeError("cannot evaluate an empty candidate answer")
        self._transition(InterviewPhase.ANSWER_COMPLETE, "candidate_answer_complete")
        self._transition(InterviewPhase.EVALUATING, "evaluation_requested")
        return answer

    def record_evaluation(self, *, follow_up: str | None) -> None:
        if self.state.phase is not InterviewPhase.EVALUATING:
            raise RuntimeError("evaluation can be recorded only after an answer")
        self.state.pending_follow_up = (follow_up or "").strip() or None
        self._transition(
            InterviewPhase.FOLLOW_UP_READY
            if self.state.pending_follow_up is not None
            else InterviewPhase.NEXT_QUESTION,
            "evaluation_recorded",
        )

    def prepare_follow_up(self) -> str:
        if self.state.phase is not InterviewPhase.FOLLOW_UP_READY:
            raise RuntimeError("no follow-up is ready")
        assert self.state.pending_follow_up is not None
        self.state.pending_candidate_prompt = self.state.pending_follow_up
        self.state.pending_prompt_kind = PendingPromptKind.FOLLOW_UP
        self._transition(InterviewPhase.HUMAN_APPROVAL, "follow_up_pending_approval")
        return self.state.pending_candidate_prompt

    def skip_optional_follow_up(self) -> None:
        """Record a human choice to omit a suggested follow-up for this turn."""

        if self.state.phase is not InterviewPhase.FOLLOW_UP_READY:
            raise RuntimeError("no follow-up is available to skip")
        self.state.pending_follow_up = None
        self._transition(InterviewPhase.NEXT_QUESTION, "follow_up_skipped")

    def prepare_next_question(self) -> str | None:
        if self.state.phase is not InterviewPhase.NEXT_QUESTION:
            raise RuntimeError("the next question is not ready")
        if self.state.current_question_index + 1 >= len(self.state.questions):
            self._transition(InterviewPhase.DONE, "all_questions_complete")
            return None
        self.state.current_question_index += 1
        self.state.pending_candidate_prompt = self._current_question().question_text
        self.state.pending_prompt_kind = PendingPromptKind.QUESTION
        self._transition(
            InterviewPhase.HUMAN_APPROVAL, "next_question_pending_approval"
        )
        return self.state.pending_candidate_prompt
