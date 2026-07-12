"""State contracts for the supervised interview controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from app.questions import InterviewQuestion


class InterviewPhase(StrEnum):
    START = "START"
    HUMAN_APPROVAL = "HUMAN_APPROVAL"
    LISTENING = "LISTENING"
    ANSWER_COMPLETE = "ANSWER_COMPLETE"
    EVALUATING = "EVALUATING"
    FOLLOW_UP_READY = "FOLLOW_UP_READY"
    NEXT_QUESTION = "NEXT_QUESTION"
    DONE = "DONE"
    ERROR = "ERROR"


class PendingPromptKind(StrEnum):
    INTRODUCTION = "introduction"
    QUESTION = "question"
    FOLLOW_UP = "follow_up"


@dataclass(slots=True)
class InterviewState:
    candidate_name: str
    questions: tuple[InterviewQuestion, ...]
    current_question_index: int = 0
    current_question_id: int | None = None
    candidate_answer_segments: list[str] = field(default_factory=list)
    pending_follow_up: str | None = None
    pending_candidate_prompt: str | None = None
    pending_prompt_kind: PendingPromptKind | None = None
    phase: InterviewPhase = InterviewPhase.START
