"""State contracts for the supervised interview controller."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import StrEnum
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from app.questions import InterviewQuestion
from llm.schemas import QuestionEvaluation
from orchestrator.effects import EffectRequest, EffectResult
from orchestrator.events import InterviewEvent
from orchestrator.intents import IntentDecision
from orchestrator.reducers import append_interview_events, append_skill_evidence


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


class DynamicInterviewPhase(StrEnum):
    """Durable phases owned by the event-driven controller."""

    START = "START"
    LOAD_ARTIFACTS = "LOAD_ARTIFACTS"
    PLAN_INTERVIEW = "PLAN_INTERVIEW"
    PREFLIGHT = "PREFLIGHT"
    AWAIT_START_APPROVAL = "AWAIT_START_APPROVAL"
    JOINING = "JOINING"
    WAITING_FOR_CANDIDATE = "WAITING_FOR_CANDIDATE"
    DISCLOSURE = "DISCLOSURE"
    INTRODUCTION = "INTRODUCTION"
    SELECT_QUESTION = "SELECT_QUESTION"
    RUN_TURN = "RUN_TURN"
    UPDATE_COVERAGE = "UPDATE_COVERAGE"
    CANDIDATE_QUESTIONS = "CANDIDATE_QUESTIONS"
    CLOSING = "CLOSING"
    AGGREGATE_EVALUATION = "AGGREGATE_EVALUATION"
    HUMAN_FINAL_REVIEW = "HUMAN_FINAL_REVIEW"
    DONE = "DONE"
    PAUSED = "PAUSED"
    NEEDS_OPERATOR = "NEEDS_OPERATOR"
    RECOVERY_REVIEW = "RECOVERY_REVIEW"
    ERROR = "ERROR"


class QuestionContentType(StrEnum):
    """How a scanned source card participates in an interview plan."""

    INSTRUCTION = "instruction"
    INTERVIEW_QUESTION = "interview_question"
    CODING_QUESTION = "coding_question"
    MALFORMED = "malformed"


class QuestionMappingSource(StrEnum):
    """Origin of a question-to-skill mapping."""

    EXPLICIT_DOM = "explicit_dom"
    DETERMINISTIC = "deterministic"
    LLM_INFERRED = "llm_inferred"


class SkillAssessmentStatus(StrEnum):
    """Whether available evidence supports a proposed skill score."""

    ASSESSED = "assessed"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class CoverageStatus(StrEnum):
    """Current planning view of one skill parameter."""

    UNASSESSED = "unassessed"
    PARTIALLY_ASSESSED = "partially_assessed"
    SUFFICIENT = "sufficient"
    INSUFFICIENT_EVIDENCE = "insufficient_evidence"


class QuestionState(BaseModel):
    """Read-only question data retained by the dynamic controller."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: int = Field(ge=1)
    question_text: str = Field(min_length=1)
    ideal_answer: str = ""
    has_code_editor: bool = False


class SkillParameter(BaseModel):
    """One read-only FloCareer skill assessment dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1)
    requirement: str = Field(min_length=1)
    level: str = Field(min_length=1)
    rating_scale: int = Field(ge=1, le=5)
    source: str = "flocareer_dom"


class QuestionPlanItem(BaseModel):
    """The audited decision to ask or skip one scanned source question."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: int = Field(ge=1)
    content_type: QuestionContentType
    target_skill_ids: list[str] = Field(default_factory=list)
    mandatory_skill_coverage: list[str] = Field(default_factory=list)
    estimated_minutes: float = Field(gt=0)
    priority: int = Field(ge=0)
    selected: bool
    skip_reason: str | None = None
    mapping_source: QuestionMappingSource
    mapping_confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def selection_and_skip_reason_agree(self) -> QuestionPlanItem:
        if self.selected and self.skip_reason is not None:
            raise ValueError("selected question-plan items cannot have a skip_reason")
        if not self.selected and not (self.skip_reason or "").strip():
            raise ValueError("unselected question-plan items require a skip_reason")
        return self


class SkippedQuestion(BaseModel):
    """The runtime reason a planned question was not asked."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    question_id: int = Field(ge=1)
    reason: str = Field(min_length=1)


class TurnState(BaseModel):
    """Bounded active-turn data; full transcripts remain external artifacts."""

    model_config = ConfigDict(extra="forbid")

    question_id: int = Field(ge=1)
    answer_segments: list[str] = Field(default_factory=list)
    control_utterances: list[str] = Field(default_factory=list)
    answer_started_at: datetime | None = None
    silence_started_at: datetime | None = None


class SkillEvidence(BaseModel):
    """One question-grounded citation supporting one skill assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    question_id: int = Field(ge=1)
    transcript_evidence: str = Field(min_length=1)
    question_score: int = Field(ge=1, le=5)
    relevance_weight: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class SkillAssessment(BaseModel):
    """Evidence-grounded proposed assessment, never an unattended platform rating."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    skill_id: str = Field(min_length=1)
    proposed_score: int | None = Field(default=None, ge=1, le=5)
    status: SkillAssessmentStatus
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def assessment_requires_consistent_evidence(self) -> SkillAssessment:
        if self.status is SkillAssessmentStatus.ASSESSED:
            if self.proposed_score is None or not self.evidence_ids:
                raise ValueError("assessed skills require a score and evidence IDs")
        elif self.proposed_score is not None:
            raise ValueError(
                "insufficient-evidence skills cannot have a proposed score"
            )
        return self


class CoverageState(BaseModel):
    """Planning confidence and evidence status for one skill parameter."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    status: CoverageStatus
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)


class InterruptRequest(BaseModel):
    """A human decision the graph cannot safely make alone."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)


class DynamicInterviewState(BaseModel):
    """JSON-safe, bounded state checkpointed for one isolated interview thread."""

    model_config = ConfigDict(extra="forbid")

    schema_version: Literal[1] = 1
    thread_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    candidate_identifier: str = Field(min_length=1)
    phase: DynamicInterviewPhase = DynamicInterviewPhase.START
    mode: Literal["offline", "shadow", "supervised_live"] = "offline"

    questions: list[QuestionState] = Field(default_factory=list)
    skill_parameters: list[SkillParameter] = Field(default_factory=list)
    question_plan: list[QuestionPlanItem] = Field(default_factory=list)
    current_plan_index: int | None = Field(default=None, ge=0)
    current_question_id: int | None = Field(default=None, ge=1)
    completed_question_ids: list[int] = Field(default_factory=list)
    skipped_questions: list[SkippedQuestion] = Field(default_factory=list)

    current_turn: TurnState | None = None
    recent_events: Annotated[list[InterviewEvent], append_interview_events] = Field(
        default_factory=list
    )
    intent_history: list[IntentDecision] = Field(default_factory=list)
    repeat_counts: dict[str, int] = Field(default_factory=dict)
    clarification_counts: dict[str, int] = Field(default_factory=dict)
    audio_problem_count: int = Field(default=0, ge=0)

    question_evaluations: list[QuestionEvaluation] = Field(default_factory=list)
    skill_evidence: Annotated[list[SkillEvidence], append_skill_evidence] = Field(
        default_factory=list
    )
    skill_assessments: list[SkillAssessment] = Field(default_factory=list)
    coverage: dict[str, CoverageState] = Field(default_factory=dict)

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    remaining_seconds: float = Field(default=0, ge=0)
    timer_events_emitted: list[str] = Field(default_factory=list)

    pending_effect: EffectRequest | None = None
    last_effect_result: EffectResult | None = None
    pending_interrupt: InterruptRequest | None = None
    recovery_reason: str | None = None
    operator_mode: Literal["monitor", "paused", "takeover"] = "monitor"
    disclosure_status: Literal["pending", "accepted", "declined", "unknown"] = "pending"
    final_review_status: Literal["not_ready", "pending", "approved", "rejected"] = (
        "not_ready"
    )
