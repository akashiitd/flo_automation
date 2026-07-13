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
from orchestrator.reducers import (
    RECENT_EVENT_LIMIT,
    SKILL_EVIDENCE_LIMIT,
    append_interview_events,
    append_skill_evidence,
)

MAX_QUESTIONS = 100
MAX_INTENT_HISTORY = 100
MAX_TURN_SEGMENTS = 100
MAX_SEGMENT_CHARACTERS = 4_000
MAX_TIMER_EVENTS = 10


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


class DynamicQuestionEvaluation(QuestionEvaluation):
    """Strict evaluation record accepted by persisted dynamic-controller state."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


class QuestionState(BaseModel):
    """Read-only question data retained by the dynamic controller."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: int = Field(ge=1)
    question_text: str = Field(min_length=1, max_length=10_000)
    ideal_answer: str = Field(default="", max_length=10_000)
    has_code_editor: bool = False


class SkillParameter(BaseModel):
    """One read-only FloCareer skill assessment dimension."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    id: str = Field(min_length=1)
    name: str = Field(min_length=1, max_length=500)
    requirement: str = Field(min_length=1, max_length=100)
    level: str = Field(min_length=1, max_length=100)
    rating_scale: int = Field(ge=1, le=5)
    source: str = "flocareer_dom"


class SkillParametersArtifact(BaseModel):
    """Versioned, read-only artifact extracted from FloCareer skill controls."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    read_only: Literal[True] = True
    parameters: list[SkillParameter] = Field(
        default_factory=list, max_length=MAX_QUESTIONS
    )

    @model_validator(mode="after")
    def parameters_must_have_unique_ids(self) -> SkillParametersArtifact:
        parameter_ids = {parameter.id for parameter in self.parameters}
        if len(parameter_ids) != len(self.parameters):
            raise ValueError("skill parameters must have unique IDs")
        return self


class QuestionPlanItem(BaseModel):
    """The audited decision to ask or skip one scanned source question."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

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
    mapping_evidence: list[Annotated[str, Field(min_length=1, max_length=500)]] = Field(
        default_factory=list, max_length=MAX_QUESTIONS
    )

    @model_validator(mode="after")
    def selection_and_skip_reason_agree(self) -> QuestionPlanItem:
        if self.selected and self.skip_reason is not None:
            raise ValueError("selected question-plan items cannot have a skip_reason")
        if not self.selected and not (self.skip_reason or "").strip():
            raise ValueError("unselected question-plan items require a skip_reason")
        return self


class QuestionPlanArtifact(BaseModel):
    """Versioned offline output that records every ask-or-skip decision."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    items: list[QuestionPlanItem] = Field(
        default_factory=list, max_length=MAX_QUESTIONS
    )

    @model_validator(mode="after")
    def items_must_have_unique_question_ids(self) -> QuestionPlanArtifact:
        question_ids = {item.question_id for item in self.items}
        if len(question_ids) != len(self.items):
            raise ValueError("question-plan items must have unique question IDs")
        return self


class SkippedQuestion(BaseModel):
    """The runtime reason a planned question was not asked."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    question_id: int = Field(ge=1)
    reason: str = Field(min_length=1)


class TurnState(BaseModel):
    """Bounded active-turn data; full transcripts remain external artifacts."""

    model_config = ConfigDict(extra="forbid", strict=True)

    question_id: int = Field(ge=1)
    answer_segments: list[Annotated[str, Field(max_length=MAX_SEGMENT_CHARACTERS)]] = (
        Field(default_factory=list, max_length=MAX_TURN_SEGMENTS)
    )
    control_utterances: list[
        Annotated[str, Field(max_length=MAX_SEGMENT_CHARACTERS)]
    ] = Field(default_factory=list, max_length=MAX_TURN_SEGMENTS)
    answer_started_at: datetime | None = None
    silence_started_at: datetime | None = None


class SkillEvidence(BaseModel):
    """One question-grounded citation supporting one skill assessment."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    evidence_id: str = Field(min_length=1)
    skill_id: str = Field(min_length=1)
    question_id: int = Field(ge=1)
    transcript_evidence: str = Field(min_length=1, max_length=MAX_SEGMENT_CHARACTERS)
    question_score: int = Field(ge=1, le=5)
    relevance_weight: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)


class SkillAssessment(BaseModel):
    """Evidence-grounded proposed assessment, never an unattended platform rating."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    skill_id: str = Field(min_length=1)
    proposed_score: int | None = Field(default=None, ge=1, le=5)
    status: SkillAssessmentStatus
    evidence_ids: list[str] = Field(default_factory=list)
    rationale: str = Field(min_length=1, max_length=MAX_SEGMENT_CHARACTERS)
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

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    status: CoverageStatus
    confidence: float = Field(ge=0, le=1)
    evidence_ids: list[str] = Field(default_factory=list)


class InterruptRequest(BaseModel):
    """A human decision the graph cannot safely make alone."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    kind: str = Field(min_length=1)
    reason: str = Field(min_length=1)
    options: list[str] = Field(default_factory=list)


class DynamicInterviewState(BaseModel):
    """JSON-safe, bounded state checkpointed for one isolated interview thread."""

    model_config = ConfigDict(extra="forbid", strict=True)

    schema_version: Literal[1] = 1
    thread_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    candidate_identifier: str = Field(min_length=1)
    phase: DynamicInterviewPhase = DynamicInterviewPhase.START
    mode: Literal["offline", "shadow", "supervised_live"] = "offline"

    questions: list[QuestionState] = Field(
        default_factory=list, max_length=MAX_QUESTIONS
    )
    skill_parameters: list[SkillParameter] = Field(
        default_factory=list, max_length=MAX_QUESTIONS
    )
    question_plan: list[QuestionPlanItem] = Field(
        default_factory=list, max_length=MAX_QUESTIONS
    )
    current_plan_index: int | None = Field(default=None, ge=0)
    current_question_id: int | None = Field(default=None, ge=1)
    completed_question_ids: list[int] = Field(
        default_factory=list, max_length=MAX_QUESTIONS
    )
    skipped_questions: list[SkippedQuestion] = Field(
        default_factory=list, max_length=MAX_QUESTIONS
    )

    # Event ingress is transient: the parent graph consumes it into
    # ``recent_events`` before routing a state transition.
    pending_event: InterviewEvent | None = None
    current_turn: TurnState | None = None
    recent_events: Annotated[list[InterviewEvent], append_interview_events] = Field(
        default_factory=list, max_length=RECENT_EVENT_LIMIT
    )
    intent_history: list[IntentDecision] = Field(
        default_factory=list, max_length=MAX_INTENT_HISTORY
    )
    repeat_counts: dict[str, int] = Field(default_factory=dict)
    clarification_counts: dict[str, int] = Field(default_factory=dict)
    audio_problem_count: int = Field(default=0, ge=0)

    question_evaluations: list[DynamicQuestionEvaluation] = Field(
        default_factory=list, max_length=MAX_QUESTIONS
    )
    skill_evidence: Annotated[list[SkillEvidence], append_skill_evidence] = Field(
        default_factory=list, max_length=SKILL_EVIDENCE_LIMIT
    )
    skill_assessments: list[SkillAssessment] = Field(
        default_factory=list, max_length=MAX_QUESTIONS
    )
    coverage: dict[str, CoverageState] = Field(default_factory=dict)

    started_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    remaining_seconds: float = Field(default=0, ge=0)
    timer_events_emitted: list[str] = Field(
        default_factory=list, max_length=MAX_TIMER_EVENTS
    )

    pending_effect: EffectRequest | None = None
    last_effect_request: EffectRequest | None = None
    last_effect_result: EffectResult | None = None
    pending_interrupt: InterruptRequest | None = None
    recovery_reason: str | None = None
    operator_mode: Literal["monitor", "paused", "takeover"] = "monitor"
    disclosure_status: Literal["pending", "accepted", "declined", "unknown"] = "pending"
    final_review_status: Literal["not_ready", "pending", "approved", "rejected"] = (
        "not_ready"
    )

    @model_validator(mode="after")
    def references_must_belong_to_this_session(self) -> DynamicInterviewState:
        """Keep one checkpoint isolated from other sessions and source artifacts."""

        question_ids = {question.id for question in self.questions}
        if len(question_ids) != len(self.questions):
            raise ValueError("questions must have unique IDs")
        skill_ids = {skill.id for skill in self.skill_parameters}
        if len(skill_ids) != len(self.skill_parameters):
            raise ValueError("skill_parameters must have unique IDs")

        def require_question_id(question_id: int, *, field_name: str) -> None:
            if question_id not in question_ids:
                raise ValueError(f"{field_name} references an unknown question_id")

        def require_skill_id(skill_id: str, *, field_name: str) -> None:
            if skill_id not in skill_ids:
                raise ValueError(f"{field_name} references an unknown skill_id")

        for plan_item in self.question_plan:
            require_question_id(plan_item.question_id, field_name="question_plan")
            for skill_id in [
                *plan_item.target_skill_ids,
                *plan_item.mandatory_skill_coverage,
            ]:
                require_skill_id(skill_id, field_name="question_plan")
        if self.current_plan_index is not None:
            if self.current_plan_index >= len(self.question_plan):
                raise ValueError("current_plan_index is outside question_plan")
            if (
                self.current_question_id is not None
                and self.question_plan[self.current_plan_index].question_id
                != self.current_question_id
            ):
                raise ValueError("current_plan_index must point to current_question_id")
        for skipped in self.skipped_questions:
            require_question_id(skipped.question_id, field_name="skipped_questions")
        for question_id in [*self.completed_question_ids, *self.recent_question_ids]:
            require_question_id(question_id, field_name="question state")
        if self.current_question_id is not None:
            require_question_id(
                self.current_question_id, field_name="current_question_id"
            )
        if self.current_turn is not None:
            require_question_id(
                self.current_turn.question_id, field_name="current_turn"
            )
        for evaluation in self.question_evaluations:
            require_question_id(
                evaluation.question_id, field_name="question_evaluations"
            )
        for evidence in self.skill_evidence:
            require_question_id(evidence.question_id, field_name="skill_evidence")
            require_skill_id(evidence.skill_id, field_name="skill_evidence")
            mapped_skill_ids = {
                skill_id
                for plan_item in self.question_plan
                if plan_item.question_id == evidence.question_id
                for skill_id in [
                    *plan_item.target_skill_ids,
                    *plan_item.mandatory_skill_coverage,
                ]
            }
            if evidence.skill_id not in mapped_skill_ids:
                raise ValueError(
                    "skill_evidence must map its question to its referenced skill"
                )

        evidence_ids = {evidence.evidence_id for evidence in self.skill_evidence}
        if len(evidence_ids) != len(self.skill_evidence):
            raise ValueError("skill_evidence must have unique evidence IDs")
        evidence_skill_ids = {
            evidence.evidence_id: evidence.skill_id for evidence in self.skill_evidence
        }
        for assessment in self.skill_assessments:
            require_skill_id(assessment.skill_id, field_name="skill_assessments")
            if not set(assessment.evidence_ids).issubset(evidence_ids):
                raise ValueError("skill_assessments reference unknown evidence IDs")
            if any(
                evidence_skill_ids[evidence_id] != assessment.skill_id
                for evidence_id in assessment.evidence_ids
            ):
                raise ValueError(
                    "skill_assessments can cite only evidence for their skill"
                )
        for skill_id, coverage in self.coverage.items():
            require_skill_id(skill_id, field_name="coverage")
            if not set(coverage.evidence_ids).issubset(evidence_ids):
                raise ValueError("coverage references unknown evidence IDs")
            if any(
                evidence_skill_ids[evidence_id] != skill_id
                for evidence_id in coverage.evidence_ids
            ):
                raise ValueError("coverage can cite only evidence for its skill")

        for event in self.recent_events:
            if event.session_id != self.session_id:
                raise ValueError("recent_events must use this state session_id")
            if event.question_id is not None:
                require_question_id(event.question_id, field_name="recent_events")
        if self.pending_event is not None:
            if self.pending_event.session_id != self.session_id:
                raise ValueError("pending_event must use this state session_id")
            if self.pending_event.question_id is not None:
                require_question_id(
                    self.pending_event.question_id, field_name="pending_event"
                )
        if self.pending_effect is not None and (
            self.pending_effect.session_id != self.session_id
        ):
            raise ValueError("pending_effect must use this state session_id")
        for effect_request, field_name in (
            (self.pending_effect, "pending_effect"),
            (self.last_effect_request, "last_effect_request"),
        ):
            if effect_request is not None and effect_request.question_id is not None:
                require_question_id(effect_request.question_id, field_name=field_name)
        if self.last_effect_request is not None and (
            self.last_effect_request.session_id != self.session_id
        ):
            raise ValueError("last_effect_request must use this state session_id")
        if self.last_effect_result is not None:
            effect_request = self.last_effect_request or self.pending_effect
            if effect_request is None:
                raise ValueError("last_effect_result requires its effect request")
            if (
                self.last_effect_result.effect_id != effect_request.effect_id
                or self.last_effect_result.session_id != effect_request.session_id
                or self.last_effect_result.effect_type != effect_request.effect_type
                or self.last_effect_result.idempotency_key
                != effect_request.idempotency_key
                or self.last_effect_result.payload_hash != effect_request.payload_hash
            ):
                raise ValueError("last_effect_result does not match its effect request")
        return self

    @property
    def recent_question_ids(self) -> list[int]:
        """Question IDs currently represented by repeat/clarification counters."""

        return [
            int(question_id)
            for question_id in {*self.repeat_counts, *self.clarification_counts}
        ]
