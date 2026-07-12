"""Validated contracts shared by every LLM provider."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


RatingLabel = Literal["Excellent", "Good", "Average", "Weak", "Poor"]
ModelClass = Literal["fast", "deep"]


class EvaluationInput(BaseModel):
    """Question, rubric, and candidate answer supplied to the evaluator."""

    model_config = ConfigDict(extra="forbid")

    question_id: int = Field(ge=1)
    question: str = Field(min_length=1)
    ideal_answer: str = Field(min_length=1)
    candidate_answer: str = Field(min_length=1)


class QuestionEvaluation(BaseModel):
    """Structured scoring result required by the automation plan."""

    model_config = ConfigDict(extra="forbid")

    question_id: int = Field(ge=1)
    score: int = Field(ge=1, le=5)
    rating_label: RatingLabel
    evidence: list[str] = Field(min_length=1)
    follow_up: str = Field(min_length=1)
    feedback: str = Field(min_length=1)
    confidence: float = Field(ge=0, le=1)

    @model_validator(mode="after")
    def rating_matches_score(self) -> QuestionEvaluation:
        expected = {
            5: "Excellent",
            4: "Good",
            3: "Average",
            2: "Weak",
            1: "Poor",
        }[self.score]
        if self.rating_label != expected:
            raise ValueError(
                f"rating_label must be {expected!r} when score is {self.score}"
            )
        return self


class JobDescriptionAnswer(BaseModel):
    """A candidate-facing answer constrained to an extracted job description."""

    model_config = ConfigDict(extra="forbid")

    answer: str = Field(min_length=1)
    grounded: bool
    evidence: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def grounded_answers_require_evidence(self) -> JobDescriptionAnswer:
        if self.grounded and not self.evidence:
            raise ValueError("grounded answers must cite job-description evidence")
        if any(not item.strip() for item in self.evidence):
            raise ValueError("job-description evidence must not be blank")
        return self


class ProviderMetadata(BaseModel):
    """Normalized runtime and cost data emitted for every generation."""

    model_config = ConfigDict(extra="forbid")

    provider: str
    model: str
    request_purpose: str
    latency_ms: int = Field(ge=0)
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    estimated_cost_usd: float = Field(ge=0)
    fallback_used: bool = False
    fallback_reason: str | None = None
    pii_redaction_ran: bool = False
    recorded_at: datetime = Field(default_factory=lambda: datetime.now(UTC))


class StructuredGeneration(BaseModel):
    """Provider-neutral response envelope consumed by the evaluator."""

    model_config = ConfigDict(extra="forbid")

    output: dict[str, object]
    metadata: ProviderMetadata
