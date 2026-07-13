"""Closed, evidence-grounded interpretations of candidate speech."""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CandidateIntent(StrEnum):
    """Candidate intents accepted by deterministic interview policy."""

    ANSWER_CONTENT = "ANSWER_CONTENT"
    ANSWER_CONTINUATION = "ANSWER_CONTINUATION"
    ANSWER_COMPLETE = "ANSWER_COMPLETE"
    REPEAT_REQUEST = "REPEAT_REQUEST"
    CLARIFICATION_REQUEST = "CLARIFICATION_REQUEST"
    AUDIO_PROBLEM = "AUDIO_PROBLEM"
    THINKING_TIME_REQUEST = "THINKING_TIME_REQUEST"
    SKIP_OR_RETURN_LATER_REQUEST = "SKIP_OR_RETURN_LATER_REQUEST"
    CORRECTION_TO_PRIOR_ANSWER = "CORRECTION_TO_PRIOR_ANSWER"
    JOB_DESCRIPTION_QUESTION = "JOB_DESCRIPTION_QUESTION"
    INTERVIEW_PROCESS_QUESTION = "INTERVIEW_PROCESS_QUESTION"
    IDENTITY_QUESTION = "IDENTITY_QUESTION"
    COACHING_OR_ANSWER_REQUEST = "COACHING_OR_ANSWER_REQUEST"
    OFF_TOPIC = "OFF_TOPIC"
    CANDIDATE_WITHDRAWAL = "CANDIDATE_WITHDRAWAL"
    UNKNOWN = "UNKNOWN"


class SafeRoute(StrEnum):
    """Routes an intent classifier may recommend, subject to policy approval."""

    CONTINUE_LISTENING = "CONTINUE_LISTENING"
    COMPLETE_TURN = "COMPLETE_TURN"
    REPEAT_CURRENT_QUESTION = "REPEAT_CURRENT_QUESTION"
    SAFE_CLARIFICATION = "SAFE_CLARIFICATION"
    AUDIO_RECOVERY = "AUDIO_RECOVERY"
    EXTEND_THINKING_TIME = "EXTEND_THINKING_TIME"
    DEFER_CURRENT_QUESTION = "DEFER_CURRENT_QUESTION"
    HANDLE_CORRECTION = "HANDLE_CORRECTION"
    ANSWER_JOB_DESCRIPTION = "ANSWER_JOB_DESCRIPTION"
    ANSWER_INTERVIEW_PROCESS = "ANSWER_INTERVIEW_PROCESS"
    IDENTITY_BOUNDARY = "IDENTITY_BOUNDARY"
    COACHING_BOUNDARY = "COACHING_BOUNDARY"
    OFF_TOPIC_BOUNDARY = "OFF_TOPIC_BOUNDARY"
    HANDLE_WITHDRAWAL = "HANDLE_WITHDRAWAL"
    NEEDS_OPERATOR = "NEEDS_OPERATOR"


class IntentDecision(BaseModel):
    """A structured recommendation grounded in an observed candidate utterance."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    intent: CandidateIntent
    confidence: float = Field(ge=0, le=1)
    evidence_span: str = Field(min_length=1)
    answer_text_to_keep: str
    candidate_requested_action: str | None = None
    safe_route: SafeRoute

    @field_validator("evidence_span")
    @classmethod
    def evidence_span_must_not_be_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("evidence_span must not be blank")
        return value

    def validate_against(self, candidate_transcript: str) -> IntentDecision:
        """Reject a decision whose cited evidence is absent from candidate speech."""

        if self.evidence_span.casefold() not in candidate_transcript.casefold():
            raise ValueError("evidence_span must occur in the candidate transcript")
        return self


__all__ = ["CandidateIntent", "IntentDecision", "SafeRoute"]
