"""Provider-neutral scoring entry point used by CLI and orchestration."""

from __future__ import annotations

import re
from collections.abc import Sequence
from typing import Protocol

from llm.prompts import (
    GENERIC_FOLLOW_UP_QUESTION,
    IDENTITY_BOUNDARY_RESPONSE,
    NO_COACHING_BOUNDARY_RESPONSE,
    OFF_TOPIC_BOUNDARY_RESPONSE,
    scoring_messages,
)
from llm.provider import ChatMessage
from llm.schemas import (
    EvaluationInput,
    ModelClass,
    QuestionEvaluation,
    StructuredGeneration,
)
from llm.usage_tracker import UsageTracker


class StructuredGenerator(Protocol):
    async def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[QuestionEvaluation],
        model_class: ModelClass,
        *,
        request_purpose: str,
    ) -> dict[str, object]: ...


_IDENTITY_REQUEST = re.compile(
    r"\b(?:who|what) (?:are|is) (?:you|this)|\bare you (?:an? )?"
    r"(?:ai|bot|robot|human|real person|interviewer|virtual interviewer|"
    r"automated)|\bare you\b.{0,40}\b(?:ai|bot|robot|human|interviewer|"
    r"automated)\b|\b(?:who|what) is akash\b|\bis this (?:an? )?"
    r"(?:ai|bot|chatgpt|claude|gemini|llm)\b",
    re.IGNORECASE,
)
_COACHING_REQUEST = re.compile(
    r"\b(?:can|could|would|will) you (?:please )?(?:give|tell|show|provide|"
    r"write|help)(?: me)?(?: the)? (?:answer|hint|solution|code|rubric)|"
    r"\b(?:what(?:'s| is) the (?:answer|solution)|give me (?:an? )?(?:answer|"
    r"hint|solution|code|rubric))\b",
    re.IGNORECASE,
)
_OFF_TOPIC_REQUEST = re.compile(
    r"\b(?:tell (?:me )?a joke|what(?:'s| is) the weather|"
    r"(?:let's|can we) discuss (?:sports|movies|music|politics|religion)|"
    r"how was your weekend|tell me about your personal life)\b",
    re.IGNORECASE,
)
_FIXED_BOUNDARY_RESPONSES = {
    IDENTITY_BOUNDARY_RESPONSE,
    NO_COACHING_BOUNDARY_RESPONSE,
    OFF_TOPIC_BOUNDARY_RESPONSE,
}


def candidate_boundary_response(candidate_answer: str) -> str | None:
    """Recognize high-confidence identity and real-time coaching requests."""

    if _IDENTITY_REQUEST.search(candidate_answer):
        return IDENTITY_BOUNDARY_RESPONSE
    if _COACHING_REQUEST.search(candidate_answer):
        return NO_COACHING_BOUNDARY_RESPONSE
    if _OFF_TOPIC_REQUEST.search(candidate_answer):
        return OFF_TOPIC_BOUNDARY_RESPONSE
    return None


def safe_candidate_visible_follow_up(value: str) -> str:
    """Never allow model-authored technical detail to reach the candidate."""

    follow_up = value.strip()
    if follow_up in _FIXED_BOUNDARY_RESPONSES:
        return follow_up
    return GENERIC_FOLLOW_UP_QUESTION


async def evaluate_answer(
    request: EvaluationInput,
    generator: StructuredGenerator,
    *,
    model_class: ModelClass = "fast",
    usage_tracker: UsageTracker | None = None,
) -> StructuredGeneration:
    response = await generator.generate_structured(
        scoring_messages(request),
        QuestionEvaluation,
        model_class,
        request_purpose="feedback_draft",
    )
    generation = StructuredGeneration.model_validate(response)
    evaluation = QuestionEvaluation.model_validate(generation.output)
    if evaluation.question_id != request.question_id:
        raise ValueError(
            "evaluation question_id does not match the requested question_id"
        )
    evaluation.follow_up = candidate_boundary_response(
        request.candidate_answer
    ) or safe_candidate_visible_follow_up(evaluation.follow_up)
    generation.output = evaluation.model_dump(mode="json")
    if usage_tracker is not None:
        usage_tracker.record(generation.metadata)
    return generation
