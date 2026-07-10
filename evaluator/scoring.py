"""Provider-neutral scoring entry point used by CLI and orchestration."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from llm.prompts import scoring_messages
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
    generation.output = evaluation.model_dump(mode="json")
    if usage_tracker is not None:
        usage_tracker.record(generation.metadata)
    return generation
