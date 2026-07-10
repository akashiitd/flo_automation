"""Policy-aware routing between the local primary and guarded cloud fallback."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol, TypeVar

from pydantic import BaseModel

from app.config import Settings
from llm.lmstudio_provider import LMStudioProvider
from llm.openrouter_provider import OpenRouterProvider
from llm.provider import ChatMessage
from llm.schemas import ModelClass, StructuredGeneration


SchemaModel = TypeVar("SchemaModel", bound=BaseModel)


class HumanReviewRequired(RuntimeError):
    """Visible stop when safe automatic evaluation cannot continue."""

    def __init__(
        self,
        message: str,
        *,
        local_response: dict[str, object] | None = None,
    ) -> None:
        super().__init__(message)
        self.local_response = local_response


class RoutedProvider(Protocol):
    name: str

    async def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[SchemaModel],
        model_class: ModelClass,
        *,
        request_purpose: str,
    ) -> dict[str, object]: ...


class ProviderRouter:
    """Keep fallback policy out of evaluator and LangGraph state code."""

    def __init__(
        self,
        settings: Settings,
        *,
        primary: RoutedProvider | None = None,
        fallback: RoutedProvider | None = None,
    ) -> None:
        self.settings = settings
        self.primary = primary or LMStudioProvider(settings)
        self.fallback = fallback or OpenRouterProvider(settings)

    def _fallback_allowed(self) -> bool:
        if self.fallback.name != "openrouter":
            return True
        return self.settings.llm_allow_cloud_candidate_data and bool(
            self.settings.openrouter_api_key
        )

    async def _use_fallback(
        self,
        messages: Sequence[ChatMessage],
        schema: type[SchemaModel],
        model_class: ModelClass,
        *,
        request_purpose: str,
        reason: str,
        local_response: dict[str, object] | None = None,
    ) -> dict[str, object]:
        if not self._fallback_allowed():
            raise HumanReviewRequired(
                f"{reason}; cloud fallback is blocked by configuration, so human review is required",
                local_response=local_response,
            )

        response = await self.fallback.generate_structured(
            messages,
            schema,
            model_class,
            request_purpose=request_purpose,
        )
        generation = StructuredGeneration.model_validate(response)
        generation.metadata.fallback_used = True
        generation.metadata.fallback_reason = reason
        return generation.model_dump(mode="json")

    async def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[SchemaModel],
        model_class: ModelClass,
        *,
        request_purpose: str,
    ) -> dict[str, object]:
        try:
            primary_response = await self.primary.generate_structured(
                messages,
                schema,
                model_class,
                request_purpose=request_purpose,
            )
        except Exception as error:
            detail = str(error) or type(error).__name__
            return await self._use_fallback(
                messages,
                schema,
                model_class,
                request_purpose=request_purpose,
                reason=f"{self.primary.name} failed: {detail}",
            )

        generation = StructuredGeneration.model_validate(primary_response)
        confidence = generation.output.get("confidence")
        if (
            isinstance(confidence, int | float)
            and confidence < self.settings.llm_fallback_confidence_threshold
        ):
            return await self._use_fallback(
                messages,
                schema,
                model_class,
                request_purpose=request_purpose,
                reason=(
                    f"{self.primary.name} confidence {confidence:.2f} below "
                    f"{self.settings.llm_fallback_confidence_threshold:.2f}"
                ),
                local_response=primary_response,
            )
        return primary_response
