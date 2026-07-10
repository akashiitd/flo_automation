"""Guarded OpenRouter provider with mandatory candidate-data redaction."""

from __future__ import annotations

from collections.abc import Sequence

import httpx

from app.config import Settings
from llm.privacy_redactor import redact_messages
from llm.provider import ChatMessage, OpenAICompatibleProvider, ProviderError


class CloudDataNotAllowedError(ProviderError):
    """Raised before a request when candidate data may not leave the machine."""


class OpenRouterProvider(OpenAICompatibleProvider):
    name = "openrouter"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.cloud_candidate_data_allowed = settings.llm_allow_cloud_candidate_data
        super().__init__(
            base_url=settings.openrouter_base_url,
            api_key=settings.openrouter_api_key,
            fast_model=settings.openrouter_fast_model,
            deep_model=settings.openrouter_deep_model,
            fast_timeout_seconds=settings.llm_fast_timeout_seconds,
            deep_timeout_seconds=settings.llm_deep_timeout_seconds,
            client=client,
            extra_headers={
                "HTTP-Referer": settings.openrouter_site_url,
                "X-Title": settings.openrouter_app_name,
            },
        )

    def _prepare_messages(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[list[dict[str, str]], bool]:
        if not self.cloud_candidate_data_allowed:
            raise CloudDataNotAllowedError(
                "OpenRouter request blocked: LLM_ALLOW_CLOUD_CANDIDATE_DATA=false"
            )
        if not self.api_key:
            raise CloudDataNotAllowedError(
                "OpenRouter request blocked: OPENROUTER_API_KEY is not configured"
            )
        redacted = redact_messages(messages)
        return redacted.messages, True
