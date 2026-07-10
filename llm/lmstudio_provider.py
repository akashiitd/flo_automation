"""LM Studio provider using its local OpenAI-compatible API."""

from __future__ import annotations

import httpx

from app.config import Settings
from llm.provider import OpenAICompatibleProvider


class LMStudioProvider(OpenAICompatibleProvider):
    name = "lmstudio"

    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        super().__init__(
            base_url=settings.lmstudio_base_url,
            api_key=settings.lmstudio_api_key,
            fast_model=settings.lmstudio_fast_model,
            deep_model=settings.lmstudio_deep_model,
            fast_timeout_seconds=settings.llm_fast_timeout_seconds,
            deep_timeout_seconds=settings.llm_deep_timeout_seconds,
            client=client,
        )
