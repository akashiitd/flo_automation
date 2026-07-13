"""Provider protocol and OpenAI-compatible transport implementation."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Mapping, Sequence
from typing import Protocol, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from llm.json_repair import JsonRepairError, parse_json_object
from llm.schemas import ModelClass, ProviderMetadata, StructuredGeneration


SchemaModel = TypeVar("SchemaModel", bound=BaseModel)
ChatMessage = Mapping[str, str]


class ProviderError(RuntimeError):
    """Base error for provider failures visible to the router."""


class StructuredOutputError(ProviderError):
    """Raised after parsing and strict retry both fail."""


class LLMProvider(Protocol):
    name: str

    async def health(self) -> dict[str, object]: ...

    async def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[SchemaModel],
        model_class: ModelClass,
        *,
        request_purpose: str,
    ) -> dict[str, object]: ...

    async def stream_text(
        self,
        messages: Sequence[ChatMessage],
        model_class: ModelClass,
    ) -> AsyncIterator[str]: ...


class OpenAICompatibleProvider:
    """Shared transport for LM Studio and OpenRouter chat-completion APIs."""

    name = "openai-compatible"

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        fast_model: str,
        deep_model: str,
        fast_timeout_seconds: float,
        deep_timeout_seconds: float,
        client: httpx.AsyncClient | None = None,
        extra_headers: Mapping[str, str] | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.fast_model = fast_model
        self.deep_model = deep_model
        self.fast_timeout_seconds = fast_timeout_seconds
        self.deep_timeout_seconds = deep_timeout_seconds
        self._extra_headers = dict(extra_headers or {})
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json", **self._extra_headers}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _model(self, model_class: ModelClass) -> str:
        return self.fast_model if model_class == "fast" else self.deep_model

    def _timeout(self, model_class: ModelClass) -> float:
        if model_class == "fast":
            return self.fast_timeout_seconds
        return self.deep_timeout_seconds

    def _prepare_messages(
        self, messages: Sequence[ChatMessage]
    ) -> tuple[list[dict[str, str]], bool]:
        return [dict(message) for message in messages], False

    def _estimated_cost(self, usage: Mapping[str, object]) -> float:
        raw_cost = usage.get("cost", 0.0)
        if isinstance(raw_cost, int | float) and raw_cost >= 0:
            return float(raw_cost)
        return 0.0

    async def health(self) -> dict[str, object]:
        started = time.perf_counter()
        try:
            response = await self._client.get(
                f"{self.base_url}/models",
                headers=self._headers(),
                timeout=self.fast_timeout_seconds,
            )
            response.raise_for_status()
            payload = response.json()
            raw_models = payload.get("data", []) if isinstance(payload, dict) else []
            models = [
                model["id"]
                for model in raw_models
                if isinstance(model, dict) and isinstance(model.get("id"), str)
            ]
            return {
                "provider": self.name,
                "available": True,
                "models": models,
                "latency_ms": round((time.perf_counter() - started) * 1000),
            }
        except Exception as error:
            return {
                "provider": self.name,
                "available": False,
                "error": str(error),
                "latency_ms": round((time.perf_counter() - started) * 1000),
            }

    @staticmethod
    def _message_content(payload: Mapping[str, object]) -> str:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise StructuredOutputError(
                "provider response did not contain choices[0].message.content"
            )
        first_choice = choices[0]
        if not isinstance(first_choice, Mapping):
            raise StructuredOutputError("provider returned an invalid first choice")
        message = first_choice.get("message")
        if not isinstance(message, Mapping):
            raise StructuredOutputError("provider returned an invalid message")
        content = message.get("content")

        if isinstance(content, str) and content.strip():
            return content
        if isinstance(content, list):
            text_parts: list[str] = []
            for part in content:
                if isinstance(part, Mapping):
                    part_text = part.get("text")
                    if isinstance(part_text, str):
                        text_parts.append(part_text)
            if text_parts:
                return "".join(text_parts)
        reasoning_content = message.get("reasoning_content")
        if isinstance(reasoning_content, str) and reasoning_content.strip():
            return reasoning_content
        raise StructuredOutputError("provider returned non-text message content")

    async def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[SchemaModel],
        model_class: ModelClass,
        *,
        request_purpose: str,
    ) -> dict[str, object]:
        async with asyncio.timeout(self._timeout(model_class)):
            return await self._generate_structured_with_retries(
                messages,
                schema,
                model_class,
                request_purpose=request_purpose,
            )

    async def _generate_structured_with_retries(
        self,
        messages: Sequence[ChatMessage],
        schema: type[SchemaModel],
        model_class: ModelClass,
        *,
        request_purpose: str,
    ) -> dict[str, object]:
        prepared_messages, pii_redaction_ran = self._prepare_messages(messages)
        model = self._model(model_class)
        started = time.perf_counter()
        last_error: Exception | None = None

        for attempt in range(2):
            attempt_messages = list(prepared_messages)
            if attempt:
                attempt_messages.append(
                    {
                        "role": "system",
                        "content": (
                            "Your prior output was invalid. Return exactly one JSON object "
                            "matching the supplied schema, with no reasoning or markdown."
                        ),
                    }
                )
            response = await self._client.post(
                f"{self.base_url}/chat/completions",
                headers=self._headers(),
                timeout=self._timeout(model_class),
                json={
                    "model": model,
                    "messages": attempt_messages,
                    "temperature": 0.1,
                    "max_tokens": 900,
                    "response_format": {
                        "type": "json_schema",
                        "json_schema": {
                            "name": schema.__name__,
                            "strict": True,
                            "schema": schema.model_json_schema(),
                        },
                    },
                },
            )
            response.raise_for_status()
            payload = response.json()
            if not isinstance(payload, dict):
                raise StructuredOutputError("provider response must be a JSON object")

            try:
                parsed = parse_json_object(self._message_content(payload))
                # Strict schema contracts should validate the model's JSON as JSON.
                # This preserves enum and scalar types while rejecting non-JSON forms.
                validated = schema.model_validate_json(json.dumps(parsed))
            except (JsonRepairError, StructuredOutputError, ValidationError) as error:
                last_error = error
                continue

            raw_usage = payload.get("usage", {})
            usage = raw_usage if isinstance(raw_usage, dict) else {}
            input_tokens = usage.get("prompt_tokens", 0)
            output_tokens = usage.get("completion_tokens", 0)
            metadata = ProviderMetadata(
                provider=self.name,
                model=model,
                request_purpose=request_purpose,
                latency_ms=round((time.perf_counter() - started) * 1000),
                input_tokens=input_tokens if isinstance(input_tokens, int) else 0,
                output_tokens=output_tokens if isinstance(output_tokens, int) else 0,
                estimated_cost_usd=self._estimated_cost(usage),
                pii_redaction_ran=pii_redaction_ran,
            )
            return StructuredGeneration(
                output=validated.model_dump(mode="json"),
                metadata=metadata,
            ).model_dump(mode="json")

        raise StructuredOutputError(
            f"provider returned invalid structured output after strict retry: {last_error}"
        ) from last_error

    async def stream_text(
        self,
        messages: Sequence[ChatMessage],
        model_class: ModelClass,
    ) -> AsyncIterator[str]:
        prepared_messages, _ = self._prepare_messages(messages)
        async with self._client.stream(
            "POST",
            f"{self.base_url}/chat/completions",
            headers=self._headers(),
            timeout=self._timeout(model_class),
            json={
                "model": self._model(model_class),
                "messages": prepared_messages,
                "stream": True,
            },
        ) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    return
                try:
                    event = json.loads(data)
                    content = event["choices"][0]["delta"].get("content")
                except (json.JSONDecodeError, KeyError, IndexError, TypeError):
                    continue
                if isinstance(content, str) and content:
                    yield content

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()
