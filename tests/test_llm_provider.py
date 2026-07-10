from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest

from app.config import Settings
from llm.lmstudio_provider import LMStudioProvider
from llm.openrouter_provider import OpenRouterProvider
from llm.schemas import QuestionEvaluation, StructuredGeneration


def test_lmstudio_provider_returns_the_normalized_contract(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        assert payload["model"] == "ornith-1.0-35b"
        assert payload["response_format"]["type"] == "json_schema"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": "",
                            "reasoning_content": """
                            <think>private reasoning</think>
                            ```json
                            {
                              "question_id": 1,
                              "score": 3,
                              "rating_label": "Average",
                              "evidence": ["Candidate mentioned an API layer",],
                              "follow_up": "How would you handle retries?",
                              "feedback": "Basic understanding with production gaps.",
                              "confidence": 0.72,
                            }
                            ```
                            """,
                        }
                    }
                ],
                "usage": {"prompt_tokens": 120, "completion_tokens": 80},
            },
        )

    settings = Settings.load(
        project_root=tmp_path,
        environ={"LMSTUDIO_BASE_URL": "http://lmstudio.test/v1"},
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = LMStudioProvider(settings, client=client)

    response = asyncio.run(
        provider.generate_structured(
            [{"role": "user", "content": "Evaluate this answer"}],
            QuestionEvaluation,
            "fast",
            request_purpose="feedback_draft",
        )
    )
    asyncio.run(client.aclose())

    generation = StructuredGeneration.model_validate(response)
    assert generation.output["score"] == 3
    assert generation.metadata.provider == "lmstudio"
    assert generation.metadata.input_tokens == 120
    assert generation.metadata.output_tokens == 80
    assert generation.metadata.estimated_cost_usd == 0
    assert generation.metadata.fallback_used is False


def test_openrouter_provider_redacts_pii_before_the_request(tmp_path: Path) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        payload = json.loads(request.content)
        rendered_messages = json.dumps(payload["messages"])
        assert "alice@example.com" not in rendered_messages
        assert "98765 43210" not in rendered_messages
        assert "[REDACTED_EMAIL]" in rendered_messages
        assert request.headers["Authorization"] == "Bearer test-key"
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "question_id": 1,
                                    "score": 3,
                                    "rating_label": "Average",
                                    "evidence": ["Candidate described an API layer"],
                                    "follow_up": "How would you add retries?",
                                    "feedback": "Basic understanding with gaps.",
                                    "confidence": 0.72,
                                }
                            )
                        }
                    }
                ],
                "usage": {
                    "prompt_tokens": 100,
                    "completion_tokens": 50,
                    "cost": 0.001,
                },
            },
        )

    settings = Settings.load(
        project_root=tmp_path,
        environ={
            "LLM_ALLOW_CLOUD_CANDIDATE_DATA": "true",
            "OPENROUTER_API_KEY": "test-key",
            "OPENROUTER_BASE_URL": "https://openrouter.test/api/v1",
        },
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    provider = OpenRouterProvider(settings, client=client)

    response = asyncio.run(
        provider.generate_structured(
            [
                {
                    "role": "user",
                    "content": "Email alice@example.com, phone +91 98765 43210",
                }
            ],
            QuestionEvaluation,
            "fast",
            request_purpose="feedback_draft",
        )
    )
    asyncio.run(client.aclose())

    generation = StructuredGeneration.model_validate(response)
    assert generation.metadata.provider == "openrouter"
    assert generation.metadata.pii_redaction_ran is True
    assert generation.metadata.estimated_cost_usd == 0.001


def test_provider_enforces_a_wall_clock_generation_deadline(tmp_path: Path) -> None:
    async def slow_handler(request: httpx.Request) -> httpx.Response:
        await asyncio.sleep(0.03)
        return httpx.Response(200, json={})

    settings = Settings.load(
        project_root=tmp_path,
        environ={"LLM_FAST_TIMEOUT_SECONDS": "0.005"},
    )
    client = httpx.AsyncClient(transport=httpx.MockTransport(slow_handler))
    provider = LMStudioProvider(settings, client=client)

    with pytest.raises(TimeoutError):
        asyncio.run(
            provider.generate_structured(
                [{"role": "user", "content": "Evaluate this answer"}],
                QuestionEvaluation,
                "fast",
                request_purpose="feedback_draft",
            )
        )
    asyncio.run(client.aclose())
