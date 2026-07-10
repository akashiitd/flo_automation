from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from app.config import Settings
from llm.provider_router import HumanReviewRequired, ProviderRouter
from llm.schemas import ProviderMetadata, QuestionEvaluation, StructuredGeneration


def generation(provider: str, *, confidence: float = 0.8) -> dict[str, object]:
    return StructuredGeneration(
        output={
            "question_id": 1,
            "score": 3,
            "rating_label": "Average",
            "evidence": ["Candidate described an API layer"],
            "follow_up": "How would you add retries?",
            "feedback": "Basic understanding with production gaps.",
            "confidence": confidence,
        },
        metadata=ProviderMetadata(
            provider=provider,
            model="test-model",
            request_purpose="feedback_draft",
            latency_ms=10,
            input_tokens=10,
            output_tokens=10,
            estimated_cost_usd=0,
            pii_redaction_ran=provider == "openrouter",
        ),
    ).model_dump(mode="json")


class StubProvider:
    def __init__(
        self,
        name: str,
        *,
        response: dict[str, object] | None = None,
        error: Exception | None = None,
    ) -> None:
        self.name = name
        self.response = response
        self.error = error
        self.calls = 0

    async def generate_structured(
        self, *args: object, **kwargs: object
    ) -> dict[str, object]:
        self.calls += 1
        if self.error:
            raise self.error
        assert self.response is not None
        return self.response


def test_router_stops_for_review_when_cloud_fallback_is_disabled(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        project_root=tmp_path,
        environ={"LLM_ALLOW_CLOUD_CANDIDATE_DATA": "false"},
    )
    primary = StubProvider("lmstudio", error=TimeoutError("local timeout"))
    fallback = StubProvider("openrouter", response=generation("openrouter"))
    router = ProviderRouter(settings, primary=primary, fallback=fallback)

    with pytest.raises(HumanReviewRequired, match="cloud fallback is blocked"):
        asyncio.run(
            router.generate_structured(
                [{"role": "user", "content": "candidate answer"}],
                QuestionEvaluation,
                "fast",
                request_purpose="feedback_draft",
            )
        )

    assert primary.calls == 1
    assert fallback.calls == 0


def test_router_marks_an_allowed_fallback_and_preserves_the_schema(
    tmp_path: Path,
) -> None:
    settings = Settings.load(
        project_root=tmp_path,
        environ={
            "LLM_ALLOW_CLOUD_CANDIDATE_DATA": "true",
            "OPENROUTER_API_KEY": "test-key",
        },
    )
    primary = StubProvider("lmstudio", error=TimeoutError("local timeout"))
    fallback = StubProvider("openrouter", response=generation("openrouter"))
    router = ProviderRouter(settings, primary=primary, fallback=fallback)

    response = asyncio.run(
        router.generate_structured(
            [{"role": "user", "content": "candidate answer"}],
            QuestionEvaluation,
            "fast",
            request_purpose="feedback_draft",
        )
    )

    result = StructuredGeneration.model_validate(response)
    QuestionEvaluation.model_validate(result.output)
    assert result.metadata.fallback_used is True
    assert result.metadata.fallback_reason == "lmstudio failed: local timeout"
    assert primary.calls == 1
    assert fallback.calls == 1
