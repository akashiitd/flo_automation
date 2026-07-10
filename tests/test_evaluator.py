from __future__ import annotations

import asyncio
import json
from pathlib import Path

from evaluator.scoring import evaluate_answer
from llm.schemas import (
    EvaluationInput,
    ProviderMetadata,
    StructuredGeneration,
)
from llm.usage_tracker import UsageTracker


class RecordingRouter:
    def __init__(self) -> None:
        self.messages: list[dict[str, str]] = []

    async def generate_structured(
        self,
        messages: list[dict[str, str]],
        *args: object,
        **kwargs: object,
    ) -> dict[str, object]:
        self.messages = messages
        return StructuredGeneration(
            output={
                "question_id": 1,
                "score": 3,
                "rating_label": "Average",
                "evidence": ["Candidate mentioned an API layer"],
                "follow_up": "How would you handle retries?",
                "feedback": "Basic understanding with production gaps.",
                "confidence": 0.72,
            },
            metadata=ProviderMetadata(
                provider="lmstudio",
                model="google/gemma-4-12b",
                request_purpose="feedback_draft",
                latency_ms=50,
                input_tokens=100,
                output_tokens=50,
                estimated_cost_usd=0,
            ),
        ).model_dump(mode="json")


def test_evaluator_uses_the_shared_contract_and_logs_usage(tmp_path: Path) -> None:
    router = RecordingRouter()
    usage_path = tmp_path / "llm_usage.jsonl"

    generation = asyncio.run(
        evaluate_answer(
            EvaluationInput(
                question_id=1,
                question="How would you build a LangChain microservice?",
                ideal_answer="API layer, orchestration, tracing, and retries.",
                candidate_answer="I would create an API and call the model.",
            ),
            router,
            usage_tracker=UsageTracker(usage_path),
        )
    )

    assert generation.output["score"] == 3
    assert (
        "How would you build a LangChain microservice?" in router.messages[1]["content"]
    )
    usage = json.loads(usage_path.read_text(encoding="utf-8"))
    assert usage["provider"] == "lmstudio"
    assert usage["request_purpose"] == "feedback_draft"
    assert usage["pii_redaction_ran"] is False
