"""Durable answer-evaluation effect tests."""

from __future__ import annotations

import asyncio
from pathlib import Path

from llm.schemas import ProviderMetadata, StructuredGeneration
from orchestrator.effect_ledger import EffectLedger
from orchestrator.effects import EffectRequest, EffectStatus, EffectType
from orchestrator.evaluation_executor import EvaluationEffectExecutor
from orchestrator.event_adapters import EventNormalizer
from orchestrator.events import EventType
from orchestrator.state import QuestionState


class FakeEvaluator:
    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self, *args: object, **kwargs: object
    ) -> dict[str, object]:
        self.calls += 1
        return StructuredGeneration(
            output={
                "question_id": 1,
                "score": 4,
                "rating_label": "Good",
                "evidence": ["Candidate described bounded retries."],
                "follow_up": "What would you change?",
                "feedback": "Solid operational reasoning.",
                "confidence": 0.9,
            },
            metadata=ProviderMetadata(
                provider="test",
                model="test-model",
                request_purpose="feedback_draft",
                latency_ms=1,
                input_tokens=1,
                output_tokens=1,
                estimated_cost_usd=0,
            ),
        ).model_dump(mode="json")


def _request(*, session_id: str = "session-1") -> EffectRequest:
    return EffectRequest(
        effect_id="evaluate-1",
        effect_type=EffectType.EVALUATE_ANSWER,
        idempotency_key=f"{session_id}:evaluate:1:1",
        session_id=session_id,
        question_id=1,
        payload={"offline_only": True, "candidate_answer": "I use bounded retries."},
    )


def _executor(
    tmp_path: Path, evaluator: FakeEvaluator, *, session_id: str = "session-1"
) -> EvaluationEffectExecutor:
    return EvaluationEffectExecutor(
        ledger=EffectLedger(tmp_path / "effects.sqlite"),
        result_path=tmp_path / "evaluation-results.sqlite",
        normalizer=EventNormalizer(session_id=session_id),
        evaluator=evaluator,
        questions=[
            QuestionState(
                id=1,
                question_text="How do you handle retries?",
                ideal_answer="Use bounded retries with idempotency.",
            )
        ],
    )


def test_evaluation_executor_caches_output_and_reemits_the_same_result(
    tmp_path: Path,
) -> None:
    async def run() -> tuple[object, object, FakeEvaluator]:
        evaluator = FakeEvaluator()
        executor = _executor(tmp_path, evaluator)
        first = await executor.execute(_request())
        second = await executor.execute(_request())
        return first, second, evaluator

    first, second, evaluator = asyncio.run(run())

    assert first.result.status is EffectStatus.COMPLETED
    assert second.result.status is EffectStatus.COMPLETED
    assert first.events[0].event_type is EventType.EVALUATION_COMPLETED
    assert second.events[0].event_id == first.events[0].event_id
    assert first.events[0].payload["output"]["follow_up"] == (
        "Could you expand on your approach and reasoning?"
    )
    assert evaluator.calls == 1


def test_evaluation_executor_reports_provider_failure_as_a_graph_event(
    tmp_path: Path,
) -> None:
    class FailingEvaluator(FakeEvaluator):
        async def generate_structured(
            self, *args: object, **kwargs: object
        ) -> dict[str, object]:
            raise TimeoutError("local evaluator unavailable")

    async def run() -> object:
        return await _executor(tmp_path, FailingEvaluator()).execute(_request())

    execution = asyncio.run(run())

    assert execution.result.status is EffectStatus.FAILED
    assert execution.events[0].event_type is EventType.EVALUATION_FAILED


def test_evaluation_executor_recovers_cached_output_after_a_crash_window(
    tmp_path: Path,
) -> None:
    async def run() -> tuple[object, FakeEvaluator]:
        evaluator = FakeEvaluator()
        executor = _executor(tmp_path, evaluator)
        request = _request()
        executor._ledger.prepare(request)
        executor._ledger.claim_start(request)
        executor._store_output(
            request,
            {
                "question_id": 1,
                "score": 4,
                "rating_label": "Good",
                "evidence": ["Candidate described bounded retries."],
                "follow_up": "Could you expand on your approach and reasoning?",
                "feedback": "Solid operational reasoning.",
                "confidence": 0.9,
            },
        )
        return await executor.execute(request), evaluator

    execution, evaluator = asyncio.run(run())

    assert execution.result.status is EffectStatus.COMPLETED
    assert execution.events[0].event_type is EventType.EVALUATION_COMPLETED
    assert evaluator.calls == 0


def test_evaluation_executor_completes_a_prepared_cached_result(tmp_path: Path) -> None:
    async def run() -> tuple[object, FakeEvaluator]:
        evaluator = FakeEvaluator()
        executor = _executor(tmp_path, evaluator)
        request = _request()
        executor._ledger.prepare(request)
        executor._store_output(
            request,
            {
                "question_id": 1,
                "score": 4,
                "rating_label": "Good",
                "evidence": ["Candidate described bounded retries."],
                "follow_up": "Could you expand on your approach and reasoning?",
                "feedback": "Solid operational reasoning.",
                "confidence": 0.9,
            },
        )
        return await executor.execute(request), evaluator

    execution, evaluator = asyncio.run(run())

    assert execution.result.status is EffectStatus.COMPLETED
    assert execution.events[0].event_type is EventType.EVALUATION_COMPLETED
    assert evaluator.calls == 0


def test_evaluation_cache_cannot_cross_session_boundaries(tmp_path: Path) -> None:
    async def run() -> FakeEvaluator:
        evaluator = FakeEvaluator()
        await _executor(tmp_path, evaluator).execute(_request())
        await _executor(tmp_path, evaluator, session_id="session-2").execute(
            _request(session_id="session-2")
        )
        return evaluator

    evaluator = asyncio.run(run())

    assert evaluator.calls == 2
