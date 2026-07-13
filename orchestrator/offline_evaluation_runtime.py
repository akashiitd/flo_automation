"""Offline wiring for the evaluation effect boundary and local preview artifacts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from evaluator.scoring import StructuredGenerator
from evaluator.skill_evaluation import write_skill_assessment_preview
from langgraph.checkpoint.base import BaseCheckpointSaver
from orchestrator.effect_ledger import EffectLedger
from orchestrator.evaluation_executor import EvaluationEffectExecutor
from orchestrator.effects import EffectType
from orchestrator.event_adapters import EventNormalizer
from orchestrator.intent_routing import IntentRouter
from orchestrator.langgraph_builder import build_offline_interview_graph
from orchestrator.state import DynamicInterviewState, QuestionState


@dataclass(frozen=True, slots=True)
class OfflineEvaluationUpdate:
    """One graph advance and any local-only skill-preview artifacts it produced."""

    state: dict[str, object]
    preview_paths: tuple[Path, Path] | None


class OfflineEvaluationRuntime:
    """Run offline evaluation effects outside LangGraph and persist safe previews.

    This adapter is deliberately local-only. It does not know about browser
    controls or FloCareer ratings, and it writes previews only after the graph
    has accepted an evaluation-result event.
    """

    def __init__(
        self,
        *,
        graph: Any,
        executor: EvaluationEffectExecutor,
        preview_root: Path,
    ) -> None:
        self._graph = graph
        self._executor = executor
        self._preview_root = preview_root

    async def advance(
        self, graph_input: dict[str, object], *, config: object
    ) -> OfflineEvaluationUpdate:
        """Advance one event and reduce a requested answer evaluation if present."""

        state = await self._graph.ainvoke(graph_input, config=config)
        pending_effect = state.get("pending_effect")
        if getattr(pending_effect, "effect_type", None) is EffectType.EVALUATE_ANSWER:
            execution = await self._executor.execute(pending_effect)
            if execution.events:
                state = await self._graph.ainvoke(
                    {"pending_event": execution.events[0]}, config=config
                )
        return OfflineEvaluationUpdate(
            state=state,
            preview_paths=self._write_preview_if_available(state),
        )

    def _write_preview_if_available(
        self, state: dict[str, object]
    ) -> tuple[Path, Path] | None:
        if not state.get("skill_assessments"):
            return None
        typed_state = DynamicInterviewState.model_validate(state)
        return write_skill_assessment_preview(
            self._preview_root / typed_state.session_id,
            assessments=typed_state.skill_assessments,
            evidence=typed_state.skill_evidence,
        )


def build_offline_evaluation_runtime(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    evaluator: StructuredGenerator,
    questions: list[QuestionState],
    session_id: str,
    ledger_path: Path,
    result_path: Path,
    preview_root: Path,
    intent_router: IntentRouter | None = None,
) -> OfflineEvaluationRuntime:
    """Build the Phase 9 offline path with evaluation enabled by default."""

    return OfflineEvaluationRuntime(
        graph=build_offline_interview_graph(
            checkpointer=checkpointer,
            intent_router=intent_router,
            evaluation_enabled=True,
        ),
        executor=EvaluationEffectExecutor(
            ledger=EffectLedger(ledger_path),
            result_path=result_path,
            normalizer=EventNormalizer(session_id=session_id),
            evaluator=evaluator,
            questions=questions,
        ),
        preview_root=preview_root,
    )


__all__ = [
    "OfflineEvaluationRuntime",
    "OfflineEvaluationUpdate",
    "build_offline_evaluation_runtime",
]
