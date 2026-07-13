"""Offline contract tests for the Phase 5 dynamic LangGraph foundation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.questions import InterviewQuestion
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from orchestrator.effects import EffectType
from orchestrator.events import InterviewEvent
from orchestrator.graph import InterviewController
from orchestrator.langgraph_builder import build_offline_interview_graph
from orchestrator.state import (
    CoverageState,
    CoverageStatus,
    DynamicInterviewPhase,
    DynamicInterviewState,
    QuestionContentType,
    QuestionMappingSource,
    QuestionPlanItem,
    QuestionState,
    SkillParameter,
)

_FIXTURE_PATH = Path(__file__).parent / "fixtures" / "offline_dynamic_events.json"
_TRANSCRIPT_FIXTURE_PATH = (
    Path(__file__).parent / "fixtures" / "offline_dynamic_transcript.json"
)


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def _plan_item(
    question_id: int,
    skill_id: str,
    *,
    estimated_minutes: float = 3.0,
    priority: int = 100,
) -> QuestionPlanItem:
    return QuestionPlanItem(
        question_id=question_id,
        content_type=QuestionContentType.INTERVIEW_QUESTION,
        target_skill_ids=[skill_id],
        mandatory_skill_coverage=[skill_id],
        estimated_minutes=estimated_minutes,
        priority=priority,
        selected=True,
        mapping_source=QuestionMappingSource.DETERMINISTIC,
        mapping_confidence=1.0,
        mapping_evidence=["offline fixture mapping"],
    )


def _state(*, remaining_seconds: float = 600.0) -> DynamicInterviewState:
    return DynamicInterviewState(
        thread_id="offline-fixture-thread",
        session_id="offline-fixture-session",
        candidate_identifier="offline-fixture-candidate",
        questions=[
            QuestionState(id=1, question_text="Explain API retry handling."),
            QuestionState(id=2, question_text="Explain Python exception handling."),
        ],
        skill_parameters=[
            SkillParameter(
                id="api",
                name="API",
                requirement="Mandatory",
                level="Professional",
                rating_scale=5,
            ),
            SkillParameter(
                id="python",
                name="Python",
                requirement="Mandatory",
                level="Professional",
                rating_scale=5,
            ),
        ],
        question_plan=[_plan_item(1, "api"), _plan_item(2, "python")],
        remaining_seconds=remaining_seconds,
    )


def _load_events() -> list[InterviewEvent]:
    raw = json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))
    return [
        InterviewEvent.model_validate_json(json.dumps(event)) for event in raw["events"]
    ]


def _load_transcript_segments() -> dict[int, str]:
    raw = json.loads(_TRANSCRIPT_FIXTURE_PATH.read_text(encoding="utf-8"))
    return {
        int(turn["question_id"]): " ".join(turn["segments"]) for turn in raw["turns"]
    }


def _legacy_question_trajectory() -> list[str]:
    controller = InterviewController(
        candidate_name="Offline Fixture",
        questions=(
            InterviewQuestion(
                id=1, question_text="Explain API retry handling.", ideal_answer=""
            ),
            InterviewQuestion(
                id=2,
                question_text="Explain Python exception handling.",
                ideal_answer="",
            ),
        ),
    )
    controller.start()
    controller.approve_candidate_prompt()
    trajectory = [controller.approve_candidate_prompt()]
    transcript_segments = _load_transcript_segments()
    controller.record_candidate_segment(transcript_segments[1])
    controller.complete_answer()
    controller.record_evaluation(follow_up=None)
    assert controller.prepare_next_question() is not None
    trajectory.append(controller.approve_candidate_prompt())
    controller.record_candidate_segment(transcript_segments[2])
    return trajectory


def test_offline_parent_graph_replays_saved_events_and_matches_legacy_trajectory() -> (
    None
):
    async def run() -> tuple[list[str], dict[str, object], list[dict[str, object]]]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        config = _config("offline-fixture-thread")
        state = _state()
        asked: list[str] = []
        checkpoint_values: list[dict[str, object]] = []
        result: dict[str, object] = state.model_dump(mode="python")
        for position, event in enumerate(_load_events()):
            payload = (
                {**state.model_dump(mode="python"), "pending_event": event}
                if position == 0
                else {"pending_event": event}
            )
            result = await graph.ainvoke(payload, config=config)
            snapshot = await graph.aget_state(config)
            checkpoint_values.append(snapshot.values)
            request = result.get("pending_effect")
            if request is not None:
                asked.append(request.payload["text"])
        return asked, result, checkpoint_values

    asked, result, checkpoint_values = asyncio.run(run())

    assert asked == _legacy_question_trajectory()
    assert result["phase"] == DynamicInterviewPhase.DONE
    assert result["completed_question_ids"] == [1, 2]
    assert result["pending_effect"] is None
    assert [values["phase"] for values in checkpoint_values] == [
        DynamicInterviewPhase.RUN_TURN,
        DynamicInterviewPhase.RUN_TURN,
        DynamicInterviewPhase.DONE,
    ]
    assert [len(values["recent_events"]) for values in checkpoint_values] == [1, 2, 3]


def test_timer_selection_skips_sufficient_and_over_budget_questions_with_fake_effects() -> (
    None
):
    async def run() -> dict[str, object]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state(remaining_seconds=240.0).model_copy(
            update={
                "questions": [
                    QuestionState(id=1, question_text="Explain API retry handling."),
                    QuestionState(
                        id=2, question_text="Explain Python exception handling."
                    ),
                    QuestionState(id=3, question_text="Implement a Python parser."),
                ],
                "question_plan": [
                    _plan_item(1, "api", priority=300),
                    _plan_item(2, "python", priority=100),
                    _plan_item(3, "python", estimated_minutes=6.0, priority=500),
                ],
                "coverage": {
                    "api": CoverageState(
                        status=CoverageStatus.SUFFICIENT,
                        confidence=1.0,
                    )
                },
            }
        )
        event = InterviewEvent.model_validate_json(
            json.dumps(
                {
                    "event_id": "timer-warning",
                    "event_type": "TIMER_WARNING",
                    "occurred_at": "2026-07-13T08:00:00+00:00",
                    "source": "timer",
                    "session_id": state.session_id,
                    "payload": {},
                }
            )
        )
        return await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": event},
            config=_config(state.thread_id),
        )

    result = asyncio.run(run())

    assert result["phase"] == DynamicInterviewPhase.RUN_TURN
    assert result["current_question_id"] == 2
    effect = result["pending_effect"]
    assert effect.effect_type is EffectType.SPEAK_TEXT
    assert effect.payload["offline_only"] is True
    assert effect.payload["text"] == ("Explain Python exception handling.")
    skipped = {item.question_id: item.reason for item in result["skipped_questions"]}
    assert skipped[1] == "runtime coverage already sufficient"
    assert skipped[3] == "runtime time budget exhausted"


def test_zero_remaining_time_routes_to_done_without_an_effect_request() -> None:
    async def run() -> dict[str, object]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state(remaining_seconds=0.0)
        event = _load_events()[0]
        return await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": event},
            config=_config(state.thread_id),
        )

    result = asyncio.run(run())

    assert result["phase"] is DynamicInterviewPhase.DONE
    assert result["pending_effect"] is None
    assert {item.question_id for item in result["skipped_questions"]} == {1, 2}


def test_time_limit_event_routes_to_done_with_auditable_skip_reasons() -> None:
    async def run() -> dict[str, object]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        event = InterviewEvent.model_validate_json(
            json.dumps(
                {
                    "event_id": "time-limit-reached",
                    "event_type": "TIME_LIMIT_REACHED",
                    "occurred_at": "2026-07-13T08:00:00+00:00",
                    "source": "timer",
                    "session_id": state.session_id,
                    "payload": {},
                }
            )
        )
        return await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": event},
            config=_config(state.thread_id),
        )

    result = asyncio.run(run())

    assert result["phase"] is DynamicInterviewPhase.DONE
    assert result["pending_effect"] is None
    assert {item.question_id: item.reason for item in result["skipped_questions"]} == {
        1: "runtime time limit reached",
        2: "runtime time limit reached",
    }
