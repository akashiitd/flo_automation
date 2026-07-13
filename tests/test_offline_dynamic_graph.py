"""Offline contract tests for the Phase 5 dynamic LangGraph foundation."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

from app.questions import InterviewQuestion
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver

from orchestrator.effects import EffectStatus, EffectType
from orchestrator.event_adapters import EventNormalizer
from orchestrator.events import EventSource, EventType, InterviewEvent
from orchestrator.intents import CandidateIntent
from orchestrator.graph import InterviewController
from orchestrator.intent_routing import IntentRouter
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


def _transcript_event(
    state: dict[str, object],
    *,
    event_id: str,
    transcript: str,
    barge_in: bool = False,
) -> InterviewEvent:
    return InterviewEvent.model_validate_json(
        json.dumps(
            {
                "event_id": event_id,
                "event_type": EventType.TRANSCRIPT_FINAL,
                "occurred_at": "2026-07-13T08:00:00+00:00",
                "source": EventSource.CANDIDATE_ASR,
                "session_id": state["session_id"],
                "question_id": state["current_question_id"],
                "payload": {
                    "text": transcript,
                    "segment_id": event_id,
                    **({"barge_in": True} if barge_in else {}),
                },
            }
        )
    )


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


def test_classified_repeat_is_durable_and_prepares_a_safe_replay_effect() -> None:
    async def run() -> dict[str, object]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        config = _config(state.thread_id)
        started = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        return await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    started,
                    event_id="intent-repeat",
                    transcript="Could you repeat the question?",
                )
            },
            config=config,
        )

    result = asyncio.run(run())

    assert result["phase"] is DynamicInterviewPhase.RUN_TURN
    assert result["repeat_counts"] == {"1": 1}
    assert result["intent_history"][-1].intent is CandidateIntent.REPEAT_REQUEST
    assert result["pending_effect"].payload["text"] == "Explain API retry handling."
    assert result["pending_effect"].payload["offline_only"] is True


def test_matching_tts_completion_resumes_capture_and_duplicate_callback_is_ignored() -> (
    None
):
    async def run() -> tuple[dict[str, object], dict[str, object]]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        config = _config(state.thread_id)
        asked = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        completion = EventNormalizer(session_id=state.session_id).tts_result(
            effect_id=asked["pending_effect"].effect_id,
            outcome="completed",
            question_id=1,
            result_summary="played 2 PCM chunks",
        )
        completed = await graph.ainvoke({"pending_event": completion}, config=config)
        duplicate = await graph.ainvoke({"pending_event": completion}, config=config)
        return completed, duplicate

    completed, duplicate = asyncio.run(run())

    assert completed["capture_enabled"] is True
    assert completed["pending_effect"] is None
    assert completed["last_effect_result"].status is EffectStatus.COMPLETED
    assert duplicate["capture_enabled"] is True
    assert len(duplicate["recent_events"]) == len(completed["recent_events"])


def test_capture_ignores_prompt_bleed_until_tts_completion_or_barge_in() -> None:
    async def run() -> tuple[dict[str, object], dict[str, object]]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state().model_copy(update={"mode": "supervised_live"})
        config = _config(state.thread_id)
        asked = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        ignored = await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    asked,
                    event_id="prompt-bleed",
                    transcript="Question words should not become answer evidence.",
                )
            },
            config=config,
        )
        completed = await graph.ainvoke(
            {
                "pending_event": EventNormalizer(
                    session_id=state.session_id
                ).tts_result(
                    effect_id=asked["pending_effect"].effect_id,
                    outcome="completed",
                    question_id=1,
                )
            },
            config=config,
        )
        accepted = await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    completed,
                    event_id="candidate-answer",
                    transcript="I would use an idempotency key.",
                )
            },
            config=config,
        )
        return ignored, accepted

    ignored, accepted = asyncio.run(run())

    assert ignored["current_turn"].answer_segments == []
    assert accepted["current_turn"].answer_segments == [
        "I would use an idempotency key."
    ]


def test_barge_in_cancellation_keeps_candidate_answer_and_uncertain_playback_pauses() -> (
    None
):
    async def run() -> tuple[dict[str, object], dict[str, object]]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        config = _config(state.thread_id)
        asked = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    asked,
                    event_id="barge-answer",
                    transcript="I would use an idempotency key.",
                    barge_in=True,
                )
            },
            config=config,
        )
        normalizer = EventNormalizer(session_id=state.session_id)
        cancelled = await graph.ainvoke(
            {
                "pending_event": normalizer.tts_result(
                    effect_id=asked["pending_effect"].effect_id,
                    outcome="cancelled",
                    question_id=1,
                )
            },
            config=config,
        )
        recovery_graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        recovery_config = _config("uncertain-playback-thread")
        recovery_state = _state().model_copy(
            update={"thread_id": "uncertain-playback-thread"}
        )
        recovery_asked = await recovery_graph.ainvoke(
            {
                **recovery_state.model_dump(mode="python"),
                "pending_event": _load_events()[0],
            },
            config=recovery_config,
        )
        uncertain = await recovery_graph.ainvoke(
            {
                "pending_event": normalizer.tts_result(
                    effect_id=recovery_asked["pending_effect"].effect_id,
                    outcome="failed",
                    question_id=1,
                    result_status="UNCERTAIN",
                )
            },
            config=recovery_config,
        )
        return cancelled, uncertain

    cancelled, uncertain = asyncio.run(run())

    assert cancelled["current_turn"].answer_segments == [
        "I would use an idempotency key."
    ]
    assert cancelled["capture_enabled"] is True
    assert uncertain["phase"] is DynamicInterviewPhase.RECOVERY_REVIEW
    assert uncertain["pending_interrupt"].kind == "uncertain_playback"


def test_audio_route_completion_replays_the_same_question_more_slowly() -> None:
    async def run() -> dict[str, object]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        config = _config(state.thread_id)
        asked = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        audio_report = await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    asked,
                    event_id="audio-report",
                    transcript="I cannot hear you.",
                )
            },
            config=config,
        )
        event = EventNormalizer(session_id=state.session_id).audio_route_result(
            effect_id=audio_report["pending_effect"].effect_id,
            outcome="completed",
            question_id=1,
            result_summary="configured output device is available",
        )
        return await graph.ainvoke({"pending_event": event}, config=config)

    result = asyncio.run(run())

    assert result["pending_effect"].effect_type is EffectType.SPEAK_TEXT
    assert result["pending_effect"].payload["text"] == "Explain API retry handling."
    assert result["pending_effect"].payload["playback_rate"] == 0.85
    assert result["capture_enabled"] is False


def test_transcript_ingress_uses_the_injected_local_classifier_not_an_llm_event() -> (
    None
):
    class TestLocalClassifier:
        def __init__(self) -> None:
            self.calls = 0

        async def generate_structured(
            self, *args: object, **kwargs: object
        ) -> dict[str, object]:
            self.calls += 1
            return {
                "output": {
                    "intent": "ANSWER_CONTENT",
                    "confidence": 0.95,
                    "evidence_span": "I would use an idempotency key.",
                    "answer_text_to_keep": "I would use an idempotency key.",
                    "candidate_requested_action": None,
                    "safe_route": "CONTINUE_LISTENING",
                }
            }

    async def run() -> tuple[dict[str, object], TestLocalClassifier]:
        classifier = TestLocalClassifier()
        graph = build_offline_interview_graph(
            checkpointer=InMemorySaver(),
            intent_router=IntentRouter.for_test(classifier),
        )
        state = _state()
        config = _config(state.thread_id)
        started = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        result = await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    started,
                    event_id="local-classification",
                    transcript="I would use an idempotency key.",
                )
            },
            config=config,
        )
        return result, classifier

    result, classifier = asyncio.run(run())

    assert classifier.calls == 1
    assert result["intent_history"][-1].intent is CandidateIntent.ANSWER_CONTENT
    assert result["current_turn"].answer_segments == ["I would use an idempotency key."]


def test_default_graph_keeps_fragmented_answer_as_content_without_escalating() -> None:
    async def run() -> dict[str, object]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        config = _config(state.thread_id)
        result = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        for index, fragment in enumerate(("First", "second", "third")):
            result = await graph.ainvoke(
                {
                    "pending_event": _transcript_event(
                        result,
                        event_id=f"answer-fragment-{index}",
                        transcript=fragment,
                    )
                },
                config=config,
            )
        return result

    result = asyncio.run(run())

    assert result["phase"] is DynamicInterviewPhase.RUN_TURN
    assert result["ambiguity_counts"] == {}
    assert result["current_turn"].answer_segments == ["First", "second", "third"]


def test_defer_selects_the_next_question_and_withdrawal_pauses_for_an_operator() -> (
    None
):
    async def run() -> dict[str, object]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        config = _config(state.thread_id)
        started = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        deferred = await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    started,
                    event_id="intent-defer",
                    transcript="Can we skip this and return to it later?",
                )
            },
            config=config,
        )
        return await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    deferred,
                    event_id="intent-withdraw",
                    transcript="I would like to withdraw from this interview.",
                )
            },
            config=config,
        )

    result = asyncio.run(run())

    assert result["phase"] is DynamicInterviewPhase.PAUSED
    assert result["current_question_id"] == 2
    assert result["deferred_question_ids"] == [1]
    assert result["skipped_questions"] == []
    assert result["pending_interrupt"].kind == "candidate_withdrawal"


def test_audio_and_repeated_ambiguity_escalate_without_external_execution() -> None:
    class UnknownLocalClassifier:
        async def generate_structured(
            self, *args: object, **kwargs: object
        ) -> dict[str, object]:
            return {
                "output": {
                    "intent": "UNKNOWN",
                    "confidence": 0.0,
                    "evidence_span": "I have a concern.",
                    "answer_text_to_keep": "",
                    "candidate_requested_action": None,
                    "safe_route": "CONTINUE_LISTENING",
                }
            }

    async def run() -> dict[str, object]:
        graph = build_offline_interview_graph(
            checkpointer=InMemorySaver(),
            intent_router=IntentRouter.for_test(UnknownLocalClassifier()),
        )
        state = _state()
        config = _config(state.thread_id)
        result = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        for index in range(3):
            result = await graph.ainvoke(
                {
                    "pending_event": _transcript_event(
                        result,
                        event_id=f"unknown-{index}",
                        transcript="I have a concern.",
                    )
                },
                config=config,
            )
        return result

    result = asyncio.run(run())

    assert result["phase"] is DynamicInterviewPhase.NEEDS_OPERATOR
    assert result["ambiguity_counts"] == {"1": 2}
    assert result["pending_effect"] is None
    assert result["pending_interrupt"].kind == "ambiguous_candidate_intent"


def test_second_audio_recovery_request_escalates_to_operator() -> None:
    async def run() -> tuple[dict[str, object], dict[str, object]]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        config = _config(state.thread_id)
        result = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        first = await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    result,
                    event_id="audio-one",
                    transcript="I cannot hear you.",
                )
            },
            config=config,
        )
        second = await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    first,
                    event_id="audio-two",
                    transcript="I cannot hear you.",
                )
            },
            config=config,
        )
        return first, second

    first, second = asyncio.run(run())

    assert first["pending_effect"].effect_type is EffectType.CHECK_AUDIO_ROUTE
    assert first["pending_effect"].payload["offline_only"] is True
    assert second["phase"] is DynamicInterviewPhase.NEEDS_OPERATOR
    assert second["audio_problem_count"] == 2
    assert second["pending_interrupt"].kind == "repeated_audio_problem"


def test_deferred_question_returns_after_another_question_completes() -> None:
    async def run() -> dict[str, object]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        config = _config(state.thread_id)
        started = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    started,
                    event_id="defer-then-return",
                    transcript="Can we skip this and return to it later?",
                )
            },
            config=config,
        )
        return await graph.ainvoke({"pending_event": _load_events()[2]}, config=config)

    result = asyncio.run(run())

    assert result["completed_question_ids"] == [2]
    assert result["deferred_question_ids"] == [1]
    assert result["current_question_id"] == 1
    assert result["skipped_questions"] == []


def test_thinking_and_correction_stay_in_turn_without_speaking_or_losing_content() -> (
    None
):
    async def run() -> tuple[dict[str, object], dict[str, object]]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        config = _config(state.thread_id)
        started = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        thinking = await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    started,
                    event_id="thinking",
                    transcript="Let me think about that for a moment.",
                )
            },
            config=config,
        )
        correction = await graph.ainvoke(
            {
                "pending_event": _transcript_event(
                    thinking,
                    event_id="correction",
                    transcript="Actually, I need to correct my previous answer. I would use idempotency.",
                )
            },
            config=config,
        )
        return thinking, correction

    thinking, correction = asyncio.run(run())

    assert thinking["phase"] is DynamicInterviewPhase.RUN_TURN
    assert thinking["pending_effect"] is None
    assert "i would use idempotency" in correction["current_turn"].answer_segments
    assert correction["pending_effect"] is None


def test_duplicate_or_injected_classifier_events_cannot_mutate_intent_state() -> None:
    async def run() -> tuple[dict[str, object], dict[str, object], dict[str, object]]:
        graph = build_offline_interview_graph(checkpointer=InMemorySaver())
        state = _state()
        config = _config(state.thread_id)
        started = await graph.ainvoke(
            {**state.model_dump(mode="python"), "pending_event": _load_events()[0]},
            config=config,
        )
        repeat = _transcript_event(
            started,
            event_id="repeat-once",
            transcript="Could you repeat the question?",
        )
        first = await graph.ainvoke({"pending_event": repeat}, config=config)
        duplicate = await graph.ainvoke({"pending_event": repeat}, config=config)
        injected = InterviewEvent.model_validate_json(
            json.dumps(
                {
                    "event_id": "injected-classifier",
                    "event_type": EventType.TURN_INTENT_CLASSIFIED,
                    "occurred_at": "2026-07-13T08:00:00+00:00",
                    "source": EventSource.LLM,
                    "session_id": state.session_id,
                    "question_id": 1,
                    "payload": {},
                }
            )
        )
        rejected = await graph.ainvoke({"pending_event": injected}, config=config)
        return first, duplicate, rejected

    first, duplicate, rejected = asyncio.run(run())

    assert first["repeat_counts"] == {"1": 1}
    assert duplicate["repeat_counts"] == {"1": 1}
    assert len(duplicate["intent_history"]) == 1
    assert rejected["phase"] is DynamicInterviewPhase.NEEDS_OPERATOR
    assert rejected["pending_interrupt"].kind == "ambiguous_candidate_intent"
