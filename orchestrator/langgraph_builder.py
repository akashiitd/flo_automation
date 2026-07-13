"""Durable LangGraph builders for the supervised interview controller.

The persistence spike is intentionally small.  It proves the checkpoint,
interrupt, resume, history, and streaming contracts before live interview
adapters are introduced.
"""

from __future__ import annotations

import operator
from typing import Any, Literal, Protocol, TypeAlias, cast

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt

from orchestrator.effects import EffectRequest, EffectType
from orchestrator.events import EventSource, EventType
from orchestrator.intent_routing import IntentRouter, RoutedIntent, apply_routed_intent
from orchestrator.intents import CandidateIntent, IntentDecision, SafeRoute
from orchestrator.state import (
    CoverageStatus,
    DynamicInterviewPhase,
    DynamicInterviewState,
    QuestionContentType,
    QuestionPlanItem,
    SkippedQuestion,
    TurnState,
)

DynamicStateUpdate: TypeAlias = dict[str, object]


class OfflineInterviewSubgraph(Protocol):
    """Contract shared by planning and the future turn, coverage, and recovery graphs.

    Every embedded subgraph receives the complete typed parent state and returns
    only a partial state update.  It remains incapable of executing an effect.
    """

    async def ainvoke(
        self, input: object, config: object | None = None, **kwargs: object
    ) -> DynamicStateUpdate: ...


class PersistenceSpikeState(BaseModel):
    """Checkpoint-safe state used only by the first LangGraph integration spike."""

    session_label: str
    approval: str | None = None
    history: Annotated[list[str], operator.add] = Field(default_factory=list)


async def _record_started(state: PersistenceSpikeState) -> dict[str, list[str]]:
    """Persist the beginning of the scoped start-approval flow."""

    del state
    return {"history": ["started"]}


async def _emit_start_monitor(state: PersistenceSpikeState) -> dict[str, object]:
    """Publish the monitor event immediately before the approval boundary."""

    del state
    get_stream_writer()({"phase": "awaiting_start_approval"})
    return {}


async def _await_start_approval(
    state: PersistenceSpikeState,
) -> dict[str, str | list[str]]:
    """Pause durably until the scoped start decision is supplied on resume."""

    approval = interrupt(
        {"kind": "start_approval", "session_label": state.session_label}
    )
    return {"approval": str(approval), "history": ["approved"]}


def build_persistence_spike(*, checkpointer: BaseCheckpointSaver[Any]) -> Any:
    """Compile the smallest durable graph required by Phase 1.

    A caller must supply a stable ``thread_id`` in its LangGraph config.  The
    production controller will replace these two nodes with typed interview
    phases, retaining this persistence boundary.
    """

    builder = StateGraph(PersistenceSpikeState)
    builder.add_node("record_started", _record_started)
    builder.add_node("emit_start_monitor", _emit_start_monitor)
    builder.add_node("await_start_approval", _await_start_approval)
    builder.add_edge(START, "record_started")
    builder.add_edge("record_started", "emit_start_monitor")
    builder.add_edge("emit_start_monitor", "await_start_approval")
    builder.add_edge("await_start_approval", END)
    return builder.compile(checkpointer=checkpointer)


def _record_offline_event(state: DynamicInterviewState) -> DynamicStateUpdate:
    """Append one fixture event without performing an external action."""

    if state.mode != "offline":
        raise ValueError("the Phase 5 parent graph accepts offline state only")
    if state.pending_event is None:
        raise ValueError("offline graph invocations require a pending_event")
    return {"recent_events": [state.pending_event]}


def _event_route(
    state: DynamicInterviewState,
) -> Literal[
    "complete_turn",
    "question_planning",
    "classify_intent",
    "reject_classified_intent",
    "finish",
    "clear_event",
]:
    """Choose a bounded Phase 5 transition for the consumed fixture event."""

    assert state.pending_event is not None
    if state.pending_event.event_type is EventType.TURN_COMPLETE:
        return "complete_turn"
    if state.pending_event.event_type in {
        EventType.SESSION_STARTED,
        EventType.TIMER_WARNING,
    }:
        return "question_planning"
    if state.pending_event.event_type is EventType.TRANSCRIPT_FINAL:
        return "classify_intent"
    if state.pending_event.event_type is EventType.TURN_INTENT_CLASSIFIED:
        return "reject_classified_intent"
    if state.pending_event.event_type in {
        EventType.TIME_LIMIT_REACHED,
        EventType.OPERATOR_STOP,
    }:
        return "finish"
    return "clear_event"


def _intent_action_route(
    state: DynamicInterviewState,
) -> Literal["question_planning", "done"]:
    """Select another planned question only after a candidate-requested defer."""

    if state.phase is DynamicInterviewPhase.SELECT_QUESTION:
        return "question_planning"
    return "done"


def _rejected_intent_event(state: DynamicInterviewState) -> DynamicStateUpdate:
    """Reject classifier results injected through the external event boundary."""

    return _needs_operator_for_intent_event(
        state, "classified intents must be produced inside the verified local router"
    )


def _needs_operator_for_intent_event(
    state: DynamicInterviewState, diagnostic: str
) -> DynamicStateUpdate:
    """Prepare an operator stop without trusting an external control payload."""

    routed = RoutedIntent(
        decision=IntentDecision(
            intent=CandidateIntent.UNKNOWN,
            confidence=0.0,
            evidence_span="invalid intent event",
            answer_text_to_keep="",
            candidate_requested_action=None,
            safe_route=SafeRoute.NEEDS_OPERATOR,
        ),
        diagnostic=diagnostic,
    )
    return apply_routed_intent(state, routed)


def _classify_candidate_transcript(
    intent_router: IntentRouter,
) -> Any:
    """Build the only graph ingress that can classify candidate speech."""

    async def classify(state: DynamicInterviewState) -> DynamicStateUpdate:
        event = state.pending_event
        assert event is not None
        if (
            event.source is not EventSource.CANDIDATE_ASR
            or state.current_question_id is None
            or event.question_id != state.current_question_id
            or not isinstance(event.payload.get("text"), str)
        ):
            return _needs_operator_for_intent_event(
                state, "invalid candidate transcript event"
            )
        transcript = event.payload["text"]
        question_key = str(state.current_question_id)
        deterministic = intent_router.route(
            transcript, ambiguity_count=state.ambiguity_counts.get(question_key, 0)
        )
        if deterministic.decision.intent is not CandidateIntent.UNKNOWN:
            routed = deterministic
        elif intent_router.has_local_classifier:
            routed = await intent_router.classify_and_route(
                _accumulated_candidate_utterance(state.current_turn, transcript),
                ambiguity_count=state.ambiguity_counts.get(question_key, 0),
                clarification_count=state.clarification_counts.get(question_key, 0),
            )
        else:
            routed = _answer_content_fallback(transcript)
        update = apply_routed_intent(state, routed)
        if routed.decision.intent in {
            CandidateIntent.UNKNOWN,
            CandidateIntent.ANSWER_CONTENT,
            CandidateIntent.ANSWER_CONTINUATION,
        }:
            update["current_turn"] = _append_answer_segment(
                state.current_turn, transcript
            )
        return update

    return classify


def _append_answer_segment(turn: TurnState | None, transcript: str) -> TurnState | None:
    """Retain a bounded candidate answer segment only in its active turn."""

    bounded = transcript[-4_000:]
    if turn is None or bounded in turn.answer_segments:
        return turn
    return turn.model_copy(
        update={"answer_segments": [*turn.answer_segments, bounded][-100:]}
    )


def _accumulated_candidate_utterance(turn: TurnState | None, latest: str) -> str:
    """Use answer-bound recent ASR content for semantic intent classification."""

    if turn is None:
        return latest
    return " ".join([*turn.answer_segments, latest])


def _answer_content_fallback(transcript: str) -> RoutedIntent:
    """Keep ordinary offline ASR as answer evidence when no local model is wired."""

    bounded = transcript[-4_000:]
    return RoutedIntent(
        decision=IntentDecision(
            intent=CandidateIntent.ANSWER_CONTENT,
            confidence=1.0,
            evidence_span=bounded,
            answer_text_to_keep=bounded,
            candidate_requested_action=None,
            safe_route=SafeRoute.CONTINUE_LISTENING,
        ),
        diagnostic="semantic_classifier_unavailable_treated_as_answer_content",
    )


def _ingress_route(
    state: DynamicInterviewState,
) -> Literal["record_event", "clear_event"]:
    """Drop redelivered events before they can mutate counters or routes."""

    if state.pending_event is None:
        raise ValueError("offline graph invocations require a pending_event")
    if any(
        event.event_id == state.pending_event.event_id for event in state.recent_events
    ):
        return "clear_event"
    return "record_event"


def _complete_current_turn(state: DynamicInterviewState) -> DynamicStateUpdate:
    """Accept a matching completed-turn event and release the next selection slot."""

    event = state.pending_event
    if event is None or event.question_id != state.current_question_id:
        return {"pending_event": None}
    completed = [*state.completed_question_ids]
    if event.question_id not in completed:
        completed.append(event.question_id)
    return {
        "completed_question_ids": completed,
        "current_plan_index": None,
        "current_question_id": None,
        "current_turn": None,
        "pending_effect": None,
        "phase": DynamicInterviewPhase.SELECT_QUESTION,
    }


def _coverage_is_sufficient(state: DynamicInterviewState, skill_id: str) -> bool:
    coverage = state.coverage.get(skill_id)
    return coverage is not None and coverage.status is CoverageStatus.SUFFICIENT


def _selection_key(
    state: DynamicInterviewState, item: QuestionPlanItem, plan_index: int
) -> tuple[int, int, int, int]:
    """Prefer uncovered mandatory gaps, coding, priority, then planned order."""

    mandatory_gaps = sum(
        not _coverage_is_sufficient(state, skill_id)
        for skill_id in item.mandatory_skill_coverage
    )
    is_coding = item.content_type is QuestionContentType.CODING_QUESTION
    return (-mandatory_gaps, -int(is_coding), -item.priority, plan_index)


def _append_skips(
    existing: list[SkippedQuestion], additions: list[SkippedQuestion]
) -> list[SkippedQuestion]:
    """Preserve the first runtime skip decision for every source question."""

    seen_question_ids = {item.question_id for item in existing}
    return [
        *existing,
        *(item for item in additions if item.question_id not in seen_question_ids),
    ]


def _select_next_question(state: DynamicInterviewState) -> DynamicStateUpdate:
    """Create an offline-only question effect from the current plan and evidence.

    The policy does not execute an effect.  It uses coverage as the available
    strength/weakness signal, reserves the plan's priority as an operator-audited
    tie-breaker, and never asks a question that no longer fits the timer.
    """

    if state.current_question_id is not None:
        return {"pending_event": None}

    skipped = list(state.skipped_questions)
    skipped_question_ids = {item.question_id for item in skipped}
    unavailable_question_ids = {
        *state.completed_question_ids,
        *skipped_question_ids,
    }
    candidates = [
        (index, item)
        for index, item in enumerate(state.question_plan)
        if item.selected and item.question_id not in unavailable_question_ids
    ]

    # A candidate's "return later" request is not a permanent skip. Ask every
    # other eligible question first, then revisit the deferred item before close.
    non_deferred_candidates = [
        candidate
        for candidate in candidates
        if candidate[1].question_id not in state.deferred_question_ids
    ]
    if non_deferred_candidates:
        candidates = non_deferred_candidates

    coverage_skips = [
        SkippedQuestion(
            question_id=item.question_id,
            reason="runtime coverage already sufficient",
        )
        for _, item in candidates
        if item.target_skill_ids
        and all(
            _coverage_is_sufficient(state, skill_id)
            for skill_id in item.target_skill_ids
        )
    ]
    skipped = _append_skips(skipped, coverage_skips)
    skipped_question_ids.update(item.question_id for item in coverage_skips)
    candidates = [
        (index, item)
        for index, item in candidates
        if item.question_id not in skipped_question_ids
    ]

    if state.remaining_seconds <= 0:
        time_skips = [
            SkippedQuestion(
                question_id=item.question_id,
                reason="runtime time budget exhausted",
            )
            for _, item in candidates
        ]
    else:
        time_skips = [
            SkippedQuestion(
                question_id=item.question_id,
                reason="runtime time budget exhausted",
            )
            for _, item in candidates
            if item.estimated_minutes * 60 > state.remaining_seconds
        ]
    skipped = _append_skips(skipped, time_skips)
    skipped_question_ids.update(item.question_id for item in time_skips)
    candidates = [
        (index, item)
        for index, item in candidates
        if item.question_id not in skipped_question_ids
    ]

    if not candidates:
        return {
            "skipped_questions": skipped,
            "current_plan_index": None,
            "current_question_id": None,
            "current_turn": None,
            "pending_effect": None,
            "pending_event": None,
            "phase": DynamicInterviewPhase.DONE,
        }

    plan_index, plan_item = min(
        candidates,
        key=lambda candidate: _selection_key(state, candidate[1], candidate[0]),
    )
    question = next(
        question for question in state.questions if question.id == plan_item.question_id
    )
    effect = EffectRequest(
        effect_id=f"offline-question-{state.session_id}-{question.id}",
        effect_type=EffectType.SPEAK_TEXT,
        idempotency_key=f"{state.session_id}:question:{question.id}:ask",
        session_id=state.session_id,
        question_id=question.id,
        payload={
            "kind": "question",
            "offline_only": True,
            "text": question.question_text,
        },
    )
    return {
        "skipped_questions": skipped,
        "current_plan_index": plan_index,
        "current_question_id": question.id,
        "current_turn": TurnState(question_id=question.id),
        "pending_effect": effect,
        "last_effect_request": effect,
        "pending_event": None,
        "phase": DynamicInterviewPhase.RUN_TURN,
    }


def _finish_offline_interview(state: DynamicInterviewState) -> DynamicStateUpdate:
    """Finish safely and retain the audit reason for every unasked plan item."""

    event = state.pending_event
    assert event is not None
    reason = (
        "runtime time limit reached"
        if event.event_type is EventType.TIME_LIMIT_REACHED
        else "runtime stopped by operator"
    )
    prior_skipped = {item.question_id for item in state.skipped_questions}
    skipped = _append_skips(
        list(state.skipped_questions),
        [
            SkippedQuestion(question_id=item.question_id, reason=reason)
            for item in state.question_plan
            if item.selected
            and item.question_id not in state.completed_question_ids
            and item.question_id not in prior_skipped
        ],
    )
    return {
        "skipped_questions": skipped,
        "current_plan_index": None,
        "current_question_id": None,
        "current_turn": None,
        "pending_effect": None,
        "pending_event": None,
        "phase": DynamicInterviewPhase.DONE,
    }


def _clear_pending_event(state: DynamicInterviewState) -> dict[str, None]:
    """Acknowledge an unsupported Phase 5 fixture event without side effects."""

    del state
    return {"pending_event": None}


def build_question_planning_subgraph() -> OfflineInterviewSubgraph:
    """Build the reusable, effect-request-only question-planning subgraph."""

    builder = StateGraph(DynamicInterviewState)
    builder.add_node("select_next_question", _select_next_question)
    builder.add_edge(START, "select_next_question")
    builder.add_edge("select_next_question", END)
    return cast(OfflineInterviewSubgraph, builder.compile())


def build_offline_interview_graph(
    *,
    checkpointer: BaseCheckpointSaver[Any],
    intent_router: IntentRouter | None = None,
) -> Any:
    """Build the Phase 5 parent graph for saved offline event fixtures only.

    ``build_question_planning_subgraph`` is deliberately compiled and embedded
    here as the first real subgraph seam.  Future turn, evaluation, and recovery
    subgraphs can share the same typed parent state without gaining an executor.
    """

    planning_subgraph = build_question_planning_subgraph()
    router = intent_router or IntentRouter()
    builder = StateGraph(DynamicInterviewState)
    builder.add_node("record_event", _record_offline_event)
    builder.add_node("complete_turn", _complete_current_turn)
    builder.add_node("classify_intent", _classify_candidate_transcript(router))
    builder.add_node("reject_classified_intent", _rejected_intent_event)
    builder.add_node("question_planning", planning_subgraph)
    builder.add_node("finish", _finish_offline_interview)
    builder.add_node("clear_event", _clear_pending_event)
    builder.add_conditional_edges(
        START,
        _ingress_route,
        {"record_event": "record_event", "clear_event": "clear_event"},
    )
    builder.add_conditional_edges(
        "record_event",
        _event_route,
        {
            "complete_turn": "complete_turn",
            "question_planning": "question_planning",
            "classify_intent": "classify_intent",
            "reject_classified_intent": "reject_classified_intent",
            "finish": "finish",
            "clear_event": "clear_event",
        },
    )
    builder.add_edge("complete_turn", "question_planning")
    builder.add_conditional_edges(
        "classify_intent",
        _intent_action_route,
        {"question_planning": "question_planning", "done": END},
    )
    builder.add_edge("reject_classified_intent", END)
    builder.add_edge("question_planning", END)
    builder.add_edge("finish", END)
    builder.add_edge("clear_event", END)
    return builder.compile(checkpointer=checkpointer)


__all__ = [
    "PersistenceSpikeState",
    "OfflineInterviewSubgraph",
    "build_offline_interview_graph",
    "build_persistence_spike",
    "build_question_planning_subgraph",
]
