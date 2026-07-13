"""Public contracts for the durable, dynamic interview controller."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from orchestrator.effects import EffectRequest, EffectResult, EffectStatus, EffectType
from orchestrator.events import EventSource, EventType, InterviewEvent
from orchestrator.intents import CandidateIntent, IntentDecision, SafeRoute
from orchestrator.reducers import (
    DynamicStateConflictError,
    append_interview_events,
    append_skill_evidence,
)
from orchestrator.state import (
    DynamicInterviewState,
    QuestionContentType,
    QuestionMappingSource,
    QuestionPlanItem,
    QuestionState,
    SkillEvidence,
    SkillParameter,
)


def test_interview_events_are_strict_json_round_trippable_and_deduplicated() -> None:
    event = InterviewEvent(
        event_id="event-001",
        event_type=EventType.TRANSCRIPT_FINAL,
        occurred_at=datetime(2026, 7, 13, 10, 0, tzinfo=UTC),
        source=EventSource.CANDIDATE_ASR,
        session_id="session-001",
        question_id=5,
        payload={"segment_id": "segment-001", "text": "My answer is retries."},
    )

    restored = InterviewEvent.model_validate_json(event.model_dump_json())

    assert restored == event
    assert append_interview_events([event], [restored]) == [event]
    with pytest.raises(DynamicStateConflictError, match="event_id"):
        append_interview_events(
            [event], [event.model_copy(update={"payload": {"text": "changed"}})]
        )
    with pytest.raises(ValidationError):
        InterviewEvent.model_validate({**event.model_dump(), "unexpected": True})


def test_intent_requires_candidate_evidence_and_effects_are_idempotent_contracts() -> (
    None
):
    decision = IntentDecision(
        intent=CandidateIntent.REPEAT_REQUEST,
        confidence=0.96,
        evidence_span="Could you repeat that?",
        answer_text_to_keep="",
        candidate_requested_action="repeat the current question",
        safe_route=SafeRoute.REPEAT_CURRENT_QUESTION,
    )

    assert decision.validate_against("I missed it. Could you repeat that?") == decision
    with pytest.raises(ValueError, match="evidence_span"):
        decision.validate_against("I understand the question.")

    request = EffectRequest(
        effect_id="effect-001",
        effect_type=EffectType.SPEAK_TEXT,
        idempotency_key="session-001:question-5:replay-1",
        session_id="session-001",
        payload={"text": "Please explain retries."},
    )
    result = EffectResult(
        effect_id=request.effect_id,
        status=EffectStatus.UNCERTAIN,
        result_summary="process stopped after playback began",
    )

    assert EffectRequest.model_validate_json(request.model_dump_json()) == request
    assert EffectResult.model_validate_json(result.model_dump_json()) == result


def test_dynamic_state_round_trips_with_skill_evidence_and_explicit_reducers() -> None:
    question = QuestionState(
        id=5,
        question_text="How would you secure a REST API?",
        ideal_answer="Use authentication, authorization, validation, and rate limits.",
    )
    skill = SkillParameter(
        id="rest-api-security",
        name="REST API Development & Security",
        requirement="Mandatory",
        level="Professional",
        rating_scale=5,
    )
    evidence = SkillEvidence(
        evidence_id="evidence-001",
        skill_id=skill.id,
        question_id=question.id,
        transcript_evidence="I would validate input and require authorization.",
        question_score=4,
        relevance_weight=0.9,
        confidence=0.8,
    )
    state = DynamicInterviewState(
        thread_id="thread-001",
        session_id="session-001",
        candidate_identifier="candidate-hash",
        questions=[question],
        skill_parameters=[skill],
        question_plan=[
            QuestionPlanItem(
                question_id=question.id,
                content_type=QuestionContentType.INTERVIEW_QUESTION,
                target_skill_ids=[skill.id],
                estimated_minutes=3,
                priority=100,
                selected=True,
                mapping_source=QuestionMappingSource.DETERMINISTIC,
                mapping_confidence=1,
            )
        ],
        skill_evidence=[evidence],
    )

    restored = DynamicInterviewState.model_validate_json(state.model_dump_json())

    assert restored == state
    assert append_skill_evidence([evidence], [evidence]) == [evidence]
    with pytest.raises(DynamicStateConflictError, match="evidence_id"):
        append_skill_evidence(
            [evidence], [evidence.model_copy(update={"confidence": 0.2})]
        )
