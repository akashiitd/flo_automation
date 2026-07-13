"""Public contracts for the durable, dynamic interview controller."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import pytest
from pydantic import ValidationError

from orchestrator.effects import EffectRequest, EffectResult, EffectStatus, EffectType
from orchestrator.event_ledger import EventLedger, EventLedgerConflictError
from orchestrator.events import EventSource, EventType, InterviewEvent
from orchestrator.intents import CandidateIntent, IntentDecision, SafeRoute
from orchestrator.reducers import (
    RECENT_EVENT_LIMIT,
    DynamicStateConflictError,
    append_interview_events,
    append_skill_evidence,
)
from orchestrator.state import (
    DynamicInterviewState,
    QuestionContentType,
    QuestionMappingSource,
    QuestionPlanItem,
    QuestionPlanArtifact,
    QuestionState,
    SkillAssessment,
    SkillAssessmentStatus,
    SkillEvidence,
    SkillParameter,
    SkillParametersArtifact,
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
    with pytest.raises(ValidationError):
        InterviewEvent.model_validate({**event.model_dump(), "question_id": "5"})

    event_window = [
        event.model_copy(
            update={
                "event_id": f"event-{index}",
                "payload": {"segment_id": f"segment-{index}"},
            }
        )
        for index in range(RECENT_EVENT_LIMIT + 1)
    ]
    retained_events = append_interview_events([], event_window)
    assert len(retained_events) == RECENT_EVENT_LIMIT
    assert retained_events[0].event_id == "event-1"


def test_event_ledger_deduplicates_across_a_reopened_process_boundary(
    tmp_path: Path,
) -> None:
    event = InterviewEvent(
        event_id="durable-event-001",
        event_type=EventType.SESSION_STARTED,
        occurred_at=datetime(2026, 7, 13, 10, 0, tzinfo=UTC),
        source=EventSource.OPERATOR,
        session_id="session-001",
    )
    ledger_path = tmp_path / "graph_events.jsonl"

    assert EventLedger(ledger_path).append(event) is True
    reopened_ledger = EventLedger(ledger_path)

    assert reopened_ledger.append(event) is False
    with pytest.raises(EventLedgerConflictError, match="event_id"):
        reopened_ledger.append(event.model_copy(update={"payload": {"changed": True}}))


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
    with pytest.raises(ValidationError):
        QuestionState(
            id=cast(Any, "5"),
            question_text=question.question_text,
            ideal_answer=question.ideal_answer,
        )
    with pytest.raises(ValidationError, match="session_id"):
        DynamicInterviewState.model_validate(
            {
                **state.model_dump(),
                "recent_events": [
                    InterviewEvent(
                        event_id="wrong-session-event",
                        event_type=EventType.SESSION_STARTED,
                        occurred_at=datetime(2026, 7, 13, 10, 0, tzinfo=UTC),
                        source=EventSource.OPERATOR,
                        session_id="other-session",
                    )
                ],
            }
        )
    with pytest.raises(ValidationError):
        DynamicInterviewState.model_validate(
            {
                **state.model_dump(),
                "question_evaluations": [
                    {
                        "question_id": question.id,
                        "score": "4",
                        "rating_label": "Good",
                        "evidence": ["Candidate described input validation."],
                        "follow_up": "Could you expand on your approach?",
                        "feedback": "Strong security fundamentals.",
                        "confidence": 0.8,
                    }
                ],
            }
        )
    assert SkillParametersArtifact(parameters=[skill]).schema_version == 1
    assert QuestionPlanArtifact(items=state.question_plan).schema_version == 1
    with pytest.raises(ValidationError):
        SkillParametersArtifact.model_validate({"schema_version": 2, "parameters": []})
    assessment = SkillAssessment(
        skill_id=skill.id,
        proposed_score=4,
        status=SkillAssessmentStatus.ASSESSED,
        evidence_ids=[evidence.evidence_id],
        rationale="The candidate cited input validation and authorization.",
        confidence=0.8,
    )
    assert (
        SkillAssessment.model_validate_json(assessment.model_dump_json()) == assessment
    )
