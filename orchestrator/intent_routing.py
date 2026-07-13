"""Safety-first classification and durable routing of candidate turn controls."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Protocol
from urllib.parse import urlparse

import httpx
from pydantic import ValidationError

from llm.lmstudio_provider import LMStudioProvider
from llm.prompts import (
    IDENTITY_BOUNDARY_RESPONSE,
    NO_COACHING_BOUNDARY_RESPONSE,
    OFF_TOPIC_BOUNDARY_RESPONSE,
    SAFE_CLARIFICATION_RESPONSE,
    candidate_intent_messages,
)
from llm.provider import StructuredOutputError
from llm.schemas import ModelClass
from orchestrator.effects import EffectRequest, EffectType
from orchestrator.intents import CandidateIntent, IntentDecision, SafeRoute
from orchestrator.state import (
    DynamicInterviewPhase,
    DynamicInterviewState,
    InterruptRequest,
    TurnState,
)

MAX_CLASSIFICATION_TRANSCRIPT = 4_000
MAX_AUTOMATIC_REPEATS = 2
MAX_AUDIO_RECOVERY_ATTEMPTS = 1


@dataclass(frozen=True, slots=True)
class RoutedIntent:
    """One policy-approved route, safe response, and non-sensitive diagnosis."""

    decision: IntentDecision
    candidate_response: str | None = None
    diagnostic: str | None = None


class LocalStructuredIntentProvider(Protocol):
    """Minimal structured-generation seam, used only by a verified local provider."""

    async def generate_structured(
        self,
        messages: object,
        schema: object,
        model_class: ModelClass,
        *,
        request_purpose: str,
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class _IntentRule:
    pattern: re.Pattern[str]
    intent: CandidateIntent
    route: SafeRoute
    action: str
    response: str | None = None


class IntentRouter:
    """Classify candidate controls without permitting cloud fallback or coaching.

    Production construction accepts only an actual ``LMStudioProvider`` at a
    loopback URL. ``for_test`` is deliberately separate so test doubles cannot
    accidentally become a production data-egress path.
    """

    def __init__(self, *, local_provider: LMStudioProvider | None = None):
        if local_provider is not None and not _is_verified_loopback_lmstudio(
            local_provider
        ):
            raise ValueError(
                "intent classification requires an LMStudioProvider at a loopback URL"
            )
        self._local_provider: LocalStructuredIntentProvider | None = local_provider

    @classmethod
    def for_test(cls, local_provider: LocalStructuredIntentProvider) -> IntentRouter:
        """Create the explicit test-only structured-classifier seam."""

        router = cls()
        router._local_provider = local_provider
        return router

    @property
    def has_local_classifier(self) -> bool:
        """Whether semantic routing can use the verified local classifier."""

        return self._local_provider is not None

    def route(
        self, candidate_transcript: str, *, ambiguity_count: int = 0
    ) -> RoutedIntent:
        """Detect bounded, whole-utterance controls before semantic classification."""

        transcript, truncated = _bounded_transcript(candidate_transcript)
        normalized = _normalize(transcript)
        for rule in _DETERMINISTIC_RULES:
            if rule.pattern.fullmatch(normalized):
                return self._routed(
                    transcript,
                    intent=rule.intent,
                    route=rule.route,
                    action=rule.action,
                    response=rule.response,
                    diagnostic="input_truncated" if truncated else None,
                )
        return self._unknown(
            transcript,
            ambiguity_count=ambiguity_count,
            diagnostic="input_truncated" if truncated else None,
        )

    async def classify_and_route(
        self,
        candidate_transcript: str,
        *,
        ambiguity_count: int = 0,
        clarification_count: int = 0,
    ) -> RoutedIntent:
        """Use validated local structured output only after deterministic rules miss."""

        deterministic = self.route(
            candidate_transcript, ambiguity_count=ambiguity_count
        )
        if deterministic.decision.intent is not CandidateIntent.UNKNOWN:
            return deterministic
        if self._local_provider is None:
            return deterministic
        transcript, truncated = _bounded_transcript(candidate_transcript)
        try:
            generation = await self._local_provider.generate_structured(
                candidate_intent_messages(transcript),
                IntentDecision,
                "fast",
                request_purpose="candidate_intent_classification",
            )
        except (TimeoutError, httpx.HTTPError, StructuredOutputError):
            return self._unknown(
                transcript,
                ambiguity_count=ambiguity_count,
                diagnostic="local_classifier_unavailable",
            )
        try:
            output = generation.get("output")
            if not isinstance(output, dict):
                raise ValueError("local intent output must be an object")
            decision = IntentDecision.model_validate_json(json.dumps(output))
        except (AttributeError, TypeError, ValueError, ValidationError):
            return self._unknown(
                transcript,
                ambiguity_count=ambiguity_count,
                diagnostic="invalid_local_classifier_output",
            )
        routed = self.route_decision(
            decision,
            transcript,
            ambiguity_count=ambiguity_count,
            clarification_count=clarification_count,
        )
        if truncated and routed.diagnostic is None:
            return RoutedIntent(
                decision=routed.decision,
                candidate_response=routed.candidate_response,
                diagnostic="input_truncated",
            )
        return routed

    @staticmethod
    def route_decision(
        decision: IntentDecision,
        candidate_transcript: str,
        *,
        ambiguity_count: int = 0,
        clarification_count: int = 0,
    ) -> RoutedIntent:
        """Revalidate a structured decision before a graph may enact it."""

        transcript, _ = _bounded_transcript(candidate_transcript)
        try:
            decision.validate_against(transcript)
        except ValueError:
            return IntentRouter._unknown(
                transcript,
                ambiguity_count=ambiguity_count,
                diagnostic="decision_not_grounded_in_candidate_speech",
            )
        expected_route = _SAFE_ROUTE_BY_INTENT[decision.intent]
        if decision.safe_route is not expected_route:
            return IntentRouter._unknown(
                transcript,
                ambiguity_count=ambiguity_count,
                diagnostic="decision_route_not_allowed",
            )
        if decision.intent is CandidateIntent.UNKNOWN:
            return IntentRouter._unknown(transcript, ambiguity_count=ambiguity_count)
        if decision.confidence >= 0.85:
            return RoutedIntent(
                decision=decision,
                candidate_response=_boundary_response(decision.intent),
            )
        if (
            decision.confidence >= 0.60
            and clarification_count == 0
            and ambiguity_count == 0
        ):
            return RoutedIntent(
                decision=IntentDecision(
                    intent=CandidateIntent.CLARIFICATION_REQUEST,
                    confidence=decision.confidence,
                    evidence_span=decision.evidence_span,
                    answer_text_to_keep="",
                    candidate_requested_action="request neutral clarification due to uncertainty",
                    safe_route=SafeRoute.SAFE_CLARIFICATION,
                ),
                candidate_response=SAFE_CLARIFICATION_RESPONSE,
                diagnostic="medium_confidence_neutral_clarification",
            )
        return IntentRouter._unknown(
            transcript,
            ambiguity_count=ambiguity_count,
            diagnostic="classifier_confidence_too_low_or_clarification_used",
        )

    @staticmethod
    def _routed(
        transcript: str,
        *,
        intent: CandidateIntent,
        route: SafeRoute,
        action: str | None = None,
        response: str | None = None,
        diagnostic: str | None = None,
    ) -> RoutedIntent:
        return RoutedIntent(
            decision=IntentDecision(
                intent=intent,
                confidence=1.0 if intent is not CandidateIntent.UNKNOWN else 0.0,
                evidence_span=transcript,
                answer_text_to_keep="",
                candidate_requested_action=action,
                safe_route=route,
            ),
            candidate_response=response,
            diagnostic=diagnostic,
        )

    @staticmethod
    def _unknown(
        transcript: str,
        *,
        ambiguity_count: int,
        diagnostic: str | None = None,
    ) -> RoutedIntent:
        route = (
            SafeRoute.NEEDS_OPERATOR
            if ambiguity_count >= 2
            else SafeRoute.CONTINUE_LISTENING
        )
        return IntentRouter._routed(
            transcript,
            intent=CandidateIntent.UNKNOWN,
            route=route,
            diagnostic=diagnostic,
        )


def apply_routed_intent(
    state: DynamicInterviewState, routed: RoutedIntent
) -> dict[str, object]:
    """Convert an approved route into bounded state/effect/interrupt updates.

    This is deliberately pure: it prepares offline-safe effects and operator
    interrupts, but it neither plays audio nor controls the interview page.
    """

    decision = routed.decision
    question_id = state.current_question_id
    question_key = str(question_id) if question_id is not None else "session"
    history = [*state.intent_history, decision][-100:]
    update: dict[str, object] = {
        "intent_history": history,
        "pending_event": None,
    }
    route = decision.safe_route

    if question_id is None and route in {
        SafeRoute.COMPLETE_TURN,
        SafeRoute.REPEAT_CURRENT_QUESTION,
        SafeRoute.SAFE_CLARIFICATION,
        SafeRoute.AUDIO_RECOVERY,
        SafeRoute.EXTEND_THINKING_TIME,
        SafeRoute.DEFER_CURRENT_QUESTION,
        SafeRoute.HANDLE_CORRECTION,
    }:
        return _needs_operator_update(
            update, "intent route requires an active question"
        )

    if route is SafeRoute.NEEDS_OPERATOR:
        return {
            **update,
            "phase": DynamicInterviewPhase.NEEDS_OPERATOR,
            "pending_effect": None,
            "pending_interrupt": InterruptRequest(
                kind="ambiguous_candidate_intent",
                reason=routed.diagnostic or "candidate intent needs human review",
                options=["continue", "clarify", "pause", "takeover"],
            ),
        }
    if decision.intent is CandidateIntent.UNKNOWN:
        ambiguity_counts = dict(state.ambiguity_counts)
        ambiguity_counts[question_key] = ambiguity_counts.get(question_key, 0) + 1
        return {**update, "ambiguity_counts": ambiguity_counts}
    if route is SafeRoute.REPEAT_CURRENT_QUESTION:
        repeat_counts = dict(state.repeat_counts)
        attempts = repeat_counts.get(question_key, 0) + 1
        repeat_counts[question_key] = attempts
        if attempts > MAX_AUTOMATIC_REPEATS + 1:
            return {
                **update,
                "repeat_counts": repeat_counts,
                "phase": DynamicInterviewPhase.NEEDS_OPERATOR,
                "pending_effect": None,
                "pending_interrupt": InterruptRequest(
                    kind="repeated_question_replay",
                    reason="candidate requested more than two replays and a wording/audio check",
                    options=["replay", "clarify", "pause", "takeover"],
                ),
            }
        if attempts == MAX_AUTOMATIC_REPEATS + 1:
            return {
                **update,
                "repeat_counts": repeat_counts,
                **_speak_update(
                    state,
                    question_id,
                    "Would you like me to repeat the wording, or are you having an audio problem?",
                    "repeat-disambiguation",
                ),
            }
        replay_extra = (
            {"playback_rate": 0.85, "chunked": True}
            if attempts == MAX_AUTOMATIC_REPEATS
            else {}
        )
        return {
            **update,
            "repeat_counts": repeat_counts,
            **_speak_update(
                state,
                question_id,
                _current_question_text(state),
                f"repeat-{attempts}",
                extra_payload=replay_extra,
            ),
        }
    if route is SafeRoute.SAFE_CLARIFICATION:
        clarification_counts = dict(state.clarification_counts)
        attempts = clarification_counts.get(question_key, 0) + 1
        clarification_counts[question_key] = attempts
        if attempts > 1:
            return {
                **update,
                "clarification_counts": clarification_counts,
                "phase": DynamicInterviewPhase.NEEDS_OPERATOR,
                "pending_effect": None,
                "pending_interrupt": InterruptRequest(
                    kind="repeated_clarification_request",
                    reason="one neutral clarification was already provided",
                    options=["continue", "pause", "takeover"],
                ),
            }
        return {
            **update,
            "clarification_counts": clarification_counts,
            **_speak_update(
                state,
                question_id,
                routed.candidate_response or SAFE_CLARIFICATION_RESPONSE,
                "clarification",
            ),
        }
    if route is SafeRoute.AUDIO_RECOVERY:
        attempts = state.audio_problem_count + 1
        if attempts > MAX_AUDIO_RECOVERY_ATTEMPTS:
            return {
                **update,
                "audio_problem_count": attempts,
                "phase": DynamicInterviewPhase.NEEDS_OPERATOR,
                "pending_effect": None,
                "pending_interrupt": InterruptRequest(
                    kind="repeated_audio_problem",
                    reason="audio recovery did not resolve the candidate report",
                    options=["retry_audio", "pause", "takeover"],
                ),
            }
        effect = EffectRequest(
            effect_id=f"offline-audio-check-{state.session_id}-{attempts}",
            effect_type=EffectType.CHECK_AUDIO_ROUTE,
            idempotency_key=f"{state.session_id}:audio-check:{attempts}",
            session_id=state.session_id,
            question_id=question_id,
            payload={"offline_only": True, "kind": "candidate_audio_recovery"},
        )
        return {
            **update,
            "audio_problem_count": attempts,
            "pending_effect": effect,
            "last_effect_request": effect,
            "phase": DynamicInterviewPhase.RUN_TURN,
        }
    if route is SafeRoute.EXTEND_THINKING_TIME:
        return {
            **update,
            "current_turn": _append_control_utterance(
                state.current_turn, decision.evidence_span
            ),
            "pending_effect": None,
            "phase": DynamicInterviewPhase.RUN_TURN,
        }
    if route is SafeRoute.DEFER_CURRENT_QUESTION:
        if question_id is None:
            return _needs_operator_update(
                update, "cannot defer without an active question"
            )
        deferred = [*state.deferred_question_ids]
        if question_id not in deferred:
            deferred.append(question_id)
        return {
            **update,
            "deferred_question_ids": deferred,
            "current_plan_index": None,
            "current_question_id": None,
            "current_turn": None,
            "pending_effect": None,
            "phase": DynamicInterviewPhase.SELECT_QUESTION,
        }
    if route is SafeRoute.HANDLE_CORRECTION:
        corrected_answer = decision.answer_text_to_keep or _correction_answer_text(
            decision.evidence_span
        )
        return {
            **update,
            "current_turn": _record_correction(
                state.current_turn, decision.evidence_span, corrected_answer
            ),
            "phase": DynamicInterviewPhase.RUN_TURN,
        }
    if route is SafeRoute.HANDLE_WITHDRAWAL:
        return {
            **update,
            "phase": DynamicInterviewPhase.PAUSED,
            "pending_effect": None,
            "pending_interrupt": InterruptRequest(
                kind="candidate_withdrawal",
                reason="candidate asked to withdraw from the interview",
                options=["confirm_withdrawal", "resume", "takeover"],
            ),
        }
    if route is SafeRoute.COMPLETE_TURN:
        return {**update, "phase": DynamicInterviewPhase.UPDATE_COVERAGE}
    if route is SafeRoute.ANSWER_JOB_DESCRIPTION:
        return {
            **update,
            **_speak_update(
                state,
                question_id,
                "I do not have additional verified role details in this interview session. Please ask the recruiter for that information.",
                "job-description-boundary",
            ),
        }
    if route is SafeRoute.ANSWER_INTERVIEW_PROCESS:
        return {
            **update,
            **_speak_update(
                state,
                question_id,
                "I do not have verified details about next steps. Please check with your recruiter.",
                "interview-process-boundary",
            ),
        }
    if route in {
        SafeRoute.IDENTITY_BOUNDARY,
        SafeRoute.COACHING_BOUNDARY,
        SafeRoute.OFF_TOPIC_BOUNDARY,
    }:
        response = routed.candidate_response or _boundary_response(decision.intent)
        assert response is not None
        return {**update, **_speak_update(state, question_id, response, route.value)}
    return {**update, "phase": DynamicInterviewPhase.RUN_TURN}


def _needs_operator_update(update: dict[str, object], reason: str) -> dict[str, object]:
    return {
        **update,
        "phase": DynamicInterviewPhase.NEEDS_OPERATOR,
        "pending_effect": None,
        "pending_interrupt": InterruptRequest(
            kind="intent_route_invalid_for_state",
            reason=reason,
            options=["continue", "pause", "takeover"],
        ),
    }


def _append_control_utterance(
    turn: TurnState | None, utterance: str
) -> TurnState | None:
    if turn is None or utterance in turn.control_utterances:
        return turn
    return turn.model_copy(
        update={"control_utterances": [*turn.control_utterances, utterance][-100:]}
    )


def _correction_answer_text(utterance: str) -> str:
    """Keep the substantive portion after a correction request, if present."""

    match = re.match(
        r"(?:actually,? )?(?:i need to )?correct (?:my )?(?:previous )?answer[.:;]?\s*(.*)",
        _normalize(utterance),
    )
    return match.group(1).strip() if match is not None else ""


def _record_correction(
    turn: TurnState | None, utterance: str, corrected_answer: str
) -> TurnState | None:
    turn = _append_control_utterance(turn, utterance)
    if turn is None or not corrected_answer or corrected_answer in turn.answer_segments:
        return turn
    return turn.model_copy(
        update={"answer_segments": [*turn.answer_segments, corrected_answer][-100:]}
    )


def _current_question_text(state: DynamicInterviewState) -> str:
    if state.current_question_id is None:
        return "Could you continue with the current interview question?"
    return next(
        question.question_text
        for question in state.questions
        if question.id == state.current_question_id
    )


def _speak_update(
    state: DynamicInterviewState,
    question_id: int | None,
    text: str,
    kind: str,
    *,
    extra_payload: dict[str, object] | None = None,
) -> dict[str, object]:
    effect = EffectRequest(
        effect_id=f"offline-intent-{state.session_id}-{question_id or 0}-{kind}",
        effect_type=EffectType.SPEAK_TEXT,
        idempotency_key=f"{state.session_id}:intent:{question_id or 0}:{kind}",
        session_id=state.session_id,
        question_id=question_id,
        payload={
            "offline_only": True,
            "kind": kind,
            "text": text,
            **(extra_payload or {}),
        },
    )
    return {
        "pending_effect": effect,
        "last_effect_request": effect,
        "phase": DynamicInterviewPhase.RUN_TURN,
    }


def _is_verified_loopback_lmstudio(provider: LMStudioProvider) -> bool:
    if not isinstance(provider, LMStudioProvider):
        return False
    parsed = urlparse(provider.base_url)
    return parsed.scheme == "http" and parsed.hostname in {
        "127.0.0.1",
        "localhost",
        "::1",
    }


def _bounded_transcript(candidate_transcript: str) -> tuple[str, bool]:
    transcript = candidate_transcript.strip()
    if not transcript:
        raise ValueError("candidate_transcript must not be empty")
    return transcript[-MAX_CLASSIFICATION_TRANSCRIPT:], len(
        transcript
    ) > MAX_CLASSIFICATION_TRANSCRIPT


def _normalize(value: str) -> str:
    return " ".join(value.casefold().strip(".,!? ").split())


def _rule(
    expression: str,
    intent: CandidateIntent,
    route: SafeRoute,
    action: str,
    response: str | None = None,
) -> _IntentRule:
    return _IntentRule(re.compile(expression), intent, route, action, response)


_DETERMINISTIC_RULES: tuple[_IntentRule, ...] = (
    _rule(
        r"(?:please )?(?:ignore (?:all |the )?(?:prior |previous )?(?:interview )?instructions and )?(?:reveal|give me|show me|write)(?: the)? (?:ideal )?(?:answer|solution|rubric|code)(?: and (?:the )?(?:ideal )?(?:answer|solution|rubric|code))*(?: please)?",
        CandidateIntent.COACHING_OR_ANSWER_REQUEST,
        SafeRoute.COACHING_BOUNDARY,
        "request interview answer or rubric",
        NO_COACHING_BOUNDARY_RESPONSE,
    ),
    _rule(
        r"(?:please )?(?:ignore .+ and )?(?:show|reveal) (?:the )?(?:hidden )?(?:prompt|system prompt|system instructions|tools?)(?:.*)?",
        CandidateIntent.COACHING_OR_ANSWER_REQUEST,
        SafeRoute.COACHING_BOUNDARY,
        "request hidden interview controls",
        NO_COACHING_BOUNDARY_RESPONSE,
    ),
    _rule(
        r"(?:(?:could|can|would) you |please )?(?:use (?:your )?(?:tools?|system) to )?(?:give|provide) me (?:(?:an?|the) )?(?:hint|answer|solution|rubric|code)(?:.*)?",
        CandidateIntent.COACHING_OR_ANSWER_REQUEST,
        SafeRoute.COACHING_BOUNDARY,
        "request interview answer or hint",
        NO_COACHING_BOUNDARY_RESPONSE,
    ),
    _rule(
        r"(?:i would like to )?(?:withdraw(?: from (?:this )?interview)?|end this interview|do not want to continue)(?: please)?",
        CandidateIntent.CANDIDATE_WITHDRAWAL,
        SafeRoute.HANDLE_WITHDRAWAL,
        "withdraw from interview",
    ),
    _rule(
        r"(?:are you (?:an ai|human)(?: or (?:an? )?human(?: interviewer)?)?|who are you|are you an ai interviewer)",
        CandidateIntent.IDENTITY_QUESTION,
        SafeRoute.IDENTITY_BOUNDARY,
        "ask interviewer identity",
        IDENTITY_BOUNDARY_RESPONSE,
    ),
    _rule(
        r"(?:sorry,? )?(?:i )?(?:cannot|can't) hear (?:you|anything)(?:.*)?",
        CandidateIntent.AUDIO_PROBLEM,
        SafeRoute.AUDIO_RECOVERY,
        "report audio problem",
    ),
    _rule(
        r"(?:could|can|would) you (?:please )?(?:repeat|say) (?:the )?(?:question|that|it)(?: again)?",
        CandidateIntent.REPEAT_REQUEST,
        SafeRoute.REPEAT_CURRENT_QUESTION,
        "repeat current question",
    ),
    _rule(
        r"(?:what do you mean by .+|(?:could|can) you (?:please )?clarify (?:the )?(?:question|that))",
        CandidateIntent.CLARIFICATION_REQUEST,
        SafeRoute.SAFE_CLARIFICATION,
        "clarify question wording",
        SAFE_CLARIFICATION_RESPONSE,
    ),
    _rule(
        r"(?:please )?(?:let me think(?: about that)?(?: for a moment)?|give me a moment|i need a moment)(?: please)?",
        CandidateIntent.THINKING_TIME_REQUEST,
        SafeRoute.EXTEND_THINKING_TIME,
        "request thinking time",
    ),
    _rule(
        r"(?:can we )?(?:please )?(?:skip this(?: and return to it later)?|return to (?:this|it) later|come back to (?:this|it) later)(?: please)?",
        CandidateIntent.SKIP_OR_RETURN_LATER_REQUEST,
        SafeRoute.DEFER_CURRENT_QUESTION,
        "defer current question",
    ),
    _rule(
        r"(?:actually,? )?(?:i need to )?correct (?:my )?(?:previous )?answer(?:.*)?",
        CandidateIntent.CORRECTION_TO_PRIOR_ANSWER,
        SafeRoute.HANDLE_CORRECTION,
        "correct prior answer",
    ),
    _rule(
        r"(?:what are )?(?:the )?(?:next steps|interview process|how long is the interview)(?:.*)?",
        CandidateIntent.INTERVIEW_PROCESS_QUESTION,
        SafeRoute.ANSWER_INTERVIEW_PROCESS,
        "ask interview process",
    ),
    _rule(
        r"(?:what would i be working on|what are the (?:role )?responsibilities|can you describe the job)(?:.*)?",
        CandidateIntent.JOB_DESCRIPTION_QUESTION,
        SafeRoute.ANSWER_JOB_DESCRIPTION,
        "ask job description",
    ),
    _rule(
        r"(?:please )?(?:tell me a joke|what(?:'s| is) the weather|what(?:'s| is) the sports score)(?:.*)?",
        CandidateIntent.OFF_TOPIC,
        SafeRoute.OFF_TOPIC_BOUNDARY,
        "ask unrelated question",
        OFF_TOPIC_BOUNDARY_RESPONSE,
    ),
    _rule(
        r"(?:that is all|that's all|thats all|done with my answer)(?:.*)?",
        CandidateIntent.ANSWER_COMPLETE,
        SafeRoute.COMPLETE_TURN,
        "complete answer",
    ),
)


_SAFE_ROUTE_BY_INTENT = {
    CandidateIntent.ANSWER_CONTENT: SafeRoute.CONTINUE_LISTENING,
    CandidateIntent.ANSWER_CONTINUATION: SafeRoute.CONTINUE_LISTENING,
    CandidateIntent.ANSWER_COMPLETE: SafeRoute.COMPLETE_TURN,
    CandidateIntent.REPEAT_REQUEST: SafeRoute.REPEAT_CURRENT_QUESTION,
    CandidateIntent.CLARIFICATION_REQUEST: SafeRoute.SAFE_CLARIFICATION,
    CandidateIntent.AUDIO_PROBLEM: SafeRoute.AUDIO_RECOVERY,
    CandidateIntent.THINKING_TIME_REQUEST: SafeRoute.EXTEND_THINKING_TIME,
    CandidateIntent.SKIP_OR_RETURN_LATER_REQUEST: SafeRoute.DEFER_CURRENT_QUESTION,
    CandidateIntent.CORRECTION_TO_PRIOR_ANSWER: SafeRoute.HANDLE_CORRECTION,
    CandidateIntent.JOB_DESCRIPTION_QUESTION: SafeRoute.ANSWER_JOB_DESCRIPTION,
    CandidateIntent.INTERVIEW_PROCESS_QUESTION: SafeRoute.ANSWER_INTERVIEW_PROCESS,
    CandidateIntent.IDENTITY_QUESTION: SafeRoute.IDENTITY_BOUNDARY,
    CandidateIntent.COACHING_OR_ANSWER_REQUEST: SafeRoute.COACHING_BOUNDARY,
    CandidateIntent.OFF_TOPIC: SafeRoute.OFF_TOPIC_BOUNDARY,
    CandidateIntent.CANDIDATE_WITHDRAWAL: SafeRoute.HANDLE_WITHDRAWAL,
    CandidateIntent.UNKNOWN: SafeRoute.CONTINUE_LISTENING,
}


def _boundary_response(intent: CandidateIntent) -> str | None:
    return {
        CandidateIntent.IDENTITY_QUESTION: IDENTITY_BOUNDARY_RESPONSE,
        CandidateIntent.COACHING_OR_ANSWER_REQUEST: NO_COACHING_BOUNDARY_RESPONSE,
        CandidateIntent.OFF_TOPIC: OFF_TOPIC_BOUNDARY_RESPONSE,
        CandidateIntent.CLARIFICATION_REQUEST: SAFE_CLARIFICATION_RESPONSE,
    }.get(intent)


__all__ = [
    "IntentRouter",
    "LocalStructuredIntentProvider",
    "MAX_CLASSIFICATION_TRANSCRIPT",
    "RoutedIntent",
    "apply_routed_intent",
]
