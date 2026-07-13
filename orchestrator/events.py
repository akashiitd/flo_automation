"""Typed external observations accepted by the dynamic interview graph."""

from __future__ import annotations

import json
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue, field_validator

MAX_EVENT_PAYLOAD_BYTES = 16_000


class EventSource(StrEnum):
    """Trusted adapter that observed or issued an event."""

    BROWSER = "browser"
    CANDIDATE_ASR = "candidate_asr"
    TTS = "tts"
    TIMER = "timer"
    LLM = "llm"
    OPERATOR = "operator"


class EventType(StrEnum):
    """Closed set of events that can enter an interview session."""

    SESSION_STARTED = "SESSION_STARTED"
    PREFLIGHT_PASSED = "PREFLIGHT_PASSED"
    PREFLIGHT_FAILED = "PREFLIGHT_FAILED"
    JOIN_APPROVED = "JOIN_APPROVED"
    JOINED = "JOINED"
    CANDIDATE_CONNECTED = "CANDIDATE_CONNECTED"
    CANDIDATE_DISCONNECTED = "CANDIDATE_DISCONNECTED"
    CANDIDATE_RECONNECTED = "CANDIDATE_RECONNECTED"
    DISCLOSURE_ACCEPTED = "DISCLOSURE_ACCEPTED"
    DISCLOSURE_DECLINED = "DISCLOSURE_DECLINED"
    TTS_STARTED = "TTS_STARTED"
    TTS_COMPLETED = "TTS_COMPLETED"
    TTS_CANCELLED = "TTS_CANCELLED"
    TTS_FAILED = "TTS_FAILED"
    AUDIO_ROUTE_COMPLETED = "AUDIO_ROUTE_COMPLETED"
    AUDIO_ROUTE_FAILED = "AUDIO_ROUTE_FAILED"
    TRANSCRIPT_PARTIAL = "TRANSCRIPT_PARTIAL"
    TRANSCRIPT_FINAL = "TRANSCRIPT_FINAL"
    SPEECH_STARTED = "SPEECH_STARTED"
    SILENCE_STARTED = "SILENCE_STARTED"
    SILENCE_TIMEOUT = "SILENCE_TIMEOUT"
    TURN_INTENT_CLASSIFIED = "TURN_INTENT_CLASSIFIED"
    TURN_COMPLETE = "TURN_COMPLETE"
    EVALUATION_COMPLETED = "EVALUATION_COMPLETED"
    EVALUATION_FAILED = "EVALUATION_FAILED"
    TIMER_WARNING = "TIMER_WARNING"
    TIME_LIMIT_REACHED = "TIME_LIMIT_REACHED"
    OPERATOR_PAUSE = "OPERATOR_PAUSE"
    OPERATOR_TAKEOVER = "OPERATOR_TAKEOVER"
    OPERATOR_RESUME = "OPERATOR_RESUME"
    OPERATOR_STOP = "OPERATOR_STOP"
    BROWSER_EFFECT_COMPLETED = "BROWSER_EFFECT_COMPLETED"
    BROWSER_EFFECT_FAILED = "BROWSER_EFFECT_FAILED"


class InterviewEvent(BaseModel):
    """One immutable, JSON-safe event accepted by a single interview session."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    event_id: str = Field(min_length=1)
    event_type: EventType
    occurred_at: datetime
    source: EventSource
    session_id: str = Field(min_length=1)
    question_id: int | None = Field(default=None, ge=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @field_validator("occurred_at")
    @classmethod
    def occurred_at_must_be_timezone_aware(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("occurred_at must include a timezone")
        return value

    @field_validator("payload")
    @classmethod
    def payload_must_fit_the_checkpoint_window(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if len(json.dumps(value, separators=(",", ":")).encode("utf-8")) > (
            MAX_EVENT_PAYLOAD_BYTES
        ):
            raise ValueError("payload exceeds the event checkpoint size limit")
        return value


__all__ = [
    "EventSource",
    "EventType",
    "InterviewEvent",
    "MAX_EVENT_PAYLOAD_BYTES",
]
