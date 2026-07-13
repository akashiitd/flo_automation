"""Typed requests and recorded outcomes for idempotent external work."""

from __future__ import annotations

from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, JsonValue


class EffectType(StrEnum):
    """External operations emitted by the graph but executed outside it."""

    SPEAK_TEXT = "SPEAK_TEXT"
    BROWSER_ACTION = "BROWSER_ACTION"
    EVALUATE_ANSWER = "EVALUATE_ANSWER"
    APPEND_TRANSCRIPT = "APPEND_TRANSCRIPT"
    CHECK_AUDIO_ROUTE = "CHECK_AUDIO_ROUTE"


class EffectStatus(StrEnum):
    """Durable lifecycle status for a requested external operation."""

    PREPARED = "PREPARED"
    STARTED = "STARTED"
    COMPLETED = "COMPLETED"
    FAILED = "FAILED"
    UNCERTAIN = "UNCERTAIN"


class EffectRequest(BaseModel):
    """An idempotent request ready for the separately supervised executor."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    effect_id: str = Field(min_length=1)
    effect_type: EffectType
    idempotency_key: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)


class EffectResult(BaseModel):
    """The append-only execution result used by recovery policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    effect_id: str = Field(min_length=1)
    status: EffectStatus
    result_summary: str = Field(min_length=1)
    started_at: datetime | None = None
    completed_at: datetime | None = None


__all__ = ["EffectRequest", "EffectResult", "EffectStatus", "EffectType"]
