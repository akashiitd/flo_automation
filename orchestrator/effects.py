"""Typed requests and recorded outcomes for idempotent external work."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Literal

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    JsonValue,
    field_validator,
    model_validator,
)

MAX_EFFECT_PAYLOAD_BYTES = 16_000


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
    question_id: int | None = Field(default=None, ge=1)
    payload: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def payload_hash(self) -> str:
        """Stable payload identity used to reconcile a recorded effect result."""

        payload_json = json.dumps(
            self.payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
        )
        return hashlib.sha256(payload_json.encode("utf-8")).hexdigest()

    @field_validator("payload")
    @classmethod
    def payload_must_fit_the_checkpoint_window(
        cls, value: dict[str, JsonValue]
    ) -> dict[str, JsonValue]:
        if len(json.dumps(value, separators=(",", ":")).encode("utf-8")) > (
            MAX_EFFECT_PAYLOAD_BYTES
        ):
            raise ValueError("payload exceeds the effect checkpoint size limit")
        return value


class EffectResult(BaseModel):
    """The append-only execution result used by recovery policy."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)

    schema_version: Literal[1] = 1
    effect_id: str = Field(min_length=1)
    session_id: str = Field(min_length=1)
    effect_type: EffectType
    idempotency_key: str = Field(min_length=1)
    payload_hash: str = Field(pattern=r"^[0-9a-f]{64}$")
    status: EffectStatus
    result_summary: str = Field(min_length=1, max_length=4_000)
    started_at: datetime | None = None
    completed_at: datetime | None = None

    @model_validator(mode="after")
    def lifecycle_timestamps_must_match_status(self) -> EffectResult:
        if self.status is EffectStatus.STARTED and self.started_at is None:
            raise ValueError("started effects require started_at")
        if self.status is EffectStatus.COMPLETED and self.completed_at is None:
            raise ValueError("completed effects require completed_at")
        return self


__all__ = ["EffectRequest", "EffectResult", "EffectStatus", "EffectType"]
