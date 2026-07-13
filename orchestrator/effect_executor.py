"""Supervised executor for the narrowly typed Qwen/audio effects."""

from __future__ import annotations

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Literal, Protocol

from orchestrator.effect_ledger import EffectLedger
from orchestrator.effects import EffectRequest, EffectResult, EffectStatus, EffectType
from orchestrator.event_adapters import EventNormalizer
from orchestrator.events import InterviewEvent
from tts.audio_output import (
    AudioOutputBackend,
    PCMPlaybackSession,
    PlaybackBargeInController,
    PlaybackResult,
    play_pcm_stream,
)
from tts.schemas import SpeechPCMChunk


class QwenStreamingClient(Protocol):
    """The small local-only streaming seam required by the executor."""

    def stream_synthesize(self, text: str) -> AsyncIterator[SpeechPCMChunk]: ...


@dataclass(frozen=True, slots=True)
class EffectExecution:
    """One ledger outcome plus the graph events an executor callback should ingest."""

    result: EffectResult
    events: tuple[InterviewEvent, ...]


class QwenEffectExecutor:
    """Execute only prepared graph effects after explicit supervised construction.

    It deliberately has no graph reference. The caller must feed ``events`` back
    through normal event ingress, preserving checkpointed routing and allowing
    restart reconciliation without rerunning a LangGraph node.
    """

    def __init__(
        self,
        *,
        ledger: EffectLedger,
        normalizer: EventNormalizer,
        qwen: QwenStreamingClient,
        audio_backend: AudioOutputBackend,
        output_device: str,
        barge_in: PlaybackBargeInController | None = None,
        supervised: bool = False,
        allow_offline_test_effects: bool = False,
    ) -> None:
        if not supervised:
            raise ValueError("Qwen effect execution requires explicit supervision")
        if not output_device.strip():
            raise ValueError("output_device must not be empty")
        self._ledger = ledger
        self._normalizer = normalizer
        self._qwen = qwen
        self._audio_backend = audio_backend
        self._output_device = output_device
        self._barge_in = barge_in
        self._allow_offline_test_effects = allow_offline_test_effects

    async def execute(self, request: EffectRequest) -> EffectExecution:
        """Run one prepared Qwen/audio effect exactly once per idempotency key."""

        if request.effect_type not in {
            EffectType.SPEAK_TEXT,
            EffectType.CHECK_AUDIO_ROUTE,
        }:
            raise ValueError(f"unsupported Phase 8 effect: {request.effect_type}")
        if (
            request.payload.get("offline_only") is True
            and not self._allow_offline_test_effects
        ):
            raise ValueError(
                "offline-only graph effects cannot reach the Qwen executor"
            )

        entry = self._ledger.prepare(request)
        if entry.result.status in {
            EffectStatus.COMPLETED,
            EffectStatus.CANCELLED,
            EffectStatus.FAILED,
            EffectStatus.UNCERTAIN,
        }:
            return EffectExecution(
                entry.result, self._events_for_result(request, entry.result)
            )
        if entry.result.status is EffectStatus.STARTED:
            return EffectExecution(entry.result, ())

        started_entry, claimed = self._ledger.claim_start(request)
        if not claimed:
            return EffectExecution(
                started_entry.result,
                self._events_for_result(request, started_entry.result),
            )
        started = started_entry.result
        events: list[InterviewEvent] = []
        if request.effect_type is EffectType.SPEAK_TEXT:
            events.append(
                self._normalizer.tts_result(
                    effect_id=request.effect_id,
                    outcome="started",
                    question_id=request.question_id,
                    result_summary=started.result_summary,
                )
            )
        try:
            if request.effect_type is EffectType.SPEAK_TEXT:
                playback = await self._play(request)
                status = (
                    EffectStatus.CANCELLED
                    if playback.cancelled
                    else EffectStatus.COMPLETED
                )
                summary = (
                    "candidate barge-in cancelled playback"
                    if playback.cancelled
                    else f"played {playback.chunk_count} PCM chunks"
                )
                result = self._ledger.transition(
                    request, status=status, result_summary=summary
                ).result
                events.append(
                    self._normalizer.tts_result(
                        effect_id=request.effect_id,
                        outcome="cancelled" if playback.cancelled else "completed",
                        question_id=request.question_id,
                        result_summary=result.result_summary,
                    )
                )
            else:
                result = self._check_audio_route(request)
                events.append(
                    self._normalizer.audio_route_result(
                        effect_id=request.effect_id,
                        outcome="completed",
                        question_id=request.question_id,
                        result_summary=result.result_summary,
                    )
                )
        except Exception as error:
            result = self._ledger.transition(
                request,
                status=EffectStatus.FAILED,
                result_summary=f"effect failed: {type(error).__name__}",
            ).result
            if request.effect_type is EffectType.SPEAK_TEXT:
                events.append(
                    self._normalizer.tts_result(
                        effect_id=request.effect_id,
                        outcome="failed",
                        question_id=request.question_id,
                        result_summary=result.result_summary,
                    )
                )
            else:
                events.append(
                    self._normalizer.audio_route_result(
                        effect_id=request.effect_id,
                        outcome="failed",
                        question_id=request.question_id,
                        result_summary=result.result_summary,
                    )
                )
        return EffectExecution(result=result, events=tuple(events))

    def reconcile_after_restart(self, request: EffectRequest) -> EffectExecution:
        """Convert ambiguous in-flight playback into a graph-visible recovery event."""

        entry = self._ledger.reconcile_after_restart(request)
        return EffectExecution(
            entry.result, self._events_for_result(request, entry.result)
        )

    async def _play(self, request: EffectRequest) -> PlaybackResult:
        text = request.payload.get("text")
        if not isinstance(text, str) or not text.strip():
            raise ValueError("SPEAK_TEXT effects require non-empty payload.text")
        playback = PCMPlaybackSession(
            self._audio_backend.open_output(self._output_device)
        )
        return await play_pcm_stream(
            self._qwen.stream_synthesize(text), playback, barge_in=self._barge_in
        )

    def _check_audio_route(self, request: EffectRequest) -> EffectResult:
        if self._output_device not in self._audio_backend.list_output_devices():
            raise RuntimeError("configured output device is unavailable")
        return self._ledger.transition(
            request,
            status=EffectStatus.COMPLETED,
            result_summary="configured audio output device is available",
        ).result

    def _events_for_result(
        self, request: EffectRequest, result: EffectResult
    ) -> tuple[InterviewEvent, ...]:
        """Reissue a stable lifecycle observation for graph/event-ledger recovery."""

        if request.effect_type is EffectType.SPEAK_TEXT:
            outcome_by_status: dict[
                EffectStatus, Literal["started", "completed", "cancelled", "failed"]
            ] = {
                EffectStatus.STARTED: "started",
                EffectStatus.COMPLETED: "completed",
                EffectStatus.CANCELLED: "cancelled",
                EffectStatus.FAILED: "failed",
                EffectStatus.UNCERTAIN: "failed",
            }
            outcome = outcome_by_status.get(result.status)
            if outcome is None:
                return ()
            return (
                self._normalizer.tts_result(
                    effect_id=request.effect_id,
                    outcome=outcome,
                    question_id=request.question_id,
                    result_summary=result.result_summary,
                    result_status=(
                        EffectStatus.UNCERTAIN
                        if result.status is EffectStatus.UNCERTAIN
                        else None
                    ),
                ),
            )
        if request.effect_type is EffectType.CHECK_AUDIO_ROUTE and result.status in {
            EffectStatus.COMPLETED,
            EffectStatus.FAILED,
        }:
            return (
                self._normalizer.audio_route_result(
                    effect_id=request.effect_id,
                    outcome=(
                        "completed"
                        if result.status is EffectStatus.COMPLETED
                        else "failed"
                    ),
                    question_id=request.question_id,
                    result_summary=result.result_summary,
                ),
            )
        return ()


__all__ = ["EffectExecution", "QwenEffectExecutor", "QwenStreamingClient"]
