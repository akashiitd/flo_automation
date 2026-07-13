"""Offline trajectories for the Phase 8 Qwen effect boundary."""

from __future__ import annotations

import asyncio
import json
import struct
from collections.abc import AsyncIterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from orchestrator.effect_executor import QwenEffectExecutor
from orchestrator.effect_ledger import EffectLedger, EffectLedgerConflictError
from orchestrator.effects import EffectRequest, EffectStatus, EffectType
from orchestrator.event_adapters import EventNormalizer
from orchestrator.events import EventType
from tts.audio_output import PlaybackBargeInController
from tts.schemas import SpeechPCMChunk


class FakeOutput:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)

    def close(self) -> None:
        self.closed = True


class FakeAudioBackend:
    def __init__(self) -> None:
        self.outputs: list[FakeOutput] = []

    def list_output_devices(self) -> tuple[str, ...]:
        return ("INTERVIEWER_TO_CALL",)

    def open_output(self, device_name: str) -> FakeOutput:
        assert device_name == "INTERVIEWER_TO_CALL"
        output = FakeOutput()
        self.outputs.append(output)
        return output


class FakeQwen:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def stream_synthesize(self, text: str) -> AsyncIterator[SpeechPCMChunk]:
        self.texts.append(text)
        yield SpeechPCMChunk(
            audio=struct.pack("<h", 1),
            sample_rate=24_000,
            duration_seconds=1 / 24_000,
        )


def _request(*, effect_id: str = "speak-1") -> EffectRequest:
    return EffectRequest(
        effect_id=effect_id,
        effect_type=EffectType.SPEAK_TEXT,
        idempotency_key=f"session-1:{effect_id}",
        session_id="session-1",
        question_id=1,
        payload={"text": "Please explain retries."},
    )


def _audio_check_request() -> EffectRequest:
    return EffectRequest(
        effect_id="audio-check-1",
        effect_type=EffectType.CHECK_AUDIO_ROUTE,
        idempotency_key="session-1:audio-check:1",
        session_id="session-1",
        question_id=1,
        payload={"kind": "candidate_audio_recovery"},
    )


def _audio_check_request() -> EffectRequest:
    return EffectRequest(
        effect_id="audio-check-1",
        effect_type=EffectType.CHECK_AUDIO_ROUTE,
        idempotency_key="session-1:audio-check:1",
        session_id="session-1",
        question_id=1,
        payload={"kind": "candidate_audio_recovery"},
    )


def test_effect_ledger_marks_started_playback_uncertain_after_a_restart(
    tmp_path: Path,
) -> None:
    ledger = EffectLedger(tmp_path / "effects.sqlite")
    request = _request()

    assert ledger.prepare(request).result.status is EffectStatus.PREPARED
    assert (
        ledger.transition(
            request,
            status=EffectStatus.STARTED,
            result_summary="playing",
        ).result.status
        is EffectStatus.STARTED
    )

    recovered = ledger.reconcile_after_restart(request)

    assert recovered.result.status is EffectStatus.UNCERTAIN
    assert recovered.result.completed_at is not None
    audit_records = [
        json.loads(line)
        for line in (tmp_path / "effects.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
    ]
    assert [record["result"]["status"] for record in audit_records] == [
        "PREPARED",
        "STARTED",
        "UNCERTAIN",
    ]


def test_only_one_executor_can_claim_a_prepared_effect(tmp_path: Path) -> None:
    path = tmp_path / "effects.sqlite"
    request = _request()
    EffectLedger(path).prepare(request)

    def claim() -> bool:
        _, claimed = EffectLedger(path).claim_start(request)
        return claimed

    with ThreadPoolExecutor(max_workers=2) as executor:
        claims = list(executor.map(lambda _: claim(), range(2)))

    assert claims.count(True) == 1
    assert claims.count(False) == 1


def test_effect_ledger_rejects_idempotency_key_reuse_with_changed_text(
    tmp_path: Path,
) -> None:
    ledger = EffectLedger(tmp_path / "effects.sqlite")
    request = _request()
    ledger.prepare(request)

    with pytest.raises(EffectLedgerConflictError, match="reused with different"):
        ledger.prepare(
            request.model_copy(update={"payload": {"text": "Different speech."}})
        )


def test_executor_streams_once_and_redelivery_returns_the_recorded_result(
    tmp_path: Path,
) -> None:
    async def run() -> tuple[object, object, FakeQwen, FakeAudioBackend]:
        qwen = FakeQwen()
        backend = FakeAudioBackend()
        executor = QwenEffectExecutor(
            ledger=EffectLedger(tmp_path / "effects.sqlite"),
            normalizer=EventNormalizer(session_id="session-1"),
            qwen=qwen,
            audio_backend=backend,
            output_device="INTERVIEWER_TO_CALL",
            supervised=True,
        )
        first = await executor.execute(_request())
        second = await executor.execute(_request())
        return first, second, qwen, backend

    first, second, qwen, backend = asyncio.run(run())

    assert first.result.status is EffectStatus.COMPLETED
    assert [event.event_type for event in first.events] == [
        EventType.TTS_STARTED,
        EventType.TTS_COMPLETED,
    ]
    assert [event.event_type for event in second.events] == [EventType.TTS_COMPLETED]
    assert second.events[0].event_id == first.events[-1].event_id
    assert qwen.texts == ["Please explain retries."]
    assert backend.outputs[0].closed is True


def test_executor_preserves_barge_in_as_a_cancelled_tts_result(tmp_path: Path) -> None:
    class BargeInQwen(FakeQwen):
        async def stream_synthesize(self, text: str) -> AsyncIterator[SpeechPCMChunk]:
            async for chunk in super().stream_synthesize(text):
                yield chunk
            barge_in.on_transcript_segment(
                type("Segment", (), {"source": "system", "text": "answer"})()
            )
            yield SpeechPCMChunk(
                audio=struct.pack("<h", 2),
                sample_rate=24_000,
                duration_seconds=1 / 24_000,
            )

    async def run() -> object:
        executor = QwenEffectExecutor(
            ledger=EffectLedger(tmp_path / "effects.sqlite"),
            normalizer=EventNormalizer(session_id="session-1"),
            qwen=BargeInQwen(),
            audio_backend=FakeAudioBackend(),
            output_device="INTERVIEWER_TO_CALL",
            barge_in=barge_in,
            supervised=True,
        )
        return await executor.execute(_request())

    barge_in = PlaybackBargeInController()
    execution = asyncio.run(run())

    assert execution.result.status is EffectStatus.CANCELLED
    assert execution.events[-1].event_type is EventType.TTS_CANCELLED


def test_restart_reconciliation_emits_one_uncertain_playback_event(
    tmp_path: Path,
) -> None:
    ledger = EffectLedger(tmp_path / "effects.sqlite")
    request = _request()
    ledger.prepare(request)
    ledger.transition(request, status=EffectStatus.STARTED, result_summary="playing")
    executor = QwenEffectExecutor(
        ledger=ledger,
        normalizer=EventNormalizer(session_id="session-1"),
        qwen=FakeQwen(),
        audio_backend=FakeAudioBackend(),
        output_device="INTERVIEWER_TO_CALL",
        supervised=True,
    )

    execution = executor.reconcile_after_restart(request)

    assert execution.result.status is EffectStatus.UNCERTAIN
    assert execution.events[0].event_type is EventType.TTS_FAILED
    assert execution.events[0].payload["result_status"] == "UNCERTAIN"


def test_executor_records_audio_route_before_emitting_a_retry_event(
    tmp_path: Path,
) -> None:
    async def run() -> object:
        executor = QwenEffectExecutor(
            ledger=EffectLedger(tmp_path / "effects.sqlite"),
            normalizer=EventNormalizer(session_id="session-1"),
            qwen=FakeQwen(),
            audio_backend=FakeAudioBackend(),
            output_device="INTERVIEWER_TO_CALL",
            supervised=True,
        )
        return await executor.execute(_audio_check_request())

    execution = asyncio.run(run())

    assert execution.result.status is EffectStatus.COMPLETED
    assert execution.events[0].event_type is EventType.AUDIO_ROUTE_COMPLETED
