from __future__ import annotations

import asyncio
import struct
from collections.abc import AsyncIterator

import pytest

from tts.audio_output import (
    PCMPlaybackError,
    PCMPlaybackSession,
    SoundDeviceOutputBackend,
    play_pcm_stream,
)
from tts.schemas import SpeechPCMChunk


class FakeOutputStream:
    def __init__(self) -> None:
        self.writes: list[bytes] = []
        self.closed = False

    def write(self, pcm: bytes) -> None:
        self.writes.append(pcm)

    def close(self) -> None:
        self.closed = True


class FakeRawOutputStream(FakeOutputStream):
    def __init__(self, **options: object) -> None:
        super().__init__()
        self.options = options
        self.started = False

    def start(self) -> None:
        self.started = True

    def stop(self) -> None:
        pass


class FakeSoundDevice:
    def __init__(self) -> None:
        self.streams: list[FakeRawOutputStream] = []

    def query_devices(self) -> list[dict[str, object]]:
        return [
            {
                "name": "Mac mini Speakers",
                "max_input_channels": 0,
                "max_output_channels": 2,
            },
            {
                "name": "INTERVIEWER_TO_CALL",
                "max_input_channels": 2,
                "max_output_channels": 2,
            },
            {
                "name": "CANDIDATE_ONLY",
                "max_input_channels": 2,
                "max_output_channels": 0,
            },
        ]

    def RawOutputStream(self, **options: object) -> FakeRawOutputStream:  # noqa: N802
        stream = FakeRawOutputStream(**options)
        self.streams.append(stream)
        return stream


def test_pcm_playback_converts_qwen_pcm_to_loopback_format_and_closes() -> None:
    output = FakeOutputStream()
    playback = PCMPlaybackSession(output)

    accepted = playback.write(
        SpeechPCMChunk(
            audio=struct.pack("<hh", 1_000, -2_000),
            sample_rate=24_000,
            duration_seconds=2 / 24_000,
        )
    )
    playback.close()

    assert accepted is True
    assert output.writes == [
        struct.pack(
            "<hhhhhhhh", 1_000, 1_000, 1_000, 1_000, -2_000, -2_000, -2_000, -2_000
        )
    ]
    assert output.closed is True


def test_pcm_playback_cancellation_stops_later_qwen_chunks() -> None:
    first_written = asyncio.Event()
    release_second = asyncio.Event()

    async def chunks() -> AsyncIterator[SpeechPCMChunk]:
        yield SpeechPCMChunk(
            audio=struct.pack("<h", 10),
            sample_rate=24_000,
            duration_seconds=1 / 24_000,
        )
        await release_second.wait()
        yield SpeechPCMChunk(
            audio=struct.pack("<h", 20),
            sample_rate=24_000,
            duration_seconds=1 / 24_000,
        )

    class SignallingOutputStream(FakeOutputStream):
        def write(self, pcm: bytes) -> None:
            super().write(pcm)
            first_written.set()

    async def run() -> None:
        output = SignallingOutputStream()
        playback = PCMPlaybackSession(output)
        task = asyncio.create_task(play_pcm_stream(chunks(), playback))

        await first_written.wait()
        assert len(output.writes) == 1
        playback.cancel()
        release_second.set()

        result = await task
        assert result.chunk_count == 1
        assert result.cancelled is True
        assert output.closed is True

    asyncio.run(run())


def test_sounddevice_backend_opens_only_the_exact_named_output_device() -> None:
    sounddevice = FakeSoundDevice()
    backend = SoundDeviceOutputBackend(sounddevice)

    output = backend.open_output("INTERVIEWER_TO_CALL")
    output.close()

    assert sounddevice.streams[0].started is True
    assert sounddevice.streams[0].closed is True
    assert sounddevice.streams[0].options == {
        "device": 1,
        "samplerate": 48_000,
        "channels": 2,
        "dtype": "int16",
    }
    assert backend.list_input_devices() == ("INTERVIEWER_TO_CALL", "CANDIDATE_ONLY")


def test_sounddevice_backend_rejects_a_missing_or_input_only_output_device() -> None:
    backend = SoundDeviceOutputBackend(FakeSoundDevice())

    with pytest.raises(PCMPlaybackError, match="exact output device"):
        backend.open_output("CANDIDATE_ONLY")
