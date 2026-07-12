"""Local, cancellable playback for Qwen's streamed PCM speech."""

from __future__ import annotations

import struct
from collections.abc import AsyncIterable
from dataclasses import dataclass
from typing import Any, Protocol

from tts.schemas import SpeechPCMChunk


QWEN_SAMPLE_RATE = 24_000
LOOPBACK_SAMPLE_RATE = 48_000
LOOPBACK_CHANNELS = 2


class PCMPlaybackError(RuntimeError):
    """A local audio output stream cannot safely play Qwen PCM."""


class PCMOutputStream(Protocol):
    """Minimal writable PCM stream used by a playback session."""

    def write(self, pcm: bytes) -> None: ...

    def close(self) -> None: ...


class AudioOutputBackend(Protocol):
    """Discover and open explicitly selected local output devices."""

    def list_output_devices(self) -> tuple[str, ...]: ...

    def open_output(self, device_name: str) -> PCMOutputStream: ...


@dataclass(frozen=True, slots=True)
class PlaybackResult:
    """Observable outcome from one streamed playback run."""

    chunk_count: int
    cancelled: bool


def _qwen_pcm_to_loopback_pcm(chunk: SpeechPCMChunk) -> bytes:
    """Convert 24 kHz signed-16 mono Qwen PCM to 48 kHz stereo PCM."""

    if chunk.sample_rate != QWEN_SAMPLE_RATE:
        raise PCMPlaybackError(
            f"Qwen PCM sample rate must be {QWEN_SAMPLE_RATE}, got {chunk.sample_rate}"
        )
    if len(chunk.audio) % 2:
        raise PCMPlaybackError("Qwen PCM must contain complete signed-16 samples")

    output = bytearray(len(chunk.audio) * 4)
    offset = 0
    for (sample,) in struct.iter_unpack("<h", chunk.audio):
        # Duplicate each mono frame for left/right and duplicate it once more to
        # upsample 24 kHz to the Loopback device's 48 kHz transport.
        struct.pack_into("<hhhh", output, offset, sample, sample, sample, sample)
        offset += 8
    return bytes(output)


class PCMPlaybackSession:
    """Write Qwen PCM to one already-open local output stream."""

    def __init__(self, output: PCMOutputStream) -> None:
        self._output = output
        self._cancelled = False
        self._closed = False

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def write(self, chunk: SpeechPCMChunk) -> bool:
        """Play one chunk, returning false when cancellation has taken effect."""

        if self._closed:
            raise PCMPlaybackError("cannot write to a closed playback session")
        if self._cancelled:
            return False
        self._output.write(_qwen_pcm_to_loopback_pcm(chunk))
        return True

    def cancel(self) -> None:
        """Prevent later chunks from reaching the local output device."""

        self._cancelled = True

    def close(self) -> None:
        """Release the local output device exactly once."""

        if not self._closed:
            self._output.close()
            self._closed = True


async def play_pcm_stream(
    chunks: AsyncIterable[SpeechPCMChunk], playback: PCMPlaybackSession
) -> PlaybackResult:
    """Play a Qwen stream chunk-by-chunk and always release its output device."""

    chunk_count = 0
    try:
        async for chunk in chunks:
            if not playback.write(chunk):
                break
            chunk_count += 1
    finally:
        playback.close()
    return PlaybackResult(chunk_count=chunk_count, cancelled=playback.cancelled)


class _SoundDeviceOutputStream:
    """Adapt sounddevice's raw stream to the narrow local playback contract."""

    def __init__(self, stream: Any) -> None:
        self._stream = stream
        self._closed = False

    def write(self, pcm: bytes) -> None:
        self._stream.write(pcm)

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._stream.stop()
        finally:
            self._stream.close()
            self._closed = True


class SoundDeviceOutputBackend:
    """CoreAudio output access through python-sounddevice, selected by exact name."""

    def __init__(self, sounddevice_module: Any | None = None) -> None:
        if sounddevice_module is None:
            try:
                import sounddevice
            except ImportError as error:
                raise PCMPlaybackError(
                    "python-sounddevice is unavailable; run uv sync"
                ) from error
            sounddevice_module = sounddevice
        self._sounddevice = sounddevice_module

    def list_output_devices(self) -> tuple[str, ...]:
        devices = self._sounddevice.query_devices()
        return tuple(
            str(device["name"])
            for device in devices
            if int(device.get("max_output_channels", 0)) > 0
        )

    def list_input_devices(self) -> tuple[str, ...]:
        """Return capture-capable devices for candidate-route diagnostics."""

        devices = self._sounddevice.query_devices()
        return tuple(
            str(device["name"])
            for device in devices
            if int(device.get("max_input_channels", 0)) > 0
        )

    def open_output(self, device_name: str) -> PCMOutputStream:
        normalized = device_name.strip()
        if not normalized:
            raise PCMPlaybackError("audio output device name must not be empty")

        matches = [
            index
            for index, device in enumerate(self._sounddevice.query_devices())
            if device.get("name") == normalized
            and int(device.get("max_output_channels", 0)) >= LOOPBACK_CHANNELS
        ]
        if len(matches) != 1:
            raise PCMPlaybackError(
                f"exact output device {normalized!r} with two channels is unavailable"
            )
        try:
            stream = self._sounddevice.RawOutputStream(
                device=matches[0],
                samplerate=LOOPBACK_SAMPLE_RATE,
                channels=LOOPBACK_CHANNELS,
                dtype="int16",
            )
            stream.start()
        except Exception as error:
            raise PCMPlaybackError(
                f"could not open local output device {normalized!r}: {error}"
            ) from error
        return _SoundDeviceOutputStream(stream)


__all__ = [
    "LOOPBACK_CHANNELS",
    "LOOPBACK_SAMPLE_RATE",
    "AudioOutputBackend",
    "PCMOutputStream",
    "PCMPlaybackError",
    "PCMPlaybackSession",
    "PlaybackResult",
    "QWEN_SAMPLE_RATE",
    "SoundDeviceOutputBackend",
    "play_pcm_stream",
]
