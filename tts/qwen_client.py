"""Async client for the loopback-only Qwen voice service."""

from __future__ import annotations

import base64
import json
import math
from collections.abc import AsyncIterator
from urllib.parse import urlparse

import httpx

from tts.schemas import SpeechAudio, SpeechPCMChunk


class QwenTTSError(RuntimeError):
    """A local Qwen voice service request failed."""


class QwenTTSClient:
    """Send approved interviewer text to the locally hosted Qwen voice worker."""

    def __init__(
        self,
        base_url: str,
        *,
        timeout_seconds: float = 45,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        if urlparse(self.base_url).hostname != "127.0.0.1":
            raise ValueError("Qwen TTS client requires a 127.0.0.1 URL")
        self.timeout_seconds = timeout_seconds
        self._client = client or httpx.AsyncClient()
        self._owns_client = client is None

    async def synthesize(self, text: str) -> SpeechAudio:
        normalized = text.strip()
        if not normalized:
            raise ValueError("speech text must not be empty")

        try:
            response = await self._client.post(
                f"{self.base_url}/v1/audio/speech",
                json={"input": normalized},
                timeout=self.timeout_seconds,
            )
            response.raise_for_status()
        except httpx.HTTPError as error:
            raise QwenTTSError(f"Qwen speech request failed: {error}") from error

        try:
            duration_seconds = float(response.headers.get("x-audio-duration", "0"))
        except ValueError as error:
            raise QwenTTSError(
                "Qwen speech service returned an invalid duration"
            ) from error
        if not math.isfinite(duration_seconds) or duration_seconds < 0:
            raise QwenTTSError("Qwen speech service returned an invalid duration")
        if not response.content:
            raise QwenTTSError("Qwen speech service returned no audio")
        return SpeechAudio(response.content, duration_seconds)

    async def stream_synthesize(self, text: str) -> AsyncIterator[SpeechPCMChunk]:
        normalized = text.strip()
        if not normalized:
            raise ValueError("speech text must not be empty")

        try:
            async with self._client.stream(
                "POST",
                f"{self.base_url}/v1/audio/speech/stream",
                json={"input": normalized},
                timeout=self.timeout_seconds,
            ) as response:
                response.raise_for_status()
                async for line in response.aiter_lines():
                    if not line:
                        continue
                    event = json.loads(line)
                    if event.get("type") == "end":
                        return
                    if event.get("type") == "error":
                        raise QwenTTSError("Qwen speech stream failed")
                    if event.get("type") != "audio":
                        raise QwenTTSError(
                            "Qwen speech stream returned an invalid event"
                        )
                    audio = base64.b64decode(event["audio_b64"], validate=True)
                    sample_rate = int(event["sample_rate"])
                    duration_seconds = float(event["duration_seconds"])
                    if (
                        not audio
                        or sample_rate <= 0
                        or not math.isfinite(duration_seconds)
                        or duration_seconds <= 0
                    ):
                        raise QwenTTSError(
                            "Qwen speech stream returned an invalid chunk"
                        )
                    yield SpeechPCMChunk(audio, sample_rate, duration_seconds)
        except (
            httpx.HTTPError,
            KeyError,
            TypeError,
            ValueError,
            json.JSONDecodeError,
        ) as error:
            if isinstance(error, QwenTTSError):
                raise
            raise QwenTTSError(f"Qwen speech stream failed: {error}") from error

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


__all__ = ["QwenTTSClient", "QwenTTSError", "SpeechAudio", "SpeechPCMChunk"]
