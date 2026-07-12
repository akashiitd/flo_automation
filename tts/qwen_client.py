"""Async client for the loopback-only Qwen voice service."""

from __future__ import annotations

import math
from urllib.parse import urlparse

import httpx

from tts.schemas import SpeechAudio


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

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()


__all__ = ["QwenTTSClient", "QwenTTSError", "SpeechAudio"]
