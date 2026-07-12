"""Shared, provider-neutral audio response contracts."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class SpeechAudio:
    """One synthesized WAV response returned by a local speech engine."""

    audio: bytes
    duration_seconds: float
