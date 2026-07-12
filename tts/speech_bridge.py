"""Forward completed local-LLM sentences to a local speech service."""

from __future__ import annotations

import re
from collections.abc import AsyncIterable, AsyncIterator
from typing import Protocol

from tts.schemas import SpeechAudio, SpeechPCMChunk


class SpeechClient(Protocol):
    async def synthesize(self, text: str) -> SpeechAudio: ...


class StreamingSpeechClient(Protocol):
    def stream_synthesize(self, text: str) -> AsyncIterator[SpeechPCMChunk]: ...


_SENTENCE_END = re.compile(r"(?<=[.!?])(?:\s+|$)")


def _pop_completed_sentences(buffer: str) -> tuple[list[str], str]:
    """Return complete sentence units while retaining an unfinished tail."""

    completed: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(buffer):
        sentence = buffer[start : match.start() + 1].strip()
        if sentence:
            completed.append(sentence)
        start = match.end()
    return completed, buffer[start:]


async def iter_provider_speech(
    text_chunks: AsyncIterable[str],
    speech_client: SpeechClient,
) -> AsyncIterator[SpeechAudio]:
    """Yield spoken sentences as soon as the local LLM completes each one."""

    pending = ""
    async for chunk in text_chunks:
        pending += chunk
        completed, pending = _pop_completed_sentences(pending)
        for sentence in completed:
            yield await speech_client.synthesize(sentence)
    if pending.strip():
        yield await speech_client.synthesize(pending.strip())


async def iter_provider_pcm(
    text_chunks: AsyncIterable[str],
    speech_client: StreamingSpeechClient,
) -> AsyncIterator[SpeechPCMChunk]:
    """Yield PCM as soon as Qwen creates it for each completed LLM sentence."""

    pending = ""
    async for chunk in text_chunks:
        pending += chunk
        completed, pending = _pop_completed_sentences(pending)
        for sentence in completed:
            async for audio in speech_client.stream_synthesize(sentence):
                yield audio
    if pending.strip():
        async for audio in speech_client.stream_synthesize(pending.strip()):
            yield audio


async def speak_provider_stream(
    text_chunks: AsyncIterable[str],
    speech_client: SpeechClient,
) -> list[SpeechAudio]:
    """Collect speech chunks for callers that do not need streaming playback."""

    return [speech async for speech in iter_provider_speech(text_chunks, speech_client)]
