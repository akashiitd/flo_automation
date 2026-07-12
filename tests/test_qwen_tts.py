from __future__ import annotations

import asyncio
import json
import threading
from collections.abc import AsyncIterator
from dataclasses import dataclass

import httpx
import pytest

from tts.qwen_client import QwenTTSClient, SpeechAudio
from tts.qwen_service import create_server
from tts.speech_bridge import iter_provider_speech, speak_provider_stream


@dataclass
class FakeSpeechEngine:
    calls: list[str]

    def synthesize(self, text: str) -> SpeechAudio:
        self.calls.append(text)
        return SpeechAudio(audio=b"RIFFfake-wav", duration_seconds=1.25)


class LocalServer:
    def __init__(self, engine: FakeSpeechEngine) -> None:
        self.server = create_server(engine, host="127.0.0.1", port=0)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)

    def __enter__(self) -> str:
        self.thread.start()
        host, port = self.server.server_address[:2]
        return f"http://{host}:{port}"

    def __exit__(self, *args: object) -> None:
        self.server.shutdown()
        self.thread.join(timeout=2)
        self.server.server_close()


def test_local_qwen_service_returns_wav_for_valid_speech_request() -> None:
    engine = FakeSpeechEngine(calls=[])

    with LocalServer(engine) as base_url:
        response = httpx.post(
            f"{base_url}/v1/audio/speech",
            json={"input": "Please explain your reasoning."},
            timeout=2,
        )

    assert response.status_code == 200
    assert response.content == b"RIFFfake-wav"
    assert response.headers["content-type"] == "audio/wav"
    assert response.headers["x-audio-duration"] == "1.250"
    assert engine.calls == ["Please explain your reasoning."]


def test_local_qwen_service_rejects_missing_speech_text() -> None:
    engine = FakeSpeechEngine(calls=[])

    with LocalServer(engine) as base_url:
        response = httpx.post(f"{base_url}/v1/audio/speech", json={}, timeout=2)

    assert response.status_code == 400
    assert response.json()["error"] == "input must be a non-empty string"
    assert engine.calls == []


def test_local_qwen_service_factory_rejects_network_binding() -> None:
    with pytest.raises(ValueError, match="127.0.0.1"):
        create_server(FakeSpeechEngine(calls=[]), host="0.0.0.0", port=0)


def test_qwen_client_sends_only_text_to_local_speech_service() -> None:
    requests: list[dict[str, object]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            content=b"RIFFaudio",
            headers={"x-audio-duration": "2.5"},
        )

    async def run() -> SpeechAudio:
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as http:
            client = QwenTTSClient("http://127.0.0.1:7789", client=http)
            return await client.synthesize("What trade-offs would you consider?")

    result = asyncio.run(run())

    assert result.audio == b"RIFFaudio"
    assert result.duration_seconds == 2.5
    assert requests == [{"input": "What trade-offs would you consider?"}]


class FakeProvider:
    async def stream_text(self) -> AsyncIterator[str]:
        yield "First sentence. Second"
        yield " sentence? Third sentence."


class FakeSpeechClient:
    def __init__(self) -> None:
        self.texts: list[str] = []

    async def synthesize(self, text: str) -> SpeechAudio:
        self.texts.append(text)
        return SpeechAudio(audio=text.encode(), duration_seconds=1.0)


def test_lm_stream_is_forwarded_sentence_by_sentence_to_qwen() -> None:
    async def run() -> list[SpeechAudio]:
        return await speak_provider_stream(
            FakeProvider().stream_text(), FakeSpeechClient()
        )

    client = FakeSpeechClient()

    async def run_with_client() -> list[SpeechAudio]:
        return await speak_provider_stream(FakeProvider().stream_text(), client)

    results = asyncio.run(run_with_client())

    assert client.texts == ["First sentence.", "Second sentence?", "Third sentence."]
    assert [result.audio for result in results] == [
        b"First sentence.",
        b"Second sentence?",
        b"Third sentence.",
    ]


def test_first_qwen_audio_chunk_is_available_before_lm_stream_finishes() -> None:
    release_second_chunk = asyncio.Event()

    async def text_chunks() -> AsyncIterator[str]:
        yield "First sentence."
        await release_second_chunk.wait()
        yield "Second sentence."

    async def run() -> None:
        client = FakeSpeechClient()
        speech = iter_provider_speech(text_chunks(), client)

        first = await anext(speech)

        assert first.audio == b"First sentence."
        assert client.texts == ["First sentence."]
        release_second_chunk.set()
        await speech.aclose()

    asyncio.run(run())
