"""Loopback-only, persistent Qwen3-TTS voice-cloning HTTP service.

Run this module with the Python environment that contains ``mlx-audio``.  The
FloCareer application itself talks to it through a small HTTP client and never
imports MLX or model weights into its normal Python environment.
"""

from __future__ import annotations

import argparse
import base64
import importlib
import io
import json
import os
import threading
import wave
from collections.abc import Iterator, Sequence
from http import HTTPStatus
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Protocol

from tts.schemas import SpeechAudio, SpeechPCMChunk


MAX_REQUEST_BYTES = 16_384
MAX_INPUT_CHARACTERS = 1_200


class SpeechEngine(Protocol):
    def synthesize(self, text: str) -> SpeechAudio: ...

    def stream_synthesize(self, text: str) -> Iterator[SpeechPCMChunk]: ...


class QwenSpeechEngine:
    """One loaded Qwen model with a reusable private voice reference."""

    def __init__(
        self,
        *,
        model_id: str,
        reference_audio: str,
        reference_text: str,
        language: str = "English",
    ) -> None:
        if not reference_audio:
            raise ValueError("QWEN_TTS_REFERENCE_AUDIO is required")
        if not reference_text.strip():
            raise ValueError("QWEN_TTS_REFERENCE_TEXT is required")

        self._np = importlib.import_module("numpy")
        tts_utils = importlib.import_module("mlx_audio.tts.utils")
        self._model = tts_utils.load_model(model_id)
        self._reference_audio = reference_audio
        self._reference_text = reference_text
        self._language = language
        self._lock = threading.Lock()

    def synthesize(self, text: str) -> SpeechAudio:
        with self._lock:
            results = list(
                self._model.generate(
                    text=text,
                    ref_audio=self._reference_audio,
                    ref_text=self._reference_text,
                    lang_code=self._language,
                    verbose=False,
                )
            )
        if not results:
            raise RuntimeError("Qwen generated no audio")

        sample_rate = results[0].sample_rate
        samples = self._np.concatenate(
            [
                self._np.asarray(result.audio, dtype=self._np.float32)
                for result in results
            ]
        )
        duration_seconds = len(samples) / sample_rate
        pcm = self._np.clip(samples, -1, 1)
        pcm = (pcm * 32767).astype("<i2").tobytes()
        buffer = io.BytesIO()
        with wave.open(buffer, "wb") as output:
            output.setnchannels(1)
            output.setsampwidth(2)
            output.setframerate(sample_rate)
            output.writeframes(pcm)
        return SpeechAudio(buffer.getvalue(), duration_seconds)

    def stream_synthesize(self, text: str) -> Iterator[SpeechPCMChunk]:
        with self._lock:
            for result in self._model.generate(
                text=text,
                ref_audio=self._reference_audio,
                ref_text=self._reference_text,
                lang_code=self._language,
                stream=True,
                streaming_interval=0.32,
                verbose=False,
            ):
                samples = self._np.asarray(result.audio, dtype=self._np.float32)
                if not len(samples):
                    continue
                pcm = self._np.clip(samples, -1, 1)
                pcm = (pcm * 32767).astype("<i2").tobytes()
                yield SpeechPCMChunk(
                    audio=pcm,
                    sample_rate=result.sample_rate,
                    duration_seconds=len(samples) / result.sample_rate,
                )


class _RequestHandler(BaseHTTPRequestHandler):
    speech_engine: SpeechEngine
    protocol_version = "HTTP/1.1"

    def log_message(self, format: str, *args: object) -> None:
        """Avoid recording interview text in the HTTP access log."""

    def _write_json(self, status: HTTPStatus, payload: dict[str, object]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler convention.
        if self.path != "/health":
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        self._write_json(HTTPStatus.OK, {"status": "ok"})

    def do_POST(self) -> None:  # noqa: N802 - BaseHTTPRequestHandler convention.
        if self.path not in {"/v1/audio/speech", "/v1/audio/speech/stream"}:
            self._write_json(HTTPStatus.NOT_FOUND, {"error": "not found"})
            return
        text = self._read_speech_text()
        if text is None:
            return
        if self.path == "/v1/audio/speech/stream":
            self._stream_speech(text)
            return
        try:
            speech = self.speech_engine.synthesize(text)
        except Exception:
            self._write_json(
                HTTPStatus.INTERNAL_SERVER_ERROR, {"error": "speech synthesis failed"}
            )
            return
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "audio/wav")
        self.send_header("Content-Length", str(len(speech.audio)))
        self.send_header("X-Audio-Duration", f"{speech.duration_seconds:.3f}")
        self.end_headers()
        self.wfile.write(speech.audio)

    def _read_speech_text(self) -> str | None:
        content_length = self.headers.get("Content-Length")
        try:
            length = int(content_length or "")
        except ValueError:
            self._write_json(
                HTTPStatus.BAD_REQUEST, {"error": "invalid content length"}
            )
            return
        if length < 1 or length > MAX_REQUEST_BYTES:
            self._write_json(
                HTTPStatus.BAD_REQUEST, {"error": "invalid content length"}
            )
            return
        try:
            payload = json.loads(self.rfile.read(length))
        except (UnicodeDecodeError, json.JSONDecodeError):
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "invalid JSON"})
            return
        text = payload.get("input") if isinstance(payload, dict) else None
        if not isinstance(text, str) or not text.strip():
            self._write_json(
                HTTPStatus.BAD_REQUEST, {"error": "input must be a non-empty string"}
            )
            return
        if len(text) > MAX_INPUT_CHARACTERS:
            self._write_json(HTTPStatus.BAD_REQUEST, {"error": "input is too long"})
            return None
        return text.strip()

    def _write_chunk(self, payload: dict[str, object]) -> None:
        data = json.dumps(payload, separators=(",", ":")).encode("utf-8") + b"\n"
        self.wfile.write(f"{len(data):X}\r\n".encode("ascii"))
        self.wfile.write(data)
        self.wfile.write(b"\r\n")
        self.wfile.flush()

    def _stream_speech(self, text: str) -> None:
        self.send_response(HTTPStatus.OK)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Transfer-Encoding", "chunked")
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        try:
            for speech in self.speech_engine.stream_synthesize(text):
                self._write_chunk(
                    {
                        "type": "audio",
                        "audio_b64": base64.b64encode(speech.audio).decode("ascii"),
                        "sample_rate": speech.sample_rate,
                        "duration_seconds": speech.duration_seconds,
                    }
                )
        except Exception:
            self._write_chunk({"type": "error"})
        else:
            self._write_chunk({"type": "end"})
        finally:
            self.wfile.write(b"0\r\n\r\n")
            self.wfile.flush()


def create_server(engine: SpeechEngine, *, host: str, port: int) -> ThreadingHTTPServer:
    """Create a loopback HTTP service without loading Qwen in tests."""

    if host != "127.0.0.1":
        raise ValueError("Qwen TTS must bind to 127.0.0.1 only")
    handler = type(
        "QwenSpeechRequestHandler", (_RequestHandler,), {"speech_engine": engine}
    )
    return ThreadingHTTPServer((host, port), handler)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Persistent local Qwen3-TTS service")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7789)
    parser.add_argument(
        "--model",
        default=os.environ.get(
            "QWEN_TTS_MODEL", "mlx-community/Qwen3-TTS-12Hz-0.6B-Base-bf16"
        ),
    )
    parser.add_argument(
        "--reference-audio", default=os.environ.get("QWEN_TTS_REFERENCE_AUDIO", "")
    )
    parser.add_argument(
        "--reference-text", default=os.environ.get("QWEN_TTS_REFERENCE_TEXT", "")
    )
    parser.add_argument(
        "--reference-text-file",
        default=os.environ.get("QWEN_TTS_REFERENCE_TEXT_FILE", ""),
        help="private local file containing the exact reference transcript",
    )
    parser.add_argument(
        "--language", default=os.environ.get("QWEN_TTS_LANGUAGE", "English")
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.host != "127.0.0.1":
        raise ValueError("Qwen TTS must bind to 127.0.0.1 only")
    reference_text = args.reference_text
    if args.reference_text_file:
        reference_text = Path(args.reference_text_file).read_text(encoding="utf-8")
    engine = QwenSpeechEngine(
        model_id=args.model,
        reference_audio=args.reference_audio,
        reference_text=reference_text,
        language=args.language,
    )
    server = create_server(engine, host=args.host, port=args.port)
    print(f"Qwen TTS listening on http://{args.host}:{args.port}", flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        return 0
    finally:
        server.server_close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
