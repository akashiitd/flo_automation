"""Adapter around the existing Meeting Transcriber's Apple Speech backend."""

from __future__ import annotations

import importlib
import logging
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol, cast

from app.config import Settings
from transcriber.transcript_store import TranscriptSegmentLike, TranscriptStore


logger = logging.getLogger(__name__)


class RealtimeTranscriberLike(Protocol):
    def start(self) -> bool: ...

    def stop(self) -> list[object]: ...


TranscriberFactory = Callable[..., tuple[RealtimeTranscriberLike, object | None]]


@dataclass(frozen=True, slots=True)
class ListenSummary:
    session_id: str
    segment_count: int
    json_path: Path
    text_path: Path


class AppleSpeechAdapter:
    """Capture candidate system audio while keeping microphone capture disabled."""

    def __init__(
        self,
        settings: Settings,
        *,
        session_id: str,
        transcriber_factory: TranscriberFactory | None = None,
        on_segment: Callable[[object], None] | None = None,
    ) -> None:
        if not settings.transcribe_system_audio:
            raise ValueError(
                "Apple Speech listen-test requires system audio to be enabled"
            )
        if settings.transcribe_microphone:
            raise ValueError(
                "Apple Speech listen-test requires microphone capture to remain disabled"
            )
        if settings.transcription_backend != "apple-speech":
            raise ValueError("TRANSCRIPTION_BACKEND must be apple-speech")

        self.settings = settings
        self.session_id = session_id
        self.store = TranscriptStore(settings.runs_dir, session_id)
        self._factory = transcriber_factory
        self._on_segment = on_segment
        self._transcriber: RealtimeTranscriberLike | None = None
        self._running = False

    def _load_factory(self) -> TranscriberFactory:
        if self._factory is not None:
            return self._factory

        source_file = (
            self.settings.meeting_transcriber_path / "src" / "realtime_transcriber.py"
        )
        if not source_file.is_file():
            raise FileNotFoundError(
                f"Meeting Transcriber factory not found: {source_file}"
            )

        root = str(self.settings.meeting_transcriber_path)
        if root not in sys.path:
            sys.path.insert(0, root)
        module = importlib.import_module("src.realtime_transcriber")
        factory: Any = getattr(module, "create_realtime_transcriber", None)
        if not callable(factory):
            raise ImportError(
                "src.realtime_transcriber.create_realtime_transcriber is unavailable"
            )
        return factory

    def _handle_segment(self, segment: object) -> None:
        self.store.append(cast(TranscriptSegmentLike, segment))
        if self._on_segment is not None:
            try:
                self._on_segment(segment)
            except Exception:
                logger.exception("Apple Speech segment observer failed")

    def start(self) -> bool:
        if self._running:
            return True

        factory = self._load_factory()
        transcriber, _ = factory(
            model_size="small",
            language="en",
            enable_system_audio=True,
            enable_microphone=False,
            callback=self._handle_segment,
            session_name=self.session_id,
            # TranscriptStore persists directly into runs/<session_id>.
            enable_live_logging=False,
            transcription_backend="apple-speech",
        )
        self._transcriber = transcriber
        self._running = bool(transcriber.start())
        return self._running

    def stop(self) -> ListenSummary:
        if self._transcriber is not None:
            self._transcriber.stop()
        self._running = False
        json_path, text_path = self.store.finalize()
        return ListenSummary(
            session_id=self.session_id,
            segment_count=self.store.segment_count,
            json_path=json_path,
            text_path=text_path,
        )
