"""Thread-safe, per-session persistence for live transcript segments."""

from __future__ import annotations

import json
import re
import threading
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Protocol


_SAFE_SESSION_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


class TranscriptSegmentLike(Protocol):
    text: str
    start_time: float
    end_time: float
    speaker: str | None
    source: str
    confidence: float
    timestamp: datetime


@dataclass(frozen=True, slots=True)
class TranscriptRecord:
    text: str
    start_time: float
    end_time: float
    speaker: str
    source: str
    confidence: float
    timestamp: str
    question_id: int | None = None


class TranscriptStore:
    """Persist every callback before returning control to the audio thread."""

    def __init__(self, runs_dir: Path, session_id: str) -> None:
        if not _SAFE_SESSION_ID.fullmatch(session_id):
            raise ValueError(
                "session_id may contain only letters, numbers, dots, dashes, and underscores"
            )
        self.session_id = session_id
        self.session_dir = runs_dir / session_id
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.json_path = self.session_dir / "transcript.json"
        self.text_path = self.session_dir / "transcript.txt"
        self.started_at = datetime.now(UTC).isoformat()
        self._segments: list[TranscriptRecord] = []
        self._lock = threading.Lock()
        with self._lock:
            self._write_locked()

    @property
    def segment_count(self) -> int:
        with self._lock:
            return len(self._segments)

    def append(
        self, segment: TranscriptSegmentLike, *, question_id: int | None = None
    ) -> None:
        text = str(segment.text).strip()
        if not text:
            return
        timestamp = segment.timestamp
        timestamp_text = (
            timestamp.isoformat()
            if isinstance(timestamp, datetime)
            else str(timestamp or datetime.now(UTC).isoformat())
        )
        record = TranscriptRecord(
            text=text,
            start_time=float(segment.start_time),
            end_time=float(segment.end_time),
            speaker=str(segment.speaker or "Other"),
            source=str(segment.source or "system"),
            confidence=float(segment.confidence or 0.0),
            timestamp=timestamp_text,
            question_id=question_id,
        )
        with self._lock:
            self._segments.append(record)
            self._write_locked()

    def finalize(self) -> tuple[Path, Path]:
        with self._lock:
            self._write_locked()
        return self.json_path, self.text_path

    def _write_locked(self) -> None:
        payload = {
            "session_id": self.session_id,
            "started_at": self.started_at,
            "updated_at": datetime.now(UTC).isoformat(),
            "segments": [
                self._serialize_segment(segment) for segment in self._segments
            ],
        }
        json_temp = self.json_path.with_suffix(".json.tmp")
        json_temp.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        json_temp.replace(self.json_path)

        lines = [
            (
                f"[{segment.start_time:.2f}s - {segment.end_time:.2f}s] "
                f"[{segment.speaker}]: {segment.text}"
            )
            for segment in self._segments
        ]
        text_temp = self.text_path.with_suffix(".txt.tmp")
        text_temp.write_text(
            "\n".join(lines) + ("\n" if lines else ""),
            encoding="utf-8",
        )
        text_temp.replace(self.text_path)

    @staticmethod
    def _serialize_segment(segment: TranscriptRecord) -> dict[str, object]:
        payload = asdict(segment)
        if payload["question_id"] is None:
            del payload["question_id"]
        return payload
