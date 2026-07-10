from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from transcriber.transcript_store import TranscriptStore


def test_store_persists_system_audio_segments_immediately(tmp_path: Path) -> None:
    store = TranscriptStore(tmp_path / "runs", "listen_test")
    store.append(
        SimpleNamespace(
            text="System audio validation phrase",
            start_time=1.25,
            end_time=3.5,
            speaker="Other",
            source="system",
            confidence=0.91,
            timestamp=datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
        )
    )

    payload = json.loads(store.json_path.read_text(encoding="utf-8"))
    assert payload["session_id"] == "listen_test"
    assert payload["segments"] == [
        {
            "text": "System audio validation phrase",
            "start_time": 1.25,
            "end_time": 3.5,
            "speaker": "Other",
            "source": "system",
            "confidence": 0.91,
            "timestamp": "2026-07-10T12:00:00+00:00",
        }
    ]
    assert "[Other]: System audio validation phrase" in store.text_path.read_text(
        encoding="utf-8"
    )
    assert store.segment_count == 1
