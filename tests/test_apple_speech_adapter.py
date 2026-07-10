from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from types import SimpleNamespace

from app.config import Settings
from transcriber.apple_speech_adapter import AppleSpeechAdapter


def test_adapter_uses_system_audio_only_and_saves_callbacks(tmp_path: Path) -> None:
    transcriber_root = tmp_path / "Meeting_transcriber_with_LLM"
    (transcriber_root / "src").mkdir(parents=True)
    (transcriber_root / "src" / "realtime_transcriber.py").touch()
    captured_options: dict[str, object] = {}

    class FakeTranscriber:
        def start(self) -> bool:
            callback = captured_options["callback"]
            callback(
                SimpleNamespace(
                    text="Candidate system audio answer",
                    start_time=0.5,
                    end_time=2.0,
                    speaker="Other",
                    source="system",
                    confidence=0.94,
                    timestamp=datetime(2026, 7, 10, 12, 0, tzinfo=UTC),
                )
            )
            return True

        def stop(self) -> list[object]:
            return []

    def factory(**options: object) -> tuple[FakeTranscriber, None]:
        captured_options.update(options)
        return FakeTranscriber(), None

    settings = Settings.load(
        project_root=tmp_path,
        environ={
            "MEETING_TRANSCRIBER_PATH": str(transcriber_root),
            "RUNS_DIR": str(tmp_path / "runs"),
            "TRANSCRIPTION_BACKEND": "apple-speech",
            "TRANSCRIBE_SYSTEM_AUDIO": "true",
            "TRANSCRIBE_MICROPHONE": "false",
        },
    )
    adapter = AppleSpeechAdapter(
        settings,
        session_id="listen_test",
        transcriber_factory=factory,
    )

    assert adapter.start() is True
    summary = adapter.stop()

    assert captured_options["enable_system_audio"] is True
    assert captured_options["enable_microphone"] is False
    assert captured_options["transcription_backend"] == "apple-speech"
    assert captured_options["enable_live_logging"] is False
    assert summary.segment_count == 1
    assert summary.json_path.is_file()
    assert summary.text_path.read_text(encoding="utf-8").endswith(
        "[Other]: Candidate system audio answer\n"
    )
