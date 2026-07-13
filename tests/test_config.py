from __future__ import annotations

from pathlib import Path

import pytest

from app.config import Settings


def test_settings_load_dotenv_and_allow_environment_overrides(tmp_path: Path) -> None:
    (tmp_path / ".env").write_text(
        "LMSTUDIO_BASE_URL=http://from-dotenv.test/v1\n"
        "LLM_FAST_TIMEOUT_SECONDS=7\n"
        "TRANSCRIBE_SYSTEM_AUDIO=false\n",
        encoding="utf-8",
    )

    settings = Settings.load(
        project_root=tmp_path,
        environ={
            "LLM_FAST_TIMEOUT_SECONDS": "3.5",
            "TRANSCRIBE_SYSTEM_AUDIO": "true",
            "QWEN_TTS_BASE_URL": "http://127.0.0.1:7789/",
        },
    )

    assert settings.lmstudio_base_url == "http://from-dotenv.test/v1"
    assert settings.llm_fast_timeout_seconds == 3.5
    assert settings.transcribe_system_audio is True
    assert settings.qwen_tts_base_url == "http://127.0.0.1:7789"
    assert settings.interviewer_audio_output_device == "INTERVIEWER_TO_CALL"
    assert settings.candidate_audio_input_device == "CANDIDATE_ONLY"
    assert settings.flocareer_speaker_output_device == "Jabra Evolve2 65 Flex (Bluetooth)"


def test_settings_reject_non_loopback_qwen_service_url(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="QWEN_TTS_BASE_URL must use 127.0.0.1"):
        Settings.load(
            project_root=tmp_path,
            environ={"QWEN_TTS_BASE_URL": "http://qwen.example.test:7789"},
        )


def test_safe_dump_never_exposes_api_keys(tmp_path: Path) -> None:
    settings = Settings.load(
        project_root=tmp_path,
        environ={
            "LMSTUDIO_API_KEY": "local-secret",
            "OPENROUTER_API_KEY": "cloud-secret",
        },
    )

    rendered = settings.safe_dump()

    assert "local-secret" not in rendered
    assert "cloud-secret" not in rendered
    assert "configured" in rendered
