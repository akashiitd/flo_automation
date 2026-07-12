from __future__ import annotations

import asyncio
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as cli
from app.config import Settings


def test_timer_demo_reports_synthetic_warnings_without_waiting(
    tmp_path: Path, capsys: object
) -> None:
    exit_code = cli.main(
        ["timer-demo", "--minutes", "1"], project_root=tmp_path, environ={}
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Timer simulation only; no interview is started." in output
    assert "TIME_LIMIT_REACHED" in output


def test_session_commands_are_exposed_as_offline_file_workflows() -> None:
    parser = cli.build_parser()

    assert (
        parser.parse_args(["evaluate", "--session", "runs/example"]).command
        == "evaluate"
    )
    assert (
        parser.parse_args(["simulate-interview", "--session", "runs/example"]).command
        == "simulate-interview"
    )
    assert (
        parser.parse_args(
            [
                "answer-job-question",
                "--session",
                "runs/example",
                "--question",
                "Which technologies are used?",
            ]
        ).command
        == "answer-job-question"
    )
    assert (
        parser.parse_args(
            [
                "answer-job-question",
                "--session",
                "runs/example",
                "--question",
                "What technologies are used?",
                "--speak",
            ]
        ).speak
        is True
    )


def test_barge_in_route_test_requires_an_explicit_loopback_confirmation() -> None:
    parser = cli.build_parser()

    args = parser.parse_args(
        [
            "qwen-tts-barge-in-test",
            "--text",
            "Hello",
            "--confirm-selected-loopback-route",
        ]
    )

    assert args.command == "qwen-tts-barge-in-test"
    assert args.confirm_selected_loopback_route is True


def test_answer_job_question_speaks_only_when_explicitly_requested(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    session = tmp_path / "runs" / "job-question"
    session.mkdir(parents=True)
    (session / "job_description.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "read_only": True,
                "source": "FloCareer Job Description tab",
                "description": "Build RAG pipelines.",
            }
        ),
        encoding="utf-8",
    )
    spoken: list[str] = []

    class FakeProvider:
        async def aclose(self) -> None:
            return None

    async def fake_answer(**kwargs: object) -> object:
        return SimpleNamespace(
            answer=SimpleNamespace(
                answer="Build RAG pipelines.", grounded=True, evidence=["RAG pipelines"]
            )
        )

    async def fake_playback(settings: object, *, text: str) -> object:
        spoken.append(text)
        return SimpleNamespace(chunk_count=1)

    monkeypatch.setattr(cli, "LMStudioProvider", lambda settings: FakeProvider())
    monkeypatch.setattr(cli, "answer_job_description_question", fake_answer)
    monkeypatch.setattr(cli, "_play_qwen_text", fake_playback)
    settings = Settings.load(project_root=tmp_path, environ={})

    exit_code = asyncio.run(
        cli._answer_job_question(
            settings,
            session_path=session,
            question="What would I work with?",
            model_class="fast",
            speak=True,
        )
    )

    assert exit_code == 0
    assert spoken == ["Build RAG pipelines."]

    exit_code = asyncio.run(
        cli._answer_job_question(
            settings,
            session_path=session,
            question="What would I work with?",
            model_class="fast",
            speak=False,
        )
    )

    assert exit_code == 0
    assert spoken == ["Build RAG pipelines."]


def test_answer_job_question_rejects_audio_without_disclosure_confirmation(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(
        [
            "answer-job-question",
            "--session",
            "runs/example",
            "--question",
            "What technologies are used?",
            "--speak",
        ],
        project_root=tmp_path,
        environ={},
    )

    assert exit_code == 2
    assert "requires --confirm-disclosed-audio-output" in capsys.readouterr().err


def test_barge_in_route_test_fails_closed_without_loopback_confirmation(
    tmp_path: Path, capsys: object
) -> None:
    exit_code = cli.main(
        ["qwen-tts-barge-in-test", "--text", "Hello"],
        project_root=tmp_path,
        environ={},
    )

    assert exit_code == 2
    assert "requires --confirm-selected-loopback-route" in capsys.readouterr().err


def test_supervised_voice_loop_requires_disclosure_confirmation(
    tmp_path: Path, capsys: object
) -> None:
    exit_code = cli.main(
        [
            "supervise-voice-loop",
            "--session",
            "runs/example",
            "--candidate",
            "Candidate Alpha",
        ],
        project_root=tmp_path,
        environ={},
    )

    assert exit_code == 2
    assert "requires --confirm-disclosed-supervision" in capsys.readouterr().err
