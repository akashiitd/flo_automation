from __future__ import annotations

from pathlib import Path

import main as cli


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
