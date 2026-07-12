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
