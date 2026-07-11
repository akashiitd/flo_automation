from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import main as cli


def test_join_command_requires_explicit_dry_run_flag() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["join", "--candidate", "Candidate Alpha"])


def test_join_command_runs_only_the_dry_run_controller(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    calls: list[tuple[str, float]] = []
    session_dir = tmp_path / "runs" / "join_test"
    screenshots = session_dir / "screenshots"
    screenshots.mkdir(parents=True)
    result = SimpleNamespace(
        candidate_identifier="candidate-a1b2c3",
        candidate_found_screenshot=screenshots / "candidate_found.png",
        join_dry_run_screenshot=screenshots / "join_dry_run.png",
        action_log_path=session_dir / "action_log.jsonl",
    )

    def fake_join(
        settings: object,
        *,
        candidate_name: str,
        login_timeout_seconds: float,
        progress: object,
    ) -> object:
        calls.append((candidate_name, login_timeout_seconds))
        return result

    monkeypatch.setattr(cli, "join_candidate_dry_run", fake_join)
    monkeypatch.setattr(
        cli,
        "run_health_checks",
        lambda settings: SimpleNamespace(overall="READY_FOR_BROWSER_SCAN"),
    )

    exit_code = cli.main(
        [
            "join",
            "--candidate",
            "Candidate Alpha",
            "--dry-run",
            "--login-timeout",
            "45",
        ],
        project_root=tmp_path,
        environ={},
    )

    assert exit_code == 0
    assert calls == [("Candidate Alpha", 45.0)]
    assert "Validation passed: launch control found and blocked by dry run" in (
        capsys.readouterr().out
    )
