from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace

import pytest

import main as cli
from browser.action_guard import BrowserAction, approval_token_for


def test_join_command_requires_explicit_dry_run_flag() -> None:
    with pytest.raises(SystemExit):
        cli.build_parser().parse_args(["join", "--candidate", "Candidate Alpha"])


def test_no_show_command_rejects_a_wait_shorter_than_seven_minutes(
    tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    exit_code = cli.main(
        [
            "no-show",
            "--candidate",
            "Candidate Alpha",
            "--wait-seconds",
            "419",
        ],
        project_root=tmp_path,
        environ={},
    )

    assert exit_code == 2
    assert "at least 420" in capsys.readouterr().err


def test_questions_scan_reports_coding_questions_without_join(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    session = tmp_path / "runs" / "questions_scan_test"
    result = SimpleNamespace(
        questions=(
            SimpleNamespace(id=1, has_code_editor=False),
            SimpleNamespace(id=2, has_code_editor=True),
        ),
        code_editor_dom_observations=(
            SimpleNamespace(question_id=2, association_status="unique"),
        ),
        questions_path=session / "questions.json",
        job_description_path=session / "job_description.json",
        code_editor_dom_path=session / "code_editor_dom.json",
        screenshot_path=session / "screenshots" / "questions_expanded.png",
        action_log_path=session / "action_log.jsonl",
    )

    def fake_scan(settings: object, **kwargs: object) -> object:
        approval = kwargs["request_approval"]
        assert isinstance(approval, Callable)
        return result

    monkeypatch.setattr(cli, "scan_candidate_questions", fake_scan)
    monkeypatch.setattr(
        cli,
        "run_health_checks",
        lambda settings: SimpleNamespace(overall="READY_FOR_BROWSER_SCAN"),
    )

    exit_code = cli.main(
        ["questions-scan", "--candidate", "Candidate Alpha"],
        project_root=tmp_path,
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Extracted questions: 2" in output
    assert "Coding question IDs: 2" in output
    assert "Code editor DOM associations: 2=unique" in output
    assert "Code editor DOM capture: complete" in output
    assert "Code editor DOM: " in output
    assert "Job description: " in output
    assert "without clicking Join" in output


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


def test_live_join_prompts_for_launch_consent_and_join_approvals(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_identifier = "candidate-a1b2c3"
    requested_actions: list[BrowserAction] = []
    typed_tokens = iter(
        [
            approval_token_for(BrowserAction.LAUNCH_INTERVIEW, candidate_identifier),
            approval_token_for(BrowserAction.CLICK_CONSENT_OK, candidate_identifier),
            approval_token_for(BrowserAction.CLICK_JOIN, candidate_identifier),
            f"CONFIRM-INTERVIEW-ENDED {candidate_identifier}",
        ]
    )
    session_dir = tmp_path / "runs" / "join_live_test"
    screenshots = session_dir / "screenshots"
    screenshots.mkdir(parents=True)
    result = SimpleNamespace(
        candidate_identifier=candidate_identifier,
        candidate_found_screenshot=screenshots / "candidate_found.png",
        launch_approval_screenshot=screenshots / "launch_approval.png",
        consent_screenshot=screenshots / "consent.png",
        pre_call_screenshot=screenshots / "pre_call.png",
        joined_screenshot=screenshots / "joined.png",
        room_state_log_path=session_dir / "room_state_log.jsonl",
        code_editor_result=None,
        action_log_path=session_dir / "action_log.jsonl",
    )

    def fake_live_join(
        settings: object,
        *,
        candidate_name: str,
        login_timeout_seconds: float,
        progress: object,
        request_approval: Callable[[BrowserAction, str], str | None],
        wait_for_manual_end: Callable[[str], None],
        enable_code_editor_question: int | None,
        request_code_editor_approval: object,
        candidate_wait_timeout_seconds: float | None,
        configure_flocareer_audio: bool,
    ) -> object:
        assert enable_code_editor_question is None
        assert request_code_editor_approval is None
        assert candidate_wait_timeout_seconds is None
        assert configure_flocareer_audio is True
        for action in (
            BrowserAction.LAUNCH_INTERVIEW,
            BrowserAction.CLICK_CONSENT_OK,
            BrowserAction.CLICK_JOIN,
        ):
            requested_actions.append(action)
            request_approval(action, candidate_identifier)
        wait_for_manual_end(candidate_identifier)
        return result

    monkeypatch.setattr(cli, "join_candidate_live", fake_live_join)
    monkeypatch.setattr(
        cli,
        "run_health_checks",
        lambda settings: SimpleNamespace(overall="READY_FOR_BROWSER_SCAN"),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(typed_tokens))

    exit_code = cli.main(
        ["join", "--candidate", "Candidate Alpha", "--live"],
        project_root=tmp_path,
        environ={},
    )

    assert exit_code == 0
    assert requested_actions == [
        BrowserAction.LAUNCH_INTERVIEW,
        BrowserAction.CLICK_CONSENT_OK,
        BrowserAction.CLICK_JOIN,
    ]
    output = capsys.readouterr().out
    assert "Launch and Join require separate approvals" in output
    assert "Consent OK requires another approval when the form is shown" in output
    assert "The browser will remain open until you confirm it has ended" in output
    assert "interview joined after all required approvals" in output


def test_live_join_requests_question_bound_editor_approval_after_connection(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    candidate_identifier = "candidate-a1b2c3"
    typed_tokens = iter(
        [
            approval_token_for(BrowserAction.LAUNCH_INTERVIEW, candidate_identifier),
            approval_token_for(BrowserAction.CLICK_CONSENT_OK, candidate_identifier),
            approval_token_for(BrowserAction.CLICK_JOIN, candidate_identifier),
            approval_token_for(
                BrowserAction.SHOW_CODE_EDITOR_TO_CANDIDATE,
                candidate_identifier,
                question_id=13,
            ),
            f"CONFIRM-INTERVIEW-ENDED {candidate_identifier}",
        ]
    )
    session_dir = tmp_path / "runs" / "join_live_editor_test"
    screenshots = session_dir / "screenshots"
    screenshots.mkdir(parents=True)
    result = SimpleNamespace(
        candidate_identifier=candidate_identifier,
        consent_screenshot=None,
        pre_call_screenshot=screenshots / "pre_call.png",
        joined_screenshot=screenshots / "joined.png",
        room_state_log_path=session_dir / "room_state_log.jsonl",
        action_log_path=session_dir / "action_log.jsonl",
        code_editor_result=SimpleNamespace(
            changed=True,
            question_id=13,
            before_screenshot=screenshots / "editor_before.png",
            after_screenshot=screenshots / "editor_after.png",
        ),
    )

    def fake_live_join(settings: object, **kwargs: object) -> object:
        request = kwargs["request_approval"]
        request_editor = kwargs["request_code_editor_approval"]
        manual_end = kwargs["wait_for_manual_end"]
        assert isinstance(request, Callable)
        assert isinstance(request_editor, Callable)
        assert kwargs["enable_code_editor_question"] == 13
        assert kwargs["configure_flocareer_audio"] is True
        for action in (
            BrowserAction.LAUNCH_INTERVIEW,
            BrowserAction.CLICK_CONSENT_OK,
            BrowserAction.CLICK_JOIN,
        ):
            request(action, candidate_identifier)
        request_editor(
            BrowserAction.SHOW_CODE_EDITOR_TO_CANDIDATE, candidate_identifier, 13
        )
        manual_end(candidate_identifier)
        return result

    monkeypatch.setattr(cli, "join_candidate_live", fake_live_join)
    monkeypatch.setattr(
        cli,
        "run_health_checks",
        lambda settings: SimpleNamespace(overall="READY_FOR_BROWSER_SCAN"),
    )
    monkeypatch.setattr("builtins.input", lambda prompt: next(typed_tokens))

    exit_code = cli.main(
        [
            "join",
            "--candidate",
            "Candidate Alpha",
            "--live",
            "--enable-code-editor-question",
            "13",
        ],
        project_root=tmp_path,
        environ={},
    )

    assert exit_code == 0
    output = capsys.readouterr().out
    assert "Code editor requires a separate candidate-and-question approval" in output
    assert "Code editor: enabled for question 13" in output
