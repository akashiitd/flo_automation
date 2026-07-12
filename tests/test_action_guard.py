from __future__ import annotations

import json
from pathlib import Path

import pytest

from browser.action_guard import ActionGuard, BrowserAction, approval_token_for
from browser.action_router import ActionRouter


def test_dry_run_guard_allows_only_reversible_discovery_actions() -> None:
    guard = ActionGuard.dry_run()

    assert guard.decide(BrowserAction.OPEN_DASHBOARD).allowed is True
    assert guard.decide(BrowserAction.FIND_CANDIDATE).allowed is True
    assert guard.decide(BrowserAction.OPEN_CANDIDATE_MENU).allowed is True


def test_dry_run_guard_blocks_interview_mutations() -> None:
    guard = ActionGuard.dry_run()

    blocked = {
        BrowserAction.LAUNCH_INTERVIEW,
        BrowserAction.CLICK_JOIN,
        BrowserAction.HANG_UP,
        BrowserAction.FILL_FEEDBACK,
        BrowserAction.FINISH_INTERVIEW,
    }

    assert all(guard.decide(action).allowed is False for action in blocked)


def test_router_does_not_execute_a_blocked_action_and_writes_audit_log(
    tmp_path: Path,
) -> None:
    calls: list[str] = []
    log_path = tmp_path / "action_log.jsonl"
    router = ActionRouter(ActionGuard.dry_run(), log_path)

    decision = router.route(
        BrowserAction.LAUNCH_INTERVIEW,
        operation=lambda: calls.append("launched"),
        candidate_identifier="candidate-a1b2c3",
        screenshot_path=tmp_path / "join_dry_run.png",
    )

    assert decision.allowed is False
    assert calls == []
    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["action"] == "LAUNCH_INTERVIEW"
    assert record["decision"] == "BLOCK"
    assert record["candidate_identifier"] == "candidate-a1b2c3"
    assert record["screenshot_path"].endswith("join_dry_run.png")


def test_router_does_not_execute_a_blocked_join_action(tmp_path: Path) -> None:
    calls: list[str] = []
    router = ActionRouter(ActionGuard.dry_run(), tmp_path / "action_log.jsonl")

    decision = router.route(
        BrowserAction.CLICK_JOIN,
        operation=lambda: calls.append("joined"),
    )

    assert decision.allowed is False
    assert calls == []


def test_live_join_requires_separate_candidate_bound_approvals() -> None:
    guard = ActionGuard.live_join()
    candidate = "candidate-a1b2c3"
    launch_token = approval_token_for(BrowserAction.LAUNCH_INTERVIEW, candidate)
    consent_token = approval_token_for(BrowserAction.CLICK_CONSENT_OK, candidate)
    join_token = approval_token_for(BrowserAction.CLICK_JOIN, candidate)

    assert guard.decide(BrowserAction.LAUNCH_INTERVIEW).allowed is False
    assert (
        guard.decide(
            BrowserAction.LAUNCH_INTERVIEW,
            candidate_identifier=candidate,
            approval_token=join_token,
        ).allowed
        is False
    )
    assert guard.decide(
        BrowserAction.LAUNCH_INTERVIEW,
        candidate_identifier=candidate,
        approval_token=launch_token,
    ).allowed
    assert guard.decide(
        BrowserAction.CLICK_CONSENT_OK,
        candidate_identifier=candidate,
        approval_token=consent_token,
    ).allowed
    assert guard.decide(
        BrowserAction.CLICK_JOIN,
        candidate_identifier=candidate,
        approval_token=join_token,
    ).allowed


def test_live_join_never_approves_hang_up_or_finish() -> None:
    guard = ActionGuard.live_join()
    candidate = "candidate-a1b2c3"

    for action in (BrowserAction.HANG_UP, BrowserAction.FINISH_INTERVIEW):
        decision = guard.decide(
            action,
            candidate_identifier=candidate,
            approval_token=f"APPROVE-{action.value}-{candidate}",
        )
        assert decision.allowed is False


def test_no_show_requires_its_own_candidate_bound_approval() -> None:
    candidate = "candidate-a1b2c3"
    token = approval_token_for(BrowserAction.MARK_NO_SHOW, candidate)

    assert token == "APPROVE-MARK-NO-SHOW candidate-a1b2c3"
    assert (
        ActionGuard.live_join()
        .decide(
            BrowserAction.MARK_NO_SHOW,
            candidate_identifier=candidate,
            approval_token=token,
        )
        .allowed
        is False
    )
    assert (
        ActionGuard.no_show()
        .decide(
            BrowserAction.MARK_NO_SHOW,
            candidate_identifier=candidate,
            approval_token=token,
        )
        .allowed
        is True
    )


def test_code_editor_show_requires_candidate_and_question_bound_approval() -> None:
    guard = ActionGuard.code_editor()
    candidate = "candidate-a1b2c3"
    token = approval_token_for(
        BrowserAction.SHOW_CODE_EDITOR_TO_CANDIDATE,
        candidate,
        question_id=13,
    )

    assert token == "APPROVE-SHOW-CODE-EDITOR candidate-a1b2c3 question-13"
    assert guard.decide(BrowserAction.OPEN_CODE_EDITOR_TAB).allowed is True
    assert (
        guard.decide(
            BrowserAction.SHOW_CODE_EDITOR_TO_CANDIDATE,
            candidate_identifier=candidate,
            question_id=13,
            approval_token=token,
        ).allowed
        is True
    )
    assert (
        guard.decide(
            BrowserAction.SHOW_CODE_EDITOR_TO_CANDIDATE,
            candidate_identifier=candidate,
            question_id=12,
            approval_token=token,
        ).allowed
        is False
    )
    assert (
        guard.decide(
            BrowserAction.SHOW_CODE_EDITOR_TO_CANDIDATE,
            candidate_identifier="candidate-other",
            question_id=13,
            approval_token=token,
        ).allowed
        is False
    )


def test_router_does_not_write_approval_tokens_to_audit_log(tmp_path: Path) -> None:
    candidate = "candidate-a1b2c3"
    token = approval_token_for(BrowserAction.LAUNCH_INTERVIEW, candidate)
    log_path = tmp_path / "action_log.jsonl"
    router = ActionRouter(ActionGuard.live_join(), log_path)

    decision = router.route(
        BrowserAction.LAUNCH_INTERVIEW,
        operation=lambda: None,
        candidate_identifier=candidate,
        approval_token=token,
    )

    assert decision.allowed is True
    assert token not in log_path.read_text(encoding="utf-8")


def test_router_consumes_a_live_approval_token_once(tmp_path: Path) -> None:
    candidate = "candidate-a1b2c3"
    token = approval_token_for(BrowserAction.LAUNCH_INTERVIEW, candidate)
    calls: list[str] = []
    router = ActionRouter(ActionGuard.live_join(), tmp_path / "action_log.jsonl")

    first = router.route(
        BrowserAction.LAUNCH_INTERVIEW,
        operation=lambda: calls.append("launched"),
        candidate_identifier=candidate,
        approval_token=token,
    )
    replay = router.route(
        BrowserAction.LAUNCH_INTERVIEW,
        operation=lambda: calls.append("launched-again"),
        candidate_identifier=candidate,
        approval_token=token,
    )

    assert first.allowed is True
    assert replay.allowed is False
    assert calls == ["launched"]


def test_router_records_failed_action_execution(tmp_path: Path) -> None:
    candidate = "candidate-a1b2c3"
    token = approval_token_for(BrowserAction.LAUNCH_INTERVIEW, candidate)
    log_path = tmp_path / "action_log.jsonl"
    router = ActionRouter(ActionGuard.live_join(), log_path)

    with pytest.raises(RuntimeError, match="simulated click failure"):
        router.route(
            BrowserAction.LAUNCH_INTERVIEW,
            operation=lambda: (_ for _ in ()).throw(
                RuntimeError("simulated click failure")
            ),
            candidate_identifier=candidate,
            approval_token=token,
        )

    record = json.loads(log_path.read_text(encoding="utf-8"))
    assert record["decision"] == "ALLOW"
    assert record["execution_outcome"] == "ERROR"
