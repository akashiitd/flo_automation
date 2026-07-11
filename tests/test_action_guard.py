from __future__ import annotations

import json
from pathlib import Path

from browser.action_guard import ActionGuard, BrowserAction
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
