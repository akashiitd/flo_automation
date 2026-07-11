"""Guarded execution and audit logging for browser actions."""

from __future__ import annotations

import json
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path

from browser.action_guard import ActionDecision, ActionGuard, BrowserAction


class ActionRouter:
    def __init__(self, guard: ActionGuard, log_path: Path) -> None:
        self._guard = guard
        self._log_path = log_path
        self._consumed_approval_tokens: set[str] = set()

    @property
    def log_path(self) -> Path:
        return self._log_path

    def route(
        self,
        action: BrowserAction,
        *,
        operation: Callable[[], None] | None = None,
        candidate_identifier: str | None = None,
        approval_token: str | None = None,
        screenshot_path: Path | None = None,
    ) -> ActionDecision:
        unused_approval = (
            approval_token
            if approval_token not in self._consumed_approval_tokens
            else None
        )
        decision = self._guard.decide(
            action,
            candidate_identifier=candidate_identifier,
            approval_token=unused_approval,
        )
        self._append_record(
            decision,
            candidate_identifier=candidate_identifier,
            screenshot_path=screenshot_path,
        )
        if decision.allowed and approval_token is not None:
            self._consumed_approval_tokens.add(approval_token)
        if decision.allowed and operation is not None:
            operation()
        return decision

    def _append_record(
        self,
        decision: ActionDecision,
        *,
        candidate_identifier: str | None,
        screenshot_path: Path | None,
    ) -> None:
        self._log_path.parent.mkdir(parents=True, exist_ok=True)
        record = {
            "timestamp": datetime.now(UTC).isoformat(),
            "action": decision.action.value,
            "decision": "ALLOW" if decision.allowed else "BLOCK",
            "reason": decision.reason,
            "candidate_identifier": candidate_identifier,
            "screenshot_path": str(screenshot_path) if screenshot_path else None,
        }
        with self._log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
