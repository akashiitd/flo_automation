"""Typed safety policy for browser actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BrowserAction(str, Enum):
    OPEN_DASHBOARD = "OPEN_DASHBOARD"
    FIND_CANDIDATE = "FIND_CANDIDATE"
    OPEN_CANDIDATE_MENU = "OPEN_CANDIDATE_MENU"
    LAUNCH_INTERVIEW = "LAUNCH_INTERVIEW"
    CLICK_CONSENT_OK = "CLICK_CONSENT_OK"
    CLICK_JOIN = "CLICK_JOIN"
    HANG_UP = "HANG_UP"
    FILL_FEEDBACK = "FILL_FEEDBACK"
    FINISH_INTERVIEW = "FINISH_INTERVIEW"


_APPROVAL_LABELS = {
    BrowserAction.LAUNCH_INTERVIEW: "APPROVE-LAUNCH",
    BrowserAction.CLICK_CONSENT_OK: "APPROVE-CONSENT",
    BrowserAction.CLICK_JOIN: "APPROVE-JOIN",
}


def approval_token_for(
    action: BrowserAction,
    candidate_identifier: str,
) -> str:
    """Return the exact operator phrase required for one guarded action."""

    if action not in _APPROVAL_LABELS:
        raise ValueError(f"{action.value} does not support operator approval")
    if not candidate_identifier.strip():
        raise ValueError("candidate identifier is required for approval")
    return f"{_APPROVAL_LABELS[action]} {candidate_identifier}"


@dataclass(frozen=True, slots=True)
class ActionDecision:
    action: BrowserAction
    allowed: bool
    reason: str


@dataclass(frozen=True, slots=True)
class ActionGuard:
    """Decide whether a requested browser action is safe in the active mode."""

    allowed_actions: frozenset[BrowserAction]
    mode: str

    @classmethod
    def dry_run(cls) -> ActionGuard:
        return cls(
            allowed_actions=frozenset(
                {
                    BrowserAction.OPEN_DASHBOARD,
                    BrowserAction.FIND_CANDIDATE,
                    BrowserAction.OPEN_CANDIDATE_MENU,
                }
            ),
            mode="dry_run",
        )

    @classmethod
    def live_join(cls) -> ActionGuard:
        return cls(
            allowed_actions=frozenset(
                {
                    BrowserAction.OPEN_DASHBOARD,
                    BrowserAction.FIND_CANDIDATE,
                    BrowserAction.OPEN_CANDIDATE_MENU,
                }
            ),
            mode="live_join",
        )

    def decide(
        self,
        action: BrowserAction,
        *,
        candidate_identifier: str | None = None,
        approval_token: str | None = None,
    ) -> ActionDecision:
        allowed = action in self.allowed_actions
        if (
            not allowed
            and self.mode == "live_join"
            and action in _APPROVAL_LABELS
            and candidate_identifier is not None
        ):
            allowed = approval_token == approval_token_for(action, candidate_identifier)
        reason = (
            f"{action.value} is permitted in {self.mode} mode"
            if allowed
            else (
                f"{action.value} requires its candidate-bound approval token"
                if self.mode == "live_join" and action in _APPROVAL_LABELS
                else f"{action.value} is blocked in {self.mode} mode"
            )
        )
        return ActionDecision(action=action, allowed=allowed, reason=reason)
