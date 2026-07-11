"""Typed safety policy for browser actions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


class BrowserAction(str, Enum):
    OPEN_DASHBOARD = "OPEN_DASHBOARD"
    FIND_CANDIDATE = "FIND_CANDIDATE"
    OPEN_CANDIDATE_MENU = "OPEN_CANDIDATE_MENU"
    LAUNCH_INTERVIEW = "LAUNCH_INTERVIEW"
    CLICK_JOIN = "CLICK_JOIN"
    HANG_UP = "HANG_UP"
    FILL_FEEDBACK = "FILL_FEEDBACK"
    FINISH_INTERVIEW = "FINISH_INTERVIEW"


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

    def decide(self, action: BrowserAction) -> ActionDecision:
        allowed = action in self.allowed_actions
        reason = (
            f"{action.value} is permitted in {self.mode} mode"
            if allowed
            else f"{action.value} is blocked in {self.mode} mode"
        )
        return ActionDecision(action=action, allowed=allowed, reason=reason)
