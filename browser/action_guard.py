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
    OPEN_CODE_EDITOR_TAB = "OPEN_CODE_EDITOR_TAB"
    SHOW_CODE_EDITOR_TO_CANDIDATE = "SHOW_CODE_EDITOR_TO_CANDIDATE"
    HANG_UP = "HANG_UP"
    FILL_FEEDBACK = "FILL_FEEDBACK"
    FINISH_INTERVIEW = "FINISH_INTERVIEW"


_APPROVAL_LABELS = {
    BrowserAction.LAUNCH_INTERVIEW: "APPROVE-LAUNCH",
    BrowserAction.CLICK_CONSENT_OK: "APPROVE-CONSENT",
    BrowserAction.CLICK_JOIN: "APPROVE-JOIN",
}

_QUESTION_APPROVAL_LABELS = {
    BrowserAction.SHOW_CODE_EDITOR_TO_CANDIDATE: "APPROVE-SHOW-CODE-EDITOR",
}


def approval_token_for(
    action: BrowserAction,
    candidate_identifier: str,
    *,
    question_id: int | None = None,
) -> str:
    """Return the exact operator phrase required for one guarded action."""

    if action not in _APPROVAL_LABELS and action not in _QUESTION_APPROVAL_LABELS:
        raise ValueError(f"{action.value} does not support operator approval")
    if not candidate_identifier.strip():
        raise ValueError("candidate identifier is required for approval")
    if action in _QUESTION_APPROVAL_LABELS:
        if question_id is None or question_id < 1:
            raise ValueError("positive question id is required for approval")
        return (
            f"{_QUESTION_APPROVAL_LABELS[action]} {candidate_identifier} "
            f"question-{question_id}"
        )
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

    @classmethod
    def code_editor(cls) -> ActionGuard:
        return cls(
            allowed_actions=frozenset({BrowserAction.OPEN_CODE_EDITOR_TAB}),
            mode="code_editor",
        )

    def decide(
        self,
        action: BrowserAction,
        *,
        candidate_identifier: str | None = None,
        question_id: int | None = None,
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
        if (
            not allowed
            and self.mode == "code_editor"
            and action in _QUESTION_APPROVAL_LABELS
            and candidate_identifier is not None
            and question_id is not None
        ):
            allowed = approval_token == approval_token_for(
                action,
                candidate_identifier,
                question_id=question_id,
            )
        reason = (
            f"{action.value} is permitted in {self.mode} mode"
            if allowed
            else (
                f"{action.value} requires its candidate-bound approval token"
                if self.mode == "live_join" and action in _APPROVAL_LABELS
                else (
                    f"{action.value} requires its candidate-and-question-bound "
                    "approval token"
                    if self.mode == "code_editor"
                    and action in _QUESTION_APPROVAL_LABELS
                    else f"{action.value} is blocked in {self.mode} mode"
                )
            )
        )
        return ActionDecision(action=action, allowed=allowed, reason=reason)
