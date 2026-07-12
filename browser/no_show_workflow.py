"""Candidate-bound, approval-gated marking of an observed interview no-show."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from browser.action_guard import BrowserAction
from browser.action_router import ActionRouter
from browser.join_workflow import JoinLiveResult


class NoShowWorkflowError(RuntimeError):
    """The no-show UI cannot be proven safe to act on."""


class NoShowPage(Protocol):
    def capture_screenshot(self, directory: Path, name: str) -> Path: ...

    def visible_mark_no_show_count(self) -> int: ...

    def click_mark_no_show(self) -> None: ...

    def wait_for_mark_no_show_applied(self) -> None: ...

    def read_interview_level(self) -> str: ...

    def candidate_is_connected(self) -> bool: ...


NoShowApprovalRequester = Callable[[BrowserAction, str], str | None]


@dataclass(frozen=True, slots=True)
class NoShowResult:
    candidate_identifier: str
    level: str
    before_screenshot: Path
    after_screenshot: Path
    action_log_path: Path


def mark_no_show(
    page: NoShowPage,
    *,
    joined: JoinLiveResult,
    session_dir: Path,
    action_router: ActionRouter,
    request_approval: NoShowApprovalRequester,
) -> NoShowResult:
    """Verify the shown level, then click exactly one approved no-show control."""

    required_level = "Intermediate"
    screenshots = session_dir / "screenshots"
    before = page.capture_screenshot(screenshots, "no_show_before_approval")
    controls = page.visible_mark_no_show_count()
    if controls != 1:
        raise NoShowWorkflowError(
            f"Expected one visible Mark No-show control; found {controls}. "
            f"Screenshot: {before}"
        )
    actual_level = page.read_interview_level()
    if actual_level != required_level:
        raise NoShowWorkflowError(
            f"Expected interview level {required_level!r}; found {actual_level!r}. "
            "Set it manually, then restart the guarded no-show workflow."
        )

    approval_token = request_approval(
        BrowserAction.MARK_NO_SHOW, joined.candidate_identifier
    )

    def apply_no_show() -> None:
        if page.visible_mark_no_show_count() != 1:
            raise NoShowWorkflowError("Mark No-show control changed before approval")
        if page.candidate_is_connected():
            raise NoShowWorkflowError(
                "Candidate connected before Mark No-show could be applied"
            )
        if page.read_interview_level() != required_level:
            raise NoShowWorkflowError("Interview level changed before no-show approval")
        page.click_mark_no_show()
        page.wait_for_mark_no_show_applied()

    decision = action_router.route(
        BrowserAction.MARK_NO_SHOW,
        operation=apply_no_show,
        candidate_identifier=joined.candidate_identifier,
        approval_token=approval_token,
        screenshot_path=before,
    )
    if not decision.allowed:
        raise NoShowWorkflowError("Mark No-show approval was not granted")
    after = page.capture_screenshot(screenshots, "no_show_after_marked")
    return NoShowResult(
        candidate_identifier=joined.candidate_identifier,
        level=required_level,
        before_screenshot=before,
        after_screenshot=after,
        action_log_path=action_router.log_path,
    )
