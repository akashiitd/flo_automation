"""Guarded, question-scoped code-editor visibility workflow."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol

from browser.action_guard import BrowserAction
from browser.action_router import ActionRouter


class CodeEditorWorkflowError(RuntimeError):
    """Raised when the requested editor action cannot be proven safe."""


class CandidateDisconnectedError(CodeEditorWorkflowError):
    """Raised when the candidate disconnects before the editor can be shown."""


class CodeEditorVisibility(str, Enum):
    HIDDEN = "hidden"
    VISIBLE = "visible"


class CodeEditorPage(Protocol):
    def active_candidate_matches(self, candidate_identifier: str) -> bool: ...

    def candidate_is_connected(self) -> bool: ...

    def capture_screenshot(self, directory: Path, name: str) -> Path: ...

    def open_code_editor_tab(self, question_id: int) -> None: ...

    def read_code_editor_visibility(self, question_id: int) -> CodeEditorVisibility: ...

    def click_show_code_editor(self, question_id: int) -> None: ...

    def code_editor_tab_is_active(self, question_id: int) -> bool: ...

    def wait_for_code_editor_visibility(
        self,
        question_id: int,
        expected: CodeEditorVisibility,
    ) -> None: ...


CodeEditorApprovalRequester = Callable[[BrowserAction, str, int], str | None]


@dataclass(frozen=True, slots=True)
class CodeEditorResult:
    candidate_identifier: str
    question_id: int
    changed: bool
    visibility: CodeEditorVisibility
    before_screenshot: Path
    after_screenshot: Path
    action_log_path: Path


def _require_active_candidate(
    page: CodeEditorPage,
    candidate_identifier: str,
) -> None:
    if not page.active_candidate_matches(candidate_identifier):
        raise CodeEditorWorkflowError(
            "The active candidate no longer matches the approved session"
        )


def _unchanged_result(
    page: CodeEditorPage,
    *,
    candidate_identifier: str,
    question_id: int,
    screenshots_dir: Path,
    evidence_prefix: str,
    before: Path,
    action_router: ActionRouter,
) -> CodeEditorResult:
    after = page.capture_screenshot(
        screenshots_dir,
        f"{evidence_prefix}_already_visible",
    )
    return CodeEditorResult(
        candidate_identifier=candidate_identifier,
        question_id=question_id,
        changed=False,
        visibility=CodeEditorVisibility.VISIBLE,
        before_screenshot=before,
        after_screenshot=after,
        action_log_path=action_router.log_path,
    )


def run_show_code_editor(
    page: CodeEditorPage,
    *,
    candidate_identifier: str,
    question_id: int,
    session_dir: Path,
    action_router: ActionRouter,
    request_approval: CodeEditorApprovalRequester,
    allow_prejoin: bool = False,
) -> CodeEditorResult:
    """Show one coding editor after exact candidate/question approval."""

    if not candidate_identifier.strip():
        raise ValueError("candidate identifier is required")
    if question_id < 1:
        raise ValueError("question id must be positive")

    screenshots_dir = session_dir / "screenshots"
    evidence_prefix = (
        f"code_editor_question_{question_id}_"
        f"{datetime.now(UTC).strftime('%Y%m%d_%H%M%S_%f')}"
    )
    _require_active_candidate(page, candidate_identifier)
    if not allow_prejoin and not page.candidate_is_connected():
        raise CandidateDisconnectedError(
            "Candidate is not connected; editor visibility was not changed"
        )
    open_decision = action_router.route(
        BrowserAction.OPEN_CODE_EDITOR_TAB,
        operation=lambda: page.open_code_editor_tab(question_id),
        candidate_identifier=candidate_identifier,
        question_id=question_id,
    )
    if not open_decision.allowed:
        raise CodeEditorWorkflowError("Opening the Code Editor tab is blocked")

    visibility = page.read_code_editor_visibility(question_id)
    before = page.capture_screenshot(
        screenshots_dir,
        f"{evidence_prefix}_before",
    )
    if visibility is CodeEditorVisibility.VISIBLE:
        return _unchanged_result(
            page,
            candidate_identifier=candidate_identifier,
            question_id=question_id,
            screenshots_dir=screenshots_dir,
            evidence_prefix=evidence_prefix,
            before=before,
            action_router=action_router,
        )

    token = request_approval(
        BrowserAction.SHOW_CODE_EDITOR_TO_CANDIDATE,
        candidate_identifier,
        question_id,
    )

    # The operator pause can be long enough for another actor to change the state.
    _require_active_candidate(page, candidate_identifier)
    if not allow_prejoin and not page.candidate_is_connected():
        raise CandidateDisconnectedError(
            "Candidate disconnected while awaiting editor approval"
        )
    visibility = page.read_code_editor_visibility(question_id)
    if visibility is CodeEditorVisibility.VISIBLE:
        return _unchanged_result(
            page,
            candidate_identifier=candidate_identifier,
            question_id=question_id,
            screenshots_dir=screenshots_dir,
            evidence_prefix=evidence_prefix,
            before=before,
            action_router=action_router,
        )

    def click_after_final_revalidation() -> None:
        _require_active_candidate(page, candidate_identifier)
        if not allow_prejoin and not page.candidate_is_connected():
            raise CandidateDisconnectedError(
                "Candidate disconnected before the editor could be shown"
            )
        if not page.code_editor_tab_is_active(question_id):
            raise CodeEditorWorkflowError(
                f"Code Editor tab for question {question_id} is no longer active"
            )
        if (
            page.read_code_editor_visibility(question_id)
            is not CodeEditorVisibility.HIDDEN
        ):
            raise CodeEditorWorkflowError(
                f"Code editor for question {question_id} is no longer hidden"
            )
        page.click_show_code_editor(question_id)
        page.wait_for_code_editor_visibility(
            question_id,
            CodeEditorVisibility.VISIBLE,
        )

    decision = action_router.route(
        BrowserAction.SHOW_CODE_EDITOR_TO_CANDIDATE,
        operation=click_after_final_revalidation,
        candidate_identifier=candidate_identifier,
        question_id=question_id,
        approval_token=token,
        screenshot_path=before,
    )
    if not decision.allowed:
        raise CodeEditorWorkflowError(
            "Show-editor approval was not granted; the switch was not clicked"
        )

    after = page.capture_screenshot(
        screenshots_dir,
        f"{evidence_prefix}_after",
    )
    return CodeEditorResult(
        candidate_identifier=candidate_identifier,
        question_id=question_id,
        changed=True,
        visibility=CodeEditorVisibility.VISIBLE,
        before_screenshot=before,
        after_screenshot=after,
        action_log_path=action_router.log_path,
    )
