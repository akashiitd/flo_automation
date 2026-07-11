"""Approved Launch-only extraction of FloCareer interview questions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

from browser.action_guard import BrowserAction
from browser.action_router import ActionRouter
from browser.join_workflow import (
    ApprovalRequester,
    JoinWorkflowError,
    LaunchWorkflowPage,
    PostLaunchState,
    prepare_launch_control,
)


@dataclass(frozen=True, slots=True)
class ExtractedQuestion:
    id: int
    question_text: str
    has_code_editor: bool
    ideal_answer: str
    guidelines: Mapping[str, str]
    feedback_field_locator_hint: str
    rating_locator_hint: str
    mark_as_locator_hint: str


@dataclass(frozen=True, slots=True)
class QuestionScanResult:
    candidate_identifier: str
    questions: tuple[ExtractedQuestion, ...]
    questions_path: Path
    screenshot_path: Path
    action_log_path: Path


class QuestionScanPage(LaunchWorkflowPage, Protocol):
    def wait_for_questions_or_consent(self) -> PostLaunchState: ...
    def visible_consent_ok_count(self) -> int: ...
    def click_consent_ok(self) -> None: ...
    def wait_for_question_panel(self) -> None: ...
    def extract_questions(self) -> list[ExtractedQuestion]: ...


def _write_questions(path: Path, questions: tuple[ExtractedQuestion, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps([asdict(question) for question in questions], indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def run_question_scan(
    page: QuestionScanPage,
    *,
    candidate_name: str,
    session_dir: Path,
    action_router: ActionRouter,
    request_approval: ApprovalRequester,
) -> QuestionScanResult:
    """Launch one candidate page and read questions without clicking Join."""

    prepared = prepare_launch_control(
        page,
        candidate_name=candidate_name,
        session_dir=session_dir,
        action_router=action_router,
        launch_screenshot_name="questions_launch_approval",
    )
    identifier = prepared.candidate_identifier
    launch = action_router.route(
        BrowserAction.LAUNCH_INTERVIEW,
        operation=page.click_launch_interview,
        candidate_identifier=identifier,
        approval_token=request_approval(BrowserAction.LAUNCH_INTERVIEW, identifier),
        screenshot_path=prepared.launch_control_screenshot,
    )
    if not launch.allowed:
        raise JoinWorkflowError("Launch approval was not granted; nothing launched")

    screenshots_dir = session_dir / "screenshots"
    try:
        state = page.wait_for_questions_or_consent()
    except Exception as error:
        diagnostic = page.capture_screenshot(
            screenshots_dir, "question_panel_wait_error"
        )
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error
    if state is PostLaunchState.CONSENT:
        consent = page.capture_screenshot(screenshots_dir, "questions_consent")
        if page.visible_consent_ok_count() != 1:
            raise JoinWorkflowError("Expected exactly one consent OK control")
        decision = action_router.route(
            BrowserAction.CLICK_CONSENT_OK,
            operation=page.click_consent_ok,
            candidate_identifier=identifier,
            approval_token=request_approval(BrowserAction.CLICK_CONSENT_OK, identifier),
            screenshot_path=consent,
        )
        if not decision.allowed:
            raise JoinWorkflowError(
                "Consent approval was not granted; OK was not clicked"
            )
        page.wait_for_question_panel()

    page.wait_for_question_panel()
    try:
        questions = tuple(page.extract_questions())
    except Exception as error:
        diagnostic = page.capture_screenshot(screenshots_dir, "question_extract_error")
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error
    if not questions:
        diagnostic = page.capture_screenshot(screenshots_dir, "questions_not_found")
        raise JoinWorkflowError(
            f"No question cards were extracted. Screenshot: {diagnostic}"
        )
    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)) or any(
        not question.question_text for question in questions
    ):
        raise JoinWorkflowError(
            "Question extraction returned duplicate IDs or empty text"
        )

    screenshot = page.capture_screenshot(screenshots_dir, "questions_expanded")
    questions_path = session_dir / "questions.json"
    _write_questions(questions_path, questions)
    return QuestionScanResult(
        candidate_identifier=identifier,
        questions=questions,
        questions_path=questions_path,
        screenshot_path=screenshot,
        action_log_path=action_router.log_path,
    )
