"""Candidate-scoped dry-run and explicitly approved live join workflows."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Protocol

from browser.action_guard import BrowserAction
from browser.action_router import ActionRouter


class JoinWorkflowError(RuntimeError):
    """Raised when the guarded join discovery cannot complete safely."""


class CandidateNotFoundError(JoinWorkflowError):
    pass


class AmbiguousCandidateError(JoinWorkflowError):
    pass


class PostLaunchState(str, Enum):
    CONSENT = "consent"
    PRE_CALL = "pre_call"
    QUESTIONS = "questions"


@dataclass(frozen=True, slots=True)
class CandidateCardHandle:
    value: str


@dataclass(frozen=True, slots=True)
class JoinCandidate:
    candidate_name: str
    scheduled_time: str
    card_handle: CandidateCardHandle


@dataclass(frozen=True, slots=True)
class JoinDryRunResult:
    candidate_identifier: str
    candidate_found_screenshot: Path
    join_dry_run_screenshot: Path
    action_log_path: Path


@dataclass(frozen=True, slots=True)
class JoinLiveResult:
    candidate_identifier: str
    candidate_found_screenshot: Path
    launch_approval_screenshot: Path
    consent_screenshot: Path | None
    pre_call_screenshot: Path
    joined_screenshot: Path
    action_log_path: Path


@dataclass(frozen=True, slots=True)
class PreparedLaunch:
    candidate_identifier: str
    candidate_found_screenshot: Path
    launch_control_screenshot: Path


class LaunchWorkflowPage(Protocol):
    def list_join_candidates(self) -> list[JoinCandidate]: ...

    def capture_screenshot(self, directory: Path, name: str) -> Path: ...

    def open_candidate_menu(self, candidate: JoinCandidate) -> None: ...

    def visible_launch_control_count(self) -> int: ...

    def click_launch_interview(self) -> None: ...


class JoinWorkflowPage(LaunchWorkflowPage, Protocol):
    def bind_candidate_identifier(self, candidate_identifier: str) -> None: ...

    def wait_for_consent_form(self) -> None: ...

    def wait_for_consent_or_pre_call(self) -> PostLaunchState: ...

    def visible_consent_ok_count(self) -> int: ...

    def click_consent_ok(self) -> None: ...

    def wait_for_pre_call_page(self) -> None: ...

    def visible_join_control_count(self) -> int: ...

    def click_join(self) -> None: ...

    def wait_for_joined_interview(self) -> None: ...


ApprovalRequester = Callable[[BrowserAction, str], str | None]


def normalize_candidate_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _masked_name(value: str) -> str:
    words = re.findall(r"\S+", value)
    return " ".join(word[0] + "*" * (len(word) - 1) for word in words)


def _candidate_identifier(candidate: JoinCandidate) -> str:
    source = f"{normalize_candidate_name(candidate.candidate_name)}|{candidate.scheduled_time}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"candidate-{digest}"


def prepare_launch_control(
    page: LaunchWorkflowPage,
    *,
    candidate_name: str,
    session_dir: Path,
    action_router: ActionRouter,
    launch_screenshot_name: str,
) -> PreparedLaunch:

    requested_name = normalize_candidate_name(candidate_name)
    candidates = page.list_join_candidates()
    matches = [
        candidate
        for candidate in candidates
        if normalize_candidate_name(candidate.candidate_name) == requested_name
    ]
    if not matches:
        lookup_failed = page.capture_screenshot(
            session_dir / "screenshots", "candidate_lookup_failed"
        )
        action_router.route(
            BrowserAction.FIND_CANDIDATE,
            screenshot_path=lookup_failed,
        )
        available = ", ".join(_masked_name(item.candidate_name) for item in candidates)
        detail = available or "none"
        raise CandidateNotFoundError(
            f"No exact candidate match. Sanitized available names: {detail}"
        )
    if len(matches) > 1:
        lookup_ambiguous = page.capture_screenshot(
            session_dir / "screenshots", "candidate_lookup_ambiguous"
        )
        action_router.route(
            BrowserAction.FIND_CANDIDATE,
            screenshot_path=lookup_ambiguous,
        )
        raise AmbiguousCandidateError(
            "Multiple exact candidate matches; use candidate plus date/time in a "
            "future stronger selector. No card was selected."
        )

    candidate = matches[0]
    identifier = _candidate_identifier(candidate)
    screenshots_dir = session_dir / "screenshots"
    candidate_found = page.capture_screenshot(screenshots_dir, "candidate_found")
    action_router.route(
        BrowserAction.FIND_CANDIDATE,
        candidate_identifier=identifier,
        screenshot_path=candidate_found,
    )
    try:
        action_router.route(
            BrowserAction.OPEN_CANDIDATE_MENU,
            operation=lambda: page.open_candidate_menu(candidate),
            candidate_identifier=identifier,
            screenshot_path=candidate_found,
        )
        launch_controls = page.visible_launch_control_count()
    except Exception as error:
        diagnostic = page.capture_screenshot(
            screenshots_dir, "candidate_menu_selector_error"
        )
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error

    if launch_controls != 1:
        diagnostic = page.capture_screenshot(
            screenshots_dir, "launch_control_selector_error"
        )
        raise JoinWorkflowError(
            "Expected exactly one visible Launch Video Interview control after "
            f"opening the candidate menu; found {launch_controls}. "
            f"Screenshot: {diagnostic}"
        )

    launch_control = page.capture_screenshot(screenshots_dir, launch_screenshot_name)
    return PreparedLaunch(
        candidate_identifier=identifier,
        candidate_found_screenshot=candidate_found,
        launch_control_screenshot=launch_control,
    )


def run_join_dry_run(
    page: JoinWorkflowPage,
    *,
    candidate_name: str,
    session_dir: Path,
    action_router: ActionRouter,
) -> JoinDryRunResult:
    """Find one exact candidate and prove launch is blocked by dry-run policy."""

    prepared = prepare_launch_control(
        page,
        candidate_name=candidate_name,
        session_dir=session_dir,
        action_router=action_router,
        launch_screenshot_name="join_dry_run",
    )
    decision = action_router.route(
        BrowserAction.LAUNCH_INTERVIEW,
        operation=page.click_launch_interview,
        candidate_identifier=prepared.candidate_identifier,
        screenshot_path=prepared.launch_control_screenshot,
    )
    if decision.allowed:
        raise JoinWorkflowError("Dry-run safety policy unexpectedly allowed launch")

    return JoinDryRunResult(
        candidate_identifier=prepared.candidate_identifier,
        candidate_found_screenshot=prepared.candidate_found_screenshot,
        join_dry_run_screenshot=prepared.launch_control_screenshot,
        action_log_path=action_router.log_path,
    )


def run_join_live(
    page: JoinWorkflowPage,
    *,
    candidate_name: str,
    session_dir: Path,
    action_router: ActionRouter,
    request_approval: ApprovalRequester,
) -> JoinLiveResult:
    """Launch, accept consent, and Join after three operator approvals."""

    prepared = prepare_launch_control(
        page,
        candidate_name=candidate_name,
        session_dir=session_dir,
        action_router=action_router,
        launch_screenshot_name="launch_approval",
    )
    identifier = prepared.candidate_identifier
    launch_token = request_approval(BrowserAction.LAUNCH_INTERVIEW, identifier)
    launch_decision = action_router.route(
        BrowserAction.LAUNCH_INTERVIEW,
        operation=page.click_launch_interview,
        candidate_identifier=identifier,
        approval_token=launch_token,
        screenshot_path=prepared.launch_control_screenshot,
    )
    if not launch_decision.allowed:
        raise JoinWorkflowError("Launch approval was not granted; nothing launched")

    screenshots_dir = session_dir / "screenshots"
    consent: Path | None = None
    try:
        post_launch_state = page.wait_for_consent_or_pre_call()
    except Exception as error:
        diagnostic = page.capture_screenshot(screenshots_dir, "post_launch_state_error")
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error

    if post_launch_state is PostLaunchState.CONSENT:
        consent_controls = page.visible_consent_ok_count()
        if consent_controls != 1:
            diagnostic = page.capture_screenshot(
                screenshots_dir, "consent_ok_selector_error"
            )
            raise JoinWorkflowError(
                f"Expected exactly one consent OK control; found {consent_controls}. "
                f"Screenshot: {diagnostic}"
            )

        consent = page.capture_screenshot(screenshots_dir, "consent")
        consent_token = request_approval(BrowserAction.CLICK_CONSENT_OK, identifier)
        consent_decision = action_router.route(
            BrowserAction.CLICK_CONSENT_OK,
            operation=page.click_consent_ok,
            candidate_identifier=identifier,
            approval_token=consent_token,
            screenshot_path=consent,
        )
        if not consent_decision.allowed:
            raise JoinWorkflowError(
                "Consent approval was not granted; OK was not clicked"
            )

    try:
        if post_launch_state is PostLaunchState.CONSENT:
            page.wait_for_pre_call_page()
        join_controls = page.visible_join_control_count()
    except Exception as error:
        diagnostic = page.capture_screenshot(screenshots_dir, "pre_call_error")
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error
    if join_controls != 1:
        diagnostic = page.capture_screenshot(
            screenshots_dir, "join_control_selector_error"
        )
        raise JoinWorkflowError(
            f"Expected exactly one visible Join control; found {join_controls}. "
            f"Screenshot: {diagnostic}"
        )

    pre_call = page.capture_screenshot(screenshots_dir, "pre_call")
    join_token = request_approval(BrowserAction.CLICK_JOIN, identifier)
    try:
        join_decision = action_router.route(
            BrowserAction.CLICK_JOIN,
            operation=page.click_join,
            candidate_identifier=identifier,
            approval_token=join_token,
            screenshot_path=pre_call,
        )
    except Exception as error:
        diagnostic = page.capture_screenshot(
            screenshots_dir, "join_click_revalidation_error"
        )
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error
    if not join_decision.allowed:
        raise JoinWorkflowError("Join approval was not granted; Join was not clicked")

    try:
        page.wait_for_joined_interview()
    except Exception as error:
        diagnostic = page.capture_screenshot(screenshots_dir, "join_result_error")
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error
    page.bind_candidate_identifier(identifier)
    joined = page.capture_screenshot(screenshots_dir, "joined")
    return JoinLiveResult(
        candidate_identifier=identifier,
        candidate_found_screenshot=prepared.candidate_found_screenshot,
        launch_approval_screenshot=prepared.launch_control_screenshot,
        consent_screenshot=consent,
        pre_call_screenshot=pre_call,
        joined_screenshot=joined,
        action_log_path=action_router.log_path,
    )
