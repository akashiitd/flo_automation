"""Persistent Playwright controller for guarded FloCareer workflows."""

from __future__ import annotations

import time
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import Iterator

from playwright.sync_api import sync_playwright

from app.config import Settings
from browser.action_guard import ActionGuard, BrowserAction
from browser.action_router import ActionRouter
from browser.code_editor_workflow import (
    CandidateDisconnectedError,
    CodeEditorApprovalRequester,
    CodeEditorResult,
    run_show_code_editor,
)
from browser.flocareer_page import FloCareerPage, ScheduledInterview
from browser.join_workflow import (
    ApprovalRequester,
    JoinDryRunResult,
    JoinLiveResult,
    JoinWorkflowError,
    run_join_dry_run,
    run_join_live,
    prepare_launch_control,
    PostLaunchState,
)
from browser.no_show_workflow import (
    NoShowApprovalRequester,
    NoShowResult,
    mark_no_show,
)
from browser.question_workflow import QuestionScanResult, run_question_scan
from browser.room_workflow import (
    InterviewRoomState,
    RoomMonitorResult,
    wait_for_candidate_connection,
    wait_for_no_show_eligibility,
)
from browser.screenshots import save_screenshot


class BrowserScanError(RuntimeError):
    """Raised when a safe dashboard scan cannot complete."""


@dataclass(frozen=True, slots=True)
class BrowserScanResult:
    session_id: str
    interviews: tuple[ScheduledInterview, ...]
    screenshot_path: Path
    login_was_required: bool


@dataclass(frozen=True, slots=True)
class NoShowLiveResult:
    joined: JoinLiveResult
    room: RoomMonitorResult
    no_show: NoShowResult | None


@contextmanager
def _persistent_flocareer_page(settings: Settings) -> Iterator[FloCareerPage]:
    settings.browser_user_data_dir.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(settings.browser_user_data_dir),
            headless=settings.browser_headless,
            viewport={"width": 1440, "height": 1000},
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            yield FloCareerPage(page)
        finally:
            context.close()


def _wait_for_authenticated_dashboard(
    settings: Settings,
    flocareer: FloCareerPage,
    *,
    screenshots_dir: Path,
    login_timeout_seconds: float,
    report: Callable[[str], None],
) -> bool:
    initial_state = flocareer.wait_for_initial_state(
        timeout_seconds=min(10, login_timeout_seconds)
    )
    login_was_required = initial_state != "dashboard"
    if initial_state == "dashboard" and not flocareer.remains_dashboard_ready():
        login_was_required = True

    if login_was_required:
        if settings.browser_headless:
            screenshot = save_screenshot(
                flocareer.page, screenshots_dir, "login_required"
            )
            raise BrowserScanError(
                "FloCareer login is required but BROWSER_HEADLESS=true. "
                f"Screenshot: {screenshot}"
            )
        report(
            "Dashboard is logged out or still loading. Complete your normal "
            "FloCareer login manually in the opened browser; the automation "
            "will continue automatically."
        )
        deadline = time.monotonic() + login_timeout_seconds
        while time.monotonic() < deadline:
            remaining = deadline - time.monotonic()
            state = flocareer.wait_for_initial_state(
                timeout_seconds=min(2, remaining),
                settle_seconds=min(1, remaining),
            )
            if state == "dashboard" and flocareer.remains_dashboard_ready():
                break
        else:
            screenshot = save_screenshot(
                flocareer.page, screenshots_dir, "login_timeout"
            )
            raise BrowserScanError(
                "Timed out waiting for manual FloCareer login. "
                f"Screenshot: {screenshot}"
            )

    if not flocareer.remains_dashboard_ready(duration_seconds=1):
        screenshot = save_screenshot(
            flocareer.page, screenshots_dir, "dashboard_not_ready"
        )
        raise BrowserScanError(
            f"FloCareer dashboard did not become ready. Screenshot: {screenshot}"
        )
    return login_was_required


def scan_dashboard(
    settings: Settings,
    *,
    login_timeout_seconds: float = 180,
    progress: Callable[[str], None] | None = None,
) -> BrowserScanResult:
    """Open, authenticate manually if needed, then read dashboard rows."""

    report = progress or (lambda message: None)
    session_id = f"browser_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    screenshots_dir = settings.runs_dir / session_id / "screenshots"

    with _persistent_flocareer_page(settings) as flocareer:
        page = flocareer.page
        report(f"Opening {settings.flocareer_url}")
        flocareer.open_dashboard(settings.flocareer_url)
        login_was_required = _wait_for_authenticated_dashboard(
            settings,
            flocareer,
            screenshots_dir=screenshots_dir,
            login_timeout_seconds=login_timeout_seconds,
            report=report,
        )

        page.wait_for_timeout(500)
        interviews = tuple(flocareer.scan_scheduled_interviews())
        screenshot = save_screenshot(page, screenshots_dir, "dashboard")
        if not flocareer.is_dashboard_ready():
            raise BrowserScanError(
                "FloCareer became logged out while the dashboard screenshot "
                f"was being saved. Screenshot: {screenshot}"
            )
        return BrowserScanResult(
            session_id=session_id,
            interviews=interviews,
            screenshot_path=screenshot,
            login_was_required=login_was_required,
        )


def join_candidate_dry_run(
    settings: Settings,
    *,
    candidate_name: str,
    login_timeout_seconds: float = 180,
    progress: Callable[[str], None] | None = None,
) -> JoinDryRunResult:
    """Locate one candidate's launch control without launching the interview."""

    report = progress or (lambda message: None)
    session_id = f"join_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = settings.runs_dir / session_id
    screenshots_dir = session_dir / "screenshots"
    router = ActionRouter(ActionGuard.dry_run(), session_dir / "action_log.jsonl")

    with _persistent_flocareer_page(settings) as flocareer:
        page = flocareer.page
        report(f"Opening {settings.flocareer_url}")
        router.route(
            BrowserAction.OPEN_DASHBOARD,
            operation=lambda: flocareer.open_dashboard(settings.flocareer_url),
        )
        _wait_for_authenticated_dashboard(
            settings,
            flocareer,
            screenshots_dir=screenshots_dir,
            login_timeout_seconds=login_timeout_seconds,
            report=report,
        )
        page.wait_for_timeout(500)
        return run_join_dry_run(
            flocareer,
            candidate_name=candidate_name,
            session_dir=session_dir,
            action_router=router,
        )


def join_candidate_live(
    settings: Settings,
    *,
    candidate_name: str,
    request_approval: ApprovalRequester,
    wait_for_manual_end: Callable[[str], None],
    enable_code_editor_question: int | None = None,
    request_code_editor_approval: CodeEditorApprovalRequester | None = None,
    candidate_wait_timeout_seconds: float | None = None,
    login_timeout_seconds: float = 180,
    progress: Callable[[str], None] | None = None,
) -> JoinLiveResult:
    """Keep one approved interview session open through candidate connection."""

    if enable_code_editor_question is not None:
        if enable_code_editor_question < 1:
            raise ValueError("code-editor question must be positive")
        if request_code_editor_approval is None:
            raise ValueError("code-editor approval requester is required")

    report = progress or (lambda message: None)
    session_id = f"join_live_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = settings.runs_dir / session_id
    screenshots_dir = session_dir / "screenshots"
    router = ActionRouter(ActionGuard.live_join(), session_dir / "action_log.jsonl")

    with _persistent_flocareer_page(settings) as flocareer:
        page = flocareer.page
        report(f"Opening {settings.flocareer_url}")
        router.route(
            BrowserAction.OPEN_DASHBOARD,
            operation=lambda: flocareer.open_dashboard(settings.flocareer_url),
        )
        _wait_for_authenticated_dashboard(
            settings,
            flocareer,
            screenshots_dir=screenshots_dir,
            login_timeout_seconds=login_timeout_seconds,
            report=report,
        )
        page.wait_for_timeout(500)
        result = run_join_live(
            flocareer,
            candidate_name=candidate_name,
            session_dir=session_dir,
            action_router=router,
            request_approval=request_approval,
        )
        room = wait_for_candidate_connection(
            flocareer,
            session_dir=session_dir,
            timeout_seconds=candidate_wait_timeout_seconds,
            report=report,
        )
        editor_result = None
        if enable_code_editor_question is not None:
            if request_code_editor_approval is None:
                raise AssertionError("validated editor approval requester is missing")
            while True:
                editor_router = ActionRouter(
                    ActionGuard.code_editor(), session_dir / "action_log.jsonl"
                )
                try:
                    editor_result = run_show_code_editor(
                        flocareer,
                        candidate_identifier=result.candidate_identifier,
                        question_id=enable_code_editor_question,
                        session_dir=session_dir,
                        action_router=editor_router,
                        request_approval=request_code_editor_approval,
                    )
                    break
                except CandidateDisconnectedError:
                    report("Candidate disconnected; waiting for reconnection")
                    room = wait_for_candidate_connection(
                        flocareer,
                        session_dir=session_dir,
                        timeout_seconds=candidate_wait_timeout_seconds,
                        report=report,
                        state_log_path=room.state_log_path,
                        prior_state=room.final_state,
                        prior_transitions=room.transitions,
                        initial_state=InterviewRoomState.WAITING_FOR_CANDIDATE,
                    )
        result = replace(
            result,
            room_state_log_path=room.state_log_path,
            code_editor_result=editor_result,
        )
        wait_for_manual_end(result.candidate_identifier)
        return result


def mark_candidate_no_show_live(
    settings: Settings,
    *,
    candidate_name: str,
    request_approval: ApprovalRequester,
    request_no_show_approval: NoShowApprovalRequester,
    wait_for_no_show_dialog: Callable[[str], None],
    wait_for_manual_end: Callable[[str], None],
    no_show_wait_seconds: float = 420,
    login_timeout_seconds: float = 180,
    progress: Callable[[str], None] | None = None,
) -> NoShowLiveResult:
    """Join one call, wait seven minutes, then allow a fresh no-show approval."""

    if no_show_wait_seconds < 420:
        raise ValueError("No-show wait must be at least 420 seconds")
    report = progress or (lambda message: None)
    session_id = f"no_show_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = settings.runs_dir / session_id
    screenshots_dir = session_dir / "screenshots"
    router = ActionRouter(ActionGuard.no_show(), session_dir / "action_log.jsonl")

    with _persistent_flocareer_page(settings) as flocareer:
        page = flocareer.page
        report(f"Opening {settings.flocareer_url}")
        router.route(
            BrowserAction.OPEN_DASHBOARD,
            operation=lambda: flocareer.open_dashboard(settings.flocareer_url),
        )
        _wait_for_authenticated_dashboard(
            settings,
            flocareer,
            screenshots_dir=screenshots_dir,
            login_timeout_seconds=login_timeout_seconds,
            report=report,
        )
        page.wait_for_timeout(500)
        joined = run_join_live(
            flocareer,
            candidate_name=candidate_name,
            session_dir=session_dir,
            action_router=router,
            request_approval=request_approval,
        )
        room = wait_for_no_show_eligibility(
            flocareer,
            session_dir=session_dir,
            wait_seconds=no_show_wait_seconds,
            report=report,
        )
        if room.final_state is InterviewRoomState.CANDIDATE_CONNECTED:
            report("Candidate connected; Mark No-show is blocked for this session")
            wait_for_manual_end(joined.candidate_identifier)
            return NoShowLiveResult(joined=joined, room=room, no_show=None)
        if not room.timed_out:
            raise BrowserScanError("Candidate wait ended without a no-show timeout")
        report("Seven-minute no-show window elapsed without candidate connection")
        wait_for_no_show_dialog(joined.candidate_identifier)
        no_show = mark_no_show(
            flocareer,
            joined=joined,
            session_dir=session_dir,
            action_router=router,
            request_approval=request_no_show_approval,
        )
        return NoShowLiveResult(joined=joined, room=room, no_show=no_show)


def scan_candidate_questions(
    settings: Settings,
    *,
    candidate_name: str,
    request_approval: ApprovalRequester,
    inspect_code_editor_tabs: bool = False,
    login_timeout_seconds: float = 180,
    progress: Callable[[str], None] | None = None,
) -> QuestionScanResult:
    """Launch one interview page and extract questions without clicking Join."""

    report = progress or (lambda message: None)
    session_id = f"questions_scan_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = settings.runs_dir / session_id
    screenshots_dir = session_dir / "screenshots"
    router = ActionRouter(ActionGuard.live_join(), session_dir / "action_log.jsonl")

    with _persistent_flocareer_page(settings) as flocareer:
        report(f"Opening {settings.flocareer_url}")
        router.route(
            BrowserAction.OPEN_DASHBOARD,
            operation=lambda: flocareer.open_dashboard(settings.flocareer_url),
        )
        _wait_for_authenticated_dashboard(
            settings,
            flocareer,
            screenshots_dir=screenshots_dir,
            login_timeout_seconds=login_timeout_seconds,
            report=report,
        )
        flocareer.page.wait_for_timeout(500)
        return run_question_scan(
            flocareer,
            candidate_name=candidate_name,
            session_dir=session_dir,
            action_router=router,
            request_approval=request_approval,
            inspect_code_editor_tabs=inspect_code_editor_tabs,
        )


def enable_candidate_code_editor_prejoin(
    settings: Settings,
    *,
    candidate_name: str,
    question_id: int,
    request_approval: ApprovalRequester,
    request_code_editor_approval: CodeEditorApprovalRequester,
    login_timeout_seconds: float = 180,
    progress: Callable[[str], None] | None = None,
) -> CodeEditorResult:
    """Show one exact editor after launch, without clicking pre-call Join."""

    report = progress or (lambda message: None)
    session_id = f"prejoin_code_editor_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    session_dir = settings.runs_dir / session_id
    screenshots_dir = session_dir / "screenshots"
    log_path = session_dir / "action_log.jsonl"
    live_router = ActionRouter(ActionGuard.live_join(), log_path)
    editor_router = ActionRouter(ActionGuard.code_editor(), log_path)

    with _persistent_flocareer_page(settings) as flocareer:
        report(f"Opening {settings.flocareer_url}")
        live_router.route(
            BrowserAction.OPEN_DASHBOARD,
            operation=lambda: flocareer.open_dashboard(settings.flocareer_url),
        )
        _wait_for_authenticated_dashboard(
            settings,
            flocareer,
            screenshots_dir=screenshots_dir,
            login_timeout_seconds=login_timeout_seconds,
            report=report,
        )
        prepared = prepare_launch_control(
            flocareer,
            candidate_name=candidate_name,
            session_dir=session_dir,
            action_router=live_router,
            launch_screenshot_name="prejoin_editor_launch_approval",
        )
        identifier = prepared.candidate_identifier
        decision = live_router.route(
            BrowserAction.LAUNCH_INTERVIEW,
            operation=flocareer.click_launch_interview,
            candidate_identifier=identifier,
            approval_token=request_approval(BrowserAction.LAUNCH_INTERVIEW, identifier),
            screenshot_path=prepared.launch_control_screenshot,
        )
        if not decision.allowed:
            raise JoinWorkflowError("Launch approval was not granted; nothing launched")
        state = flocareer.wait_for_questions_or_consent()
        if state is PostLaunchState.CONSENT:
            consent = flocareer.capture_screenshot(screenshots_dir, "prejoin_consent")
            if flocareer.visible_consent_ok_count() != 1:
                raise JoinWorkflowError("Expected exactly one consent OK control")
            consent_decision = live_router.route(
                BrowserAction.CLICK_CONSENT_OK,
                operation=flocareer.click_consent_ok,
                candidate_identifier=identifier,
                approval_token=request_approval(
                    BrowserAction.CLICK_CONSENT_OK, identifier
                ),
                screenshot_path=consent,
            )
            if not consent_decision.allowed:
                raise JoinWorkflowError(
                    "Consent approval was not granted; OK was not clicked"
                )
        flocareer.wait_for_question_panel()
        flocareer.bind_candidate_identifier(identifier, candidate_name=candidate_name)
        return run_show_code_editor(
            flocareer,
            candidate_identifier=identifier,
            question_id=question_id,
            session_dir=session_dir,
            action_router=editor_router,
            request_approval=request_code_editor_approval,
            allow_prejoin=True,
        )
