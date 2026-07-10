"""Persistent, read-only Playwright controller for FloCareer dashboard scans."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from playwright.sync_api import sync_playwright

from app.config import Settings
from browser.flocareer_page import FloCareerPage, ScheduledInterview
from browser.screenshots import save_screenshot


class BrowserScanError(RuntimeError):
    """Raised when a safe dashboard scan cannot complete."""


@dataclass(frozen=True, slots=True)
class BrowserScanResult:
    session_id: str
    interviews: tuple[ScheduledInterview, ...]
    screenshot_path: Path
    login_was_required: bool


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
    settings.browser_user_data_dir.mkdir(parents=True, exist_ok=True)

    with sync_playwright() as playwright:
        context = playwright.chromium.launch_persistent_context(
            user_data_dir=str(settings.browser_user_data_dir),
            headless=settings.browser_headless,
            viewport={"width": 1440, "height": 1000},
        )
        try:
            page = context.pages[0] if context.pages else context.new_page()
            flocareer = FloCareerPage(page)
            report(f"Opening {settings.flocareer_url}")
            flocareer.open_dashboard(settings.flocareer_url)
            initial_state = flocareer.wait_for_initial_state(
                timeout_seconds=min(10, login_timeout_seconds)
            )
            login_was_required = initial_state != "dashboard"
            if initial_state == "dashboard" and not flocareer.remains_dashboard_ready():
                login_was_required = True

            if login_was_required:
                if settings.browser_headless:
                    screenshot = save_screenshot(
                        page, screenshots_dir, "login_required"
                    )
                    raise BrowserScanError(
                        "FloCareer login is required but BROWSER_HEADLESS=true. "
                        f"Screenshot: {screenshot}"
                    )
                report(
                    "Dashboard is logged out or still loading. Complete your normal "
                    "FloCareer login manually in the opened browser; the scanner "
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
                    screenshot = save_screenshot(page, screenshots_dir, "login_timeout")
                    raise BrowserScanError(
                        "Timed out waiting for manual FloCareer login. "
                        f"Screenshot: {screenshot}"
                    )

            if not flocareer.remains_dashboard_ready(duration_seconds=1):
                screenshot = save_screenshot(
                    page, screenshots_dir, "dashboard_not_ready"
                )
                raise BrowserScanError(
                    f"FloCareer dashboard did not become ready. Screenshot: {screenshot}"
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
        finally:
            context.close()
