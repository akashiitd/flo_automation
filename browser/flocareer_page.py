"""Read-only FloCareer dashboard page model."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass
from typing import Literal

from playwright.sync_api import Page

from browser.selectors import (
    INTERVIEW_ROW_SELECTORS,
    LOADING_SELECTORS,
    LOGGED_OUT_TEXT,
)


@dataclass(frozen=True, slots=True)
class ScheduledInterview:
    candidate_name: str
    role: str
    company: str
    scheduled_time: str
    summary: str


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


_DATE_LINE = re.compile(
    r"^(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},\s+\d{4}$",
    re.IGNORECASE,
)
_CARD_NOISE = {
    "subscribe to whatsapp",
    "notification",
    "⋮",
    "more_vert",
}


def parse_scheduled_interviews_text(text: str) -> list[ScheduledInterview]:
    """Parse FloCareer's visible scheduled-card section without clicking it."""

    lines = [_clean_text(line) for line in text.splitlines() if _clean_text(line)]
    start = None
    for index, line in enumerate(lines):
        normalized = line.upper()
        if normalized.startswith("SCHEDULED INTERVIEWS"):
            start = index + 1
            break
        if normalized == "SCHEDULED" and index + 1 < len(lines):
            if lines[index + 1].upper().startswith("INTERVIEWS"):
                start = index + 2
                break
    if start is None:
        return []

    end = len(lines)
    for index in range(start, len(lines)):
        normalized = lines[index].upper()
        if normalized.startswith(("PENDING ACTIONS", "SET INTERVIEW STRUCTURE")):
            end = index
            break

    section = [
        line
        for line in lines[start:end]
        if line.lower() not in _CARD_NOISE
        and not line.lower().startswith("subscribe to whatsapp")
    ]
    interviews: list[ScheduledInterview] = []
    cursor = 0
    while cursor < len(section):
        date_index = next(
            (
                index
                for index in range(cursor, len(section))
                if _DATE_LINE.fullmatch(section[index])
            ),
            None,
        )
        if date_index is None:
            break

        details = [
            line
            for line in section[cursor:date_index]
            if not line.startswith("(") and not line.lower().startswith("at ")
        ]
        time_line = ""
        next_index = date_index + 1
        if next_index < len(section) and section[next_index].lower().startswith("at "):
            time_line = section[next_index]
            next_index += 1
        if next_index < len(section) and section[next_index].startswith("("):
            next_index += 1

        if len(details) >= 3:
            candidate_name = details[0]
            company = details[-1]
            role = " ".join(details[1:-1])
            scheduled_time = " ".join(
                value for value in (section[date_index], time_line) if value
            )
            summary = " | ".join((candidate_name, role, company, scheduled_time))
            interviews.append(
                ScheduledInterview(
                    candidate_name=candidate_name,
                    role=role,
                    company=company,
                    scheduled_time=scheduled_time,
                    summary=summary,
                )
            )
        cursor = next_index
    return interviews


class FloCareerPage:
    """Expose only non-destructive dashboard operations."""

    def __init__(self, page: Page) -> None:
        self.page = page

    def open_dashboard(self, url: str) -> None:
        self.page.goto(url, wait_until="domcontentloaded")

    def is_login_required(self) -> bool:
        url = self.page.url.lower()
        if any(marker in url for marker in ("login", "signin", "auth")):
            return True
        if self.page.locator('input[type="password"]').count() > 0:
            return True
        body_text = self.page.locator("body").inner_text(timeout=5_000).lower()
        return any(message in body_text for message in LOGGED_OUT_TEXT)

    def is_dashboard_ready(self) -> bool:
        if self.is_login_required():
            return False
        if self.is_loading():
            return False
        body_text = self.page.locator("body").inner_text(timeout=5_000).lower()
        return "dashboard" in body_text or "interviews" in body_text

    def is_loading(self) -> bool:
        for selector in LOADING_SELECTORS:
            loaders = self.page.locator(selector)
            for index in range(min(loaders.count(), 20)):
                if loaders.nth(index).is_visible():
                    return True
        return False

    def wait_for_initial_state(
        self,
        *,
        timeout_seconds: float,
        settle_seconds: float = 1.5,
    ) -> Literal["dashboard", "login_required", "unknown"]:
        """Wait through FloCareer's delayed authentication check."""

        started = time.monotonic()
        self.page.wait_for_timeout(min(settle_seconds, timeout_seconds) * 1_000)
        while time.monotonic() - started < timeout_seconds:
            if self.is_login_required():
                return "login_required"
            if self.is_dashboard_ready():
                return "dashboard"
            self.page.wait_for_timeout(250)
        return "unknown"

    def remains_dashboard_ready(self, *, duration_seconds: float = 3) -> bool:
        """Require readiness to remain stable across delayed auth checks."""

        deadline = time.monotonic() + duration_seconds
        while time.monotonic() < deadline:
            if not self.is_dashboard_ready():
                return False
            remaining_ms = max(1, (deadline - time.monotonic()) * 1_000)
            self.page.wait_for_timeout(min(250, remaining_ms))
        return self.is_dashboard_ready()

    def scan_scheduled_interviews(self) -> list[ScheduledInterview]:
        rows = None
        for selector in INTERVIEW_ROW_SELECTORS:
            candidate_rows = self.page.locator(selector)
            if candidate_rows.count() > 0:
                rows = candidate_rows
                break
        if rows is None:
            return parse_scheduled_interviews_text(
                self.page.locator("body").inner_text(timeout=5_000)
            )

        interviews: list[ScheduledInterview] = []
        seen_summaries: set[str] = set()
        for index in range(min(rows.count(), 100)):
            row = rows.nth(index)
            cells = row.locator("td, mat-cell, [role='cell']")
            cell_texts = [
                _clean_text(cells.nth(cell_index).inner_text())
                for cell_index in range(cells.count())
            ]
            values = [value for value in cell_texts if value]
            if not values:
                values = [
                    _clean_text(line)
                    for line in row.inner_text().splitlines()
                    if _clean_text(line)
                ]
            if not values:
                continue

            candidate_name = values[0]
            if candidate_name.lower() in {"candidate", "candidate name", "name"}:
                continue
            summary = " | ".join(values)
            if summary in seen_summaries:
                continue
            seen_summaries.add(summary)
            interviews.append(
                ScheduledInterview(
                    candidate_name=candidate_name,
                    role=values[1] if len(values) > 1 else "",
                    company=values[2] if len(values) > 2 else "",
                    scheduled_time=values[3] if len(values) > 3 else "",
                    summary=summary,
                )
            )
        if interviews:
            return interviews
        return parse_scheduled_interviews_text(
            self.page.locator("body").inner_text(timeout=5_000)
        )
