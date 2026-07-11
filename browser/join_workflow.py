"""Candidate-scoped, non-launching interview join discovery workflow."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
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


class JoinWorkflowPage(Protocol):
    def list_join_candidates(self) -> list[JoinCandidate]: ...

    def capture_screenshot(self, directory: Path, name: str) -> Path: ...

    def open_candidate_menu(self, candidate: JoinCandidate) -> None: ...

    def visible_launch_control_count(self) -> int: ...

    def click_launch_interview(self) -> None: ...


def normalize_candidate_name(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def _masked_name(value: str) -> str:
    words = re.findall(r"\S+", value)
    return " ".join(word[0] + "*" * (len(word) - 1) for word in words)


def _candidate_identifier(candidate: JoinCandidate) -> str:
    source = f"{normalize_candidate_name(candidate.candidate_name)}|{candidate.scheduled_time}"
    digest = hashlib.sha256(source.encode("utf-8")).hexdigest()[:12]
    return f"candidate-{digest}"


def run_join_dry_run(
    page: JoinWorkflowPage,
    *,
    candidate_name: str,
    session_dir: Path,
    action_router: ActionRouter,
) -> JoinDryRunResult:
    """Find one exact candidate and prove launch is blocked by dry-run policy."""

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

    join_dry_run = page.capture_screenshot(screenshots_dir, "join_dry_run")
    decision = action_router.route(
        BrowserAction.LAUNCH_INTERVIEW,
        operation=page.click_launch_interview,
        candidate_identifier=identifier,
        screenshot_path=join_dry_run,
    )
    if decision.allowed:
        raise JoinWorkflowError("Dry-run safety policy unexpectedly allowed launch")

    return JoinDryRunResult(
        candidate_identifier=identifier,
        candidate_found_screenshot=candidate_found,
        join_dry_run_screenshot=join_dry_run,
        action_log_path=action_router.log_path,
    )
