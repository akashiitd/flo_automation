"""FloCareer page model for reads and explicitly guarded reversible actions."""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal

from playwright.sync_api import Locator, Page

from browser.join_workflow import (
    CandidateCardHandle,
    JoinCandidate,
    JoinWorkflowError,
    PostLaunchState,
)
from browser.question_workflow import ExtractedQuestion
from browser.screenshots import save_screenshot
from browser.selectors import (
    ACTIVE_MENU_SELECTORS,
    CANDIDATE_MENU_BUTTON_SELECTORS,
    INTERVIEW_ROW_SELECTORS,
    JOINED_INTERVIEW_SELECTORS,
    JOIN_CARD_SELECTORS,
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


class InterviewPageState(str, Enum):
    OTHER = "other"
    CONSENT = "consent"
    PRE_CALL = "pre_call"
    JOINED = "joined"


def _clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


_DATE_LINE = re.compile(
    r"^(?:(?:Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
    r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|"
    r"Dec(?:ember)?)\s+\d{1,2},\s+\d{4}|TODAY|TOMORROW)$",
    re.IGNORECASE,
)
_CARD_NOISE = {
    "subscribe to whatsapp",
    "notification",
    "⋮",
    "more_vert",
    "it's time",
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
    """Expose dashboard reads and candidate-scoped reversible menu actions."""

    def __init__(self, page: Page) -> None:
        self.page = page
        self._candidate_cards: dict[CandidateCardHandle, Locator] = {}
        self._active_candidate_menu: Locator | None = None
        self._active_consent_dialog: Locator | None = None
        self._launch_source_page: Page | None = None
        self._pages_before_launch: tuple[Page, ...] = ()

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

    def list_join_candidates(self) -> list[JoinCandidate]:
        """Bind interviews from the proven parser to card-scoped controls."""

        interviews = self.scan_scheduled_interviews()
        best_candidates: list[JoinCandidate] = []
        best_cards: dict[CandidateCardHandle, Locator] = {}
        for selector_index, selector in enumerate(JOIN_CARD_SELECTORS):
            roots = self.page.locator(selector)
            candidates: list[JoinCandidate] = []
            cards: dict[CandidateCardHandle, Locator] = {}
            for index in range(min(roots.count(), 100)):
                root = roots.nth(index)
                matches = [
                    interview
                    for interview in interviews
                    if root.get_by_text(interview.candidate_name, exact=True).count()
                    > 0
                ]
                distinct_names = {item.candidate_name.casefold() for item in matches}
                if len(distinct_names) != 1:
                    continue
                interview = matches[0]
                handle = CandidateCardHandle(f"card-{selector_index}-{index}")
                candidates.append(
                    JoinCandidate(
                        candidate_name=interview.candidate_name,
                        scheduled_time=interview.scheduled_time,
                        card_handle=handle,
                    )
                )
                cards[handle] = root
            if len(candidates) > len(best_candidates):
                best_candidates = candidates
                best_cards = cards
            if interviews and len(candidates) == len(interviews):
                break
        if interviews and not best_candidates:
            fallback_cards: dict[CandidateCardHandle, Locator] = {}
            fallback_candidates: list[JoinCandidate] = []
            for interview_index, interview in enumerate(interviews):
                name_nodes = self.page.get_by_text(interview.candidate_name, exact=True)
                for node_index in range(name_nodes.count()):
                    node = name_nodes.nth(node_index)
                    if not node.is_visible():
                        continue
                    card = node.locator(
                        "xpath=ancestor::div[.//button["
                        "translate(@aria-label, "
                        "'ABCDEFGHIJKLMNOPQRSTUVWXYZ', "
                        "'abcdefghijklmnopqrstuvwxyz')='more']][1]"
                    )
                    if card.count() != 1:
                        continue
                    handle = CandidateCardHandle(
                        f"candidate-relative-{interview_index}-{node_index}"
                    )
                    fallback_candidates.append(
                        JoinCandidate(
                            candidate_name=interview.candidate_name,
                            scheduled_time=interview.scheduled_time,
                            card_handle=handle,
                        )
                    )
                    fallback_cards[handle] = card
            best_candidates = fallback_candidates
            best_cards = fallback_cards
        self._candidate_cards = best_cards
        return best_candidates

    def capture_screenshot(self, directory: Path, name: str) -> Path:
        return save_screenshot(self.page, directory, name)

    def open_candidate_menu(self, candidate: JoinCandidate) -> None:
        card = self._candidate_cards.get(candidate.card_handle)
        if card is None:
            raise JoinWorkflowError("Candidate card handle is no longer available")
        for selector in CANDIDATE_MENU_BUTTON_SELECTORS:
            controls = card.locator(selector)
            visible = [
                controls.nth(index)
                for index in range(controls.count())
                if controls.nth(index).is_visible()
            ]
            if len(visible) == 1:
                control = visible[0]
                control.click()
                self._active_candidate_menu = self._resolve_active_menu(control)
                return
            if len(visible) > 1:
                raise JoinWorkflowError(
                    "Matched candidate card has multiple visible menu controls"
                )
        raise JoinWorkflowError("Matched candidate card has no visible menu control")

    def _resolve_active_menu(self, control: Locator) -> Locator:
        controlled_id = control.get_attribute("aria-controls") or control.get_attribute(
            "aria-owns"
        )
        if controlled_id:
            controlled = self.page.locator(f"[id={json.dumps(controlled_id)}]")
            if controlled.count() == 1 and controlled.is_visible():
                return controlled

        for _ in range(8):
            for selector in ACTIVE_MENU_SELECTORS:
                menus = self.page.locator(selector).filter(
                    has_text="Launch Video Interview"
                )
                visible = [
                    menus.nth(index)
                    for index in range(menus.count())
                    if menus.nth(index).is_visible()
                ]
                if len(visible) == 1:
                    return visible[0]
                if len(visible) > 1:
                    raise JoinWorkflowError(
                        "Candidate menu resolved to multiple visible launch menus"
                    )
            self.page.wait_for_timeout(250)
        raise JoinWorkflowError(
            "Could not bind the candidate menu to a visible launch control"
        )

    def visible_launch_control_count(self) -> int:
        if self._active_candidate_menu is None:
            return 0
        controls = self._active_candidate_menu.get_by_text(
            "Launch Video Interview", exact=True
        )
        return sum(
            controls.nth(index).is_visible() for index in range(controls.count())
        )

    def click_launch_interview(self) -> None:
        if self._active_candidate_menu is None:
            raise JoinWorkflowError("No candidate-scoped menu is active")
        controls = self._active_candidate_menu.get_by_text(
            "Launch Video Interview", exact=True
        )
        visible = [
            controls.nth(index)
            for index in range(controls.count())
            if controls.nth(index).is_visible()
        ]
        if len(visible) != 1:
            raise JoinWorkflowError(
                f"Expected one visible launch control; found {len(visible)}"
            )
        self._launch_source_page = self.page
        self._pages_before_launch = tuple(self.page.context.pages)
        visible[0].click()

    @staticmethod
    def _visible_join_controls(page: Page) -> list[Locator]:
        controls = page.get_by_role("button", name=re.compile(r"^Join$", re.I))
        return [
            controls.nth(index)
            for index in range(controls.count())
            if controls.nth(index).is_visible()
        ]

    @staticmethod
    def _visible_consent_dialogs(page: Page) -> list[Locator]:
        dialogs = page.get_by_role("dialog").filter(has_text="Interviewer Consent Form")
        visible = [
            dialogs.nth(index)
            for index in range(dialogs.count())
            if dialogs.nth(index).is_visible()
        ]
        if visible:
            return visible

        headings = page.get_by_text("Interviewer Consent Form", exact=True)
        for index in range(headings.count()):
            heading = headings.nth(index)
            if not heading.is_visible():
                continue
            container = heading.locator(
                "xpath=ancestor::div[.//button[normalize-space()='OK']][1]"
            )
            if container.count() == 1 and container.is_visible():
                visible.append(container)
        return visible

    def _is_launch_related_page(self, candidate_page: Page) -> bool:
        is_launch_source = candidate_page is self._launch_source_page
        existed_before_launch = candidate_page in self._pages_before_launch
        return is_launch_source or not existed_before_launch

    @classmethod
    def _page_state(cls, page: Page) -> InterviewPageState:
        if len(cls._visible_consent_dialogs(page)) == 1:
            return InterviewPageState.CONSENT
        joining_as = page.get_by_text(re.compile(r"\bJoining as\b", re.I))
        visible_joining_as = any(
            joining_as.nth(index).is_visible() for index in range(joining_as.count())
        )
        if visible_joining_as and len(cls._visible_join_controls(page)) == 1:
            return InterviewPageState.PRE_CALL

        joined_markers = page.locator(", ".join(JOINED_INTERVIEW_SELECTORS))
        if not cls._visible_join_controls(page) and any(
            joined_markers.nth(index).is_visible()
            for index in range(joined_markers.count())
        ):
            return InterviewPageState.JOINED
        return InterviewPageState.OTHER

    def wait_for_consent_or_pre_call(
        self, *, timeout_seconds: float = 30
    ) -> PostLaunchState:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            consent_matches: list[tuple[Page, Locator]] = []
            pre_call_matches: list[Page] = []
            for candidate_page in reversed(self.page.context.pages):
                if candidate_page.is_closed() or not self._is_launch_related_page(
                    candidate_page
                ):
                    continue
                dialogs = self._visible_consent_dialogs(candidate_page)
                consent_matches.extend((candidate_page, dialog) for dialog in dialogs)
                if self._page_state(candidate_page) is InterviewPageState.PRE_CALL:
                    pre_call_matches.append(candidate_page)
            total_matches = len(consent_matches) + len(pre_call_matches)
            if total_matches == 1 and consent_matches:
                self.page, self._active_consent_dialog = consent_matches[0]
                return PostLaunchState.CONSENT
            if total_matches == 1:
                self.page = pre_call_matches[0]
                return PostLaunchState.PRE_CALL
            if total_matches > 1:
                raise JoinWorkflowError(
                    "Post-Launch state is ambiguous across consent and pre-call pages"
                )
            self.page.wait_for_timeout(250)
        raise JoinWorkflowError(
            "Timed out waiting for consent form or verified pre-call page"
        )

    def wait_for_consent_form(self, *, timeout_seconds: float = 30) -> None:
        state = self.wait_for_consent_or_pre_call(timeout_seconds=timeout_seconds)
        if state is not PostLaunchState.CONSENT:
            raise JoinWorkflowError(
                "Verified pre-call page appeared without a consent form"
            )

    def visible_consent_ok_count(self) -> int:
        if self._active_consent_dialog is None:
            return 0
        controls = self._active_consent_dialog.get_by_role(
            "button", name=re.compile(r"^OK$", re.I)
        )
        return sum(
            controls.nth(index).is_visible() for index in range(controls.count())
        )

    def click_consent_ok(self) -> None:
        if self._active_consent_dialog is None:
            raise JoinWorkflowError("No scoped interviewer consent form is active")
        controls = self._active_consent_dialog.get_by_role(
            "button", name=re.compile(r"^OK$", re.I)
        )
        visible = [
            controls.nth(index)
            for index in range(controls.count())
            if controls.nth(index).is_visible()
        ]
        if len(visible) != 1:
            raise JoinWorkflowError(
                f"Expected one consent OK control; found {len(visible)}"
            )
        visible[0].click()

    @staticmethod
    def _question_card_snapshots(
        page: Page, *, expand: bool
    ) -> list[dict[str, object]]:
        return page.evaluate(
            r"""
            async ({expand}) => {
              const visible = (element) => {
                const style = getComputedStyle(element);
                const rect = element.getBoundingClientRect();
                return style.display !== 'none' && style.visibility !== 'hidden' && rect.height > 0;
              };
              const roots = [];
              const add = (element) => {
                if (element && visible(element) && !roots.includes(element)) roots.push(element);
              };
              document.querySelectorAll(
                '[data-question-id], [data-testid="question-card"], .question-card, '
                + '.clMainSingleFESug[id^="container-"]'
              ).forEach(add);
              const findRoot = (start, required) => {
                let node = start;
                let visualFallback = null;
                for (let depth = 0; node && depth < 10; depth += 1, node = node.parentElement) {
                  if (node === document.body || node === document.documentElement) break;
                  const text = node.innerText || '';
                  const rect = node.getBoundingClientRect();
                  const hasNumericChild = [...node.querySelectorAll('*')]
                    .some(child => /^\d{1,3}$/.test((child.textContent || '').trim()));
                  if (!visualFallback && rect.width > 450 && rect.height > 140
                      && ((/Bookmark in Video/i.test(text) && /Mark as/i.test(text))
                          || /SHOW CODE EDITOR TO CANDIDATE/i.test(text))) {
                    visualFallback = node;
                  }
                  const bookmarkCount = (text.match(/Bookmark in Video/g) || []).length;
                  const editorToggleCount = (text.match(/SHOW CODE EDITOR TO CANDIDATE/g) || []).length;
                  if ((/^\s*\d{1,3}\s*$/m.test(text) || hasNumericChild)
                      && required.every(value => text.includes(value))
                      && bookmarkCount <= 1 && editorToggleCount <= 1) {
                    return node;
                  }
                }
                return visualFallback;
              };
              if (roots.length === 0) {
                document.querySelectorAll('textarea').forEach((node) => {
                  add(findRoot(node, ['Mark as']));
                });
                [...document.querySelectorAll('body *')]
                  .filter(node => ['Bookmark in Video', 'YOUR RATING'].includes((node.textContent || '').trim()))
                  .forEach(node => add(findRoot(node, ['Mark as'])));
                [...document.querySelectorAll('body *')]
                  .filter(node => (node.textContent || '').trim() === 'SHOW CODE EDITOR TO CANDIDATE')
                  .forEach(node => add(findRoot(node, ['Code Editor'])));
              }

              if (expand) {
                for (const root of roots) {
                  if (root.querySelector('.MuiCollapse-entered')) continue;
                  const rootTop = root.getBoundingClientRect().top;
                  const explicit = root.querySelector(
                    '.question-title, [data-testid="question-title"], [name="title"]'
                  );
                  const choices = [...root.querySelectorAll('button, [role="button"], p, span, div')]
                    .filter(node => {
                      const text = (node.innerText || '').trim();
                      const top = node.getBoundingClientRect().top;
                      return visible(node) && text.length > 20 && text.length < 600
                        && top - rootTop < 120
                        && !/Ideal Answer|Guidelines|Bookmark|Feedback|SHOW CODE EDITOR/i.test(text);
                    });
                  const target = explicit || choices[0];
                  if (target) target.click();
                }
                await new Promise(resolve => setTimeout(resolve, 500));
              }
              return roots.map(root => ({
                id: root.getAttribute('data-question-id') ||
                  [...root.querySelectorAll('*')]
                    .map(node => (node.textContent || '').trim())
                    .find(text => /^\d{1,3}$/.test(text)),
                has_code_editor: [...root.querySelectorAll('[role="tab"], button, div')]
                  .some(node => (node.textContent || '').trim() === 'Code Editor'),
                expanded_question_text: (() => {
                  const detail = root.querySelector(
                    '.MuiCollapse-entered .clFESingleSugDet, '
                    + '.MuiCollapse-root .clFESingleSugDet'
                  );
                  return detail ? detail.innerText.trim() : '';
                })(),
                text: root.innerText || '',
              }));
            }
            """,
            {"expand": expand},
        )

    @classmethod
    def _has_question_panel(cls, page: Page) -> bool:
        body_text = page.locator("body").inner_text(timeout=5_000)
        return (
            "Bookmark in Video" in body_text
            and "Mark as" in body_text
            and bool(re.search(r"^\s*\d{1,3}\s*$", body_text, re.M))
        )

    def wait_for_questions_or_consent(
        self, *, timeout_seconds: float = 30
    ) -> PostLaunchState:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            matches: list[tuple[Page, PostLaunchState, Locator | None]] = []
            for candidate_page in reversed(self.page.context.pages):
                if candidate_page.is_closed() or not self._is_launch_related_page(
                    candidate_page
                ):
                    continue
                dialogs = self._visible_consent_dialogs(candidate_page)
                matches.extend(
                    (candidate_page, PostLaunchState.CONSENT, dialog)
                    for dialog in dialogs
                )
                if self._has_question_panel(candidate_page):
                    matches.append((candidate_page, PostLaunchState.QUESTIONS, None))
            if len(matches) == 1:
                self.page, state, dialog = matches[0]
                self._active_consent_dialog = dialog
                return state
            if len(matches) > 1:
                consent = [
                    match for match in matches if match[1] is PostLaunchState.CONSENT
                ]
                if len(consent) == 1:
                    self.page, state, self._active_consent_dialog = consent[0]
                    return state
                raise JoinWorkflowError(
                    "Question page state is ambiguous across browser pages"
                )
            self.page.wait_for_timeout(250)
        raise JoinWorkflowError("Timed out waiting for consent form or question panel")

    def wait_for_question_panel(self, *, timeout_seconds: float = 30) -> None:
        deadline = time.monotonic() + timeout_seconds
        stable_polls = 0
        while time.monotonic() < deadline:
            if self._has_question_panel(self.page):
                stable_polls += 1
                if stable_polls >= 4:
                    return
            else:
                stable_polls = 0
            self.page.wait_for_timeout(250)
        raise JoinWorkflowError("Timed out waiting for the interview question panel")

    @staticmethod
    def _parse_question_snapshot(snapshot: dict[str, object]) -> ExtractedQuestion:
        raw = str(snapshot.get("text") or "")
        lines = [_clean_text(line) for line in raw.splitlines() if _clean_text(line)]
        raw_id = str(snapshot.get("id") or "")
        id_match = re.search(r"\d+", raw_id) or re.search(
            r"^\s*(\d{1,3})\s*$", raw, re.M
        )
        if id_match is None:
            raise JoinWorkflowError("A question card has no readable numeric ID")
        question_id = int(id_match.group(0))
        stop_labels = re.compile(
            r"^(?:=+|Ideal Answer|Guidelines for|Bookmark|Mark as|Feedback|YOUR RATING)",
            re.I,
        )
        question_end = next(
            (
                index
                for index, line in enumerate(lines)
                if line.startswith("=") or line.lower().startswith("ideal answer")
            ),
            len(lines),
        )
        candidates = [
            line
            for line in lines[:question_end]
            if not line.isdigit()
            and not stop_labels.match(line)
            and line not in {"Question", "Code Editor", "SHOW CODE EDITOR TO CANDIDATE"}
            and not any(
                label in line
                for label in (
                    "Bookmark in Video",
                    "SHOW CODE EDITOR TO CANDIDATE",
                    "YOUR RATING",
                    "Use Speech to Text",
                    "Enhance Feedback",
                )
            )
        ]
        question_text = max(candidates, key=len, default="")
        expanded_text = str(snapshot.get("expanded_question_text") or "").strip()
        if expanded_text:
            question_text = "\n".join(
                _clean_text(line)
                for line in expanded_text.splitlines()
                if _clean_text(line)
            )
        question_text = re.sub(
            rf"^\s*{re.escape(str(question_id))}\s*", "", question_text
        )

        def section_after(label: re.Pattern[str]) -> str:
            for index, line in enumerate(lines):
                if label.match(line):
                    values: list[str] = []
                    for following in lines[index + 1 :]:
                        if stop_labels.match(following):
                            break
                        values.append(following)
                    return " ".join(values)
            return ""

        ideal = section_after(re.compile(r"^Ideal Answer", re.I))
        guidelines: dict[str, str] = {}
        for stars in range(5, 0, -1):
            value = section_after(
                re.compile(rf"^Guidelines for {stars} star rating", re.I)
            )
            if value:
                guidelines[f"{stars}_star"] = value
        return ExtractedQuestion(
            id=question_id,
            question_text=question_text,
            has_code_editor=bool(snapshot.get("has_code_editor")),
            ideal_answer=ideal,
            guidelines=guidelines,
            feedback_field_locator_hint=f"question:{question_id}:feedback",
            rating_locator_hint=f"question:{question_id}:rating",
            mark_as_locator_hint=f"question:{question_id}:mark_as",
        )

    def extract_questions(self) -> list[ExtractedQuestion]:
        snapshots = self._question_card_snapshots(self.page, expand=True)
        questions = [self._parse_question_snapshot(snapshot) for snapshot in snapshots]
        return sorted(questions, key=lambda question: question.id)

    def wait_for_pre_call_page(self, *, timeout_seconds: float = 30) -> None:
        deadline = time.monotonic() + timeout_seconds
        while time.monotonic() < deadline:
            matches: list[Page] = []
            for candidate_page in reversed(self.page.context.pages):
                if candidate_page.is_closed():
                    continue
                if not self._is_launch_related_page(candidate_page):
                    continue
                if self._page_state(candidate_page) is InterviewPageState.PRE_CALL:
                    matches.append(candidate_page)
            if len(matches) == 1:
                self.page = matches[0]
                return
            if len(matches) > 1:
                raise JoinWorkflowError(
                    "Multiple pages expose a visible pre-call Join control"
                )
            self.page.wait_for_timeout(250)
        raise JoinWorkflowError("Timed out waiting for the pre-call Join page")

    def visible_join_control_count(self) -> int:
        return len(self._visible_join_controls(self.page))

    def click_join(self) -> None:
        if self._page_state(self.page) is not InterviewPageState.PRE_CALL:
            raise JoinWorkflowError(
                "Pre-call page is no longer verified; Join was not clicked"
            )
        visible = self._visible_join_controls(self.page)
        if len(visible) != 1:
            raise JoinWorkflowError(
                f"Expected one visible Join control; found {len(visible)}"
            )
        visible[0].click()

    def wait_for_joined_interview(self, *, timeout_seconds: float = 30) -> None:
        deadline = time.monotonic() + timeout_seconds
        stable_polls = 0
        while time.monotonic() < deadline:
            if self.page.is_closed():
                raise JoinWorkflowError("Interview page closed after clicking Join")
            if self._page_state(self.page) is InterviewPageState.JOINED:
                stable_polls += 1
                if stable_polls >= 3:
                    return
            else:
                stable_polls = 0
            self.page.wait_for_timeout(250)
        raise JoinWorkflowError(
            "Timed out waiting for a stable interview room-ready marker"
        )
