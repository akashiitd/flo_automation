"""FloCareer page model for reads and explicitly guarded reversible actions."""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Literal, cast

from playwright.sync_api import Locator, Page

from browser.code_editor_workflow import (
    CodeEditorVisibility,
    CodeEditorWorkflowError,
)
from browser.join_workflow import (
    CandidateCardHandle,
    JoinCandidate,
    JoinWorkflowError,
    PostLaunchState,
)
from browser.question_workflow import (
    CodeEditorAssociationStatus,
    CodeEditorControlObservation,
    CodeEditorDomObservation,
    CodeEditorQuestionIdSource,
    ExtractedQuestion,
    StructuralDomSnapshot,
)
from browser.room_workflow import InterviewRoomState
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

    def __init__(
        self,
        page: Page,
    ) -> None:
        self.page = page
        self._active_candidate_identifier: str | None = None
        self._active_candidate_name: str | None = None
        self._candidate_cards: dict[CandidateCardHandle, Locator] = {}
        self._active_candidate_menu: Locator | None = None
        self._active_consent_dialog: Locator | None = None
        self._launch_source_page: Page | None = None
        self._pages_before_launch: tuple[Page, ...] = ()

    def bind_candidate_identifier(
        self,
        candidate_identifier: str,
        *,
        candidate_name: str | None = None,
    ) -> None:
        if not candidate_identifier.strip():
            raise ValueError("candidate identifier is required")
        if candidate_name is not None and not candidate_name.strip():
            raise ValueError("candidate name must not be blank")
        self._active_candidate_identifier = candidate_identifier
        self._active_candidate_name = candidate_name

    def active_candidate_matches(self, candidate_identifier: str) -> bool:
        return self._active_candidate_identifier == candidate_identifier

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
            if controls.nth(index).is_visible() and controls.nth(index).is_enabled()
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

        if cls._is_interviewer_in_room(page):
            return InterviewPageState.JOINED
        return InterviewPageState.OTHER

    @classmethod
    def _is_interviewer_in_room(cls, page: Page) -> bool:
        """Require multiple independent live-room signals, not one control."""

        body_text = page.locator("body").inner_text(timeout=5_000)
        joined_markers = page.locator(", ".join(JOINED_INTERVIEW_SELECTORS))
        has_call_control = any(
            joined_markers.nth(index).is_visible()
            for index in range(joined_markers.count())
        )
        signals = (
            cls._has_question_panel(page),
            bool(re.search(r"^\s*REC\s*$", body_text, re.M)),
            bool(re.search(r"\b\d{1,2}:\d{2}:\d{2}\b", body_text)),
            has_call_control,
            bool(re.search(r"^\s*Interview room\s*$", body_text, re.M | re.I)),
        )
        return sum(signals) >= 2 and not cls._visible_join_controls(page)

    def read_interview_room_state(self) -> InterviewRoomState:
        """Return the live candidate connection state after room entry."""

        if not self._is_interviewer_in_room(self.page):
            raise JoinWorkflowError("Interview room is no longer verified")
        if self._active_candidate_name is None:
            raise JoinWorkflowError(
                "Active candidate name is not bound to this session"
            )
        candidate_names = self.page.get_by_text(self._active_candidate_name, exact=True)
        visible_names = [
            candidate_names.nth(index)
            for index in range(candidate_names.count())
            if candidate_names.nth(index).is_visible()
        ]
        if len(visible_names) != 1:
            raise JoinWorkflowError(
                "Expected one visible exact candidate name in the interview room; "
                f"found {len(visible_names)}"
            )
        has_online_status = bool(
            visible_names[0].evaluate(
                """element => {
                  const parent = element.parentElement;
                  if (!parent) return false;
                  const onlineSiblings = [...parent.children].filter(
                    child => (child.innerText || '').trim().toUpperCase() === 'ONLINE'
                  );
                  return onlineSiblings.length === 1;
                }"""
            )
        )
        return (
            InterviewRoomState.CANDIDATE_CONNECTED
            if has_online_status
            else InterviewRoomState.WAITING_FOR_CANDIDATE
        )

    def candidate_is_connected(self) -> bool:
        return (
            self.read_interview_room_state() is InterviewRoomState.CANDIDATE_CONNECTED
        )

    def wait_for_room_poll(self, seconds: float) -> None:
        self.page.wait_for_timeout(seconds * 1_000)

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
                has_code_editor: [...root.querySelectorAll('[role="tab"]')]
                  .some(node => (node.textContent || '').trim() === 'Code Editor'),
                title_text: (() => {
                  const title = root.querySelector(
                    '.question-title, [data-testid="question-title"], [name="title"]'
                  );
                  return title ? (title.innerText || '').trim() : '';
                })(),
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
        title_text = str(snapshot.get("title_text") or "").strip()
        if title_text:
            question_text = title_text
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

    def inspect_code_editor_dom(
        self,
        *,
        open_code_editor_tabs: bool = False,
        coding_question_ids: tuple[int, ...] = (),
    ) -> list[CodeEditorDomObservation]:
        """Capture coding-card structure, optionally via reversible tab navigation."""

        tabs_to_restore: list[Locator] = []

        def restore_question_tabs() -> None:
            for question_tab in tabs_to_restore:
                question_tab.click()
                if question_tab.get_attribute("aria-selected") != "true":
                    raise CodeEditorWorkflowError(
                        "Question tab did not become active again"
                    )

        if open_code_editor_tabs:
            try:
                for question_id in coding_question_ids:
                    root = self._question_root(question_id)
                    editor_tabs = self._visible_locators(
                        root.get_by_role("tab", name="Code Editor", exact=True)
                    )
                    question_tabs = self._visible_locators(
                        root.get_by_role("tab", name="Question", exact=True)
                    )
                    if len(editor_tabs) != 1 or len(question_tabs) != 1:
                        raise CodeEditorWorkflowError(
                            "Expected one visible Question and Code Editor tab for "
                            f"question {question_id}"
                        )
                    if editor_tabs[0].get_attribute("aria-selected") != "true":
                        self.open_code_editor_tab(question_id)
                        tabs_to_restore.append(question_tabs[0])
            except Exception:
                restore_question_tabs()
                raise

        try:
            raw_observations = self.page.evaluate(
                r"""
            async () => {
              const MAX_HTML_LENGTH = 50000;
              const sha256 = (value) => {
                const rightRotate = (word, amount) =>
                  (word >>> amount) | (word << (32 - amount));
                const maxWord = 2 ** 32;
                const bytes = new TextEncoder().encode(value);
                let ascii = '';
                bytes.forEach(byte => { ascii += String.fromCharCode(byte); });
                const bitLength = ascii.length * 8;
                const words = [];
                const hash = [];
                const constants = [];
                const composites = {};
                for (let candidate = 2; constants.length < 64; candidate += 1) {
                  if (composites[candidate]) continue;
                  for (let multiple = candidate; multiple < 313;
                       multiple += candidate) composites[multiple] = true;
                  hash.push((candidate ** 0.5 * maxWord) | 0);
                  constants.push((candidate ** (1 / 3) * maxWord) | 0);
                }
                ascii += '\x80';
                while (ascii.length % 64 !== 56) ascii += '\x00';
                for (let index = 0; index < ascii.length; index += 1) {
                  words[index >> 2] |= ascii.charCodeAt(index)
                    << ((3 - index) % 4) * 8;
                }
                words.push((bitLength / maxWord) | 0, bitLength);
                let state = hash.slice(0, 8);
                for (let offset = 0; offset < words.length; offset += 16) {
                  const schedule = words.slice(offset, offset + 16);
                  const previous = state.slice();
                  for (let round = 0; round < 64; round += 1) {
                    const w15 = schedule[round - 15];
                    const w2 = schedule[round - 2];
                    schedule[round] = round < 16 ? schedule[round] : (
                      schedule[round - 16]
                      + (rightRotate(w15, 7) ^ rightRotate(w15, 18) ^ (w15 >>> 3))
                      + schedule[round - 7]
                      + (rightRotate(w2, 17) ^ rightRotate(w2, 19) ^ (w2 >>> 10))
                    ) | 0;
                    const a = state[0];
                    const e = state[4];
                    const temporary1 = state[7]
                      + (rightRotate(e, 6) ^ rightRotate(e, 11)
                         ^ rightRotate(e, 25))
                      + ((e & state[5]) ^ ((~e) & state[6]))
                      + constants[round] + schedule[round];
                    const temporary2 = (rightRotate(a, 2) ^ rightRotate(a, 13)
                      ^ rightRotate(a, 22))
                      + ((a & state[1]) ^ (a & state[2])
                         ^ (state[1] & state[2]));
                    state = [(temporary1 + temporary2) | 0].concat(state);
                    state[4] = (state[4] + temporary1) | 0;
                    state.pop();
                  }
                  state = state.map((word, index) =>
                    (word + previous[index]) | 0
                  );
                }
                return state.map(word => [...Array(4)].map((_, index) => {
                  const byte = (word >>> ((3 - index) * 8)) & 255;
                  return byte.toString(16).padStart(2, '0');
                }).join('')).join('');
              };
              const rendered = (element) => {
                if (!element.isConnected) return false;
                let node = element;
                while (node && node !== document.documentElement) {
                  if (node.hasAttribute('hidden') || node.hasAttribute('inert')
                      || node.getAttribute('aria-hidden') === 'true') return false;
                  const style = getComputedStyle(node);
                  if (style.display === 'none' || style.visibility === 'hidden') {
                    return false;
                  }
                  node = node.parentElement;
                }
                const rect = element.getBoundingClientRect();
                return rect.width > 0 && rect.height > 0;
              };
              const text = (element) => (element.textContent || '').trim();
              const snapshot = async (element) => {
                if (!element) return null;
                const clone = element.cloneNode(true);
                clone.querySelectorAll('script, style').forEach(node => node.remove());
                [clone, ...clone.querySelectorAll('*')].forEach(node => {
                  [...node.attributes].forEach(attribute => {
                    const allowed = [
                      'class', 'role', 'type', 'aria-label', 'aria-selected',
                      'aria-hidden', 'data-testid', 'data-question-id',
                      'data-question-number', 'name', 'for', 'hidden', 'inert',
                      'checked', 'disabled'
                    ].includes(attribute.name)
                      || (attribute.name === 'id'
                          && attribute.value.startsWith('container-'));
                    if (!allowed) node.removeAttribute(attribute.name);
                  });
                  [...node.childNodes].forEach(child => {
                    if (child.nodeType !== 3) return;
                    const value = (child.textContent || '').trim();
                    if (!value) return;
                    const keep = /^\d{1,3}$/.test(value)
                      || ['Question', 'Code Editor',
                          'SHOW CODE EDITOR TO CANDIDATE',
                          'HIDE CODE EDITOR TO CANDIDATE'].includes(value);
                    if (!keep) child.textContent = '[redacted]';
                  });
                });
                const html = clone.outerHTML;
                return {
                  html: html.slice(0, MAX_HTML_LENGTH),
                  truncated: html.length > MAX_HTML_LENGTH,
                  sha256: sha256(html),
                };
              };
              const preferredRoots = [...document.querySelectorAll(
                '.clMainSingleFESug[id^="container-"]'
              )];
              const normalizedPreferred = preferredRoots.filter(root =>
                !preferredRoots.some(other => other !== root && other.contains(root))
              );
              const fallbackCandidates = [...document.querySelectorAll(
                '[data-question-id], [data-testid="question-card"], .question-card'
              )].filter(root => !normalizedPreferred.some(preferred =>
                preferred.contains(root) || root.contains(preferred)
              ));
              const normalizedFallback = fallbackCandidates.filter(root =>
                !fallbackCandidates.some(other =>
                  other !== root && other.contains(root)
                )
              );
              const roots = [...normalizedPreferred, ...normalizedFallback];
              const describeControl = async (control) => ({
                tag_name: control.tagName.toLowerCase(),
                role: control.getAttribute('role'),
                input_type: control.getAttribute('type'),
                aria_label: control.getAttribute('aria-label'),
                test_id: control.getAttribute('data-testid'),
                name: control.getAttribute('name'),
                class_name: control.getAttribute('class'),
                rendered: rendered(control),
                outer_html: await snapshot(control),
              });
              const switchControls = (container) => [...container.querySelectorAll(
                '[role="switch"], input[type="checkbox"]'
              )];

              const observations = [];
              for (const root of roots) {
                const tabs = [...root.querySelectorAll('[role="tab"]')]
                  .filter(tab => text(tab) === 'Code Editor');
                if (tabs.length === 0) continue;

                const explicitNumbers = [...root.querySelectorAll(
                  '[data-question-number], .question-number, .clSeqGreen'
                )].filter(node => /^\d{1,3}$/.test(text(node)));
                const numericCandidates = [...new Set(
                  [...root.querySelectorAll('*')]
                    .map(node => text(node))
                    .filter(value => /^\d{1,3}$/.test(value))
                    .map(Number)
                )].sort((left, right) => left - right);
                const dataQuestionId = root.getAttribute('data-question-id');
                let questionId = null;
                let questionIdSource = 'unresolved';
                let number = explicitNumbers.length === 1
                  ? explicitNumbers[0] : null;
                if (dataQuestionId && /^\d{1,3}$/.test(dataQuestionId)) {
                  questionId = Number(dataQuestionId);
                  questionIdSource = 'data-question-id';
                } else if (number) {
                  questionId = Number(text(number));
                  questionIdSource = 'question-number-element';
                }

                const labels = [...root.querySelectorAll('.clFloSwithTxt')];
                let wrapper = null;
                let controls = [];
                if (labels.length === 1) {
                  let node = labels[0].parentElement;
                  while (node && root.contains(node)) {
                    const candidates = switchControls(node);
                    if (candidates.length > 0) {
                      wrapper = node;
                      controls = candidates;
                      break;
                    }
                    if (node === root) break;
                    node = node.parentElement;
                  }
                }

                const recognizedLabels = labels.map(text);
                const labelIsKnown = recognizedLabels.length === 1
                  && ['SHOW CODE EDITOR TO CANDIDATE',
                      'HIDE CODE EDITOR TO CANDIDATE'].includes(recognizedLabels[0]);
                let status = 'ambiguous';
                if (questionId === null || tabs.length !== 1 || labels.length > 1) {
                  status = 'ambiguous';
                } else if (labels.length === 0 || controls.length === 0) {
                  status = 'none';
                } else if (labelIsKnown && controls.length === 1) {
                  status = 'unique';
                }

                observations.push({
                  question_id: questionId,
                  question_id_source: questionIdSource,
                  question_id_candidates: numericCandidates,
                  code_editor_tab_count: tabs.length,
                  rendered_code_editor_tab_count: tabs.filter(rendered).length,
                  visibility_labels: recognizedLabels,
                  visibility_label_rendered: labels.map(rendered),
                  switch_candidates: await Promise.all(
                    controls.map(describeControl)
                  ),
                  association_status: status,
                  question_number_outer_html: await snapshot(number),
                  control_wrapper_outer_html: await snapshot(wrapper),
                  association_container_outer_html: await snapshot(root),
                });
              }
              return observations;
            }
            """
            )
        finally:
            restore_question_tabs()

        def structural_snapshot(raw: object) -> StructuralDomSnapshot | None:
            if not isinstance(raw, dict):
                return None
            html = str(raw.get("html") or "")
            return StructuralDomSnapshot(
                html=html,
                truncated=bool(raw.get("truncated")),
                sha256=str(raw.get("sha256") or ""),
            )

        observations: list[CodeEditorDomObservation] = []
        for raw in raw_observations:
            controls = tuple(
                CodeEditorControlObservation(
                    tag_name=str(control.get("tag_name") or ""),
                    role=(str(control["role"]) if control.get("role") else None),
                    input_type=(
                        str(control["input_type"])
                        if control.get("input_type")
                        else None
                    ),
                    aria_label=(
                        str(control["aria_label"])
                        if control.get("aria_label")
                        else None
                    ),
                    test_id=(
                        str(control["test_id"]) if control.get("test_id") else None
                    ),
                    name=(str(control["name"]) if control.get("name") else None),
                    class_name=(
                        str(control["class_name"])
                        if control.get("class_name")
                        else None
                    ),
                    rendered=bool(control.get("rendered")),
                    outer_html=structural_snapshot(control.get("outer_html"))
                    or StructuralDomSnapshot(
                        html="",
                        truncated=False,
                        sha256=hashlib.sha256(b"").hexdigest(),
                    ),
                )
                for control in raw.get("switch_candidates", [])
            )
            status = str(raw.get("association_status") or "ambiguous")
            if status not in {"unique", "none", "ambiguous"}:
                status = "ambiguous"
            observations.append(
                CodeEditorDomObservation(
                    question_id=(
                        int(raw["question_id"])
                        if raw.get("question_id") is not None
                        else None
                    ),
                    question_id_source=cast(
                        CodeEditorQuestionIdSource,
                        str(raw.get("question_id_source") or "unresolved"),
                    ),
                    question_id_candidates=tuple(
                        int(value) for value in raw.get("question_id_candidates", [])
                    ),
                    code_editor_tab_count=int(raw.get("code_editor_tab_count") or 0),
                    rendered_code_editor_tab_count=int(
                        raw.get("rendered_code_editor_tab_count") or 0
                    ),
                    visibility_labels=tuple(
                        str(label) for label in raw.get("visibility_labels", [])
                    ),
                    visibility_label_rendered=tuple(
                        bool(value)
                        for value in raw.get("visibility_label_rendered", [])
                    ),
                    switch_candidates=controls,
                    association_status=cast(CodeEditorAssociationStatus, status),
                    question_number_outer_html=structural_snapshot(
                        raw.get("question_number_outer_html")
                    ),
                    control_wrapper_outer_html=structural_snapshot(
                        raw.get("control_wrapper_outer_html")
                    ),
                    association_container_outer_html=structural_snapshot(
                        raw.get("association_container_outer_html")
                    )
                    or StructuralDomSnapshot(
                        html="",
                        truncated=False,
                        sha256=hashlib.sha256(b"").hexdigest(),
                    ),
                )
            )
        return sorted(
            observations,
            key=lambda observation: (
                observation.question_id is None,
                observation.question_id or 0,
            ),
        )

    def _question_root(self, question_id: int) -> Locator:
        if question_id < 1:
            raise CodeEditorWorkflowError("question id must be positive")
        roots = self.page.locator(
            '[data-question-id], [data-testid="question-card"], .question-card, '
            '.clMainSingleFESug[id^="container-"], .clMainSingleFESug'
        )
        matches: list[Locator] = []
        for index in range(roots.count()):
            root = roots.nth(index)
            if not root.is_visible():
                continue
            if root.get_attribute("data-question-id") == str(question_id):
                matches.append(root)
                continue
            number = root.locator(
                "[data-question-number], .question-number, .clSeqGreen"
            ).get_by_text(str(question_id), exact=True)
            if any(
                number.nth(number_index).is_visible()
                for number_index in range(number.count())
            ):
                matches.append(root)
        if len(matches) != 1:
            raise CodeEditorWorkflowError(
                f"Expected one visible question card for question {question_id}; "
                f"found {len(matches)}"
            )
        return matches[0]

    @staticmethod
    def _visible_locators(locator: Locator) -> list[Locator]:
        return [
            locator.nth(index)
            for index in range(locator.count())
            if locator.nth(index).is_visible()
        ]

    def open_code_editor_tab(self, question_id: int) -> None:
        root = self._question_root(question_id)
        tabs = self._visible_locators(
            root.get_by_role("tab", name="Code Editor", exact=True)
        )
        if len(tabs) != 1:
            raise CodeEditorWorkflowError(
                f"Expected one Code Editor tab in question {question_id}; "
                f"found {len(tabs)}"
            )
        root.scroll_into_view_if_needed()
        if tabs[0].get_attribute("aria-selected") != "true":
            tabs[0].click()
        if not self.code_editor_tab_is_active(question_id):
            raise CodeEditorWorkflowError(
                f"Code Editor tab did not become active for question {question_id}"
            )

    def code_editor_tab_is_active(self, question_id: int) -> bool:
        root = self._question_root(question_id)
        tabs = self._visible_locators(
            root.get_by_role("tab", name="Code Editor", exact=True)
        )
        if len(tabs) != 1:
            raise CodeEditorWorkflowError(
                f"Expected one Code Editor tab in question {question_id}; "
                f"found {len(tabs)}"
            )
        return tabs[0].get_attribute("aria-selected") == "true"

    def _code_editor_state_label(self, question_id: int) -> Locator:
        root = self._question_root(question_id)
        labels = self._visible_locators(root.locator(".clFloSwithTxt"))
        if len(labels) != 1:
            raise CodeEditorWorkflowError(
                f"Code editor visibility state is ambiguous for question "
                f"{question_id}: found {len(labels)} visible state labels"
            )
        return labels[0]

    def read_code_editor_visibility(self, question_id: int) -> CodeEditorVisibility:
        value = _clean_text(self._code_editor_state_label(question_id).inner_text())
        if value == "SHOW CODE EDITOR TO CANDIDATE":
            return CodeEditorVisibility.HIDDEN
        if value == "HIDE CODE EDITOR TO CANDIDATE":
            return CodeEditorVisibility.VISIBLE
        raise CodeEditorWorkflowError(
            f"Code editor visibility state is ambiguous for question {question_id}"
        )

    def click_show_code_editor(self, question_id: int) -> None:
        if (
            self.read_code_editor_visibility(question_id)
            is not CodeEditorVisibility.HIDDEN
        ):
            raise CodeEditorWorkflowError(
                f"Code editor for question {question_id} is no longer hidden"
            )
        root = self._question_root(question_id)
        controls = self._visible_locators(
            root.locator('input[type="checkbox"][name^="codeSwitch-"]')
        )
        if len(controls) != 1:
            raise CodeEditorWorkflowError(
                f"Expected one code-editor switch for question {question_id}; "
                f"found {len(controls)}"
            )
        controls[0].click()

    def wait_for_code_editor_visibility(
        self,
        question_id: int,
        expected: CodeEditorVisibility,
        *,
        timeout_seconds: float = 5,
    ) -> None:
        deadline = time.monotonic() + timeout_seconds
        stable_polls = 0
        while time.monotonic() < deadline:
            try:
                matches = self.read_code_editor_visibility(question_id) is expected
            except CodeEditorWorkflowError:
                matches = False
            if matches:
                stable_polls += 1
                if stable_polls >= 3:
                    return
            else:
                stable_polls = 0
            self.page.wait_for_timeout(100)
        raise CodeEditorWorkflowError(
            f"Code editor for question {question_id} did not stabilize as "
            f"{expected.value}"
        )

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

    def wait_for_joined_interview(
        self, *, timeout_seconds: float | None = None
    ) -> None:
        deadline = (
            time.monotonic() + timeout_seconds if timeout_seconds is not None else None
        )
        stable_polls = 0
        while deadline is None or time.monotonic() < deadline:
            if self.page.is_closed():
                raise JoinWorkflowError("Interview page closed after clicking Join")
            if self._page_state(self.page) is InterviewPageState.JOINED:
                stable_polls += 1
                if stable_polls >= 3:
                    return
            else:
                stable_polls = 0
            self.page.wait_for_timeout(250)
        raise JoinWorkflowError("Configured room-entry timeout elapsed")
