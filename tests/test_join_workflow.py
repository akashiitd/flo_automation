from __future__ import annotations

import json
from pathlib import Path

import pytest

from browser.action_guard import ActionGuard, BrowserAction, approval_token_for
from browser.action_router import ActionRouter
from browser.join_workflow import (
    AmbiguousCandidateError,
    CandidateCardHandle,
    CandidateNotFoundError,
    JoinCandidate,
    JoinLiveResult,
    JoinWorkflowError,
    PostLaunchState,
    run_join_dry_run,
    run_join_live,
)


class FakeJoinPage:
    def __init__(
        self,
        candidates: list[JoinCandidate],
        *,
        menu_error: Exception | None = None,
        consent_required: bool = True,
    ) -> None:
        self.candidates = candidates
        self.menu_error = menu_error
        self.consent_required = consent_required
        self.opened_tokens: list[str] = []
        self.launch_clicks = 0
        self.consent_clicks = 0
        self.join_clicks = 0
        self.events: list[str] = []
        self.candidate_identifier: str | None = None

    def bind_candidate_identifier(
        self,
        candidate_identifier: str,
        *,
        candidate_name: str | None = None,
    ) -> None:
        self.candidate_identifier = candidate_identifier
        self.events.append("candidate_bound")

    def list_join_candidates(self) -> list[JoinCandidate]:
        return self.candidates

    def capture_screenshot(self, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        path.write_bytes(b"fictional screenshot")
        return path

    def open_candidate_menu(self, candidate: JoinCandidate) -> None:
        if self.menu_error is not None:
            raise self.menu_error
        self.opened_tokens.append(candidate.card_handle.value)

    def visible_launch_control_count(self) -> int:
        return 1

    def click_launch_interview(self) -> None:
        self.launch_clicks += 1
        self.events.append("launch_clicked")

    def wait_for_pre_call_page(self) -> None:
        self.events.append("pre_call_ready")

    def wait_for_consent_form(self) -> None:
        self.events.append("consent_ready")

    def wait_for_consent_or_pre_call(self) -> PostLaunchState:
        if self.consent_required:
            self.events.append("post_launch_consent")
            return PostLaunchState.CONSENT
        self.events.append("pre_call_ready")
        return PostLaunchState.PRE_CALL

    def visible_consent_ok_count(self) -> int:
        return 1

    def click_consent_ok(self) -> None:
        self.consent_clicks += 1
        self.events.append("consent_clicked")

    def visible_join_control_count(self) -> int:
        return 1

    def click_join(self) -> None:
        self.join_clicks += 1
        self.events.append("join_clicked")

    def wait_for_joined_interview(self) -> None:
        self.events.append("interview_joined")


def _router(tmp_path: Path) -> ActionRouter:
    return ActionRouter(ActionGuard.dry_run(), tmp_path / "action_log.jsonl")


def _live_router(tmp_path: Path) -> ActionRouter:
    return ActionRouter(ActionGuard.live_join(), tmp_path / "action_log.jsonl")


def test_dry_run_opens_only_the_exact_candidates_menu_and_blocks_launch(
    tmp_path: Path,
) -> None:
    page = FakeJoinPage(
        [
            JoinCandidate(
                "Candidate Alpha",
                "Today 11:00 AM",
                CandidateCardHandle("alpha-card"),
            ),
            JoinCandidate(
                "Candidate Beta", "Today 4:00 PM", CandidateCardHandle("beta-card")
            ),
        ]
    )

    result = run_join_dry_run(
        page,
        candidate_name="  candidate   beta ",
        session_dir=tmp_path,
        action_router=_router(tmp_path),
    )

    assert page.opened_tokens == ["beta-card"]
    assert page.launch_clicks == 0
    assert page.candidate_identifier is None
    assert result.candidate_found_screenshot.name == "candidate_found.png"
    assert result.join_dry_run_screenshot.name == "join_dry_run.png"
    records = [
        json.loads(line)
        for line in result.action_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["action"] == "LAUNCH_INTERVIEW"
    assert records[-1]["decision"] == "BLOCK"


def test_candidate_lookup_does_not_fuzzy_match(tmp_path: Path) -> None:
    page = FakeJoinPage(
        [
            JoinCandidate(
                "Candidate Alpha",
                "Today 11:00 AM",
                CandidateCardHandle("alpha-card"),
            )
        ]
    )

    with pytest.raises(CandidateNotFoundError, match=r"C\*{8} A\*{4}"):
        run_join_dry_run(
            page,
            candidate_name="Candidate Alph",
            session_dir=tmp_path,
            action_router=_router(tmp_path),
        )

    assert page.opened_tokens == []


def test_duplicate_candidate_names_stop_as_ambiguous(tmp_path: Path) -> None:
    page = FakeJoinPage(
        [
            JoinCandidate(
                "Candidate Alpha",
                "Today 11:00 AM",
                CandidateCardHandle("morning-card"),
            ),
            JoinCandidate(
                "candidate alpha",
                "Tomorrow 3:00 PM",
                CandidateCardHandle("afternoon-card"),
            ),
        ]
    )

    with pytest.raises(AmbiguousCandidateError, match="candidate plus date/time"):
        run_join_dry_run(
            page,
            candidate_name="Candidate Alpha",
            session_dir=tmp_path,
            action_router=_router(tmp_path),
        )

    assert page.opened_tokens == []


def test_menu_ui_failure_saves_a_diagnostic_screenshot(tmp_path: Path) -> None:
    page = FakeJoinPage(
        [
            JoinCandidate(
                "Candidate Alpha",
                "Today 11:00 AM",
                CandidateCardHandle("alpha-card"),
            )
        ],
        menu_error=RuntimeError("simulated Playwright failure"),
    )

    with pytest.raises(JoinWorkflowError, match="candidate_menu_selector_error.png"):
        run_join_dry_run(
            page,
            candidate_name="Candidate Alpha",
            session_dir=tmp_path,
            action_router=_router(tmp_path),
        )

    assert (tmp_path / "screenshots" / "candidate_menu_selector_error.png").is_file()


def test_live_join_requests_separate_approvals_and_joins_in_order(
    tmp_path: Path,
) -> None:
    page = FakeJoinPage(
        [
            JoinCandidate(
                "Candidate Alpha",
                "Tomorrow 11:00 AM",
                CandidateCardHandle("alpha-card"),
            )
        ]
    )
    approval_requests: list[BrowserAction] = []

    def approve(action: BrowserAction, candidate_identifier: str) -> str:
        approval_requests.append(action)
        return approval_token_for(action, candidate_identifier)

    result = run_join_live(
        page,
        candidate_name="Candidate Alpha",
        session_dir=tmp_path,
        action_router=_live_router(tmp_path),
        request_approval=approve,
    )

    assert isinstance(result, JoinLiveResult)
    assert approval_requests == [
        BrowserAction.LAUNCH_INTERVIEW,
        BrowserAction.CLICK_CONSENT_OK,
        BrowserAction.CLICK_JOIN,
    ]
    assert page.events == [
        "launch_clicked",
        "post_launch_consent",
        "consent_clicked",
        "pre_call_ready",
        "join_clicked",
        "interview_joined",
        "candidate_bound",
    ]
    assert result.consent_screenshot is not None
    assert result.consent_screenshot.name == "consent.png"
    assert result.pre_call_screenshot.name == "pre_call.png"
    assert result.joined_screenshot.name == "joined.png"
    records = [
        json.loads(line)
        for line in result.action_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["action"] for record in records[-3:]] == [
        "LAUNCH_INTERVIEW",
        "CLICK_CONSENT_OK",
        "CLICK_JOIN",
    ]
    assert [record["decision"] for record in records[-3:]] == [
        "ALLOW",
        "ALLOW",
        "ALLOW",
    ]


def test_live_join_rejects_join_without_second_stage_approval(tmp_path: Path) -> None:
    page = FakeJoinPage(
        [
            JoinCandidate(
                "Candidate Alpha",
                "Tomorrow 11:00 AM",
                CandidateCardHandle("alpha-card"),
            )
        ]
    )

    def approve_before_join(action: BrowserAction, candidate_identifier: str) -> str:
        if action in {
            BrowserAction.LAUNCH_INTERVIEW,
            BrowserAction.CLICK_CONSENT_OK,
        }:
            return approval_token_for(action, candidate_identifier)
        return "wrong-stage-token"

    with pytest.raises(JoinWorkflowError, match="Join approval was not granted"):
        run_join_live(
            page,
            candidate_name="Candidate Alpha",
            session_dir=tmp_path,
            action_router=_live_router(tmp_path),
            request_approval=approve_before_join,
        )

    assert page.launch_clicks == 1
    assert page.consent_clicks == 1
    assert page.join_clicks == 0


def test_live_join_allows_verified_pre_call_when_consent_was_already_accepted(
    tmp_path: Path,
) -> None:
    page = FakeJoinPage(
        [
            JoinCandidate(
                "Candidate Alpha",
                "Tomorrow 11:00 AM",
                CandidateCardHandle("alpha-card"),
            )
        ],
        consent_required=False,
    )
    approval_requests: list[BrowserAction] = []

    def approve(action: BrowserAction, candidate_identifier: str) -> str:
        approval_requests.append(action)
        return approval_token_for(action, candidate_identifier)

    result = run_join_live(
        page,
        candidate_name="Candidate Alpha",
        session_dir=tmp_path,
        action_router=_live_router(tmp_path),
        request_approval=approve,
    )

    assert approval_requests == [
        BrowserAction.LAUNCH_INTERVIEW,
        BrowserAction.CLICK_JOIN,
    ]
    assert result.consent_screenshot is None
    assert page.consent_clicks == 0
    assert page.join_clicks == 1


def test_live_join_does_not_launch_without_first_stage_approval(tmp_path: Path) -> None:
    page = FakeJoinPage(
        [
            JoinCandidate(
                "Candidate Alpha",
                "Tomorrow 11:00 AM",
                CandidateCardHandle("alpha-card"),
            )
        ]
    )

    with pytest.raises(JoinWorkflowError, match="nothing launched"):
        run_join_live(
            page,
            candidate_name="Candidate Alpha",
            session_dir=tmp_path,
            action_router=_live_router(tmp_path),
            request_approval=lambda action, identifier: None,
        )

    assert page.launch_clicks == 0
    assert page.consent_clicks == 0
    assert page.join_clicks == 0


def test_live_join_does_not_accept_consent_without_its_approval(
    tmp_path: Path,
) -> None:
    page = FakeJoinPage(
        [
            JoinCandidate(
                "Candidate Alpha",
                "Tomorrow 11:00 AM",
                CandidateCardHandle("alpha-card"),
            )
        ]
    )

    def approve_launch(action: BrowserAction, candidate_identifier: str) -> str | None:
        if action is BrowserAction.LAUNCH_INTERVIEW:
            return approval_token_for(action, candidate_identifier)
        return None

    with pytest.raises(JoinWorkflowError, match="Consent approval was not granted"):
        run_join_live(
            page,
            candidate_name="Candidate Alpha",
            session_dir=tmp_path,
            action_router=_live_router(tmp_path),
            request_approval=approve_launch,
        )

    assert page.launch_clicks == 1
    assert page.consent_clicks == 0
    assert page.join_clicks == 0
