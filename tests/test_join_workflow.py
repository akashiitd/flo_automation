from __future__ import annotations

import json
from pathlib import Path

import pytest

from browser.action_guard import ActionGuard
from browser.action_router import ActionRouter
from browser.join_workflow import (
    AmbiguousCandidateError,
    CandidateCardHandle,
    CandidateNotFoundError,
    JoinCandidate,
    JoinWorkflowError,
    run_join_dry_run,
)


class FakeJoinPage:
    def __init__(
        self,
        candidates: list[JoinCandidate],
        *,
        menu_error: Exception | None = None,
    ) -> None:
        self.candidates = candidates
        self.menu_error = menu_error
        self.opened_tokens: list[str] = []
        self.launch_clicks = 0

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


def _router(tmp_path: Path) -> ActionRouter:
    return ActionRouter(ActionGuard.dry_run(), tmp_path / "action_log.jsonl")


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
