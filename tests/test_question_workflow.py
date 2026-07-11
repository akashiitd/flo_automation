from __future__ import annotations

import json
from pathlib import Path

from browser.action_guard import ActionGuard, approval_token_for
from browser.action_router import ActionRouter
from browser.join_workflow import CandidateCardHandle, JoinCandidate, PostLaunchState
from browser.question_workflow import ExtractedQuestion, run_question_scan


class FakeQuestionPage:
    def __init__(self) -> None:
        self.candidate = JoinCandidate(
            "Candidate Alpha", "TODAY at 4:00 PM", CandidateCardHandle("card-1")
        )
        self.launch_clicks = 0
        self.consent_clicks = 0

    def list_join_candidates(self) -> list[JoinCandidate]:
        return [self.candidate]

    def capture_screenshot(self, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        path.write_bytes(b"screenshot")
        return path

    def open_candidate_menu(self, candidate: JoinCandidate) -> None:
        assert candidate == self.candidate

    def visible_launch_control_count(self) -> int:
        return 1

    def click_launch_interview(self) -> None:
        self.launch_clicks += 1

    def wait_for_questions_or_consent(self) -> PostLaunchState:
        return PostLaunchState.QUESTIONS

    def visible_consent_ok_count(self) -> int:
        return 0

    def click_consent_ok(self) -> None:
        self.consent_clicks += 1

    def wait_for_question_panel(self) -> None:
        pass

    def extract_questions(self) -> list[ExtractedQuestion]:
        return [
            ExtractedQuestion(
                id=1,
                question_text="Explain model drift.",
                has_code_editor=False,
                ideal_answer="Monitor distributions and outcomes.",
                guidelines={"5_star": "Complete", "4_star": "Mostly complete"},
                feedback_field_locator_hint="question:1:feedback",
                rating_locator_hint="question:1:rating",
                mark_as_locator_hint="question:1:mark_as",
            ),
            ExtractedQuestion(
                id=2,
                question_text="Implement an LRU cache.",
                has_code_editor=True,
                ideal_answer="Use a hash map and linked list.",
                guidelines={},
                feedback_field_locator_hint="question:2:feedback",
                rating_locator_hint="question:2:rating",
                mark_as_locator_hint="question:2:mark_as",
            ),
        ]


def test_question_scan_launches_but_never_joins_or_enables_editor(
    tmp_path: Path,
) -> None:
    page = FakeQuestionPage()
    router = ActionRouter(ActionGuard.live_join(), tmp_path / "action_log.jsonl")

    result = run_question_scan(
        page,
        candidate_name="Candidate Alpha",
        session_dir=tmp_path,
        action_router=router,
        request_approval=lambda action, identifier: approval_token_for(
            action, identifier
        ),
    )

    assert page.launch_clicks == 1
    assert page.consent_clicks == 0
    assert [question.id for question in result.questions] == [1, 2]
    assert result.questions[1].has_code_editor is True
    saved = json.loads(result.questions_path.read_text(encoding="utf-8"))
    assert saved[0]["question_text"] == "Explain model drift."
    assert saved[1]["has_code_editor"] is True
    assert "CLICK_JOIN" not in router.log_path.read_text(encoding="utf-8")
