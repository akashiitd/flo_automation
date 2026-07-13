from __future__ import annotations

import json
import stat
from pathlib import Path

from browser.action_guard import ActionGuard, approval_token_for
from browser.action_router import ActionRouter
from browser.join_workflow import CandidateCardHandle, JoinCandidate, PostLaunchState
from browser.question_workflow import (
    CodeEditorControlObservation,
    CodeEditorDomObservation,
    ExtractedQuestion,
    StructuralDomSnapshot,
    run_question_scan,
)
from browser.skill_workflow import ExtractedSkillParameter


class FakeQuestionPage:
    def __init__(self) -> None:
        self.candidate = JoinCandidate(
            "Candidate Alpha", "TODAY at 4:00 PM", CandidateCardHandle("card-1")
        )
        self.launch_clicks = 0
        self.consent_clicks = 0
        self.dom_inspections = 0
        self.open_code_editor_tabs = False
        self.coding_question_ids: tuple[int, ...] = ()

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

    def extract_job_description(self) -> str:
        return "Build reliable GenAI services with RAG pipelines and APIs."

    def capture_skill_section_dom(self) -> StructuralDomSnapshot:
        return StructuralDomSnapshot(
            html="<section><h2>RATE CANDIDATE'S SKILLS</h2></section>",
            truncated=False,
            sha256="skills-hash",
        )

    def extract_skill_parameters(self) -> list[ExtractedSkillParameter]:
        return [
            ExtractedSkillParameter(
                id="container-normal-402780",
                name="Coding",
                requirement="Mandatory",
                level="Professional",
                rating_scale=5,
            )
        ]

    def inspect_code_editor_dom(
        self,
        *,
        open_code_editor_tabs: bool = False,
        coding_question_ids: tuple[int, ...] = (),
    ) -> list[CodeEditorDomObservation]:
        self.dom_inspections += 1
        self.open_code_editor_tabs = open_code_editor_tabs
        self.coding_question_ids = coding_question_ids
        control_html = StructuralDomSnapshot(
            html='<input type="checkbox" role="switch">',
            truncated=False,
            sha256="control-hash",
        )
        wrapper_html = StructuralDomSnapshot(
            html=(
                '<label><input type="checkbox" role="switch">'
                '<div class="clFloSwithTxt">SHOW CODE EDITOR TO CANDIDATE</div>'
                "</label>"
            ),
            truncated=False,
            sha256="wrapper-hash",
        )
        return [
            CodeEditorDomObservation(
                question_id=2,
                question_id_source="question-number-element",
                question_id_candidates=(2,),
                code_editor_tab_count=1,
                rendered_code_editor_tab_count=1,
                visibility_labels=("SHOW CODE EDITOR TO CANDIDATE",),
                visibility_label_rendered=(True,),
                switch_candidates=(
                    CodeEditorControlObservation(
                        tag_name="input",
                        role="switch",
                        input_type="checkbox",
                        aria_label=None,
                        test_id="candidate-editor-switch",
                        name=None,
                        class_name="MuiSwitch-input",
                        rendered=True,
                        outer_html=control_html,
                    ),
                ),
                association_status="unique",
                question_number_outer_html=StructuralDomSnapshot(
                    html="<span>2</span>",
                    truncated=False,
                    sha256="number-hash",
                ),
                control_wrapper_outer_html=wrapper_html,
                association_container_outer_html=StructuralDomSnapshot(
                    html='<section class="clMainSingleFESug">...</section>',
                    truncated=False,
                    sha256="card-hash",
                ),
            )
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
    assert page.dom_inspections == 1
    assert page.open_code_editor_tabs is False
    assert page.coding_question_ids == (2,)
    assert [question.id for question in result.questions] == [1, 2]
    assert result.questions[1].has_code_editor is True
    saved = json.loads(result.questions_path.read_text(encoding="utf-8"))
    assert saved[0]["question_text"] == "Explain model drift."
    assert saved[1]["has_code_editor"] is True
    job_description = json.loads(
        result.job_description_path.read_text(encoding="utf-8")
    )
    assert job_description["description"] == (
        "Build reliable GenAI services with RAG pipelines and APIs."
    )
    assert stat.S_IMODE(result.job_description_path.stat().st_mode) == 0o600
    dom_capture = json.loads(result.code_editor_dom_path.read_text(encoding="utf-8"))
    assert dom_capture["schema_version"] == 1
    assert dom_capture["read_only"] is True
    assert dom_capture["contains_private_interview_structure"] is True
    assert dom_capture["observations"][0]["question_id"] == 2
    assert dom_capture["observations"][0]["association_status"] == "unique"
    assert stat.S_IMODE(result.code_editor_dom_path.stat().st_mode) == 0o600
    skill_parameters = json.loads(
        result.skill_parameters_path.read_text(encoding="utf-8")
    )
    assert skill_parameters["schema_version"] == 1
    assert skill_parameters["read_only"] is True
    assert skill_parameters["parameters"][0]["name"] == "Coding"
    assert skill_parameters["parameters"][0]["rating_scale"] == 5
    assert skill_parameters["parameters"][0]["source"] == "flocareer_dom"
    assert stat.S_IMODE(result.skill_parameters_path.stat().st_mode) == 0o600
    assert result.skill_parameters_before_screenshot_path.exists()
    assert result.skill_parameters_after_screenshot_path.exists()
    skill_capture = json.loads(
        result.skill_section_dom_path.read_text(encoding="utf-8")
    )
    assert skill_capture["schema_version"] == 1
    assert skill_capture["read_only"] is True
    assert "RATE CANDIDATE" in skill_capture["snapshot"]["html"]
    assert stat.S_IMODE(result.skill_section_dom_path.stat().st_mode) == 0o600
    assert "CLICK_JOIN" not in router.log_path.read_text(encoding="utf-8")


def test_question_scan_can_open_only_coding_editor_tabs_for_read_only_capture(
    tmp_path: Path,
) -> None:
    page = FakeQuestionPage()

    run_question_scan(
        page,
        candidate_name="Candidate Alpha",
        session_dir=tmp_path,
        action_router=ActionRouter(
            ActionGuard.live_join(), tmp_path / "action_log.jsonl"
        ),
        request_approval=lambda action, identifier: approval_token_for(
            action, identifier
        ),
        inspect_code_editor_tabs=True,
    )

    assert page.open_code_editor_tabs is True
    assert page.coding_question_ids == (2,)
    assert "OPEN_CODE_EDITOR_TAB" in (tmp_path / "action_log.jsonl").read_text(
        encoding="utf-8"
    )
