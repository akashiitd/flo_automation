"""Approved Launch-only extraction of FloCareer interview questions."""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal, Protocol

from browser.action_guard import BrowserAction
from browser.action_router import ActionRouter
from browser.join_workflow import (
    ApprovalRequester,
    JoinWorkflowError,
    LaunchWorkflowPage,
    PostLaunchState,
    prepare_launch_control,
)
from browser.skill_workflow import ExtractedSkillParameter
from orchestrator.state import SkillParametersArtifact


@dataclass(frozen=True, slots=True)
class ExtractedQuestion:
    id: int
    question_text: str
    has_code_editor: bool
    ideal_answer: str
    guidelines: Mapping[str, str]
    feedback_field_locator_hint: str
    rating_locator_hint: str
    mark_as_locator_hint: str


CodeEditorAssociationStatus = Literal["unique", "none", "ambiguous"]
CodeEditorQuestionIdSource = Literal[
    "data-question-id",
    "question-number-element",
    "unresolved",
]


@dataclass(frozen=True, slots=True)
class StructuralDomSnapshot:
    html: str
    truncated: bool
    sha256: str


@dataclass(frozen=True, slots=True)
class CodeEditorControlObservation:
    tag_name: str
    role: str | None
    input_type: str | None
    aria_label: str | None
    test_id: str | None
    name: str | None
    class_name: str | None
    rendered: bool
    outer_html: StructuralDomSnapshot


@dataclass(frozen=True, slots=True)
class CodeEditorDomObservation:
    question_id: int | None
    question_id_source: CodeEditorQuestionIdSource
    question_id_candidates: tuple[int, ...]
    code_editor_tab_count: int
    rendered_code_editor_tab_count: int
    visibility_labels: tuple[str, ...]
    visibility_label_rendered: tuple[bool, ...]
    switch_candidates: tuple[CodeEditorControlObservation, ...]
    association_status: CodeEditorAssociationStatus
    question_number_outer_html: StructuralDomSnapshot | None
    control_wrapper_outer_html: StructuralDomSnapshot | None
    association_container_outer_html: StructuralDomSnapshot


@dataclass(frozen=True, slots=True)
class QuestionScanResult:
    candidate_identifier: str
    questions: tuple[ExtractedQuestion, ...]
    questions_path: Path
    job_description_path: Path
    code_editor_dom_observations: tuple[CodeEditorDomObservation, ...]
    code_editor_dom_path: Path
    skill_parameters: tuple[ExtractedSkillParameter, ...]
    skill_parameters_path: Path
    skill_parameters_before_screenshot_path: Path
    skill_parameters_after_screenshot_path: Path
    skill_section_dom_path: Path
    screenshot_path: Path
    action_log_path: Path


class QuestionScanPage(LaunchWorkflowPage, Protocol):
    def wait_for_questions_or_consent(self) -> PostLaunchState: ...
    def visible_consent_ok_count(self) -> int: ...
    def click_consent_ok(self) -> None: ...
    def wait_for_question_panel(self) -> None: ...
    def extract_questions(self) -> list[ExtractedQuestion]: ...
    def extract_job_description(self) -> str: ...
    def extract_skill_parameters(self) -> list[ExtractedSkillParameter]: ...
    def capture_skill_section_dom(self) -> StructuralDomSnapshot: ...
    def inspect_code_editor_dom(
        self,
        *,
        open_code_editor_tabs: bool = False,
        coding_question_ids: tuple[int, ...] = (),
    ) -> list[CodeEditorDomObservation]: ...


def _write_questions(path: Path, questions: tuple[ExtractedQuestion, ...]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps([asdict(question) for question in questions], indent=2) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _write_job_description(path: Path, description: str) -> None:
    payload = {
        "schema_version": 1,
        "read_only": True,
        "source": "FloCareer Job Description tab",
        "description": description,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    path.chmod(0o600)


def _write_code_editor_dom(
    path: Path,
    *,
    coding_question_ids: tuple[int, ...],
    observations: tuple[CodeEditorDomObservation, ...],
) -> None:
    observation_ids = tuple(
        observation.question_id
        for observation in observations
        if observation.question_id is not None
    )
    payload = {
        "schema_version": 1,
        "read_only": True,
        "contains_private_interview_structure": True,
        "detected_coding_question_ids": coding_question_ids,
        "observation_ids": observation_ids,
        "complete": (
            len(observation_ids) == len(set(observation_ids))
            and set(observation_ids) == set(coding_question_ids)
        ),
        "observations": [asdict(observation) for observation in observations],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    path.chmod(0o600)


def _write_skill_section_dom(path: Path, snapshot: StructuralDomSnapshot) -> None:
    payload = {
        "schema_version": 1,
        "read_only": True,
        "contains_private_interview_structure": True,
        "snapshot": asdict(snapshot),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    path.chmod(0o600)


def _write_skill_parameters(
    path: Path, parameters: tuple[ExtractedSkillParameter, ...]
) -> None:
    payload = {
        "schema_version": 1,
        "read_only": True,
        "parameters": [asdict(parameter) for parameter in parameters],
    }
    SkillParametersArtifact.model_validate(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.touch(mode=0o600, exist_ok=True)
    temporary.chmod(0o600)
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)
    path.chmod(0o600)


def run_question_scan(
    page: QuestionScanPage,
    *,
    candidate_name: str,
    session_dir: Path,
    action_router: ActionRouter,
    request_approval: ApprovalRequester,
    inspect_code_editor_tabs: bool = False,
) -> QuestionScanResult:
    """Launch one candidate page and read questions without clicking Join."""

    prepared = prepare_launch_control(
        page,
        candidate_name=candidate_name,
        session_dir=session_dir,
        action_router=action_router,
        launch_screenshot_name="questions_launch_approval",
    )
    identifier = prepared.candidate_identifier
    launch = action_router.route(
        BrowserAction.LAUNCH_INTERVIEW,
        operation=page.click_launch_interview,
        candidate_identifier=identifier,
        approval_token=request_approval(BrowserAction.LAUNCH_INTERVIEW, identifier),
        screenshot_path=prepared.launch_control_screenshot,
    )
    if not launch.allowed:
        raise JoinWorkflowError("Launch approval was not granted; nothing launched")

    screenshots_dir = session_dir / "screenshots"
    try:
        state = page.wait_for_questions_or_consent()
    except Exception as error:
        diagnostic = page.capture_screenshot(
            screenshots_dir, "question_panel_wait_error"
        )
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error
    if state is PostLaunchState.CONSENT:
        consent = page.capture_screenshot(screenshots_dir, "questions_consent")
        if page.visible_consent_ok_count() != 1:
            raise JoinWorkflowError("Expected exactly one consent OK control")
        decision = action_router.route(
            BrowserAction.CLICK_CONSENT_OK,
            operation=page.click_consent_ok,
            candidate_identifier=identifier,
            approval_token=request_approval(BrowserAction.CLICK_CONSENT_OK, identifier),
            screenshot_path=consent,
        )
        if not decision.allowed:
            raise JoinWorkflowError(
                "Consent approval was not granted; OK was not clicked"
            )
        page.wait_for_question_panel()

    page.wait_for_question_panel()
    try:
        questions = tuple(page.extract_questions())
    except Exception as error:
        diagnostic = page.capture_screenshot(screenshots_dir, "question_extract_error")
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error
    if not questions:
        diagnostic = page.capture_screenshot(screenshots_dir, "questions_not_found")
        raise JoinWorkflowError(
            f"No question cards were extracted. Screenshot: {diagnostic}"
        )
    ids = [question.id for question in questions]
    if len(ids) != len(set(ids)) or any(
        not question.question_text for question in questions
    ):
        raise JoinWorkflowError(
            "Question extraction returned duplicate IDs or empty text"
        )

    try:
        job_description = page.extract_job_description().strip()
    except Exception as error:
        diagnostic = page.capture_screenshot(
            screenshots_dir, "job_description_extract_error"
        )
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error
    if not job_description:
        raise JoinWorkflowError("Job description extraction returned empty text")

    skill_parameters_before_screenshot = page.capture_screenshot(
        screenshots_dir, "skill_parameters_before"
    )
    try:
        skill_parameters = tuple(page.extract_skill_parameters())
    except Exception as error:
        diagnostic = page.capture_screenshot(
            screenshots_dir, "skill_parameters_extract_error"
        )
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error
    skill_parameters_after_screenshot = page.capture_screenshot(
        screenshots_dir, "skill_parameters_after"
    )
    if not skill_parameters:
        raise JoinWorkflowError("Skill parameter extraction returned no rows")
    skill_ids = [parameter.id for parameter in skill_parameters]
    skill_names = [parameter.name.casefold() for parameter in skill_parameters]
    if (
        len(skill_ids) != len(set(skill_ids))
        or len(skill_names) != len(set(skill_names))
        or any(
            not parameter.name or not parameter.requirement or not parameter.level
            for parameter in skill_parameters
        )
        or any(parameter.rating_scale != 5 for parameter in skill_parameters)
    ):
        raise JoinWorkflowError("Skill parameter extraction failed strict validation")

    try:
        coding_question_ids = tuple(
            question.id for question in questions if question.has_code_editor
        )
        if inspect_code_editor_tabs:
            captured: list[CodeEditorDomObservation] = []
            navigation = action_router.route(
                BrowserAction.OPEN_CODE_EDITOR_TAB,
                operation=lambda: captured.extend(
                    page.inspect_code_editor_dom(
                        open_code_editor_tabs=True,
                        coding_question_ids=coding_question_ids,
                    )
                ),
                candidate_identifier=identifier,
            )
            if not navigation.allowed:
                raise JoinWorkflowError("Code Editor tab navigation is blocked")
            code_editor_dom_observations = tuple(captured)
        else:
            code_editor_dom_observations = tuple(
                page.inspect_code_editor_dom(
                    open_code_editor_tabs=False,
                    coding_question_ids=coding_question_ids,
                )
            )
    except Exception as error:
        diagnostic = page.capture_screenshot(
            screenshots_dir, "code_editor_dom_inspection_error"
        )
        detail = str(error) or type(error).__name__
        raise JoinWorkflowError(f"{detail}. Screenshot: {diagnostic}") from error

    screenshot = page.capture_screenshot(screenshots_dir, "questions_expanded")
    questions_path = session_dir / "questions.json"
    _write_questions(questions_path, questions)
    job_description_path = session_dir / "job_description.json"
    _write_job_description(job_description_path, job_description)
    code_editor_dom_path = session_dir / "code_editor_dom.json"
    _write_code_editor_dom(
        code_editor_dom_path,
        coding_question_ids=coding_question_ids,
        observations=code_editor_dom_observations,
    )
    skill_parameters_path = session_dir / "skill_parameters.json"
    _write_skill_parameters(skill_parameters_path, skill_parameters)
    skill_section_dom_path = session_dir / "skill_section_dom.json"
    _write_skill_section_dom(skill_section_dom_path, page.capture_skill_section_dom())
    return QuestionScanResult(
        candidate_identifier=identifier,
        questions=questions,
        questions_path=questions_path,
        job_description_path=job_description_path,
        code_editor_dom_observations=code_editor_dom_observations,
        code_editor_dom_path=code_editor_dom_path,
        skill_parameters=skill_parameters,
        skill_parameters_path=skill_parameters_path,
        skill_parameters_before_screenshot_path=skill_parameters_before_screenshot,
        skill_parameters_after_screenshot_path=skill_parameters_after_screenshot,
        skill_section_dom_path=skill_section_dom_path,
        screenshot_path=screenshot,
        action_log_path=action_router.log_path,
    )
