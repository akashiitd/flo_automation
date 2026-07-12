from __future__ import annotations

import json
from pathlib import Path

import pytest

from browser.action_guard import ActionGuard, BrowserAction, approval_token_for
from browser.action_router import ActionRouter
from browser.code_editor_workflow import (
    CandidateDisconnectedError,
    CodeEditorVisibility,
    CodeEditorWorkflowError,
    run_show_code_editor,
)


class FakeCodeEditorPage:
    def __init__(
        self,
        visibility: CodeEditorVisibility,
        *,
        candidate_identifier: str = "candidate-a1b2c3",
        wait_error: Exception | None = None,
    ) -> None:
        self.visibility = visibility
        self.candidate_identifier = candidate_identifier
        self.wait_error = wait_error
        self.opened_questions: list[int] = []
        self.clicks: list[int] = []
        self.tab_active = True
        self.candidate_connected = True

    def active_candidate_matches(self, candidate_identifier: str) -> bool:
        return self.candidate_identifier == candidate_identifier

    def candidate_is_connected(self) -> bool:
        return self.candidate_connected

    def capture_screenshot(self, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        path.write_bytes(b"screenshot")
        return path

    def open_code_editor_tab(self, question_id: int) -> None:
        self.opened_questions.append(question_id)
        self.tab_active = True

    def code_editor_tab_is_active(self, question_id: int) -> bool:
        return self.tab_active

    def read_code_editor_visibility(self, question_id: int) -> CodeEditorVisibility:
        return self.visibility

    def click_show_code_editor(self, question_id: int) -> None:
        if self.visibility is not CodeEditorVisibility.HIDDEN:
            raise CodeEditorWorkflowError("editor is no longer hidden")
        self.clicks.append(question_id)
        self.visibility = CodeEditorVisibility.VISIBLE

    def wait_for_code_editor_visibility(
        self,
        question_id: int,
        expected: CodeEditorVisibility,
    ) -> None:
        if self.wait_error is not None:
            raise self.wait_error
        if self.visibility is not expected:
            raise CodeEditorWorkflowError("visibility did not stabilize")


def _router(tmp_path: Path) -> ActionRouter:
    return ActionRouter(ActionGuard.code_editor(), tmp_path / "action_log.jsonl")


def test_hidden_editor_is_shown_once_with_scoped_approval(tmp_path: Path) -> None:
    page = FakeCodeEditorPage(CodeEditorVisibility.HIDDEN)
    candidate = "candidate-a1b2c3"

    result = run_show_code_editor(
        page,
        candidate_identifier=candidate,
        question_id=13,
        session_dir=tmp_path,
        action_router=_router(tmp_path),
        request_approval=lambda action, identifier, question_id: approval_token_for(
            action,
            identifier,
            question_id=question_id,
        ),
    )

    assert page.opened_questions == [13]
    assert page.clicks == [13]
    assert result.changed is True
    assert result.visibility is CodeEditorVisibility.VISIBLE
    assert result.before_screenshot.name.startswith("code_editor_question_13_")
    assert result.before_screenshot.name.endswith("_before.png")
    assert result.after_screenshot.name.startswith("code_editor_question_13_")
    assert result.after_screenshot.name.endswith("_after.png")
    records = [
        json.loads(line)
        for line in result.action_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["action"] for record in records] == [
        "OPEN_CODE_EDITOR_TAB",
        "SHOW_CODE_EDITOR_TO_CANDIDATE",
    ]
    assert records[1]["question_id"] == 13


def test_visible_editor_is_an_idempotent_no_op(tmp_path: Path) -> None:
    page = FakeCodeEditorPage(CodeEditorVisibility.VISIBLE)
    approvals: list[int] = []

    result = run_show_code_editor(
        page,
        candidate_identifier="candidate-a1b2c3",
        question_id=13,
        session_dir=tmp_path,
        action_router=_router(tmp_path),
        request_approval=lambda action, identifier, question_id: approvals.append(
            question_id
        ),
    )

    assert result.changed is False
    assert result.visibility is CodeEditorVisibility.VISIBLE
    assert page.clicks == []
    assert approvals == []


def test_candidate_binding_is_revalidated_before_editor_navigation(
    tmp_path: Path,
) -> None:
    page = FakeCodeEditorPage(
        CodeEditorVisibility.HIDDEN,
        candidate_identifier="candidate-other",
    )

    with pytest.raises(CodeEditorWorkflowError, match="active candidate"):
        run_show_code_editor(
            page,
            candidate_identifier="candidate-a1b2c3",
            question_id=13,
            session_dir=tmp_path,
            action_router=_router(tmp_path),
            request_approval=lambda action, identifier, question_id: None,
        )

    assert page.opened_questions == []
    assert page.clicks == []


def test_rejected_or_wrong_approval_never_clicks_switch(tmp_path: Path) -> None:
    page = FakeCodeEditorPage(CodeEditorVisibility.HIDDEN)

    with pytest.raises(CodeEditorWorkflowError, match="approval was not granted"):
        run_show_code_editor(
            page,
            candidate_identifier="candidate-a1b2c3",
            question_id=13,
            session_dir=tmp_path,
            action_router=_router(tmp_path),
            request_approval=lambda action, identifier, question_id: (
                "APPROVE-SHOW-CODE-EDITOR candidate-a1b2c3 question-12"
            ),
        )

    assert page.clicks == []


def test_state_is_revalidated_after_operator_approval_pause(tmp_path: Path) -> None:
    page = FakeCodeEditorPage(CodeEditorVisibility.HIDDEN)

    def approve_after_external_change(
        action: BrowserAction,
        identifier: str,
        question_id: int,
    ) -> str:
        page.visibility = CodeEditorVisibility.VISIBLE
        return approval_token_for(action, identifier, question_id=question_id)

    result = run_show_code_editor(
        page,
        candidate_identifier="candidate-a1b2c3",
        question_id=13,
        session_dir=tmp_path,
        action_router=_router(tmp_path),
        request_approval=approve_after_external_change,
    )

    assert result.changed is False
    assert result.visibility is CodeEditorVisibility.VISIBLE
    assert page.clicks == []


def test_active_editor_tab_is_revalidated_after_operator_approval_pause(
    tmp_path: Path,
) -> None:
    page = FakeCodeEditorPage(CodeEditorVisibility.HIDDEN)

    def approve_after_tab_change(
        action: BrowserAction,
        identifier: str,
        question_id: int,
    ) -> str:
        page.tab_active = False
        return approval_token_for(action, identifier, question_id=question_id)

    with pytest.raises(CodeEditorWorkflowError, match="no longer active"):
        run_show_code_editor(
            page,
            candidate_identifier="candidate-a1b2c3",
            question_id=13,
            session_dir=tmp_path,
            action_router=_router(tmp_path),
            request_approval=approve_after_tab_change,
        )

    assert page.clicks == []


def test_candidate_disconnect_after_editor_approval_never_clicks_switch(
    tmp_path: Path,
) -> None:
    page = FakeCodeEditorPage(CodeEditorVisibility.HIDDEN)

    def approve_after_disconnect(
        action: BrowserAction,
        identifier: str,
        question_id: int,
    ) -> str:
        page.candidate_connected = False
        return approval_token_for(action, identifier, question_id=question_id)

    with pytest.raises(CandidateDisconnectedError, match="disconnected"):
        run_show_code_editor(
            page,
            candidate_identifier="candidate-a1b2c3",
            question_id=13,
            session_dir=tmp_path,
            action_router=_router(tmp_path),
            request_approval=approve_after_disconnect,
        )

    assert page.clicks == []


def test_explicit_prejoin_mode_can_show_editor_without_candidate_connection(
    tmp_path: Path,
) -> None:
    page = FakeCodeEditorPage(CodeEditorVisibility.HIDDEN)
    page.candidate_connected = False

    result = run_show_code_editor(
        page,
        candidate_identifier="candidate-a1b2c3",
        question_id=13,
        session_dir=tmp_path,
        action_router=_router(tmp_path),
        request_approval=lambda action, identifier, question_id: approval_token_for(
            action, identifier, question_id=question_id
        ),
        allow_prejoin=True,
    )

    assert result.changed is True
    assert page.clicks == [13]


def test_candidate_binding_is_revalidated_after_approval_pause(
    tmp_path: Path,
) -> None:
    page = FakeCodeEditorPage(CodeEditorVisibility.HIDDEN)

    def approve_after_candidate_change(
        action: BrowserAction,
        identifier: str,
        question_id: int,
    ) -> str:
        page.candidate_identifier = "candidate-other"
        return approval_token_for(action, identifier, question_id=question_id)

    with pytest.raises(CodeEditorWorkflowError, match="active candidate"):
        run_show_code_editor(
            page,
            candidate_identifier="candidate-a1b2c3",
            question_id=13,
            session_dir=tmp_path,
            action_router=_router(tmp_path),
            request_approval=approve_after_candidate_change,
        )

    assert page.clicks == []


def test_post_click_visibility_failure_is_audited_as_execution_error(
    tmp_path: Path,
) -> None:
    page = FakeCodeEditorPage(
        CodeEditorVisibility.HIDDEN,
        wait_error=CodeEditorWorkflowError("visibility did not stabilize"),
    )
    router = _router(tmp_path)

    with pytest.raises(CodeEditorWorkflowError, match="did not stabilize"):
        run_show_code_editor(
            page,
            candidate_identifier="candidate-a1b2c3",
            question_id=13,
            session_dir=tmp_path,
            action_router=router,
            request_approval=lambda action, identifier, question_id: approval_token_for(
                action,
                identifier,
                question_id=question_id,
            ),
        )

    records = [
        json.loads(line)
        for line in router.log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert records[-1]["action"] == "SHOW_CODE_EDITOR_TO_CANDIDATE"
    assert records[-1]["execution_outcome"] == "ERROR"


def test_retries_use_distinct_evidence_filenames(tmp_path: Path) -> None:
    def run_once() -> Path:
        page = FakeCodeEditorPage(CodeEditorVisibility.VISIBLE)
        return run_show_code_editor(
            page,
            candidate_identifier="candidate-a1b2c3",
            question_id=13,
            session_dir=tmp_path,
            action_router=_router(tmp_path),
            request_approval=lambda action, identifier, question_id: None,
        ).before_screenshot

    assert run_once() != run_once()
