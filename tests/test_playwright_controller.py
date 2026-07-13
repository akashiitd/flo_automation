from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest

import browser.playwright_controller as controller
from browser.action_guard import approval_token_for
from browser.code_editor_workflow import CodeEditorVisibility
from browser.join_workflow import JoinLiveResult
from browser.room_workflow import InterviewRoomState


class FakeLivePage:
    def __init__(self) -> None:
        self.page = SimpleNamespace(wait_for_timeout=lambda milliseconds: None)
        self.candidate_identifier = "candidate-a1b2c3"
        self.room_states = [
            InterviewRoomState.WAITING_FOR_CANDIDATE,
            InterviewRoomState.CANDIDATE_CONNECTED,
        ]
        self.room_index = 0
        self.visibility = CodeEditorVisibility.HIDDEN
        self.opened_questions: list[int] = []
        self.clicked_questions: list[int] = []
        self.audio_configuration_calls: list[tuple[str, str]] = []

    def open_dashboard(self, url: str) -> None:
        return None

    def read_interview_room_state(self) -> InterviewRoomState:
        return self.room_states[self.room_index]

    def wait_for_room_poll(self, seconds: float) -> None:
        self.room_index = min(self.room_index + 1, len(self.room_states) - 1)

    def capture_screenshot(self, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        path.write_bytes(b"fictional screenshot")
        return path

    def active_candidate_matches(self, candidate_identifier: str) -> bool:
        return candidate_identifier == self.candidate_identifier

    def candidate_is_connected(self) -> bool:
        return (
            self.read_interview_room_state() is InterviewRoomState.CANDIDATE_CONNECTED
        )

    def open_code_editor_tab(self, question_id: int) -> None:
        self.opened_questions.append(question_id)

    def code_editor_tab_is_active(self, question_id: int) -> bool:
        return True

    def read_code_editor_visibility(self, question_id: int) -> CodeEditorVisibility:
        return self.visibility

    def click_show_code_editor(self, question_id: int) -> None:
        self.clicked_questions.append(question_id)
        self.visibility = CodeEditorVisibility.VISIBLE

    def wait_for_code_editor_visibility(
        self,
        question_id: int,
        expected: CodeEditorVisibility,
    ) -> None:
        assert self.visibility is expected

    def configure_audio_devices(self, *, microphone: str, speaker: str) -> object:
        self.audio_configuration_calls.append((microphone, speaker))
        return SimpleNamespace(microphone=microphone, speaker=speaker)


def test_live_controller_waits_for_candidate_then_enables_editor_in_same_session(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    flocareer = FakeLivePage()
    session_dir = tmp_path / "runs" / "join_live_test"
    screenshots = session_dir / "screenshots"
    manual_end: list[str] = []

    @contextmanager
    def persistent_page(settings: object):
        yield flocareer

    def fake_join(*args: object, **kwargs: object) -> JoinLiveResult:
        return JoinLiveResult(
            candidate_identifier="candidate-a1b2c3",
            candidate_found_screenshot=screenshots / "candidate_found.png",
            launch_approval_screenshot=screenshots / "launch_approval.png",
            consent_screenshot=None,
            pre_call_screenshot=screenshots / "pre_call.png",
            joined_screenshot=screenshots / "joined.png",
            action_log_path=session_dir / "action_log.jsonl",
        )

    monkeypatch.setattr(controller, "_persistent_flocareer_page", persistent_page)
    monkeypatch.setattr(
        controller, "_wait_for_authenticated_dashboard", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(controller, "run_join_live", fake_join)
    monkeypatch.setattr(
        controller,
        "datetime",
        SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda _: "test")),
    )

    result = controller.join_candidate_live(
        SimpleNamespace(
            runs_dir=tmp_path / "runs", flocareer_url="https://example.test"
        ),
        candidate_name="Candidate Alpha",
        request_approval=lambda action, identifier: approval_token_for(
            action, identifier
        ),
        wait_for_manual_end=manual_end.append,
        enable_code_editor_question=13,
        request_code_editor_approval=lambda action, identifier, question_id: (
            approval_token_for(action, identifier, question_id=question_id)
        ),
    )

    assert result.code_editor_result is not None
    assert result.code_editor_result.changed is True
    assert result.room_state_log_path is not None
    assert flocareer.opened_questions == [13]
    assert flocareer.clicked_questions == [13]
    assert manual_end == ["candidate-a1b2c3"]


def test_live_controller_configures_flocareer_audio_before_waiting_for_candidate(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    flocareer = FakeLivePage()
    session_dir = tmp_path / "runs" / "join_live_audio_test"
    screenshots = session_dir / "screenshots"

    @contextmanager
    def persistent_page(settings: object):
        yield flocareer

    def fake_join(*args: object, **kwargs: object) -> JoinLiveResult:
        return JoinLiveResult(
            candidate_identifier="candidate-a1b2c3",
            candidate_found_screenshot=screenshots / "candidate_found.png",
            launch_approval_screenshot=screenshots / "launch_approval.png",
            consent_screenshot=None,
            pre_call_screenshot=screenshots / "pre_call.png",
            joined_screenshot=screenshots / "joined.png",
            action_log_path=session_dir / "action_log.jsonl",
        )

    monkeypatch.setattr(controller, "_persistent_flocareer_page", persistent_page)
    monkeypatch.setattr(
        controller, "_wait_for_authenticated_dashboard", lambda *args, **kwargs: False
    )
    monkeypatch.setattr(controller, "run_join_live", fake_join)
    monkeypatch.setattr(
        controller,
        "datetime",
        SimpleNamespace(now=lambda: SimpleNamespace(strftime=lambda _: "test")),
    )

    result = controller.join_candidate_live(
        SimpleNamespace(
            runs_dir=tmp_path / "runs",
            flocareer_url="https://example.test",
            interviewer_audio_output_device="INTERVIEWER_TO_CALL",
            flocareer_speaker_output_device="Jabra Evolve2 65 Flex (Bluetooth)",
        ),
        candidate_name="Candidate Alpha",
        request_approval=lambda action, identifier: approval_token_for(
            action, identifier
        ),
        wait_for_manual_end=lambda _: None,
        configure_flocareer_audio=True,
    )

    assert flocareer.audio_configuration_calls == [
        ("INTERVIEWER_TO_CALL", "Jabra Evolve2 65 Flex (Bluetooth)")
    ]
    assert result.audio_configuration is not None
