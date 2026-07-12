from __future__ import annotations

from pathlib import Path

from browser.action_guard import ActionGuard, approval_token_for
from browser.action_router import ActionRouter
from browser.join_workflow import JoinLiveResult
from browser.no_show_workflow import mark_no_show


class FakeNoShowPage:
    def __init__(self) -> None:
        self.level = "Intermediate"
        self.marked = False
        self.connected = False

    def capture_screenshot(self, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"{name}.png"
        path.write_bytes(b"fictional screenshot")
        return path

    def visible_mark_no_show_count(self) -> int:
        return 0 if self.marked else 1

    def click_mark_no_show(self) -> None:
        self.marked = True

    def wait_for_mark_no_show_applied(self) -> None:
        assert self.marked is True

    def read_interview_level(self) -> str:
        return self.level

    def candidate_is_connected(self) -> bool:
        return self.connected


def test_mark_no_show_verifies_intermediate_and_requires_fresh_approval(
    tmp_path: Path,
) -> None:
    page = FakeNoShowPage()
    candidate = "candidate-a1b2c3"
    router = ActionRouter(ActionGuard.no_show(), tmp_path / "action_log.jsonl")
    joined = JoinLiveResult(
        candidate_identifier=candidate,
        candidate_found_screenshot=tmp_path / "candidate.png",
        launch_approval_screenshot=tmp_path / "launch.png",
        consent_screenshot=None,
        pre_call_screenshot=tmp_path / "precall.png",
        joined_screenshot=tmp_path / "joined.png",
        action_log_path=router.log_path,
    )

    result = mark_no_show(
        page,
        joined=joined,
        session_dir=tmp_path,
        action_router=router,
        request_approval=lambda action, identifier: approval_token_for(
            action, identifier
        ),
    )

    assert page.level == "Intermediate"
    assert page.marked is True
    assert result.level == "Intermediate"
    assert result.before_screenshot.is_file()
    assert result.after_screenshot.is_file()


def test_mark_no_show_does_not_click_without_its_fresh_approval(tmp_path: Path) -> None:
    page = FakeNoShowPage()
    candidate = "candidate-a1b2c3"
    router = ActionRouter(ActionGuard.no_show(), tmp_path / "action_log.jsonl")
    joined = JoinLiveResult(
        candidate_identifier=candidate,
        candidate_found_screenshot=tmp_path / "candidate.png",
        launch_approval_screenshot=tmp_path / "launch.png",
        consent_screenshot=None,
        pre_call_screenshot=tmp_path / "precall.png",
        joined_screenshot=tmp_path / "joined.png",
        action_log_path=router.log_path,
    )

    try:
        mark_no_show(
            page,
            joined=joined,
            session_dir=tmp_path,
            action_router=router,
            request_approval=lambda action, identifier: None,
        )
    except RuntimeError as error:
        assert "approval" in str(error).lower()
    else:
        raise AssertionError("no-show must remain blocked without fresh approval")

    assert page.marked is False


def test_mark_no_show_rechecks_that_the_candidate_did_not_connect(
    tmp_path: Path,
) -> None:
    page = FakeNoShowPage()
    page.connected = True
    candidate = "candidate-a1b2c3"
    router = ActionRouter(ActionGuard.no_show(), tmp_path / "action_log.jsonl")
    joined = JoinLiveResult(
        candidate_identifier=candidate,
        candidate_found_screenshot=tmp_path / "candidate.png",
        launch_approval_screenshot=tmp_path / "launch.png",
        consent_screenshot=None,
        pre_call_screenshot=tmp_path / "precall.png",
        joined_screenshot=tmp_path / "joined.png",
        action_log_path=router.log_path,
    )

    try:
        mark_no_show(
            page,
            joined=joined,
            session_dir=tmp_path,
            action_router=router,
            request_approval=lambda action, identifier: approval_token_for(
                action, identifier
            ),
        )
    except RuntimeError as error:
        assert "connected" in str(error)
    else:
        raise AssertionError("candidate connection must block no-show")

    assert page.marked is False
