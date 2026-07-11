from __future__ import annotations

import json
from pathlib import Path

from browser.room_workflow import (
    InterviewRoomState,
    RoomStateTracker,
    wait_for_candidate_connection,
)


class FakeRoomPage:
    def __init__(self, states: list[InterviewRoomState]) -> None:
        self.states = states
        self.index = 0
        self.polls: list[float] = []
        self.screenshots: list[str] = []

    def read_interview_room_state(self) -> InterviewRoomState:
        return self.states[self.index]

    def wait_for_room_poll(self, seconds: float) -> None:
        self.polls.append(seconds)
        if self.index < len(self.states) - 1:
            self.index += 1

    def capture_screenshot(self, directory: Path, name: str) -> Path:
        directory.mkdir(parents=True, exist_ok=True)
        self.screenshots.append(name)
        path = directory / f"{name}.png"
        path.write_bytes(b"fictional screenshot")
        return path


def test_room_state_tracker_records_candidate_disconnect_and_reconnect() -> None:
    tracker = RoomStateTracker()

    transitions = [
        tracker.observe(InterviewRoomState.INTERVIEWER_IN_ROOM),
        tracker.observe(InterviewRoomState.WAITING_FOR_CANDIDATE),
        tracker.observe(InterviewRoomState.CANDIDATE_CONNECTED),
        tracker.observe(InterviewRoomState.WAITING_FOR_CANDIDATE),
        tracker.observe(InterviewRoomState.CANDIDATE_CONNECTED),
    ]

    assert [transition.current for transition in transitions if transition] == [
        InterviewRoomState.INTERVIEWER_IN_ROOM,
        InterviewRoomState.WAITING_FOR_CANDIDATE,
        InterviewRoomState.CANDIDATE_CONNECTED,
        InterviewRoomState.WAITING_FOR_CANDIDATE,
        InterviewRoomState.CANDIDATE_CONNECTED,
    ]
    assert transitions[3] is not None
    assert transitions[3].previous is InterviewRoomState.CANDIDATE_CONNECTED


def test_room_monitor_waits_past_initial_room_entry_until_candidate_connects(
    tmp_path: Path,
) -> None:
    page = FakeRoomPage(
        [
            InterviewRoomState.WAITING_FOR_CANDIDATE,
            InterviewRoomState.CANDIDATE_CONNECTED,
        ]
    )
    statuses: list[str] = []

    result = wait_for_candidate_connection(
        page,
        session_dir=tmp_path,
        poll_interval_seconds=0.01,
        report=statuses.append,
    )

    assert result.final_state is InterviewRoomState.CANDIDATE_CONNECTED
    assert [transition.current for transition in result.transitions] == [
        InterviewRoomState.LAUNCHED,
        InterviewRoomState.INTERVIEWER_IN_ROOM,
        InterviewRoomState.WAITING_FOR_CANDIDATE,
        InterviewRoomState.CANDIDATE_CONNECTED,
    ]
    assert page.polls == [0.01]
    assert page.screenshots == [
        "room_interviewer_in_room",
        "room_waiting_for_candidate",
        "room_candidate_connected",
    ]
    records = [
        json.loads(line)
        for line in result.state_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["state"] for record in records] == [
        "LAUNCHED",
        "INTERVIEWER_IN_ROOM",
        "WAITING_FOR_CANDIDATE",
        "CANDIDATE_CONNECTED",
    ]
    assert any("Waiting for candidate" in status for status in statuses)


def test_room_monitor_preserves_disconnect_and_reconnect_audit_history(
    tmp_path: Path,
) -> None:
    initial_page = FakeRoomPage(
        [
            InterviewRoomState.WAITING_FOR_CANDIDATE,
            InterviewRoomState.CANDIDATE_CONNECTED,
        ]
    )
    first = wait_for_candidate_connection(
        initial_page,
        session_dir=tmp_path,
        poll_interval_seconds=0.01,
    )

    reconnected_page = FakeRoomPage([InterviewRoomState.CANDIDATE_CONNECTED])
    second = wait_for_candidate_connection(
        reconnected_page,
        session_dir=tmp_path,
        poll_interval_seconds=0.01,
        state_log_path=first.state_log_path,
        prior_state=first.final_state,
        prior_transitions=first.transitions,
        initial_state=InterviewRoomState.WAITING_FOR_CANDIDATE,
    )

    assert [transition.current for transition in second.transitions] == [
        InterviewRoomState.LAUNCHED,
        InterviewRoomState.INTERVIEWER_IN_ROOM,
        InterviewRoomState.WAITING_FOR_CANDIDATE,
        InterviewRoomState.CANDIDATE_CONNECTED,
        InterviewRoomState.WAITING_FOR_CANDIDATE,
        InterviewRoomState.CANDIDATE_CONNECTED,
    ]
    records = [
        json.loads(line)
        for line in second.state_log_path.read_text(encoding="utf-8").splitlines()
    ]
    assert [record["state"] for record in records].count("LAUNCHED") == 1
