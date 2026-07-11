"""Persistent, observed interview-room state management."""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Protocol


class InterviewRoomState(str, Enum):
    LAUNCHED = "LAUNCHED"
    INTERVIEWER_IN_ROOM = "INTERVIEWER_IN_ROOM"
    WAITING_FOR_CANDIDATE = "WAITING_FOR_CANDIDATE"
    CANDIDATE_CONNECTED = "CANDIDATE_CONNECTED"


class RoomWorkflowError(RuntimeError):
    """Raised when a live room cannot be observed safely."""


class RoomWorkflowPage(Protocol):
    def read_interview_room_state(self) -> InterviewRoomState: ...

    def wait_for_room_poll(self, seconds: float) -> None: ...

    def capture_screenshot(self, directory: Path, name: str) -> Path: ...


@dataclass(frozen=True, slots=True)
class RoomStateTransition:
    previous: InterviewRoomState | None
    current: InterviewRoomState
    timestamp: str
    screenshot_path: Path | None = None


@dataclass(frozen=True, slots=True)
class RoomMonitorResult:
    final_state: InterviewRoomState
    transitions: tuple[RoomStateTransition, ...]
    state_log_path: Path


class RoomStateTracker:
    """Record only meaningful state changes, including reconnects."""

    def __init__(self, initial_state: InterviewRoomState | None = None) -> None:
        self._current = initial_state

    @property
    def current(self) -> InterviewRoomState | None:
        return self._current

    def observe(self, state: InterviewRoomState) -> RoomStateTransition | None:
        if state is self._current:
            return None
        transition = RoomStateTransition(
            previous=self._current,
            current=state,
            timestamp=datetime.now(UTC).isoformat(),
        )
        self._current = state
        return transition


def _write_transition(path: Path, transition: RoomStateTransition) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = asdict(transition)
    payload["previous"] = (
        transition.previous.value if transition.previous is not None else None
    )
    payload["current"] = transition.current.value
    payload["state"] = transition.current.value
    payload["screenshot_path"] = (
        str(transition.screenshot_path) if transition.screenshot_path else None
    )
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, sort_keys=True) + "\n")


def _state_label(state: InterviewRoomState) -> str:
    return state.value.lower()


def _status_message(state: InterviewRoomState) -> str:
    return {
        InterviewRoomState.LAUNCHED: "Interview launch approved; waiting for room entry",
        InterviewRoomState.INTERVIEWER_IN_ROOM: "Interviewer is in the interview room",
        InterviewRoomState.WAITING_FOR_CANDIDATE: "Waiting for candidate connection",
        InterviewRoomState.CANDIDATE_CONNECTED: "Candidate connected to the interview",
    }[state]


def wait_for_candidate_connection(
    page: RoomWorkflowPage,
    *,
    session_dir: Path,
    poll_interval_seconds: float = 2,
    status_interval_seconds: float = 15,
    timeout_seconds: float | None = None,
    report: Callable[[str], None] | None = None,
    should_continue: Callable[[], bool] | None = None,
    state_log_path: Path | None = None,
    prior_state: InterviewRoomState | None = None,
    prior_transitions: tuple[RoomStateTransition, ...] = (),
    initial_state: InterviewRoomState | None = None,
) -> RoomMonitorResult:
    """Keep the live session open until a candidate is observed connected.

    ``timeout_seconds`` is opt-in. Without it, the caller controls cancellation
    through interruption or ``should_continue`` while the browser remains open.
    """

    if poll_interval_seconds <= 0:
        raise ValueError("poll interval must be positive")
    if status_interval_seconds <= 0:
        raise ValueError("status interval must be positive")
    if timeout_seconds is not None and timeout_seconds <= 0:
        raise ValueError("timeout must be positive when configured")

    state_log_path = state_log_path or session_dir / "room_state_log.jsonl"
    screenshots_dir = session_dir / "screenshots"
    tracker = RoomStateTracker(prior_state)
    transitions = list(prior_transitions)
    started = time.monotonic()
    last_status = started
    reporter = report or (lambda message: None)
    keep_going = should_continue or (lambda: True)

    def record(state: InterviewRoomState) -> None:
        transition = tracker.observe(state)
        if transition is None:
            return
        screenshot = None
        if state is not InterviewRoomState.LAUNCHED:
            screenshot = page.capture_screenshot(
                screenshots_dir, f"room_{_state_label(state)}"
            )
        recorded = RoomStateTransition(
            previous=transition.previous,
            current=transition.current,
            timestamp=transition.timestamp,
            screenshot_path=screenshot,
        )
        transitions.append(recorded)
        _write_transition(state_log_path, recorded)
        reporter(f"{recorded.timestamp}: {_status_message(state)}")

    if prior_state is None:
        record(InterviewRoomState.LAUNCHED)
        record(InterviewRoomState.INTERVIEWER_IN_ROOM)
    if initial_state is not None:
        record(initial_state)
    while True:
        if not keep_going():
            raise RoomWorkflowError("Operator cancelled while waiting for candidate")
        if (
            timeout_seconds is not None
            and time.monotonic() - started >= timeout_seconds
        ):
            raise RoomWorkflowError("Configured candidate wait timeout elapsed")
        state = page.read_interview_room_state()
        if state is InterviewRoomState.LAUNCHED:
            raise RoomWorkflowError("Room page cannot report LAUNCHED after Join")
        record(state)
        if state is InterviewRoomState.CANDIDATE_CONNECTED:
            return RoomMonitorResult(
                final_state=state,
                transitions=tuple(transitions),
                state_log_path=state_log_path,
            )
        if time.monotonic() - last_status >= status_interval_seconds:
            reporter(
                f"{datetime.now(UTC).isoformat()}: still "
                f"{_status_message(state).lower()}"
            )
            last_status = time.monotonic()
        page.wait_for_room_poll(poll_interval_seconds)
