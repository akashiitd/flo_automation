"""Contract tests for the initial durable LangGraph integration seam."""

from __future__ import annotations

import asyncio
import stat
from pathlib import Path
from typing import Any

from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.types import Command

from orchestrator.checkpointing import open_session_checkpointer
from orchestrator.langgraph_builder import build_persistence_spike


def _config(thread_id: str) -> RunnableConfig:
    return {"configurable": {"thread_id": thread_id}}


def test_persistence_spike_interrupts_and_resumes_with_one_thread() -> None:
    async def run() -> tuple[dict[str, object], list[Any]]:
        graph = build_persistence_spike(checkpointer=InMemorySaver())
        config = _config("interview-session-123")

        interrupted = await graph.ainvoke({"session_label": "fixture"}, config=config)

        pending_interrupt = interrupted["__interrupt__"][0]
        assert pending_interrupt.value == {
            "kind": "start_approval",
            "session_label": "fixture",
        }

        completed = await graph.ainvoke(Command(resume="approved"), config=config)
        snapshots = [snapshot async for snapshot in graph.aget_state_history(config)]
        return completed, snapshots

    completed, snapshots = asyncio.run(run())
    assert completed["approval"] == "approved"
    assert completed["history"] == ["started", "approved"]
    assert len(snapshots) >= 3
    assert snapshots[0].values["approval"] == "approved"


def test_persistence_spike_streams_v2_updates_and_custom_monitor_event() -> None:
    graph = build_persistence_spike(checkpointer=InMemorySaver())

    async def collect_events() -> list[dict[str, Any]]:
        events: list[dict[str, Any]] = []
        async for event in graph.astream(
            {"session_label": "stream-fixture"},
            config=_config("stream-session-456"),
            stream_mode=["updates", "custom"],
            version="v2",
        ):
            events.append(event)
        return events

    events = asyncio.run(collect_events())

    assert any(
        event["type"] == "custom"
        and event["data"] == {"phase": "awaiting_start_approval"}
        for event in events
    )
    assert any(
        event["type"] == "updates" and "record_started" in event["data"]
        for event in events
    )
    assert any(
        event["type"] == "updates" and "emit_start_monitor" in event["data"]
        for event in events
    )


def test_sqlite_checkpointer_resumes_and_keeps_session_data_owner_only(
    tmp_path: Path,
) -> None:
    async def run() -> tuple[dict[str, object], int]:
        config = _config("sqlite-session-789")
        async with open_session_checkpointer(tmp_path) as checkpointer:
            graph = build_persistence_spike(checkpointer=checkpointer)

            interrupted = await graph.ainvoke(
                {"session_label": "sqlite-fixture"}, config=config
            )
            assert interrupted["__interrupt__"]

        async with open_session_checkpointer(tmp_path) as checkpointer:
            graph = build_persistence_spike(checkpointer=checkpointer)
            completed = await graph.ainvoke(Command(resume="approved"), config=config)
            checkpoints = [
                checkpoint async for checkpoint in checkpointer.alist(config)
            ]
            return completed, len(checkpoints)

    completed, checkpoint_count = asyncio.run(run())
    database = tmp_path / "langgraph" / "checkpoints.sqlite3"

    assert completed["history"] == ["started", "approved"]
    assert checkpoint_count >= 3
    assert stat.S_IMODE(database.stat().st_mode) == 0o600
    assert stat.S_IMODE(database.parent.stat().st_mode) == 0o700
