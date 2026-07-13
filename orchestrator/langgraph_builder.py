"""Durable LangGraph builders for the supervised interview controller.

The persistence spike is intentionally small.  It proves the checkpoint,
interrupt, resume, history, and streaming contracts before live interview
adapters are introduced.
"""

from __future__ import annotations

import operator
from typing import Any

from pydantic import BaseModel, Field
from typing_extensions import Annotated

from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.config import get_stream_writer
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt


class PersistenceSpikeState(BaseModel):
    """Checkpoint-safe state used only by the first LangGraph integration spike."""

    session_label: str
    approval: str | None = None
    history: Annotated[list[str], operator.add] = Field(default_factory=list)


def _record_started(state: PersistenceSpikeState) -> dict[str, list[str]]:
    """Emit the monitor update immediately before the approval boundary."""

    del state
    get_stream_writer()({"phase": "awaiting_start_approval"})
    return {"history": ["started"]}


def _await_start_approval(
    state: PersistenceSpikeState,
) -> dict[str, str | list[str]]:
    """Pause durably until the scoped start decision is supplied on resume."""

    approval = interrupt(
        {"kind": "start_approval", "session_label": state.session_label}
    )
    return {"approval": str(approval), "history": ["approved"]}


def build_persistence_spike(*, checkpointer: BaseCheckpointSaver[Any]) -> Any:
    """Compile the smallest durable graph required by Phase 1.

    A caller must supply a stable ``thread_id`` in its LangGraph config.  The
    production controller will replace these two nodes with typed interview
    phases, retaining this persistence boundary.
    """

    builder = StateGraph(PersistenceSpikeState)
    builder.add_node("record_started", _record_started)
    builder.add_node("await_start_approval", _await_start_approval)
    builder.add_edge(START, "record_started")
    builder.add_edge("record_started", "await_start_approval")
    builder.add_edge("await_start_approval", END)
    return builder.compile(checkpointer=checkpointer)


__all__ = ["PersistenceSpikeState", "build_persistence_spike"]
