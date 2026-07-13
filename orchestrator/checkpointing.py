"""Private, local checkpoint storage for durable interview graphs."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import aiosqlite
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver


@asynccontextmanager
async def open_session_checkpointer(
    session_dir: Path,
) -> AsyncIterator[AsyncSqliteSaver]:
    """Open a strict, owner-only SQLite checkpointer for one interview session."""

    session_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(session_dir, 0o700)
    checkpoint_dir = session_dir / "langgraph"
    checkpoint_dir.mkdir(mode=0o700, exist_ok=True)
    os.chmod(checkpoint_dir, 0o700)
    database = checkpoint_dir / "checkpoints.sqlite3"
    database.touch(mode=0o600, exist_ok=True)
    os.chmod(database, 0o600)

    serializer = JsonPlusSerializer(allowed_msgpack_modules=None)
    async with aiosqlite.connect(database) as connection:
        saver = AsyncSqliteSaver(connection, serde=serializer)
        await saver.setup()
        os.chmod(database, 0o600)
        yield saver


__all__ = ["open_session_checkpointer"]
