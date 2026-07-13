"""Durable idempotency and outcome ledger for graph-issued effects."""

from __future__ import annotations

import json
import os
import sqlite3
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

from orchestrator.effects import EffectRequest, EffectResult, EffectStatus


class EffectLedgerConflictError(ValueError):
    """An effect identity was reused for a different request."""


class EffectLifecycleError(ValueError):
    """An effect attempted an invalid durable lifecycle transition."""


@dataclass(frozen=True, slots=True)
class EffectLedgerEntry:
    """One request paired with its latest durable execution result."""

    request: EffectRequest
    result: EffectResult


_TERMINAL_STATUSES = {
    EffectStatus.COMPLETED,
    EffectStatus.CANCELLED,
    EffectStatus.FAILED,
    EffectStatus.UNCERTAIN,
}


class EffectLedger:
    """Persist prepared effects before any external TTS or audio operation."""

    def __init__(self, path: Path) -> None:
        self._path = path
        self._audit_path = path.with_suffix(".jsonl")

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)
        connection = sqlite3.connect(self._path)
        os.chmod(self._path, 0o600)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS effect_ledger (
                session_id TEXT NOT NULL,
                effect_id TEXT NOT NULL,
                idempotency_key TEXT NOT NULL,
                request_json TEXT NOT NULL,
                result_json TEXT NOT NULL,
                PRIMARY KEY (session_id, effect_id),
                UNIQUE (session_id, idempotency_key)
            )
            """
        )
        return connection

    def prepare(self, request: EffectRequest) -> EffectLedgerEntry:
        """Record a request before it can reach an external effect boundary."""

        prepared = EffectResult(
            effect_id=request.effect_id,
            session_id=request.session_id,
            effect_type=request.effect_type,
            idempotency_key=request.idempotency_key,
            payload_hash=request.payload_hash,
            status=EffectStatus.PREPARED,
            result_summary="effect prepared",
        )
        with self._connect() as connection:
            existing = self._select_identity(connection, request, missing_ok=True)
            if existing is not None:
                return existing
            try:
                connection.execute(
                    """
                    INSERT INTO effect_ledger (
                        session_id, effect_id, idempotency_key, request_json, result_json
                    ) VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        request.session_id,
                        request.effect_id,
                        request.idempotency_key,
                        request.model_dump_json(),
                        prepared.model_dump_json(),
                    ),
                )
            except sqlite3.IntegrityError as error:
                existing = self._select_identity(connection, request)
                if existing is None:
                    raise EffectLedgerConflictError(
                        "could not prepare effect"
                    ) from error
                return existing
        entry = EffectLedgerEntry(request=request, result=prepared)
        self._append_audit(entry)
        return entry

    def get(self, request: EffectRequest) -> EffectLedgerEntry | None:
        """Return the matching durable record, if this executor has seen it."""

        with self._connect() as connection:
            return self._select_identity(connection, request, missing_ok=True)

    def transition(
        self,
        request: EffectRequest,
        *,
        status: EffectStatus,
        result_summary: str,
        occurred_at: datetime | None = None,
    ) -> EffectLedgerEntry:
        """Atomically advance one effect and reject conflicting retries."""

        at = occurred_at or datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            entry = self._select_identity(connection, request)
            assert entry is not None
            current = entry.result
            if current.status is status:
                return entry
            if current.status in _TERMINAL_STATUSES:
                raise EffectLifecycleError(
                    f"cannot transition terminal effect {request.effect_id} from "
                    f"{current.status} to {status}"
                )
            if status is EffectStatus.PREPARED:
                raise EffectLifecycleError("effects cannot return to PREPARED")
            if current.status is EffectStatus.PREPARED and status not in {
                EffectStatus.STARTED,
                EffectStatus.FAILED,
                EffectStatus.UNCERTAIN,
            }:
                raise EffectLifecycleError(
                    f"prepared effect cannot transition directly to {status}"
                )
            result = EffectResult(
                effect_id=request.effect_id,
                session_id=request.session_id,
                effect_type=request.effect_type,
                idempotency_key=request.idempotency_key,
                payload_hash=request.payload_hash,
                status=status,
                result_summary=result_summary,
                started_at=at if status is EffectStatus.STARTED else current.started_at,
                completed_at=at if status in _TERMINAL_STATUSES else None,
            )
            connection.execute(
                """
                UPDATE effect_ledger SET result_json = ?
                WHERE session_id = ? AND effect_id = ?
                """,
                (result.model_dump_json(), request.session_id, request.effect_id),
            )
        entry = EffectLedgerEntry(request=request, result=result)
        self._append_audit(entry)
        return entry

    def claim_start(
        self,
        request: EffectRequest,
        *,
        occurred_at: datetime | None = None,
    ) -> tuple[EffectLedgerEntry, bool]:
        """Claim a prepared effect for one executor before external work starts.

        A redelivered request can observe an in-flight record but must never
        start a second playback stream for that same idempotency key.
        """

        at = occurred_at or datetime.now(UTC)
        with self._connect() as connection:
            connection.execute("BEGIN IMMEDIATE")
            entry = self._select_identity(connection, request)
            assert entry is not None
            if entry.result.status is not EffectStatus.PREPARED:
                return entry, False
            result = EffectResult(
                effect_id=request.effect_id,
                session_id=request.session_id,
                effect_type=request.effect_type,
                idempotency_key=request.idempotency_key,
                payload_hash=request.payload_hash,
                status=EffectStatus.STARTED,
                result_summary="effect execution started",
                started_at=at,
            )
            connection.execute(
                """
                UPDATE effect_ledger SET result_json = ?
                WHERE session_id = ? AND effect_id = ?
                """,
                (result.model_dump_json(), request.session_id, request.effect_id),
            )
        entry = EffectLedgerEntry(request=request, result=result)
        self._append_audit(entry)
        return entry, True

    def reconcile_after_restart(
        self,
        request: EffectRequest,
        *,
        occurred_at: datetime | None = None,
    ) -> EffectLedgerEntry:
        """Mark an in-flight effect uncertain; never guess whether speech occurred."""

        entry = self.get(request)
        if entry is None:
            return self.prepare(request)
        if entry.result.status is not EffectStatus.STARTED:
            return entry
        return self.transition(
            request,
            status=EffectStatus.UNCERTAIN,
            result_summary="process restarted while effect execution was in progress",
            occurred_at=occurred_at,
        )

    def _select_identity(
        self,
        connection: sqlite3.Connection,
        request: EffectRequest,
        *,
        missing_ok: bool = False,
    ) -> EffectLedgerEntry | None:
        row = connection.execute(
            """
            SELECT request_json, result_json FROM effect_ledger
            WHERE session_id = ? AND (effect_id = ? OR idempotency_key = ?)
            """,
            (request.session_id, request.effect_id, request.idempotency_key),
        ).fetchone()
        if row is None:
            if missing_ok:
                return None
            raise EffectLifecycleError(f"effect {request.effect_id!r} is not prepared")
        stored_request = EffectRequest.model_validate_json(row[0])
        if stored_request != request:
            raise EffectLedgerConflictError(
                f"effect identity for {request.effect_id!r} was reused with different content"
            )
        return EffectLedgerEntry(
            request=stored_request,
            result=EffectResult.model_validate_json(row[1]),
        )

    def _append_audit(self, entry: EffectLedgerEntry) -> None:
        """Append lifecycle observations; SQLite remains the recovery authority."""

        self._audit_path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._audit_path.parent, 0o700)
        record = {
            "request": entry.request.model_dump(mode="json"),
            "result": entry.result.model_dump(mode="json"),
        }
        with self._audit_path.open("a", encoding="utf-8") as audit_file:
            os.chmod(self._audit_path, 0o600)
            audit_file.write(json.dumps(record, sort_keys=True, separators=(",", ":")))
            audit_file.write("\n")


__all__ = [
    "EffectLedger",
    "EffectLedgerConflictError",
    "EffectLedgerEntry",
    "EffectLifecycleError",
]
