"""Durable executor for graph-issued answer evaluation effects."""

from __future__ import annotations

import json
import os
import sqlite3
from hashlib import sha256
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from pydantic import JsonValue

from evaluator.scoring import StructuredGenerator, evaluate_answer
from llm.schemas import EvaluationInput, QuestionEvaluation
from orchestrator.effect_ledger import EffectLedger
from orchestrator.effects import EffectRequest, EffectResult, EffectStatus, EffectType
from orchestrator.event_adapters import EventNormalizer
from orchestrator.events import InterviewEvent
from orchestrator.state import QuestionState

_EVALUATION_PROMPT_SCHEMA_VERSION = "question-evaluation-v1"


@dataclass(frozen=True, slots=True)
class EvaluationExecution:
    result: EffectResult
    events: tuple[InterviewEvent, ...]


class EvaluationEffectExecutor:
    """Cache validated evaluation output before emitting a graph result event."""

    def __init__(
        self,
        *,
        ledger: EffectLedger,
        result_path: Path,
        normalizer: EventNormalizer,
        evaluator: StructuredGenerator,
        questions: Sequence[QuestionState],
    ) -> None:
        self._ledger = ledger
        self._path = result_path
        self._normalizer = normalizer
        self._evaluator = evaluator
        self._questions = {question.id: question for question in questions}

    async def execute(self, request: EffectRequest) -> EvaluationExecution:
        if request.effect_type is not EffectType.EVALUATE_ANSWER:
            raise ValueError("evaluation executor accepts EVALUATE_ANSWER only")
        entry = self._ledger.prepare(request)
        cached = self._load_output(request)
        if entry.result.status in {
            EffectStatus.FAILED,
            EffectStatus.CANCELLED,
            EffectStatus.UNCERTAIN,
        }:
            return EvaluationExecution(
                entry.result, self._failed_event(request, entry.result)
            )
        if entry.result.status is EffectStatus.COMPLETED:
            if cached is not None:
                return EvaluationExecution(
                    entry.result, self._completed_event(request, cached)
                )
            return EvaluationExecution(
                entry.result, self._failed_event(request, entry.result)
            )
        if cached is not None:
            if entry.result.status is EffectStatus.PREPARED:
                entry, claimed = self._ledger.claim_start(request)
                if not claimed and entry.result.status is not EffectStatus.STARTED:
                    return EvaluationExecution(
                        entry.result, self._failed_event(request, entry.result)
                    )
            entry = self._ledger.transition(
                request,
                status=EffectStatus.COMPLETED,
                result_summary="answer evaluation recovered from local cache",
            )
            return EvaluationExecution(
                entry.result, self._completed_event(request, cached)
            )
        started, claimed = self._ledger.claim_start(request)
        if not claimed:
            recovered = self._ledger.reconcile_after_restart(request)
            return EvaluationExecution(
                recovered.result, self._failed_event(request, recovered.result)
            )
        try:
            question_id, answer = _request_input(request)
            question = self._questions[question_id]
            generation = await evaluate_answer(
                EvaluationInput(
                    question_id=question_id,
                    question=question.question_text,
                    ideal_answer=question.ideal_answer,
                    candidate_answer=answer,
                ),
                self._evaluator,
            )
            output = QuestionEvaluation.model_validate(generation.output).model_dump(
                mode="json"
            )
            self._store_output(request, output)
            result = self._ledger.transition(
                request,
                status=EffectStatus.COMPLETED,
                result_summary="answer evaluated",
            ).result
            return EvaluationExecution(result, self._completed_event(request, output))
        except Exception as error:
            result = self._ledger.transition(
                request,
                status=EffectStatus.FAILED,
                result_summary=f"evaluation failed: {type(error).__name__}",
            ).result
            return EvaluationExecution(result, self._failed_event(request, result))

    def _connect(self) -> sqlite3.Connection:
        self._path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
        os.chmod(self._path.parent, 0o700)
        connection = sqlite3.connect(self._path)
        os.chmod(self._path, 0o600)
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS evaluation_results (
                session_id TEXT NOT NULL,
                cache_key TEXT NOT NULL,
                output_json TEXT NOT NULL,
                PRIMARY KEY (session_id, cache_key)
            )
            """
        )
        return connection

    def _load_output(self, request: EffectRequest) -> dict[str, JsonValue] | None:
        with self._connect() as connection:
            row = connection.execute(
                """
                SELECT output_json FROM evaluation_results
                WHERE session_id = ? AND cache_key = ?
                """,
                (request.session_id, _evaluation_cache_key(request)),
            ).fetchone()
        return None if row is None else json.loads(row[0])

    def _store_output(
        self, request: EffectRequest, output: dict[str, JsonValue]
    ) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT OR IGNORE INTO evaluation_results
                (session_id, cache_key, output_json) VALUES (?, ?, ?)
                """,
                (
                    request.session_id,
                    _evaluation_cache_key(request),
                    json.dumps(output, sort_keys=True),
                ),
            )

    def _completed_event(
        self, request: EffectRequest, output: dict[str, JsonValue]
    ) -> tuple[InterviewEvent, ...]:
        assert request.question_id is not None
        return (
            self._normalizer.evaluation_result(
                effect_id=request.effect_id,
                outcome="completed",
                question_id=request.question_id,
                output=output,
                result_summary="answer evaluated",
            ),
        )

    def _failed_event(
        self, request: EffectRequest, result: EffectResult
    ) -> tuple[InterviewEvent, ...]:
        assert request.question_id is not None
        return (
            self._normalizer.evaluation_result(
                effect_id=request.effect_id,
                outcome="failed",
                question_id=request.question_id,
                result_summary=result.result_summary,
            ),
        )


def _request_input(request: EffectRequest) -> tuple[int, str]:
    question_id = request.question_id
    answer = request.payload.get("candidate_answer")
    if question_id is None or not isinstance(answer, str) or not answer.strip():
        raise ValueError("evaluation request requires question_id and candidate_answer")
    return question_id, answer


def _evaluation_cache_key(request: EffectRequest) -> str:
    """Version the cache by question and normalized answer, scoped by session."""

    question_id, answer = _request_input(request)
    normalized_answer = " ".join(answer.split()).casefold()
    answer_hash = sha256(normalized_answer.encode("utf-8")).hexdigest()
    return sha256(
        f"{question_id}\0{answer_hash}\0{_EVALUATION_PROMPT_SCHEMA_VERSION}".encode(
            "utf-8"
        )
    ).hexdigest()


__all__ = ["EvaluationEffectExecutor", "EvaluationExecution"]
