"""Grounded answers to candidate role questions from saved FloCareer text."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

from llm.prompts import job_description_answer_messages
from llm.provider import ChatMessage
from llm.schemas import JobDescriptionAnswer, ModelClass, StructuredGeneration
from llm.usage_tracker import UsageTracker


class JobDescriptionError(ValueError):
    """A saved job description is unavailable or invalid."""


class JobDescriptionGenerator(Protocol):
    async def generate_structured(
        self,
        messages: Sequence[ChatMessage],
        schema: type[JobDescriptionAnswer],
        model_class: ModelClass,
        *,
        request_purpose: str,
    ) -> dict[str, object]: ...


@dataclass(frozen=True, slots=True)
class JobDescriptionAnswerResult:
    answer: JobDescriptionAnswer
    metadata: StructuredGeneration


UNAVAILABLE_JOB_DESCRIPTION_ANSWER = (
    "That detail is not available in the job description. "
    "Please check with the recruiter or interviewer."
)


def _normalize_evidence_whitespace(text: str) -> str:
    """Compare source words while allowing browser line wrapping to differ."""

    return " ".join(text.split())


def _candidate_safe_answer(answer: JobDescriptionAnswer) -> JobDescriptionAnswer:
    """Return only source text or the fixed unavailable-detail response."""

    if not answer.grounded:
        return answer.model_copy(
            update={
                "answer": UNAVAILABLE_JOB_DESCRIPTION_ANSWER,
                "evidence": [],
            }
        )
    return answer.model_copy(
        update={
            "answer": "According to the job description: " + " ".join(answer.evidence)
        }
    )


def load_job_description(session_dir: Path) -> str:
    path = session_dir / "job_description.json"
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except OSError as error:
        raise JobDescriptionError(
            f"Could not read job_description.json: {error}"
        ) from error
    except json.JSONDecodeError as error:
        raise JobDescriptionError("job_description.json is not valid JSON") from error
    if not isinstance(payload, dict):
        raise JobDescriptionError("job_description.json must contain an object")
    if payload.get("schema_version") != 1:
        raise JobDescriptionError("job_description.json has an unsupported schema")
    if payload.get("read_only") is not True:
        raise JobDescriptionError("job_description.json is not marked read-only")
    if payload.get("source") != "FloCareer Job Description tab":
        raise JobDescriptionError("job_description.json has an unexpected source")
    description = str(payload.get("description") or "").strip()
    if not description:
        raise JobDescriptionError("job_description.json has no readable description")
    return description


async def answer_job_description_question(
    *,
    job_description: str,
    candidate_question: str,
    generator: JobDescriptionGenerator,
    model_class: ModelClass = "fast",
    usage_tracker: UsageTracker | None = None,
) -> JobDescriptionAnswerResult:
    question = candidate_question.strip()
    if not question:
        raise JobDescriptionError("candidate question must not be empty")
    response = await generator.generate_structured(
        job_description_answer_messages(
            job_description=job_description,
            candidate_question=question,
        ),
        JobDescriptionAnswer,
        model_class,
        request_purpose="candidate_job_question",
    )
    generation = StructuredGeneration.model_validate(response)
    answer = JobDescriptionAnswer.model_validate(generation.output)
    normalized_description = _normalize_evidence_whitespace(job_description)
    invalid_evidence = [
        item
        for item in answer.evidence
        if _normalize_evidence_whitespace(item) not in normalized_description
    ]
    if invalid_evidence:
        raise JobDescriptionError(
            "model returned evidence that is not present in the saved job description"
        )
    candidate_answer = _candidate_safe_answer(answer)
    if usage_tracker is not None:
        usage_tracker.record(generation.metadata)
    return JobDescriptionAnswerResult(answer=candidate_answer, metadata=generation)
