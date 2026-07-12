from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from evaluator.job_description import (
    JobDescriptionError,
    UNAVAILABLE_JOB_DESCRIPTION_ANSWER,
    answer_job_description_question,
    load_job_description,
)
from llm.schemas import ProviderMetadata, StructuredGeneration


class GroundedGenerator:
    async def generate_structured(
        self, *args: object, **kwargs: object
    ) -> dict[str, object]:
        return StructuredGeneration(
            output={
                "answer": "The role includes building RAG pipelines and API integrations.",
                "grounded": True,
                "evidence": [
                    "Build reliable GenAI services with RAG pipelines and APIs."
                ],
            },
            metadata=ProviderMetadata(
                provider="test",
                model="test-model",
                request_purpose="candidate_job_question",
                latency_ms=1,
                input_tokens=1,
                output_tokens=1,
                estimated_cost_usd=0,
            ),
        ).model_dump(mode="json")


def test_job_description_answers_are_generated_from_saved_session_text(
    tmp_path: Path,
) -> None:
    (tmp_path / "job_description.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "read_only": True,
                "source": "FloCareer Job Description tab",
                "description": "Build reliable GenAI services with RAG pipelines and APIs.",
            }
        ),
        encoding="utf-8",
    )

    result = asyncio.run(
        answer_job_description_question(
            job_description=load_job_description(tmp_path),
            candidate_question="What technology does this role use?",
            generator=GroundedGenerator(),
        )
    )

    assert result.answer.grounded is True
    assert result.answer.answer == (
        "According to the job description: "
        "Build reliable GenAI services with RAG pipelines and APIs."
    )


def test_job_description_answers_reject_evidence_not_present_in_source() -> None:
    class UngroundedGenerator:
        async def generate_structured(
            self, *args: object, **kwargs: object
        ) -> dict[str, object]:
            return StructuredGeneration(
                output={
                    "answer": "The project uses Kubernetes.",
                    "grounded": True,
                    "evidence": ["Kubernetes platform"],
                },
                metadata=ProviderMetadata(
                    provider="test",
                    model="test-model",
                    request_purpose="candidate_job_question",
                    latency_ms=1,
                    input_tokens=1,
                    output_tokens=1,
                    estimated_cost_usd=0,
                ),
            ).model_dump(mode="json")

    with pytest.raises(JobDescriptionError, match="not present"):
        asyncio.run(
            answer_job_description_question(
                job_description="Build RAG pipelines.",
                candidate_question="What platform do you use?",
                generator=UngroundedGenerator(),
            )
        )


def test_job_description_answers_accept_evidence_with_normalized_line_breaks() -> None:
    class MultiLineEvidenceGenerator:
        async def generate_structured(
            self, *args: object, **kwargs: object
        ) -> dict[str, object]:
            return StructuredGeneration(
                output={
                    "answer": "You will use a secret technology.",
                    "grounded": True,
                    "evidence": ["RAG pipelines Vector databases LLM APIs Agentic AI"],
                },
                metadata=ProviderMetadata(
                    provider="test",
                    model="test-model",
                    request_purpose="candidate_job_question",
                    latency_ms=1,
                    input_tokens=1,
                    output_tokens=1,
                    estimated_cost_usd=0,
                ),
            ).model_dump(mode="json")

    result = asyncio.run(
        answer_job_description_question(
            job_description=(
                "Experience with:\nRAG pipelines\nVector databases\nLLM APIs\n"
                "Agentic AI"
            ),
            candidate_question="What technologies would I use?",
            generator=MultiLineEvidenceGenerator(),
        )
    )

    assert result.answer.answer == (
        "According to the job description: "
        "RAG pipelines Vector databases LLM APIs Agentic AI"
    )


def test_job_description_answers_replace_unsupported_model_text_with_fallback() -> None:
    class UnsupportedGenerator:
        async def generate_structured(
            self, *args: object, **kwargs: object
        ) -> dict[str, object]:
            return StructuredGeneration(
                output={
                    "answer": "The project is for Client X and has a remote-first culture.",
                    "grounded": False,
                    "evidence": [],
                },
                metadata=ProviderMetadata(
                    provider="test",
                    model="test-model",
                    request_purpose="candidate_job_question",
                    latency_ms=1,
                    input_tokens=1,
                    output_tokens=1,
                    estimated_cost_usd=0,
                ),
            ).model_dump(mode="json")

    result = asyncio.run(
        answer_job_description_question(
            job_description="Build RAG pipelines.",
            candidate_question="Which client is this project for?",
            generator=UnsupportedGenerator(),
        )
    )

    assert result.answer.answer == UNAVAILABLE_JOB_DESCRIPTION_ANSWER
    assert result.answer.evidence == []


def test_job_description_load_requires_flocareer_provenance(tmp_path: Path) -> None:
    (tmp_path / "job_description.json").write_text(
        json.dumps({"description": "Build RAG pipelines."}), encoding="utf-8"
    )

    with pytest.raises(JobDescriptionError, match="unsupported schema"):
        load_job_description(tmp_path)
