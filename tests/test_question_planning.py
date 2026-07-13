from __future__ import annotations

import json
import stat
from pathlib import Path

from orchestrator.question_planning import build_interview_plan
from orchestrator.state import QuestionContentType, QuestionPlanArtifact


def _write_scan_artifacts(session: Path) -> None:
    session.mkdir()
    (session / "questions.json").write_text(
        json.dumps(
            [
                {
                    "id": 17,
                    "question_text": (
                        "Before you begin, please read these interview instructions."
                    ),
                    "ideal_answer": "",
                    "has_code_editor": False,
                },
                {
                    "id": 31,
                    "question_text": "How would you secure a REST API?",
                    "ideal_answer": "Discuss authorization and validation.",
                    "has_code_editor": False,
                },
                {
                    "id": 42,
                    "question_text": "Implement a Python cache with eviction.",
                    "ideal_answer": "Use a hash map and linked list.",
                    "has_code_editor": True,
                },
                {
                    "id": 58,
                    "question_text": "How would you build a RAG pipeline?",
                    "ideal_answer": "Cover retrieval, grounding, and evaluation.",
                    "has_code_editor": False,
                },
                {
                    "id": 59,
                    "question_text": "How would you secure a REST API endpoint?",
                    "ideal_answer": "Discuss authentication and validation.",
                    "has_code_editor": False,
                },
            ]
        ),
        encoding="utf-8",
    )
    (session / "skill_parameters.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "read_only": True,
                "parameters": [
                    {
                        "id": "rest-api-security",
                        "name": "REST API Development & Security",
                        "requirement": "Mandatory",
                        "level": "Professional",
                        "rating_scale": 5,
                        "source": "flocareer_dom",
                    },
                    {
                        "id": "python",
                        "name": "Python",
                        "requirement": "Mandatory",
                        "level": "Professional",
                        "rating_scale": 5,
                        "source": "flocareer_dom",
                    },
                    {
                        "id": "rag",
                        "name": "RAG",
                        "requirement": "Mandatory",
                        "level": "Professional",
                        "rating_scale": 5,
                        "source": "flocareer_dom",
                    },
                ],
            }
        ),
        encoding="utf-8",
    )


def test_plan_interview_excludes_setup_by_content_and_covers_mandatory_skills(
    tmp_path: Path,
) -> None:
    session = tmp_path / "questions_scan_fixture"
    _write_scan_artifacts(session)

    result = build_interview_plan(session, minutes=20)

    artifact = QuestionPlanArtifact.model_validate_json(
        result.plan_path.read_text(encoding="utf-8")
    )
    by_id = {item.question_id: item for item in artifact.items}
    assert len(artifact.items) == 5
    assert by_id[17].content_type is QuestionContentType.INSTRUCTION
    assert by_id[17].selected is False
    assert "instruction" in str(by_id[17].skip_reason)
    assert by_id[42].content_type is QuestionContentType.CODING_QUESTION
    assert by_id[42].selected is True
    assert by_id[31].mapping_evidence == ["rest-api-security: matched terms api, rest"]
    assert {"rest-api-security", "python", "rag"} <= {
        skill_id
        for item in artifact.items
        if item.selected
        for skill_id in item.mandatory_skill_coverage
    }
    assert by_id[59].selected is False
    assert "redundant" in str(by_id[59].skip_reason)
    assert "# Offline interview plan" in result.markdown_path.read_text(
        encoding="utf-8"
    )
    assert stat.S_IMODE(result.plan_path.stat().st_mode) == 0o600
    assert stat.S_IMODE(result.markdown_path.stat().st_mode) == 0o600


def test_plan_interview_applies_explicit_operator_edits(tmp_path: Path) -> None:
    session = tmp_path / "questions_scan_fixture"
    _write_scan_artifacts(session)
    edits = session / "plan_edits.json"
    edits.write_text(
        json.dumps(
            {
                "out_of_scope_skill_ids": ["python"],
                "select": [59],
                "order": [59, 31],
            }
        ),
        encoding="utf-8",
    )

    result = build_interview_plan(session, minutes=20, edits_path=edits)

    artifact = QuestionPlanArtifact.model_validate_json(
        result.plan_path.read_text(encoding="utf-8")
    )
    by_id = {item.question_id: item for item in artifact.items}
    assert by_id[42].selected is False
    assert by_id[42].skip_reason == "not selected: coding skill explicitly out of scope"
    assert by_id[59].selected is True
    assert [item.question_id for item in artifact.items if item.selected][:2] == [
        59,
        31,
    ]


def test_plan_interview_skips_near_identical_optional_coverage(tmp_path: Path) -> None:
    session = tmp_path / "questions_scan_fixture"
    session.mkdir()
    (session / "questions.json").write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "question_text": "Explain alpha beta gamma delta epsilon.",
                    "ideal_answer": "",
                    "has_code_editor": False,
                },
                {
                    "id": 2,
                    "question_text": "Explain alpha beta gamma delta zeta.",
                    "ideal_answer": "",
                    "has_code_editor": False,
                },
            ]
        ),
        encoding="utf-8",
    )
    parameters = [
        {
            "id": name,
            "name": name.title(),
            "requirement": "Mandatory" if name != "zeta" else "Optional",
            "level": "Professional",
            "rating_scale": 5,
            "source": "flocareer_dom",
        }
        for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
    ]
    (session / "skill_parameters.json").write_text(
        json.dumps({"schema_version": 1, "read_only": True, "parameters": parameters}),
        encoding="utf-8",
    )

    result = build_interview_plan(session, minutes=20)

    by_id = {item.question_id: item for item in result.artifact.items}
    assert by_id[1].selected is True
    assert by_id[2].selected is False
    assert by_id[2].skip_reason == "not selected: near-identical skill coverage"


def test_plan_interview_keeps_near_identical_question_for_mandatory_gap(
    tmp_path: Path,
) -> None:
    session = tmp_path / "questions_scan_fixture"
    session.mkdir()
    (session / "questions.json").write_text(
        json.dumps(
            [
                {
                    "id": 1,
                    "question_text": "Explain alpha beta gamma delta epsilon.",
                    "ideal_answer": "",
                    "has_code_editor": False,
                },
                {
                    "id": 2,
                    "question_text": "Explain alpha beta gamma delta zeta.",
                    "ideal_answer": "",
                    "has_code_editor": False,
                },
            ]
        ),
        encoding="utf-8",
    )
    parameters = [
        {
            "id": name,
            "name": name.title(),
            "requirement": "Mandatory",
            "level": "Professional",
            "rating_scale": 5,
            "source": "flocareer_dom",
        }
        for name in ("alpha", "beta", "gamma", "delta", "epsilon", "zeta")
    ]
    (session / "skill_parameters.json").write_text(
        json.dumps({"schema_version": 1, "read_only": True, "parameters": parameters}),
        encoding="utf-8",
    )

    result = build_interview_plan(session, minutes=20)

    assert [item.question_id for item in result.artifact.items if item.selected] == [
        1,
        2,
    ]
