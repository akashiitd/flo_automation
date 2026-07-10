from __future__ import annotations

from llm.json_repair import parse_json_object


def test_parser_removes_reasoning_fences_and_trailing_commas() -> None:
    parsed = parse_json_object(
        """
        <think>This should never enter the evaluator contract.</think>
        Here is the result:
        ```json
        {"question_id": 1, "score": 3, "evidence": ["API layer",],}
        ```
        """
    )

    assert parsed == {
        "question_id": 1,
        "score": 3,
        "evidence": ["API layer"],
    }
