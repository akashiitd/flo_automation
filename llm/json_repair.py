"""Conservative parsing and repair for local-model JSON responses."""

from __future__ import annotations

import json
import re


_THINK_BLOCK = re.compile(r"<think>.*?</think>", re.IGNORECASE | re.DOTALL)
_TRAILING_COMMA = re.compile(r",\s*([}\]])")


class JsonRepairError(ValueError):
    """Raised when no valid JSON object can be recovered from model output."""


def _first_balanced_object(text: str) -> str:
    start = text.find("{")
    if start < 0:
        raise JsonRepairError("model response did not contain a JSON object")

    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        character = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif character == "\\":
                escaped = True
            elif character == '"':
                in_string = False
            continue

        if character == '"':
            in_string = True
        elif character == "{":
            depth += 1
        elif character == "}":
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    raise JsonRepairError("model response contained an incomplete JSON object")


def parse_json_object(content: str) -> dict[str, object]:
    """Parse a JSON object after removing reasoning and safe syntax noise."""

    without_reasoning = _THINK_BLOCK.sub("", content).strip()
    candidate = _first_balanced_object(without_reasoning)

    parse_errors: list[str] = []
    for possible_json in (candidate, _TRAILING_COMMA.sub(r"\1", candidate)):
        try:
            parsed = json.loads(possible_json)
        except json.JSONDecodeError as error:
            parse_errors.append(str(error))
            continue
        if not isinstance(parsed, dict):
            raise JsonRepairError("model response JSON must be an object")
        return parsed

    raise JsonRepairError("; ".join(parse_errors))
