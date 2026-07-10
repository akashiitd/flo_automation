"""Deterministic PII redaction applied before every cloud generation."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Mapping, Sequence


_PATTERNS: tuple[tuple[re.Pattern[str], str], ...] = (
    (
        re.compile(
            r"\b[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}\b",
            re.IGNORECASE,
        ),
        "[REDACTED_EMAIL]",
    ),
    (
        re.compile(r"(?<!\w)(?:\+?\d[\d\s().-]{6,}\d)(?!\w)"),
        "[REDACTED_PHONE]",
    ),
    (
        re.compile(r"(?i)\bmy name is\s+[A-Z][A-Z .'-]{1,80}?(?=[.!?,\n]|$)"),
        "My name is [REDACTED_NAME]",
    ),
    (
        re.compile(
            r"(?i)\b(account(?:\s+(?:id|identifier|number))?\s*[:#-]?\s*)"
            r"[A-Z0-9][A-Z0-9_-]{3,}"
        ),
        r"\1[REDACTED_ACCOUNT_ID]",
    ),
    (
        re.compile(r"(?im)^(\s*address\s*:).*$"),
        r"\1 [REDACTED_ADDRESS]",
    ),
    (
        re.compile(r"(?im)^(\s*resume(?:\s+(?:field|details))?\s*:).*$"),
        r"\1 [REDACTED_RESUME_FIELD]",
    ),
)


@dataclass(frozen=True, slots=True)
class RedactionResult:
    messages: list[dict[str, str]]
    redactions: int


def redact_text(text: str) -> tuple[str, int]:
    """Replace supported PII shapes and return the number of replacements."""

    redacted = text
    count = 0
    for pattern, replacement in _PATTERNS:
        redacted, replacements = pattern.subn(replacement, redacted)
        count += replacements
    return redacted, count


def redact_messages(
    messages: Sequence[Mapping[str, str]],
) -> RedactionResult:
    """Copy chat messages while redacting PII from every content field."""

    redacted_messages: list[dict[str, str]] = []
    total = 0
    for message in messages:
        content, replacements = redact_text(message.get("content", ""))
        total += replacements
        redacted_messages.append(
            {
                "role": message.get("role", "user"),
                "content": content,
            }
        )
    return RedactionResult(redacted_messages, total)
