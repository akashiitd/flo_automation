from __future__ import annotations

from llm.privacy_redactor import redact_messages


def test_cloud_messages_redact_candidate_pii() -> None:
    result = redact_messages(
        [
            {
                "role": "user",
                "content": (
                    "My name is Alice Example. Email alice@example.com. "
                    "Phone +91 98765 43210. Account ID: ACCT-88771.\n"
                    "Address: 12 MG Road, Pune\n"
                    "Resume: previously employed by Example Private Limited"
                ),
            }
        ]
    )

    rendered = result.messages[0]["content"]
    assert "Alice Example" not in rendered
    assert "alice@example.com" not in rendered
    assert "98765 43210" not in rendered
    assert "ACCT-88771" not in rendered
    assert "12 MG Road" not in rendered
    assert "Example Private Limited" not in rendered
    assert result.redactions >= 6
