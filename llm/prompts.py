"""Versioned prompts for structured candidate-answer evaluation."""

from __future__ import annotations

from llm.schemas import EvaluationInput


SCORING_SYSTEM_PROMPT = """You evaluate one interview answer against its rubric.
Return exactly one JSON object matching the supplied schema.

Rules:
- Score from 1 to 5: 5 excellent, 4 good, 3 average, 2 weak, 1 poor.
- Quote or closely paraphrase concrete evidence from the candidate answer.
- Do not invent claims, skills, or answers not present in the candidate answer.
- Give practical, concise feedback.
- Provide one concise, relevant follow-up question only.
- The follow-up must not reveal the ideal answer.
- Confidence must be between 0 and 1.
"""


def scoring_messages(request: EvaluationInput) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": SCORING_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "Evaluate this answer using only the supplied evidence:\n"
                f"{request.model_dump_json(indent=2)}"
            ),
        },
    ]


JOB_DESCRIPTION_ANSWER_SYSTEM_PROMPT = """You answer a candidate's question about an interview role.
Use only the supplied FloCareer Job Description. Do not invent project names,
team culture, benefits, customer details, technology, or work practices.
If the description does not support an answer, say that the detail is not
available and suggest asking the recruiter or interviewer. When `grounded` is
true, every `evidence` item must contain only words copied from the Job
Description; joining source line breaks with spaces is allowed. Keep the answer
concise and professional. Return exactly one JSON object matching the schema.
"""


def job_description_answer_messages(
    *, job_description: str, candidate_question: str
) -> list[dict[str, str]]:
    return [
        {"role": "system", "content": JOB_DESCRIPTION_ANSWER_SYSTEM_PROMPT},
        {
            "role": "user",
            "content": (
                "FloCareer Job Description:\n"
                f"{job_description}\n\n"
                "Candidate question:\n"
                f"{candidate_question}"
            ),
        },
    ]
