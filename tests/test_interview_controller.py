from __future__ import annotations

from app.questions import InterviewQuestion
from orchestrator.graph import InterviewController, InterviewPhase


def test_controller_keeps_candidate_visible_prompts_pending_human_approval() -> None:
    controller = InterviewController(
        candidate_name="Candidate Alpha",
        questions=(
            InterviewQuestion(
                id=1, question_text="Explain retries.", ideal_answer="Backoff."
            ),
        ),
    )

    introduction = controller.start()

    assert controller.state.phase is InterviewPhase.HUMAN_APPROVAL
    assert introduction == (
        "Hello. I am an AI-assisted interview system operating under Akash's "
        "supervision. Akash remains responsible for this interview. Please introduce "
        "yourself briefly."
    )
    assert controller.approve_candidate_prompt() == introduction
    assert controller.state.phase is InterviewPhase.HUMAN_APPROVAL
    assert controller.approve_candidate_prompt() == "Explain retries."
    assert controller.state.phase is InterviewPhase.LISTENING

    controller.record_candidate_segment("I would add exponential backoff.")
    answer = controller.complete_answer()

    assert answer == "I would add exponential backoff."
    assert controller.state.phase is InterviewPhase.EVALUATING


def test_controller_does_not_advance_after_barge_in_until_operator_chooses_next_prompt() -> (
    None
):
    controller = InterviewController(
        candidate_name="Candidate Alpha",
        questions=(
            InterviewQuestion(id=1, question_text="First?", ideal_answer="One."),
            InterviewQuestion(id=2, question_text="Second?", ideal_answer="Two."),
        ),
    )
    controller.start()
    controller.approve_candidate_prompt()
    controller.approve_candidate_prompt()
    controller.record_candidate_segment("Please repeat that.")
    controller.complete_answer()
    controller.record_evaluation(follow_up="What trade-off matters most?")

    prompt = controller.prepare_follow_up()

    assert prompt == "What trade-off matters most?"
    assert controller.state.phase is InterviewPhase.HUMAN_APPROVAL
    assert controller.approve_candidate_prompt() == "What trade-off matters most?"
    assert controller.state.current_question_id == 1
