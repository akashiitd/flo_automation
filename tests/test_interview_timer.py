from __future__ import annotations

from orchestrator.timer import InterviewTimer


def test_timer_emits_each_warning_once_and_then_expires() -> None:
    timer = InterviewTimer(minutes=25)

    assert timer.events_at_elapsed(0) == ()
    assert timer.events_at_elapsed(10 * 60) == ("FIFTEEN_MINUTES_REMAINING",)
    assert timer.events_at_elapsed(15 * 60) == ("TEN_MINUTES_REMAINING",)
    assert timer.events_at_elapsed(20 * 60) == ("FIVE_MINUTES_REMAINING",)
    assert timer.events_at_elapsed(23 * 60) == ("TWO_MINUTES_REMAINING",)
    assert timer.events_at_elapsed(24 * 60) == ("ONE_MINUTE_REMAINING",)
    assert timer.events_at_elapsed(25 * 60) == ("TIME_LIMIT_REACHED",)
    assert timer.events_at_elapsed(26 * 60) == ()
