"""Pure interview-timer transitions, kept separate from wall-clock I/O."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(slots=True)
class InterviewTimer:
    minutes: int
    _emitted: set[str] = field(default_factory=set, init=False)

    def __post_init__(self) -> None:
        if self.minutes <= 0:
            raise ValueError("minutes must be positive")

    def events_at_elapsed(self, elapsed_seconds: float) -> tuple[str, ...]:
        """Return warnings crossed at this elapsed time, exactly once each."""

        if elapsed_seconds < 0:
            raise ValueError("elapsed_seconds cannot be negative")
        total_seconds = self.minutes * 60
        thresholds: list[tuple[int, str]] = []
        if total_seconds > 15 * 60:
            thresholds.append((total_seconds - 15 * 60, "FIFTEEN_MINUTES_REMAINING"))
        if total_seconds > 10 * 60:
            thresholds.append((total_seconds - 10 * 60, "TEN_MINUTES_REMAINING"))
        if total_seconds > 5 * 60:
            thresholds.append((total_seconds - 5 * 60, "FIVE_MINUTES_REMAINING"))
        if total_seconds > 2 * 60:
            thresholds.append((total_seconds - 2 * 60, "TWO_MINUTES_REMAINING"))
        if total_seconds > 60:
            thresholds.append((total_seconds - 60, "ONE_MINUTE_REMAINING"))
        thresholds.append((total_seconds, "TIME_LIMIT_REACHED"))
        events: list[str] = []
        for threshold, event in thresholds:
            if elapsed_seconds >= threshold and event not in self._emitted:
                self._emitted.add(event)
                events.append(event)
        return tuple(events)
