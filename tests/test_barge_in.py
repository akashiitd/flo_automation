from __future__ import annotations

from types import SimpleNamespace

from tts.audio_output import PCMPlaybackSession, PlaybackBargeInController


class FakeOutput:
    def write(self, pcm: bytes) -> None:
        pass

    def close(self) -> None:
        pass


def test_candidate_only_system_segment_cancels_current_playback() -> None:
    playback = PCMPlaybackSession(FakeOutput())
    barge_in = PlaybackBargeInController()
    barge_in.register(playback)

    cancelled = barge_in.on_transcript_segment(
        SimpleNamespace(text="Please wait", source="system")
    )

    assert cancelled is True
    assert playback.cancelled is True


def test_empty_or_non_candidate_segment_never_cancels_playback() -> None:
    playback = PCMPlaybackSession(FakeOutput())
    barge_in = PlaybackBargeInController()
    barge_in.register(playback)

    assert (
        barge_in.on_transcript_segment(SimpleNamespace(text="", source="system"))
        is False
    )
    assert (
        barge_in.on_transcript_segment(
            SimpleNamespace(text="Hello", source="microphone")
        )
        is False
    )
    assert playback.cancelled is False
