"""Command-line entry point for the FloCareer interview copilot."""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import tempfile
import time
import wave
from collections.abc import AsyncIterable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from datetime import datetime
from pathlib import Path
from typing import cast

from app.config import Settings
from app.health import run_health_checks
from browser.action_guard import BrowserAction, approval_token_for
from browser.join_workflow import JoinWorkflowError
from browser.playwright_controller import (
    BrowserScanError,
    join_candidate_dry_run,
    join_candidate_live,
    scan_candidate_questions,
    scan_dashboard,
)
from evaluator.scoring import evaluate_answer
from evaluator.session_evaluation import (
    SessionEvaluationError,
    evaluate_session,
    load_session_inputs,
    load_session_questions,
)
from llm.lmstudio_provider import LMStudioProvider
from llm.openrouter_provider import OpenRouterProvider
from llm.provider_router import HumanReviewRequired, ProviderRouter
from llm.schemas import (
    EvaluationInput,
    ModelClass,
    ProviderMetadata,
    QuestionEvaluation,
    StructuredGeneration,
)
from llm.usage_tracker import UsageTracker
from transcriber.apple_speech_adapter import AppleSpeechAdapter
from orchestrator.graph import InterviewController
from orchestrator.live_loop import CandidateTurnRouter
from orchestrator.state import InterviewPhase
from orchestrator.timer import InterviewTimer
from tts.qwen_client import QwenTTSClient, QwenTTSError
from tts.schemas import SpeechPCMChunk
from tts.audio_output import (
    PCMPlaybackError,
    PCMPlaybackSession,
    PlaybackBargeInController,
    SoundDeviceOutputBackend,
    play_pcm_stream,
)
from tts.speech_bridge import iter_provider_speech, play_provider_pcm


PROJECT_ROOT = Path(__file__).resolve().parent


def _browser_health_ready(health: object) -> bool:
    """Keep browser-only workflows independent from local model availability."""

    return bool(
        getattr(
            health,
            "browser_ready",
            getattr(health, "overall", "") == "READY_FOR_BROWSER_SCAN",
        )
    )


def _config_dump(settings: Settings) -> int:
    print("Configuration")
    print(settings.safe_dump())
    print()

    failed = False
    missing = settings.missing_required()
    if missing:
        failed = True
        print(f"[FAIL] Missing required config: {', '.join(missing)}")
    else:
        print("[OK] No missing required config")

    if settings.meeting_transcriber_path.is_dir():
        print(f"[OK] Transcriber path exists: {settings.meeting_transcriber_path}")
    else:
        failed = True
        print(f"[FAIL] Transcriber path missing: {settings.meeting_transcriber_path}")

    if settings.lmstudio_base_url:
        print(f"[OK] LM Studio URL configured: {settings.lmstudio_base_url}")

    if settings.openrouter_api_key:
        print("[OK] OpenRouter key configured")
    elif settings.llm_allow_cloud_candidate_data:
        failed = True
        print("[FAIL] OpenRouter cloud use is enabled but its key is unavailable")
    else:
        print(
            "[WARN] OpenRouter optional/unavailable; cloud candidate data is disabled"
        )

    try:
        settings.runs_dir.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile(dir=settings.runs_dir, delete=True):
            pass
    except OSError as error:
        failed = True
        print(f"[FAIL] Runs directory is not writable: {error}")
    else:
        print(f"[OK] Runs directory writable: {settings.runs_dir}")

    print()
    print(f"Overall: {'INVALID' if failed else 'VALID'}")
    return 1 if failed else 0


def _sample_evaluation_input() -> EvaluationInput:
    return EvaluationInput(
        question_id=1,
        question="How would you build a LangChain microservice?",
        ideal_answer=(
            "Mentions an API layer, model and tool orchestration, tracing, "
            "timeouts, retries, validation, and deployment concerns."
        ),
        candidate_answer=(
            "I would create an API, call the model through LangChain, validate "
            "the request, and return the response."
        ),
    )


async def _llm_test(
    settings: Settings,
    *,
    provider_name: str,
    model_class: ModelClass,
) -> int:
    provider = (
        LMStudioProvider(settings)
        if provider_name == "lmstudio"
        else OpenRouterProvider(settings)
    )
    usage_tracker = UsageTracker(settings.runs_dir / "llm_tests" / "llm_usage.jsonl")
    try:
        generation = await evaluate_answer(
            _sample_evaluation_input(),
            provider,
            model_class=model_class,
            usage_tracker=usage_tracker,
        )
    except Exception as error:
        detail = str(error) or type(error).__name__
        print(f"LLM test failed for {provider_name}: {detail}", file=sys.stderr)
        return 1
    finally:
        await provider.aclose()

    print(json.dumps(generation.model_dump(mode="json"), indent=2))
    return 0


class _TimeoutProvider:
    name = "lmstudio"

    async def generate_structured(
        self, *args: object, **kwargs: object
    ) -> dict[str, object]:
        raise TimeoutError("simulated local timeout")


class _FallbackProbeProvider:
    name = "openrouter"

    def __init__(self) -> None:
        self.calls = 0

    async def generate_structured(
        self, *args: object, **kwargs: object
    ) -> dict[str, object]:
        self.calls += 1
        return StructuredGeneration(
            output={
                "question_id": 1,
                "score": 3,
                "rating_label": "Average",
                "evidence": ["Candidate mentioned an API layer"],
                "follow_up": "How would you handle retries?",
                "feedback": "Basic understanding with production gaps.",
                "confidence": 0.8,
            },
            metadata=ProviderMetadata(
                provider="openrouter",
                model="simulated-openrouter-model",
                request_purpose="feedback_draft",
                latency_ms=1,
                input_tokens=1,
                output_tokens=1,
                estimated_cost_usd=0,
                pii_redaction_ran=True,
            ),
        ).model_dump(mode="json")


async def _llm_failover_test(settings: Settings) -> int:
    request = _sample_evaluation_input()

    blocked_fallback = _FallbackProbeProvider()
    blocked_router = ProviderRouter(
        replace(
            settings,
            llm_allow_cloud_candidate_data=False,
            openrouter_api_key="",
        ),
        primary=_TimeoutProvider(),
        fallback=blocked_fallback,
    )
    try:
        await evaluate_answer(request, blocked_router)
    except HumanReviewRequired:
        if blocked_fallback.calls != 0:
            print("[FAIL] Cloud provider was called while cloud use was disabled")
            return 1
        print("[OK] Cloud-disabled timeout stopped for human review")
    else:
        print("[FAIL] Cloud-disabled timeout did not stop for human review")
        return 1

    allowed_fallback = _FallbackProbeProvider()
    allowed_router = ProviderRouter(
        replace(
            settings,
            llm_allow_cloud_candidate_data=True,
            openrouter_api_key="simulation-only",
        ),
        primary=_TimeoutProvider(),
        fallback=allowed_fallback,
    )
    generation = await evaluate_answer(request, allowed_router)
    QuestionEvaluation.model_validate(generation.output)
    if allowed_fallback.calls != 1 or not generation.metadata.fallback_used:
        print("[FAIL] Allowed timeout did not route exactly once to the fallback")
        return 1
    print("[OK] Cloud-enabled timeout routed to the fallback")
    print("[OK] Fallback preserved the evaluator schema and metadata")
    print("Overall: PASS")
    return 0


async def _qwen_tts_test(settings: Settings, *, text: str) -> int:
    session_dir = (
        settings.runs_dir / f"qwen_tts_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    client = QwenTTSClient(
        settings.qwen_tts_base_url,
        timeout_seconds=settings.qwen_tts_timeout_seconds,
    )
    try:
        speech = await client.synthesize(text)
    except (QwenTTSError, ValueError) as error:
        print(f"Qwen TTS test failed: {error}", file=sys.stderr)
        return 1
    finally:
        await client.aclose()

    audio_path = session_dir / "speech.wav"
    audio_path.write_bytes(speech.audio)
    print(f"Audio WAV: {audio_path}")
    print(f"Audio duration: {speech.duration_seconds:.2f}s")
    return 0


def _write_pcm_wav(path: Path, *, pcm: bytes, sample_rate: int) -> None:
    with wave.open(str(path), "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(sample_rate)
        output.writeframes(pcm)


@dataclass(frozen=True, slots=True)
class PCMStreamResult:
    pcm: bytes
    sample_rate: int
    first_audio_ms: int


async def _collect_pcm_stream(
    audio_stream: AsyncIterable[SpeechPCMChunk],
) -> PCMStreamResult:
    started = time.perf_counter()
    first_audio_ms: int | None = None
    sample_rate: int | None = None
    chunks: list[bytes] = []
    async for audio in audio_stream:
        if sample_rate is None:
            sample_rate = audio.sample_rate
            first_audio_ms = round((time.perf_counter() - started) * 1000)
        elif audio.sample_rate != sample_rate:
            raise QwenTTSError("Qwen speech stream changed sample rates")
        chunks.append(audio.audio)
    if sample_rate is None or first_audio_ms is None or not chunks:
        raise QwenTTSError("Qwen streaming TTS test returned no audio")
    return PCMStreamResult(
        pcm=b"".join(chunks),
        sample_rate=sample_rate,
        first_audio_ms=first_audio_ms,
    )


async def _qwen_tts_stream_test(settings: Settings, *, text: str) -> int:
    session_dir = (
        settings.runs_dir
        / f"qwen_tts_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    client = QwenTTSClient(
        settings.qwen_tts_base_url,
        timeout_seconds=settings.qwen_tts_timeout_seconds,
    )
    try:
        stream = await _collect_pcm_stream(client.stream_synthesize(text))
    except (QwenTTSError, ValueError) as error:
        print(f"Qwen streaming TTS test failed: {error}", file=sys.stderr)
        return 1
    finally:
        await client.aclose()
    audio_path = session_dir / "speech.wav"
    _write_pcm_wav(audio_path, pcm=stream.pcm, sample_rate=stream.sample_rate)
    print(f"Audio WAV: {audio_path}")
    print(f"Time to first audio: {stream.first_audio_ms}ms")
    return 0


def _audio_devices(settings: Settings) -> int:
    """Show exact local audio devices without changing macOS device selection."""

    try:
        backend = SoundDeviceOutputBackend()
        outputs = backend.list_output_devices()
        inputs = backend.list_input_devices()
    except PCMPlaybackError as error:
        print(f"Audio device diagnostics failed: {error}", file=sys.stderr)
        return 1

    print("Output-capable audio devices:")
    for device in outputs:
        print(f"- {device}")
    print("Input-capable audio devices:")
    for device in inputs:
        print(f"- {device}")

    interviewer_ready = settings.interviewer_audio_output_device in outputs
    candidate_ready = settings.candidate_audio_input_device in inputs
    print(
        "Interviewer output bus: "
        f"{'READY' if interviewer_ready else 'MISSING'} "
        f"({settings.interviewer_audio_output_device})"
    )
    print(
        "Candidate input bus: "
        f"{'READY' if candidate_ready else 'MISSING'} "
        f"({settings.candidate_audio_input_device})"
    )
    return 0 if interviewer_ready and candidate_ready else 1


async def _qwen_tts_playback_test(settings: Settings, *, text: str) -> int:
    """Play streamed Qwen speech through the explicitly configured Loopback bus."""

    session_dir = (
        settings.runs_dir
        / f"qwen_tts_playback_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    client = QwenTTSClient(
        settings.qwen_tts_base_url,
        timeout_seconds=settings.qwen_tts_timeout_seconds,
    )
    raw_pcm: list[bytes] = []
    sample_rate: int | None = None

    async def capture_and_play() -> AsyncIterable[SpeechPCMChunk]:
        nonlocal sample_rate
        async for chunk in client.stream_synthesize(text):
            if sample_rate is None:
                sample_rate = chunk.sample_rate
            elif chunk.sample_rate != sample_rate:
                raise QwenTTSError("Qwen speech stream changed sample rates")
            raw_pcm.append(chunk.audio)
            yield chunk

    try:
        backend = SoundDeviceOutputBackend()
        playback = PCMPlaybackSession(
            backend.open_output(settings.interviewer_audio_output_device)
        )
        result = await play_pcm_stream(capture_and_play(), playback)
    except (PCMPlaybackError, QwenTTSError, ValueError) as error:
        print(f"Qwen playback test failed: {error}", file=sys.stderr)
        return 1
    finally:
        await client.aclose()

    if sample_rate is None or not raw_pcm:
        print("Qwen playback test failed: no PCM was received", file=sys.stderr)
        return 1
    audio_path = session_dir / "speech.wav"
    _write_pcm_wav(audio_path, pcm=b"".join(raw_pcm), sample_rate=sample_rate)
    print(f"Output device: {settings.interviewer_audio_output_device}")
    print(f"Played Qwen PCM chunks: {result.chunk_count}")
    print(f"Audio WAV: {audio_path}")
    return 0


async def _qwen_tts_barge_in_test(settings: Settings, *, text: str) -> int:
    """Validate selected-device capture can cancel one local Qwen playback run."""

    session_id = f"qwen_barge_in_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    client = QwenTTSClient(
        settings.qwen_tts_base_url,
        timeout_seconds=settings.qwen_tts_timeout_seconds,
    )
    barge_in = PlaybackBargeInController()
    adapter: AppleSpeechAdapter | None = None
    raw_pcm: list[bytes] = []
    sample_rate: int | None = None

    def capture_chunk(chunk: SpeechPCMChunk) -> None:
        nonlocal sample_rate
        if sample_rate is None:
            sample_rate = chunk.sample_rate
        elif chunk.sample_rate != sample_rate:
            raise QwenTTSError("Qwen speech stream changed sample rates")
        raw_pcm.append(chunk.audio)

    def observe_candidate_segment(segment: object) -> None:
        cancelled = barge_in.on_transcript_segment(segment)
        source = str(getattr(segment, "source", "unknown"))
        text_value = str(getattr(segment, "text", "")).strip()
        print(f"[{source}] {text_value}", flush=True)
        if cancelled:
            print("Candidate speech cancelled active Qwen playback", flush=True)

    async def capture_and_play() -> AsyncIterable[SpeechPCMChunk]:
        async for chunk in client.stream_synthesize(text):
            capture_chunk(chunk)
            yield chunk

    try:
        adapter = AppleSpeechAdapter(
            settings,
            session_id=session_id,
            on_segment=observe_candidate_segment,
        )
        if not adapter.start():
            raise PCMPlaybackError("Apple Speech did not start on CANDIDATE_ONLY")
        backend = SoundDeviceOutputBackend()
        playback = PCMPlaybackSession(
            backend.open_output(settings.interviewer_audio_output_device)
        )
        result = await play_pcm_stream(capture_and_play(), playback, barge_in=barge_in)
    except (PCMPlaybackError, QwenTTSError, ValueError, RuntimeError) as error:
        print(f"Qwen barge-in test failed: {error}", file=sys.stderr)
        return 1
    finally:
        await client.aclose()
        summary = adapter.stop() if adapter is not None else None

    if sample_rate is None or not raw_pcm:
        print("Qwen barge-in test failed: no PCM was received", file=sys.stderr)
        return 1
    audio_path = settings.runs_dir / session_id / "speech.wav"
    _write_pcm_wav(audio_path, pcm=b"".join(raw_pcm), sample_rate=sample_rate)
    print(f"Output device: {settings.interviewer_audio_output_device}")
    print(f"Candidate input device: {settings.candidate_audio_input_device}")
    print(f"Played Qwen PCM chunks: {result.chunk_count}")
    print(f"Playback cancelled by candidate audio: {result.cancelled}")
    print(f"Audio WAV: {audio_path}")
    assert summary is not None
    print(f"Transcript JSON: {summary.json_path}")
    if not result.cancelled:
        print(
            "Validation incomplete: no non-empty candidate-only segment cancelled "
            "the Qwen playback",
            file=sys.stderr,
        )
        return 1
    print(
        "Safety: this test does not open FloCareer or change browser, feedback, "
        "hang-up, or FINISH controls"
    )
    return 0


def _require_supervisor_token(expected: str) -> None:
    print(f"Type exactly: {expected}")
    try:
        entered = input("Approval: ").strip()
    except (EOFError, KeyboardInterrupt) as error:
        raise RuntimeError("supervised voice loop cancelled by operator") from error
    if entered != expected:
        raise RuntimeError("supervised voice loop approval did not match")


def _choose_supervisor_token(*allowed: str) -> str:
    print("Type exactly one of:")
    for token in allowed:
        print(f"- {token}")
    try:
        entered = input("Approval: ").strip()
    except (EOFError, KeyboardInterrupt) as error:
        raise RuntimeError("supervised voice loop cancelled by operator") from error
    if entered not in allowed:
        raise RuntimeError("supervised voice loop approval did not match")
    return entered


async def _speak_supervised_prompt(
    settings: Settings,
    speech_client: QwenTTSClient,
    barge_in: PlaybackBargeInController,
    prompt: str,
) -> None:
    backend = SoundDeviceOutputBackend()
    playback = PCMPlaybackSession(
        backend.open_output(settings.interviewer_audio_output_device)
    )
    result = await play_pcm_stream(
        speech_client.stream_synthesize(prompt), playback, barge_in=barge_in
    )
    print(f"Played Qwen PCM chunks: {result.chunk_count}")
    if result.cancelled:
        print("Candidate speech cancelled the prompt playback")


async def _supervise_voice_loop(
    settings: Settings,
    *,
    session_path: Path,
    candidate_name: str,
    model_class: ModelClass,
) -> int:
    """Run a human-approved local voice loop without browser-side effects."""

    try:
        questions = load_session_questions(session_path)
    except SessionEvaluationError as error:
        print(f"Supervised voice loop failed: {error}", file=sys.stderr)
        return 1

    controller = InterviewController(candidate_name=candidate_name, questions=questions)
    barge_in = PlaybackBargeInController()
    router = CandidateTurnRouter(controller, barge_in)
    speech_client = QwenTTSClient(
        settings.qwen_tts_base_url,
        timeout_seconds=settings.qwen_tts_timeout_seconds,
    )
    provider = LMStudioProvider(settings)
    adapter: AppleSpeechAdapter | None = None
    usage_tracker = UsageTracker(session_path / "llm_usage.jsonl")

    def route_candidate_segment(segment: object) -> None:
        router.on_transcript_segment(segment)

    try:
        print("Browser controls, feedback, hang-up, and FINISH are not available here.")
        print("The operator must approve every candidate-visible Qwen prompt.")
        introduction = controller.start()
        _require_supervisor_token("SPEAK INTRODUCTION")
        controller.approve_candidate_prompt()
        await _speak_supervised_prompt(settings, speech_client, barge_in, introduction)

        adapter = AppleSpeechAdapter(
            settings,
            session_id=session_path.name,
            on_segment=route_candidate_segment,
            question_id_provider=lambda: router.active_question_id,
            require_question_boundary=True,
        )
        if not adapter.start():
            raise RuntimeError("Apple Speech did not start on CANDIDATE_ONLY")

        while controller.state.phase is not InterviewPhase.DONE:
            if controller.state.phase is not InterviewPhase.HUMAN_APPROVAL:
                raise RuntimeError(
                    f"unexpected controller phase {controller.state.phase}"
                )
            question_id = controller.state.questions[
                controller.state.current_question_index
            ].id
            _require_supervisor_token(f"SPEAK QUESTION {question_id}")
            prompt = controller.approve_candidate_prompt()
            await _speak_supervised_prompt(settings, speech_client, barge_in, prompt)
            _require_supervisor_token(f"END ANSWER {question_id}")
            answer = controller.complete_answer()
            question = controller.state.questions[
                controller.state.current_question_index
            ]
            generation = await evaluate_answer(
                EvaluationInput(
                    question_id=question.id,
                    question=question.question_text,
                    ideal_answer=question.ideal_answer,
                    candidate_answer=answer,
                ),
                provider,
                model_class=model_class,
                usage_tracker=usage_tracker,
            )
            score = QuestionEvaluation.model_validate(generation.output)
            print(f"Question {question_id} draft score: {score.score}/5")
            print(f"Suggested follow-up: {score.follow_up}")
            controller.record_evaluation(follow_up=score.follow_up)

            choice = _choose_supervisor_token(
                f"SKIP FOLLOW-UP {question_id}",
                f"SPEAK FOLLOW-UP {question_id}",
            )
            if choice == f"SPEAK FOLLOW-UP {question_id}":
                follow_up = controller.prepare_follow_up()
                _require_supervisor_token(f"SPEAK FOLLOW-UP {question_id}")
                controller.approve_candidate_prompt()
                await _speak_supervised_prompt(
                    settings, speech_client, barge_in, follow_up
                )
                _require_supervisor_token(f"END FOLLOW-UP {question_id}")
                follow_up_answer = controller.complete_answer()
                await evaluate_answer(
                    EvaluationInput(
                        question_id=question.id,
                        question=question.question_text,
                        ideal_answer=question.ideal_answer,
                        candidate_answer=follow_up_answer,
                    ),
                    provider,
                    model_class=model_class,
                    usage_tracker=usage_tracker,
                )
                controller.record_evaluation(follow_up=None)
            elif choice == f"SKIP FOLLOW-UP {question_id}":
                controller.skip_optional_follow_up()
            else:
                raise RuntimeError("follow-up choice did not match an approved token")

            controller.prepare_next_question()

        summary = adapter.stop()
        adapter = None
        evaluation = await evaluate_session(
            load_session_inputs(session_path), provider, model_class=model_class
        )
        trace_path = session_path / "controller_transitions.json"
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "supervised_local_voice_loop": True,
                    "final_phase": str(controller.state.phase),
                    "transitions": [
                        asdict(transition) for transition in controller.transitions
                    ],
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
    except (
        PCMPlaybackError,
        QwenTTSError,
        RuntimeError,
        SessionEvaluationError,
        TimeoutError,
        ValueError,
    ) as error:
        print(f"Supervised voice loop failed: {error}", file=sys.stderr)
        return 1
    finally:
        if adapter is not None:
            adapter.stop()
        await provider.aclose()
        await speech_client.aclose()

    print(f"Transcript JSON: {summary.json_path}")
    print(f"Controller trace: {trace_path}")
    print(f"Evaluation JSON: {evaluation.evaluation_path}")
    print(f"Feedback preview (not submitted): {evaluation.feedback_preview_path}")
    return 0


async def _llm_speak_test(
    settings: Settings, *, prompt: str, model_class: ModelClass
) -> int:
    session_dir = (
        settings.runs_dir / f"llm_speak_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    provider = LMStudioProvider(settings)
    speech_client = QwenTTSClient(
        settings.qwen_tts_base_url,
        timeout_seconds=settings.qwen_tts_timeout_seconds,
    )
    try:
        audio_count = 0
        async for speech in iter_provider_speech(
            provider.stream_text(
                (
                    {
                        "role": "system",
                        "content": (
                            "Reply with concise spoken interview text only. "
                            "Use at most two complete sentences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ),
                model_class,
            ),
            speech_client,
        ):
            audio_count += 1
            (session_dir / f"speech_{audio_count:02d}.wav").write_bytes(speech.audio)
    except (QwenTTSError, ValueError, TimeoutError) as error:
        print(f"LM Studio to Qwen speech test failed: {error}", file=sys.stderr)
        return 1
    finally:
        await provider.aclose()
        await speech_client.aclose()

    print(f"Generated speech chunks: {audio_count}")
    print(f"Output directory: {session_dir}")
    return 0


async def _llm_speak_stream_test(
    settings: Settings, *, prompt: str, model_class: ModelClass
) -> int:
    session_dir = (
        settings.runs_dir
        / f"llm_speak_stream_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    session_dir.mkdir(parents=True, exist_ok=True)
    provider = LMStudioProvider(settings)
    speech_client = QwenTTSClient(
        settings.qwen_tts_base_url,
        timeout_seconds=settings.qwen_tts_timeout_seconds,
    )
    raw_pcm: list[bytes] = []
    sample_rate: int | None = None

    def capture_chunk(chunk: SpeechPCMChunk) -> None:
        nonlocal sample_rate
        if sample_rate is None:
            sample_rate = chunk.sample_rate
        elif chunk.sample_rate != sample_rate:
            raise QwenTTSError("Qwen speech stream changed sample rates")
        raw_pcm.append(chunk.audio)

    try:
        backend = SoundDeviceOutputBackend()
        playback = PCMPlaybackSession(
            backend.open_output(settings.interviewer_audio_output_device)
        )
        result = await play_provider_pcm(
            provider.stream_text(
                (
                    {
                        "role": "system",
                        "content": (
                            "Reply with concise spoken interview text only. "
                            "Use at most two complete sentences."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ),
                model_class,
            ),
            speech_client,
            playback,
            on_chunk=capture_chunk,
        )
    except (PCMPlaybackError, QwenTTSError, ValueError, TimeoutError) as error:
        print(f"LM Studio to Qwen streaming test failed: {error}", file=sys.stderr)
        return 1
    finally:
        await provider.aclose()
        await speech_client.aclose()
    if sample_rate is None or not raw_pcm:
        print(
            "LM Studio to Qwen streaming test failed: no PCM was received",
            file=sys.stderr,
        )
        return 1

    audio_path = session_dir / "speech.wav"
    _write_pcm_wav(audio_path, pcm=b"".join(raw_pcm), sample_rate=sample_rate)
    print(f"Output device: {settings.interviewer_audio_output_device}")
    print(f"Played Qwen PCM chunks: {result.chunk_count}")
    print(f"Audio WAV: {audio_path}")
    return 0


def _positive_seconds(value: str) -> float:
    seconds = float(value)
    if seconds <= 0:
        raise argparse.ArgumentTypeError("seconds must be greater than zero")
    return seconds


def _listen_test(settings: Settings, *, seconds: float) -> int:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    session_id = f"listen_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    adapter: AppleSpeechAdapter | None = None

    def show_segment(segment: object) -> None:
        source = getattr(segment, "source", "unknown")
        speaker = getattr(segment, "speaker", "Other") or "Other"
        text = getattr(segment, "text", "")
        print(f"[{source}] [{speaker}] {text}", flush=True)

    try:
        adapter = AppleSpeechAdapter(
            settings,
            session_id=session_id,
            on_segment=show_segment,
        )
        print("Starting Apple Speech listener")
        print("Mode: system audio only (microphone disabled)")
        print(f"Duration: {seconds:g} seconds")
        if not adapter.start():
            adapter.stop()
            print(
                "Apple Speech failed to start. Check Speech Recognition and "
                "Screen & System Audio Recording permissions.",
                file=sys.stderr,
            )
            return 1
    except Exception as error:
        if adapter is not None:
            adapter.stop()
        print(f"Apple Speech failed to start: {error}", file=sys.stderr)
        return 1

    interrupted = False
    try:
        deadline = time.monotonic() + seconds
        while time.monotonic() < deadline:
            time.sleep(min(0.1, max(0.0, deadline - time.monotonic())))
    except KeyboardInterrupt:
        interrupted = True
        print("Stopping early at user request")
    finally:
        assert adapter is not None
        summary = adapter.stop()

    print(f"Captured segments: {summary.segment_count}")
    print(f"Saved transcript JSON: {summary.json_path}")
    print(f"Saved transcript text: {summary.text_path}")

    payload = json.loads(summary.json_path.read_text(encoding="utf-8"))
    segments = payload.get("segments", [])
    system_segments = [
        segment
        for segment in segments
        if isinstance(segment, dict) and segment.get("source") == "system"
    ]
    microphone_segments = [
        segment
        for segment in segments
        if isinstance(segment, dict) and segment.get("source") == "microphone"
    ]
    if microphone_segments:
        print("Validation failed: microphone segments were captured", file=sys.stderr)
        return 1
    if not system_segments:
        suffix = " after interruption" if interrupted else ""
        print(
            f"Validation failed: no system-audio transcript was captured{suffix}",
            file=sys.stderr,
        )
        return 1
    print("Validation passed: system audio captured and microphone remained off")
    return 0


def _browser_scan(settings: Settings, *, login_timeout_seconds: float) -> int:
    print("FloCareer browser scan")
    print("Safety mode: read-only; interview launch actions are disabled")
    try:
        result = scan_dashboard(
            settings,
            login_timeout_seconds=login_timeout_seconds,
            progress=lambda message: print(message, flush=True),
        )
    except BrowserScanError as error:
        print(f"Browser scan failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        detail = str(error) or type(error).__name__
        print(f"Browser scan failed: {detail}", file=sys.stderr)
        return 1

    print("FloCareer dashboard loaded")
    print(f"Found scheduled interviews: {len(result.interviews)}")
    for index, interview in enumerate(result.interviews, start=1):
        print(f"{index}. {interview.summary}")
    if not result.interviews:
        print("No scheduled interview rows were visible on the dashboard")
        print(f"Screenshot saved: {result.screenshot_path}")
        print(
            "Validation incomplete: browser access is working, but candidate-row "
            "extraction requires at least one visible scheduled interview."
        )
        return 2
    print(f"Screenshot saved: {result.screenshot_path}")
    print("Validation passed: dashboard scanned without launching an interview")
    return 0


def _join_dry_run(
    settings: Settings,
    *,
    candidate_name: str,
    login_timeout_seconds: float,
) -> int:
    print("FloCareer guarded join discovery")
    print("Safety mode: dry run; launch and Join actions are blocked")
    health = run_health_checks(settings)
    if not _browser_health_ready(health):
        print(health.render(), file=sys.stderr)
        print(
            "Join dry run failed: health prerequisites are not ready", file=sys.stderr
        )
        return 1
    try:
        result = join_candidate_dry_run(
            settings,
            candidate_name=candidate_name,
            login_timeout_seconds=login_timeout_seconds,
            progress=lambda message: print(message, flush=True),
        )
    except (BrowserScanError, JoinWorkflowError) as error:
        print(f"Join dry run failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        detail = str(error) or type(error).__name__
        print(f"Join dry run failed: {detail}", file=sys.stderr)
        return 1

    print(f"Candidate identifier: {result.candidate_identifier}")
    print(f"Candidate screenshot: {result.candidate_found_screenshot}")
    print(f"Dry-run screenshot: {result.join_dry_run_screenshot}")
    print(f"Action log: {result.action_log_path}")
    print("Validation passed: launch control found and blocked by dry run")
    return 0


def _join_live(
    settings: Settings,
    *,
    candidate_name: str,
    login_timeout_seconds: float,
    enable_code_editor_question: int | None,
    candidate_wait_timeout_seconds: float | None,
) -> int:
    print("FloCareer approved live join")
    print("Safety mode: Launch and Join require separate approvals")
    print("Consent OK requires another approval when the form is shown")
    if enable_code_editor_question is not None:
        print("Code editor requires a separate candidate-and-question approval")
    print("Hang-up and FINISH are always blocked")
    health = run_health_checks(settings)
    if not _browser_health_ready(health):
        print(health.render(), file=sys.stderr)
        print("Live join failed: health prerequisites are not ready", file=sys.stderr)
        return 1

    def request_approval(
        action: BrowserAction, candidate_identifier: str
    ) -> str | None:
        expected = approval_token_for(action, candidate_identifier)
        print()
        print(f"Approval required for {action.value}")
        print(f"Type exactly: {expected}")
        try:
            return input("Approval token: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nApproval cancelled", file=sys.stderr)
            return None

    def wait_for_manual_end(candidate_identifier: str) -> None:
        expected = f"CONFIRM-INTERVIEW-ENDED {candidate_identifier}"
        print()
        print("Interview joined. The automation will not click hang-up.")
        print("End the interview manually in FloCareer when appropriate.")
        print("The browser will remain open until you confirm it has ended.")
        while True:
            try:
                entered = input(
                    f"After it ends, type exactly: {expected}\nConfirmation: "
                )
            except (EOFError, KeyboardInterrupt):
                print(
                    "\nConfirmation still required; browser remains under the "
                    "live command's control.",
                    file=sys.stderr,
                )
                time.sleep(1)
                continue
            if entered.strip() == expected:
                return
            print("Confirmation did not match; browser remains open.")

    def request_code_editor_approval(
        action: BrowserAction, candidate_identifier: str, question_id: int
    ) -> str | None:
        expected = approval_token_for(
            action, candidate_identifier, question_id=question_id
        )
        print()
        print(f"Approval required for {action.value}")
        print(f"Type exactly: {expected}")
        try:
            return input("Approval token: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nApproval cancelled", file=sys.stderr)
            return None

    try:
        result = join_candidate_live(
            settings,
            candidate_name=candidate_name,
            login_timeout_seconds=login_timeout_seconds,
            progress=lambda message: print(message, flush=True),
            request_approval=request_approval,
            wait_for_manual_end=wait_for_manual_end,
            enable_code_editor_question=enable_code_editor_question,
            request_code_editor_approval=(
                request_code_editor_approval
                if enable_code_editor_question is not None
                else None
            ),
            candidate_wait_timeout_seconds=candidate_wait_timeout_seconds,
        )
    except (BrowserScanError, JoinWorkflowError) as error:
        print(f"Live join failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        detail = str(error) or type(error).__name__
        print(f"Live join failed: {detail}", file=sys.stderr)
        return 1

    print(f"Candidate identifier: {result.candidate_identifier}")
    if result.consent_screenshot is not None:
        print(f"Consent screenshot: {result.consent_screenshot}")
    else:
        print("Consent form: not shown; FloCareer opened verified pre-call directly")
    print(f"Pre-call screenshot: {result.pre_call_screenshot}")
    print(f"Joined screenshot: {result.joined_screenshot}")
    print(f"Room state log: {result.room_state_log_path}")
    if result.code_editor_result is not None:
        editor = result.code_editor_result
        print(
            "Code editor: "
            f"{'enabled' if editor.changed else 'already visible'} "
            f"for question {editor.question_id}"
        )
        print(f"Code editor before screenshot: {editor.before_screenshot}")
        print(f"Code editor after screenshot: {editor.after_screenshot}")
    print(f"Action log: {result.action_log_path}")
    print("Validation passed: interview joined after all required approvals")
    return 0


def _questions_scan(
    settings: Settings,
    *,
    candidate_name: str,
    login_timeout_seconds: float,
    inspect_code_editor_tabs: bool,
) -> int:
    print("FloCareer approved question scan")
    print("Safety mode: Launch requires approval; Join is never clicked")
    print("Question cards may be expanded; evaluation controls are untouched")
    print("Code editor DOM is inspected only and is never enabled for the candidate")
    if inspect_code_editor_tabs:
        print("Coding Code Editor tabs will open only for capture, then restore")
    health = run_health_checks(settings)
    if not _browser_health_ready(health):
        print(health.render(), file=sys.stderr)
        print(
            "Question scan failed: health prerequisites are not ready", file=sys.stderr
        )
        return 1

    def request_approval(
        action: BrowserAction, candidate_identifier: str
    ) -> str | None:
        expected = approval_token_for(action, candidate_identifier)
        print()
        print(f"Approval required for {action.value}")
        print(f"Type exactly: {expected}")
        try:
            return input("Approval token: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nApproval cancelled", file=sys.stderr)
            return None

    try:
        result = scan_candidate_questions(
            settings,
            candidate_name=candidate_name,
            request_approval=request_approval,
            inspect_code_editor_tabs=inspect_code_editor_tabs,
            login_timeout_seconds=login_timeout_seconds,
            progress=lambda message: print(message, flush=True),
        )
    except (BrowserScanError, JoinWorkflowError) as error:
        print(f"Question scan failed: {error}", file=sys.stderr)
        return 1
    except Exception as error:
        detail = str(error) or type(error).__name__
        print(f"Question scan failed: {detail}", file=sys.stderr)
        return 1

    coding = [
        str(question.id) for question in result.questions if question.has_code_editor
    ]
    print(f"Extracted questions: {len(result.questions)}")
    print(f"Coding question IDs: {', '.join(coding) if coding else 'none detected'}")
    associations = ", ".join(
        (
            f"{observation.question_id}={observation.association_status}"
            if observation.question_id is not None
            else "unresolved=ambiguous"
        )
        for observation in result.code_editor_dom_observations
    )
    observation_ids = [
        observation.question_id
        for observation in result.code_editor_dom_observations
        if observation.question_id is not None
    ]
    capture_complete = len(observation_ids) == len(set(observation_ids)) and set(
        observation_ids
    ) == {int(question_id) for question_id in coding}
    print(f"Code editor DOM associations: {associations or 'none detected'}")
    print(
        f"Code editor DOM capture: {'complete' if capture_complete else 'incomplete'}"
    )
    print(f"Questions JSON: {result.questions_path}")
    print(f"Code editor DOM: {result.code_editor_dom_path}")
    print(f"Expanded screenshot: {result.screenshot_path}")
    print(f"Action log: {result.action_log_path}")
    print("Validation passed: questions read without clicking Join")
    return 0


def _session_path(settings: Settings, raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    resolved = (path if path.is_absolute() else settings.project_root / path).resolve()
    try:
        resolved.relative_to(settings.runs_dir.resolve())
    except ValueError as error:
        raise SessionEvaluationError(
            "--session must be a directory under RUNS_DIR"
        ) from error
    return resolved


async def _evaluate_saved_session(
    settings: Settings, *, session_path: Path, model_class: ModelClass
) -> int:
    try:
        inputs = load_session_inputs(session_path)
    except SessionEvaluationError as error:
        print(f"Session evaluation failed: {error}", file=sys.stderr)
        return 1
    provider = LMStudioProvider(settings)
    try:
        result = await evaluate_session(inputs, provider, model_class=model_class)
    except (SessionEvaluationError, TimeoutError, ValueError) as error:
        print(f"Session evaluation failed: {error}", file=sys.stderr)
        return 1
    finally:
        await provider.aclose()
    print(f"Overall recommendation: {result.overall_recommendation}")
    print(f"Confidence: {result.confidence:.2f}")
    print(f"Evaluation JSON: {result.evaluation_path}")
    print(f"Feedback preview (not submitted): {result.feedback_preview_path}")
    print(
        "Safety: no browser controls, ratings, feedback fields, or FINISH were touched"
    )
    return 0


async def _simulate_interview(
    settings: Settings,
    *,
    session_path: Path,
    model_class: ModelClass,
    assume_human_prompt_approvals: bool,
) -> int:
    """Run saved candidate answers through the controller without live side effects."""

    try:
        inputs = load_session_inputs(session_path)
    except SessionEvaluationError as error:
        print(f"Interview simulation failed: {error}", file=sys.stderr)
        return 1
    provider = LMStudioProvider(settings)
    try:
        evaluation = await evaluate_session(inputs, provider, model_class=model_class)
    except (SessionEvaluationError, TimeoutError, ValueError) as error:
        print(f"Interview simulation failed: {error}", file=sys.stderr)
        return 1
    finally:
        await provider.aclose()

    controller = InterviewController(
        candidate_name="Saved-session simulation",
        questions=tuple(
            question
            for question in inputs.questions
            if question.id in inputs.answers_by_question_id
        ),
    )
    scores = {score.question_id: score for score in evaluation.question_scores}
    controller.start()
    trace_path = inputs.session_dir / "controller_transitions.json"
    if not assume_human_prompt_approvals:
        trace_path.write_text(
            json.dumps(
                {
                    "schema_version": 1,
                    "simulation_only": True,
                    "candidate_visible_actions": "not_emitted",
                    "final_phase": str(controller.state.phase),
                    "transitions": [
                        asdict(transition) for transition in controller.transitions
                    ],
                },
                indent=2,
                default=str,
            )
            + "\n",
            encoding="utf-8",
        )
        print("Simulation stopped at HUMAN_APPROVAL for the introduction prompt")
        print(f"Controller trace: {trace_path}")
        print(
            "Re-run with --assume-human-prompt-approvals only to model approval "
            "inside this offline simulation."
        )
        return 0
    controller.approve_candidate_prompt()
    controller.approve_candidate_prompt()
    for question in controller.state.questions:
        controller.record_candidate_segment(inputs.answers_by_question_id[question.id])
        controller.complete_answer()
        score = scores[question.id]
        controller.record_evaluation(follow_up=score.follow_up)
        # The simulation records a human choice not to emit a candidate-visible
        # follow-up. It never invokes TTS or a browser action.
        controller.skip_optional_follow_up()
        if controller.prepare_next_question() is not None:
            controller.approve_candidate_prompt()

    trace_path = inputs.session_dir / "controller_transitions.json"
    trace_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "simulation_only": True,
                "candidate_visible_actions": "not_emitted",
                "final_phase": str(controller.state.phase),
                "transitions": [
                    asdict(transition) for transition in controller.transitions
                ],
            },
            indent=2,
            default=str,
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"Simulation final phase: {controller.state.phase}")
    print(f"Controller trace: {trace_path}")
    print(f"Feedback preview (not submitted): {evaluation.feedback_preview_path}")
    print(
        "Safety: simulation did not start audio, browser, feedback, or FINISH actions"
    )
    return 0


def _timer_demo(*, minutes: int) -> int:
    timer = InterviewTimer(minutes=minutes)
    print("Timer simulation only; no interview is started.")
    checkpoints = (
        0,
        max(0, minutes * 60 - 15 * 60),
        max(0, minutes * 60 - 10 * 60),
        max(0, minutes * 60 - 5 * 60),
        max(0, minutes * 60 - 2 * 60),
        max(0, minutes * 60 - 60),
        minutes * 60,
    )
    for elapsed in checkpoints:
        for event in timer.events_at_elapsed(elapsed):
            print(f"{elapsed // 60:02d}:{elapsed % 60:02d} {event}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Supervised FloCareer interview automation copilot"
    )
    subcommands = parser.add_subparsers(dest="command", required=True)
    subcommands.add_parser(
        "config-dump", help="validate and safely print configuration"
    )
    subcommands.add_parser("health", help="check local runtime readiness")
    llm_test = subcommands.add_parser(
        "llm-test", help="run the structured evaluator through one provider"
    )
    llm_test.add_argument(
        "--provider",
        choices=("lmstudio", "openrouter"),
        default="lmstudio",
    )
    llm_test.add_argument(
        "--model-class",
        choices=("fast", "deep"),
        default="fast",
    )
    subcommands.add_parser(
        "llm-failover-test",
        help="simulate guarded local-to-cloud failover without sending data",
    )
    qwen_tts_test = subcommands.add_parser(
        "qwen-tts-test",
        help="send supplied text to the local Qwen cloned-voice service",
    )
    qwen_tts_test.add_argument("--text", required=True)
    qwen_tts_stream_test = subcommands.add_parser(
        "qwen-tts-stream-test",
        help="stream PCM from the local Qwen cloned-voice service",
    )
    qwen_tts_stream_test.add_argument("--text", required=True)
    subcommands.add_parser(
        "audio-devices",
        help="list local audio devices and verify the configured Loopback buses",
    )
    qwen_tts_playback_test = subcommands.add_parser(
        "qwen-tts-playback-test",
        help="play streamed Qwen PCM through the configured interviewer Loopback bus",
    )
    qwen_tts_playback_test.add_argument("--text", required=True)
    qwen_tts_barge_in_test = subcommands.add_parser(
        "qwen-tts-barge-in-test",
        help="test candidate-only capture cancelling local Qwen playback on Loopback",
    )
    qwen_tts_barge_in_test.add_argument("--text", required=True)
    qwen_tts_barge_in_test.add_argument(
        "--confirm-selected-loopback-route",
        action="store_true",
        help="confirm this is a disclosed, supervised test of the selected Loopback route",
    )
    supervise = subcommands.add_parser(
        "supervise-voice-loop",
        help="run an operator-approved local Qwen, candidate-capture, and evaluation loop",
    )
    supervise.add_argument(
        "--session", required=True, help="session directory under runs/"
    )
    supervise.add_argument("--candidate", required=True)
    supervise.add_argument("--model-class", choices=("fast", "deep"), default="deep")
    supervise.add_argument(
        "--confirm-disclosed-supervision",
        action="store_true",
        help="confirm candidate disclosure/consent and a supervised selected-Loopback route",
    )
    llm_speak_test = subcommands.add_parser(
        "llm-speak-test",
        help="stream a short local LM Studio reply into the local Qwen voice service",
    )
    llm_speak_test.add_argument("--prompt", required=True)
    llm_speak_test.add_argument(
        "--model-class", choices=("fast", "deep"), default="fast"
    )
    llm_speak_stream_test = subcommands.add_parser(
        "llm-speak-stream-test",
        help="stream local LM Studio sentences through Qwen PCM synthesis",
    )
    llm_speak_stream_test.add_argument("--prompt", required=True)
    llm_speak_stream_test.add_argument(
        "--model-class", choices=("fast", "deep"), default="fast"
    )
    listen_test = subcommands.add_parser(
        "listen-test",
        help="capture Apple Speech system audio and save a session transcript",
    )
    listen_test.add_argument("--seconds", type=_positive_seconds, default=60.0)
    browser_scan = subcommands.add_parser(
        "browser-scan",
        help="open and read the FloCareer dashboard without launching interviews",
    )
    browser_scan.add_argument(
        "--login-timeout",
        type=_positive_seconds,
        default=180.0,
        help="seconds to wait for manual login in the opened browser",
    )
    join = subcommands.add_parser(
        "join",
        help="find one candidate's launch control without launching an interview",
    )
    join.add_argument("--candidate", required=True, help="exact visible candidate name")
    join_mode = join.add_mutually_exclusive_group(required=True)
    join_mode.add_argument(
        "--dry-run",
        action="store_true",
        help="discover the launch control while blocking Launch and Join",
    )
    join_mode.add_argument(
        "--live",
        action="store_true",
        help="request approvals for Launch, optional Consent OK, and Join",
    )
    join.add_argument(
        "--login-timeout",
        type=_positive_seconds,
        default=180.0,
        help="seconds to wait for manual login in the opened browser",
    )
    join.add_argument(
        "--candidate-wait-timeout",
        type=_positive_seconds,
        help="optional cutoff while waiting for the candidate after Join",
    )
    join.add_argument(
        "--enable-code-editor-question",
        type=int,
        help="show this coding question's editor after the candidate connects",
    )
    questions_scan = subcommands.add_parser(
        "questions-scan",
        help="launch with approval and read all questions without clicking Join",
    )
    questions_scan.add_argument(
        "--candidate", required=True, help="exact visible candidate name"
    )
    questions_scan.add_argument(
        "--login-timeout",
        type=_positive_seconds,
        default=180.0,
        help="seconds to wait for manual login in the opened browser",
    )
    questions_scan.add_argument(
        "--inspect-code-editor-tabs",
        action="store_true",
        help="reversibly open exact coding tabs before the read-only DOM capture",
    )
    evaluate = subcommands.add_parser(
        "evaluate",
        help="evaluate question-bound saved candidate transcripts without browser actions",
    )
    evaluate.add_argument(
        "--session", required=True, help="session directory under runs/"
    )
    evaluate.add_argument("--model-class", choices=("fast", "deep"), default="deep")
    simulate = subcommands.add_parser(
        "simulate-interview",
        help="run a saved session through the controller without browser or audio output",
    )
    simulate.add_argument(
        "--session", required=True, help="session directory under runs/"
    )
    simulate.add_argument("--model-class", choices=("fast", "deep"), default="deep")
    simulate.add_argument(
        "--assume-human-prompt-approvals",
        action="store_true",
        help="model prompt approvals in this offline-only simulation",
    )
    timer_demo = subcommands.add_parser(
        "timer-demo", help="print synthetic interview timer warnings without waiting"
    )
    timer_demo.add_argument("--minutes", type=int, default=25)
    return parser


def main(
    argv: Sequence[str] | None = None,
    *,
    environ: Mapping[str, str] | None = None,
    project_root: Path = PROJECT_ROOT,
) -> int:
    args = build_parser().parse_args(argv)
    try:
        settings = Settings.load(
            project_root=project_root,
            environ=os.environ if environ is None else environ,
        )
    except ValueError as error:
        print(f"Configuration error: {error}", file=sys.stderr)
        return 2

    if args.command == "config-dump":
        return _config_dump(settings)
    if args.command == "health":
        report = run_health_checks(settings)
        print(report.render())
        return 0 if report.overall == "READY_FOR_BROWSER_SCAN" else 1
    if args.command == "llm-test":
        return asyncio.run(
            _llm_test(
                settings,
                provider_name=args.provider,
                model_class=cast(ModelClass, args.model_class),
            )
        )
    if args.command == "llm-failover-test":
        return asyncio.run(_llm_failover_test(settings))
    if args.command == "qwen-tts-test":
        return asyncio.run(_qwen_tts_test(settings, text=args.text))
    if args.command == "qwen-tts-stream-test":
        return asyncio.run(_qwen_tts_stream_test(settings, text=args.text))
    if args.command == "audio-devices":
        return _audio_devices(settings)
    if args.command == "qwen-tts-playback-test":
        return asyncio.run(_qwen_tts_playback_test(settings, text=args.text))
    if args.command == "qwen-tts-barge-in-test":
        if not args.confirm_selected_loopback_route:
            print(
                "qwen-tts-barge-in-test requires --confirm-selected-loopback-route",
                file=sys.stderr,
            )
            return 2
        return asyncio.run(_qwen_tts_barge_in_test(settings, text=args.text))
    if args.command == "supervise-voice-loop":
        if not args.confirm_disclosed_supervision:
            print(
                "supervise-voice-loop requires --confirm-disclosed-supervision",
                file=sys.stderr,
            )
            return 2
        try:
            session_path = _session_path(settings, args.session)
        except SessionEvaluationError as error:
            print(f"Supervised voice loop failed: {error}", file=sys.stderr)
            return 2
        return asyncio.run(
            _supervise_voice_loop(
                settings,
                session_path=session_path,
                candidate_name=args.candidate,
                model_class=cast(ModelClass, args.model_class),
            )
        )
    if args.command == "llm-speak-test":
        return asyncio.run(
            _llm_speak_test(
                settings,
                prompt=args.prompt,
                model_class=cast(ModelClass, args.model_class),
            )
        )
    if args.command == "llm-speak-stream-test":
        return asyncio.run(
            _llm_speak_stream_test(
                settings,
                prompt=args.prompt,
                model_class=cast(ModelClass, args.model_class),
            )
        )
    if args.command == "listen-test":
        return _listen_test(settings, seconds=args.seconds)
    if args.command == "browser-scan":
        return _browser_scan(
            settings,
            login_timeout_seconds=args.login_timeout,
        )
    if args.command == "join":
        if args.dry_run:
            if (
                args.enable_code_editor_question is not None
                or args.candidate_wait_timeout is not None
            ):
                print(
                    "candidate waiting and code-editor options require --live",
                    file=sys.stderr,
                )
                return 2
            return _join_dry_run(
                settings,
                candidate_name=args.candidate,
                login_timeout_seconds=args.login_timeout,
            )
        if (
            args.enable_code_editor_question is not None
            and args.enable_code_editor_question < 1
        ):
            print("--enable-code-editor-question must be positive", file=sys.stderr)
            return 2
        return _join_live(
            settings,
            candidate_name=args.candidate,
            login_timeout_seconds=args.login_timeout,
            enable_code_editor_question=args.enable_code_editor_question,
            candidate_wait_timeout_seconds=args.candidate_wait_timeout,
        )
    if args.command == "questions-scan":
        return _questions_scan(
            settings,
            candidate_name=args.candidate,
            login_timeout_seconds=args.login_timeout,
            inspect_code_editor_tabs=args.inspect_code_editor_tabs,
        )
    if args.command == "evaluate":
        try:
            session_path = _session_path(settings, args.session)
        except SessionEvaluationError as error:
            print(f"Session evaluation failed: {error}", file=sys.stderr)
            return 2
        return asyncio.run(
            _evaluate_saved_session(
                settings,
                session_path=session_path,
                model_class=cast(ModelClass, args.model_class),
            )
        )
    if args.command == "simulate-interview":
        try:
            session_path = _session_path(settings, args.session)
        except SessionEvaluationError as error:
            print(f"Interview simulation failed: {error}", file=sys.stderr)
            return 2
        return asyncio.run(
            _simulate_interview(
                settings,
                session_path=session_path,
                model_class=cast(ModelClass, args.model_class),
                assume_human_prompt_approvals=args.assume_human_prompt_approvals,
            )
        )
    if args.command == "timer-demo":
        if args.minutes <= 0:
            print("--minutes must be positive", file=sys.stderr)
            return 2
        return _timer_demo(minutes=args.minutes)
    raise AssertionError(f"Unhandled command: {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
