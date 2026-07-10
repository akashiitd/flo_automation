# FloCareer Interview Automation Plan

This document is a handoff-ready implementation plan for building a supervised
FloCareer interview automation/copilot on the Mac mini.

The goal is not to create an unattended hidden interviewer. The practical and
safer v1 is a human-approved copilot that:

- Joins FloCareer interviews.
- Reads questions and rubrics.
- Enables the coding editor when needed.
- Listens to candidate answers through Apple Speech/system audio.
- Uses an LLM provider router with LM Studio local inference as primary and
  OpenRouter as the configurable cloud fallback.
- Uses Supertonic as the runtime interviewer voice from the greeting onward,
  including standard questions, generated follow-ups, coding instructions, and
  the closing statement.
- Suggests follow-up questions.
- Drafts and fills feedback.
- Stops before final submit unless explicitly approved.

Implementation order and runtime order are different. During development,
Supertonic is tested independently and integrated after transcription and LLM
responses are stable. In the finished runtime, Supertonic speaks first and is
the synthetic interviewer voice throughout the interview.

---

## 0. Known Local Context

### Main automation workspace

```text
/path/to/Flocarrer_Interview_Automation
```

### Existing transcription app

```text
/path/to/Meeting_transcriber_with_LLM
```

Use this as the transcription engine. Do not rebuild transcription from
scratch.

Important files:

```text
/path/to/Meeting_transcriber_with_LLM/src/realtime_transcriber.py
/path/to/Meeting_transcriber_with_LLM/src/apple_speech_transcriber.py
/path/to/Meeting_transcriber_with_LLM/simple_recorder.py
```

### Confirmed machine

```text
Mac mini M4 Pro
14 CPU cores
20-core Apple GPU
64 GB unified memory
macOS 26.5.1
Python 3.14.2
Node 22.22.3
LM Studio installed and running
```

### LM Studio local endpoint

```text
http://127.0.0.1:1234/v1
```

Already detected local models:

```text
google/gemma-4-12b
qwen/qwen3.6-35b-a3b
google/gemma-4-31b
google/gemma-4-31b-qat
nvidia/nemotron-3-nano-4b
tinyllama-1.1b-chat-v1.0
cohere-transcribe-03-2026-mlx
qwen3-asr-1.7b
```

Recommended local LLM order:

```text
1. google/gemma-4-12b for fast follow-ups and feedback drafts
2. qwen/qwen3.6-35b-a3b for deeper final verdicts
3. OpenRouter fallback if local output quality is weak
```

### LLM provider strategy

Do not couple the evaluator directly to LM Studio. Put every model behind a
common provider interface so the runtime can switch providers without changing
the LangGraph workflow.

```text
Primary provider: LM Studio local API
Fallback provider: OpenRouter
Optional later provider: OpenAI direct API
```

Default routing policy:

```text
Fast follow-up or feedback draft
    -> LM Studio fast model

Final verdict
    -> LM Studio deep model

Local timeout, malformed JSON, unavailable model, or low confidence
    -> redact candidate PII
    -> retry through OpenRouter
```

OpenRouter must not receive candidate information unless cloud processing is
permitted. Before a cloud request, redact email, phone number, address, account
identifiers, and any resume fields not required for evaluating the answer.

### Browser automation choice

Use local Playwright as the primary controller and Computer Use as a visual
fallback.

```text
Playwright = free, open source, local
Stagehand = optional, open source SDK, useful when selectors are messy
Computer Use = paid cloud fallback for visual UI recovery
Browserbase = paid cloud browser, not needed for v1
```

### TTS / voice output choice

Test Supertonic independently early, then connect it to the live interview only
after browser control, transcription, and answer evaluation are reliable.

Supertonic can run its own local HTTP server, so we do not need to build a
custom FastAPI TTS endpoint immediately.

Expected Supertonic local server:

```text
http://127.0.0.1:7788
```

---

## 1. Final Target Flow

```text
LangGraph interview controller
    |
    +-> Browser action router
    |       +-> Playwright for normal, deterministic FloCareer actions
    |       +-> Computer Use only when Playwright cannot recover
    |       +-> Action guard blocks FINISH, hangup, and destructive actions
    |
    +-> Apple Speech
    |       +-> Candidate/system-audio transcription
    |       +-> Transcript segments sent to LangGraph
    |
    +-> LLM provider router
    |       +-> LM Studio as local primary
    |       +-> OpenRouter fallback after PII redaction
    |       +-> Structured score, follow-up, feedback, and confidence
    |
    +-> Supertonic
    |       +-> Approved question/follow-up converted to custom-voice audio
    |       +-> Audio queue sends output to the virtual microphone
    |
    +-> Human approval
            +-> Review final recommendation and evidence
            +-> Approve feedback autofill
            +-> Approve final FINISH action
```

### Runtime conversation loop

Supertonic is the text-to-speech layer. It does not decide what to ask. The LLM
and LangGraph decide the next text; Supertonic turns that text into audio using
the configured custom voice style.

```text
1. Preflight
   -> Start LM Studio and configure OpenRouter fallback
   -> Start Apple Speech system-audio listener
   -> Start Supertonic and load interviewer_voice.json
   -> Start Playwright and join FloCareer
   -> Pre-generate the greeting and standard questions

2. Greeting
   -> LangGraph selects the approved introduction text
   -> Supertonic synthesizes it in the custom voice
   -> Audio queue plays it through the virtual microphone into FloCareer

3. Candidate introduction
   -> Supertonic asks the candidate to introduce themselves
   -> State changes from AI_SPEAKING to LISTENING
   -> Apple Speech captures candidate/system audio
   -> Final transcript segments are appended to the current answer buffer

4. Answer completion
   -> Voice activity and silence detection identify the end of the answer
   -> LangGraph packages the question, rubric, answer, remaining time, and
      previous context
   -> LLM provider router calls LM Studio first
   -> OpenRouter is used only when local output fails or confidence is low

5. Next response
   -> LLM returns structured output:
      assessment, score_draft, follow_up_needed, next_spoken_text
   -> LangGraph either chooses one follow-up or advances to the next question
   -> Supertonic converts next_spoken_text into the custom voice
   -> Audio is played through the virtual microphone
   -> Loop returns to LISTENING

6. Coding question
   -> Browser action router opens the code editor and selects the language
   -> Supertonic reads the coding question and instructions
   -> Apple Speech captures the candidate's explanation
   -> Browser controller records editor state/code where technically available
   -> LLM evaluates explanation, approach, correctness, and complexity

7. Closing
   -> Supertonic asks whether the candidate has questions
   -> Supertonic plays the approved closing statement
   -> LangGraph creates per-question feedback and final recommendation
   -> Human reviews evidence and approves feedback/FINISH
```

### Required voice states

```text
PREPARING
AI_SPEAKING
LISTENING
CANDIDATE_SPEAKING
CANDIDATE_SILENCE
THINKING
TTS_GENERATING
INTERRUPTED
STOPPED
```

While `AI_SPEAKING` or `TTS_GENERATING` is active, the system must not treat the
generated voice as a candidate answer. Route Supertonic directly to a virtual
microphone, use headphones, and suppress transcript segments that match the
known generated text. If the candidate starts speaking, stop the TTS audio and
return to `LISTENING`.

The synthetic interviewer voice should be disclosed and authorized by the
company/platform and candidate. The human remains responsible for the final
hiring feedback and submission.

---

## 2. Recommended Project Structure

Create this structure under:

```text
/path/to/Flocarrer_Interview_Automation
```

```text
.
├── FLOCAREER_AUTOMATION_PLAN.md
├── README.md
├── pyproject.toml
├── .env.example
├── main.py
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── health.py
│   ├── models.py
│   └── logging_config.py
├── browser/
│   ├── __init__.py
│   ├── action_router.py
│   ├── action_guard.py
│   ├── playwright_controller.py
│   ├── computer_use_controller.py
│   ├── visual_verifier.py
│   ├── selectors.py
│   └── screenshots.py
├── transcriber/
│   ├── __init__.py
│   ├── apple_speech_adapter.py
│   └── transcript_store.py
├── llm/
│   ├── __init__.py
│   ├── provider.py
│   ├── provider_router.py
│   ├── lmstudio_provider.py
│   ├── openrouter_provider.py
│   ├── privacy_redactor.py
│   ├── usage_tracker.py
│   ├── prompts.py
│   ├── schemas.py
│   └── json_repair.py
├── evaluator/
│   ├── __init__.py
│   ├── scoring.py
│   ├── feedback.py
│   └── verdict.py
├── orchestrator/
│   ├── __init__.py
│   ├── graph.py
│   ├── state.py
│   └── timer.py
├── tts/
│   ├── __init__.py
│   ├── supertonic_client.py
│   └── audio_output.py
├── ui/
│   ├── __init__.py
│   └── server.py
├── tests/
│   ├── test_health.py
│   ├── test_llm_client.py
│   ├── test_evaluator.py
│   ├── test_timer.py
│   └── fixtures/
│       ├── sample_questions.json
│       └── sample_transcript.json
└── runs/
    └── .gitkeep
```

---

## 3. Configuration

Create `.env.example`:

```bash
# LLM routing
LLM_PROVIDER_MODE=auto
LLM_PRIMARY_PROVIDER=lmstudio
LLM_FALLBACK_PROVIDER=openrouter
LLM_FAST_TIMEOUT_SECONDS=8
LLM_DEEP_TIMEOUT_SECONDS=20
LLM_FALLBACK_CONFIDENCE_THRESHOLD=0.65
LLM_ALLOW_CLOUD_CANDIDATE_DATA=false

# Local LM Studio provider
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_API_KEY=lm-studio
LMSTUDIO_FAST_MODEL=google/gemma-4-12b
LMSTUDIO_DEEP_MODEL=qwen/qwen3.6-35b-a3b

# OpenRouter provider
OPENROUTER_API_KEY=
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
OPENROUTER_FAST_MODEL=google/gemini-2.5-flash-lite
OPENROUTER_DEEP_MODEL=google/gemini-2.5-flash
OPENROUTER_SITE_URL=http://localhost
OPENROUTER_APP_NAME=FloCareer Interview Copilot

# Existing transcriber app
MEETING_TRANSCRIBER_PATH=/path/to/Meeting_transcriber_with_LLM
TRANSCRIPTION_BACKEND=apple-speech
TRANSCRIBE_SYSTEM_AUDIO=true
TRANSCRIBE_MICROPHONE=false

# Browser
BROWSER_HEADLESS=false
BROWSER_USER_DATA_DIR=.browser-profile
FLOCAREER_URL=https://app.flocareer.com/interviewer/

# Optional Supertonic
SUPERTONIC_BASE_URL=http://127.0.0.1:7788
SUPERTONIC_VOICE=interviewer_voice

# Runtime
RUNS_DIR=runs
DEFAULT_INTERVIEW_MINUTES=25
REQUIRE_APPROVAL_BEFORE_FINISH=true
```

Test after creating `.env`:

```bash
python main.py config-dump
```

Pass criteria:

```text
No missing required config
Transcriber path exists
LM Studio URL configured
OpenRouter is configured or reported as optional/unavailable
Runs directory exists or can be created
```

---

## 4. Milestone 1: Health Check

### Objective

Build a single command that checks whether the local environment is ready.

### What to implement

`main.py health` should check:

- Python can start.
- `MEETING_TRANSCRIBER_PATH` exists.
- Apple Speech transcriber file exists.
- LM Studio `/v1/models` responds.
- At least one local LLM model is available.
- OpenRouter key is valid when cloud fallback is enabled.
- Playwright is installed.
- Browser can launch.
- `runs/` is writable.
- Supertonic is optional and reported as available/unavailable.

### Command

```bash
python main.py health
```

### Expected output

```text
Health check
[OK] Python
[OK] Runs directory writable
[OK] Meeting_transcriber_with_LLM path found
[OK] Apple Speech adapter found
[OK] LM Studio reachable
[OK] Local models found: google/gemma-4-12b, qwen/qwen3.6-35b-a3b
[OK] LLM primary provider: lmstudio
[WARN] OpenRouter fallback disabled because cloud candidate data is not allowed
[OK] Playwright browser launch
[WARN] Supertonic not running

Overall: READY_FOR_BROWSER_SCAN
```

### Failure handling

If LM Studio is not reachable:

```text
Open LM Studio -> Developer/API server -> Start server
Confirm URL: http://127.0.0.1:1234/v1
```

If Playwright is missing:

```bash
pip install playwright
python -m playwright install chromium
```

If Apple Speech permissions fail later:

```text
System Settings -> Privacy and Security
Enable:
- Speech Recognition
- Screen and System Audio Recording
- Microphone if microphone capture is used
```

---

## 5. Milestone 2: LLM Provider Router and Evaluator

### Objective

Prove that LM Studio and OpenRouter use the same structured evaluator contract,
and that automatic fallback works without changing LangGraph state.

### What to implement

Create:

```text
llm/provider.py
llm/provider_router.py
llm/lmstudio_provider.py
llm/openrouter_provider.py
llm/privacy_redactor.py
llm/usage_tracker.py
llm/schemas.py
evaluator/scoring.py
```

Every provider must implement the same operations:

```python
async def health() -> dict: ...
async def generate_structured(messages, schema, model_class) -> dict: ...
async def stream_text(messages, model_class): ...
```

Provider classes must return normalized metadata:

```json
{
  "provider": "lmstudio",
  "model": "google/gemma-4-12b",
  "latency_ms": 1840,
  "input_tokens": 1220,
  "output_tokens": 310,
  "estimated_cost_usd": 0.0,
  "fallback_used": false
}
```

Expected evaluator input:

```json
{
  "question_id": 1,
  "question": "How would you build a LangChain microservice?",
  "ideal_answer": "Mentions API layer, tool orchestration, tracing, retries...",
  "candidate_answer": "I would create an API, call the model, and return response..."
}
```

Expected evaluator output:

```json
{
  "question_id": 1,
  "score": 3,
  "rating_label": "Average",
  "evidence": [
    "Candidate mentioned API layer",
    "Candidate missed observability and retries"
  ],
  "follow_up": "How would you handle retries and timeout failures?",
  "feedback": "Candidate showed basic understanding but lacked production depth.",
  "confidence": 0.72
}
```

### Command

```bash
python main.py llm-test --provider lmstudio
python main.py llm-test --provider openrouter
python main.py llm-failover-test
```

### Pass criteria

- Response is valid JSON.
- Score is between 1 and 5.
- Feedback is not empty.
- Follow-up is relevant.
- Runtime is acceptable for live use:
  - Fast model target: under 8 seconds.
  - Deep model target: under 20 seconds.
- The same schema is returned by both providers.
- A simulated LM Studio timeout routes to OpenRouter only when cloud use is
  enabled.
- PII redaction runs before every OpenRouter request.
- Provider, model, latency, token usage, and estimated cost are logged.

### Important implementation detail

Local models may emit hidden reasoning or malformed JSON. Always:

1. Ask for JSON.
2. Parse JSON.
3. If parse fails, run a repair step.
4. If repair fails, retry once with a stricter prompt.
5. If still failed and cloud use is allowed, redact PII and fall back to
   OpenRouter.
6. If cloud use is disabled, return a visible error and request human review.

---

## 6. Milestone 3: Apple Speech Candidate Audio Test

### Objective

Use the existing Meeting Transcriber app as the "ears" of the automation.

### Integration approach

Create:

```text
transcriber/apple_speech_adapter.py
transcriber/transcript_store.py
```

Adapter behavior:

1. Add this path to `sys.path`:

```text
/path/to/Meeting_transcriber_with_LLM
```

2. Import:

```python
from src.realtime_transcriber import create_realtime_transcriber
```

3. Start with:

```python
transcriber, live_logger = create_realtime_transcriber(
    model_size="small",
    language="en",
    enable_system_audio=True,
    enable_microphone=False,
    callback=on_segment,
    session_name=session_id,
    enable_live_logging=True,
    transcription_backend="apple-speech",
)
```

4. Store segments to:

```text
runs/<session_id>/transcript.json
runs/<session_id>/transcript.txt
```

### Why microphone is false

For FloCareer, the candidate voice comes from Chrome/system audio.

```text
enable_system_audio=true  -> candidate audio
enable_microphone=false   -> avoid capturing your voice or AI TTS output
```

### Command

```bash
python main.py listen-test --seconds 60
```

### Manual test setup

1. Open any YouTube/interview audio or FloCareer meeting audio in Chrome.
2. Start the command.
3. Let candidate/system audio play.
4. Confirm transcript appears.

### Expected output

```text
Starting Apple Speech listener
Mode: system audio only
[Other] candidate answer text here
Saved transcript: runs/<session_id>/transcript.json
```

### Pass criteria

- Transcript captures Chrome/system audio.
- Speaker/source is `Other` or `system`.
- No microphone/self voice appears when microphone is off.
- JSON transcript is saved.

---

## 7. Milestone 4: Browser Automation Dry Run

### Objective

Use local Playwright to scan FloCareer dashboard without launching interviews.

### What to implement

Create:

```text
browser/flocareer_page.py
browser/selectors.py
browser/screenshots.py
```

Initial functions:

```python
open_dashboard()
wait_for_login_or_dashboard()
scan_scheduled_interviews()
save_screenshot(name)
```

### Command

```bash
python main.py browser-scan
```

### Expected behavior

1. Open local Chrome/Chromium browser.
2. Navigate to:

```text
https://app.flocareer.com/interviewer/
```

3. If login is required, stop and ask user to login manually.
4. Once logged in, list scheduled interviews.

### Expected output

```text
FloCareer dashboard loaded
Found scheduled interviews:
1. Candidate Alpha | Platform Engineer | Example Corp | Today 11:00 AM
2. Candidate Beta | Data Scientist | Northwind Labs | Today 4:00 PM

Screenshot saved: runs/<session_id>/screenshots/dashboard.png
```

### Pass criteria

- Browser opens.
- Login session can persist.
- Candidate list is detected.
- No interview is launched in this step.

---

## 8. Milestone 5: Launch and Join Interview

### Objective

Automate only the join workflow.

### Flow from screenshots

Dashboard:

1. Locate candidate row.
2. Click three-dot menu beside candidate.
3. Click `Launch Video Interview`.
4. Wait for pre-call page.
5. Click `Join`.

Pre-call page:

```text
Joining as AS971
Join button
```

### Command

```bash
python main.py join --candidate "Candidate Name"
```

Add dry-run support:

```bash
python main.py join --candidate "Candidate Name" --dry-run
```

### Pass criteria

- Candidate row is found.
- Menu opens.
- `Launch Video Interview` is clicked only when dry-run is false.
- Join page is reached.
- `Join` button is clicked.
- Screenshot after joining is saved.

### Safety rule

Never auto-click the red hangup button.

---

## 9. Milestone 6: Question Extraction

### Objective

Extract all question cards and store them in JSON.

### What to capture

For each question:

```json
{
  "id": 5,
  "question_text": "How would you use LangChain...",
  "has_code_editor": false,
  "ideal_answer": "Ideal Answer text if visible",
  "guidelines": {
    "5_star": "...",
    "4_star": "...",
    "3_star": "...",
    "2_star": "...",
    "1_star": "..."
  },
  "feedback_field_locator_hint": "...",
  "rating_locator_hint": "...",
  "mark_as_locator_hint": "..."
}
```

### Command

```bash
python main.py extract-questions
```

### Expected output

```text
Extracted 10 questions
Coding questions: 9, 10
Saved: runs/<session_id>/questions.json
```

### Pass criteria

- Question numbers match UI.
- Question text is not truncated if expandable text is available.
- Coding questions are detected.
- Screenshot saved for extraction debugging.

---

## 10. Milestone 7: Code Editor Automation

### Objective

Enable the FloCareer code editor for coding questions.

### Flow from screenshots

For a coding question:

1. Scroll to question card.
2. Click `Code Editor` tab.
3. Select language, usually `Python`.
4. Check toggle text.
5. If it says `SHOW CODE EDITOR TO CANDIDATE`, click it.
6. Pass when it says `HIDE CODE EDITOR TO CANDIDATE`.

Meaning:

```text
SHOW CODE EDITOR TO CANDIDATE -> currently hidden, click to show
HIDE CODE EDITOR TO CANDIDATE -> currently visible, correct state
```

### Command

```bash
python main.py enable-code-editor --question 9 --language Python
```

### Expected output

```text
Question 9 found
Code Editor tab opened
Language set to Python
Candidate visibility: enabled
Screenshot saved
```

### Pass criteria

- Code editor tab is active.
- Language dropdown shows Python.
- Toggle says `HIDE CODE EDITOR TO CANDIDATE`.
- Candidate can see editor.

---

## 11. Milestone 8: Offline Evaluation From Transcript

### Objective

Before live use, prove scoring works with saved transcripts.

### Input files

```text
runs/test_session/questions.json
runs/test_session/transcript.json
```

### Command

```bash
python main.py evaluate --session runs/test_session
```

### Expected output

```json
{
  "candidate": "Test Candidate",
  "overall_recommendation": "Borderline",
  "confidence": 0.74,
  "question_scores": [
    {
      "question_id": 1,
      "score": 3,
      "feedback": "..."
    }
  ],
  "strengths": [],
  "risks": [],
  "final_feedback_to_submit": "..."
}
```

### Verdict labels

Use exactly these:

```text
Strong Hire
Hire
Borderline
No Hire
```

### Scoring guide

```text
5 = excellent, production-ready, clear tradeoffs
4 = good, mostly complete, minor gaps
3 = average, understands basics but lacks depth
2 = weak, vague or incomplete
1 = poor, incorrect or unable to answer
```

### Pass criteria

- Generates per-question feedback.
- Generates final verdict.
- Includes evidence from transcript.
- Does not invent answers not present in transcript.
- Saves:

```text
runs/<session_id>/evaluation.json
runs/<session_id>/feedback_preview.md
```

---

## 12. Milestone 9: LangGraph Interview Controller

### Objective

Implement stateful interview flow.

### Why LangGraph

This is a timed workflow, not just chat:

```text
join -> ask -> listen -> evaluate -> follow-up -> code -> feedback -> approve -> fill -> finish
```

LangGraph is better than a free-form agent here because every step has state,
timing, and safety gates.

### Graph state

Create:

```text
orchestrator/state.py
orchestrator/graph.py
orchestrator/timer.py
```

State fields:

```python
candidate_name: str
role: str
company: str
session_id: str
started_at: datetime
time_limit_minutes: int
current_question_id: int
questions: list
transcript_segments: list
question_scores: list
pending_follow_up: str | None
feedback_drafts: list
final_verdict: dict | None
requires_human_approval: bool
```

### States

```text
START
HEALTH_OK
BROWSER_READY
JOINED
QUESTIONS_EXTRACTED
ASK_QUESTION
LISTENING
ANSWER_COMPLETE
EVALUATING
FOLLOW_UP_READY
CODING_MODE
FEEDBACK_READY
NEXT_QUESTION
FINAL_VERDICT_READY
HUMAN_APPROVAL
FILL_FEEDBACK
FINISH_APPROVED
DONE
ERROR
```

### Command

```bash
python main.py simulate-interview --session runs/test_session
```

### Pass criteria

- Graph can run from sample transcript without browser.
- Every state transition is logged.
- It produces final feedback.
- Human approval state blocks final submit.

---

## 13. Milestone 10: Feedback Autofill

### Objective

Fill FloCareer feedback fields automatically after user approval.

### Fields to fill

Per question:

```text
Feedback *
YOUR RATING stars
Mark as dropdown
```

Top right:

```text
Report dropdown
Level dropdown, e.g. Intermediate
FINISH button
```

### Command

```bash
python main.py fill-feedback --session runs/<session_id>
```

Approval mode:

```bash
python main.py fill-feedback --session runs/<session_id> --approve
```

### Safety rule

Default behavior:

```text
Fill feedback fields: allowed after preview approval
Click FINISH: never without explicit separate approval
```

### Pass criteria

- Feedback text appears in correct question boxes.
- Stars match score.
- `Mark as` is set.
- Final submit is not clicked automatically.
- Before/after screenshots saved.

---

## 14. Milestone 11: 25-Minute Interview Timer

### Objective

Keep the interview inside 25 minutes.

### Suggested timeline

```text
00:00 - 02:00  Greeting and format
02:00 - 05:00  Candidate intro/project
05:00 - 12:00  AI/ML architecture questions
12:00 - 20:00  Coding question
20:00 - 23:00  Follow-up and edge cases
23:00 - 25:00  Candidate questions and wrap-up
```

### Timer warnings

```text
15 min left
10 min left
5 min left
2 min left
1 min left
time over
```

### Command

```bash
python main.py timer-demo --minutes 1
```

### Pass criteria

- Timer emits warnings.
- Graph receives timer events.
- At 5 minutes left, graph prioritizes coding/final feedback.
- At 2 minutes left, graph moves to wrap-up.

---

## 15. Milestone 12: Local Dashboard

### Objective

Create one local UI for supervising the interview.

### Minimum UI panels

```text
Candidate
Timer
Current question
Live transcript
Suggested follow-up
Question score
Feedback draft
Final verdict
Browser action log
Approval buttons
```

### Buttons

```text
Start Session
Scan Browser
Join Interview
Extract Questions
Start Listening
Ask Follow-up
Enable Code Editor
Generate Feedback
Fill Feedback
Approve Finish
Stop
```

### Command

```bash
python main.py ui
```

### Pass criteria

- Dashboard opens locally.
- Live transcript updates.
- Evaluation appears.
- Buttons call backend actions.
- No final submit without approval.

---

## 16. Milestone 13: Supertonic Live Interview Voice Integration

### Objective

Connect the already-tested Supertonic service to the live LangGraph interview
loop so it speaks the introduction, questions, follow-ups, coding instructions,
time notices, and closing statement in the configured custom voice.

### Why postpone this

Voice introduces:

- Latency.
- Audio routing complexity.
- Feedback loops.
- Consent/platform policy risk.
- More live failure modes.

The isolated Supertonic health and voice-quality tests happen early. This
milestone is postponed only for full duplex audio routing and live turn-taking,
which should be connected after browser, transcript, and LLM response tests are
stable.

### Install

```bash
python -m venv .venv-supertonic
source .venv-supertonic/bin/activate
pip install 'supertonic[serve]'
```

### Start server

```bash
supertonic serve --host 127.0.0.1 --port 7788
```

### Import voice

If using a custom voice from Supertonic Voice Builder:

```bash
curl -X POST http://127.0.0.1:7788/v1/styles/import \
  -F "file=@voices/interviewer_voice.json"
```

### Generate WAV

```bash
curl -X POST http://127.0.0.1:7788/v1/tts \
  -H 'content-type: application/json' \
  -d '{"text":"Let us move to the coding question.","voice":"interviewer_voice","lang":"en"}' \
  -o runs/tts_test.wav
```

### Test command

```bash
python main.py tts-test --text "Let us move to the coding question."
```

### Pass criteria

- WAV file is generated.
- Audio sounds acceptable.
- Latency is measured.
- Standard scripts are pre-generated successfully.
- Dynamic follow-up text can be synthesized within the latency target.
- Audio can be interrupted when candidate speech begins.
- Generated speech is not added to the candidate transcript.
- Voice use follows the configured approval/disclosure policy.

### Later virtual microphone

To send TTS into FloCareer:

```text
Supertonic WAV/audio stream -> virtual audio device -> Chrome microphone input
```

Possible virtual audio tools:

```text
BlackHole
Loopback
VB-Cable equivalent for macOS
```

Do not enable this until text-only mode is stable.

---

## 17. Operating Modes

### Mode A: Safest v1

```text
Human speaks.
Tool listens, scores, suggests, fills feedback.
```

Use this first.

### Mode B: Click-to-speak assistant

```text
Tool suggests next question.
User clicks Speak.
Supertonic speaks through selected output.
```

Use after v1.

### Mode C: Semi-autonomous co-interviewer

```text
Tool asks approved questions/follow-ups.
User monitors.
Tool fills feedback.
User approves final submit.
```

Use only if platform/company/candidate consent is clear.

### Mode D: Fully unattended hidden clone

Do not build this as the production target. It is risky for hiring integrity,
platform policy, and candidate trust.

---

## 18. Cost Strategy

### Best ROI setup

```text
Browser automation: local/free
Transcription: local/free using existing app
LLM primary: LM Studio local/free
LLM fallback: OpenRouter only on timeout, invalid output, or low confidence
Voice: local Supertonic later
```

Record LLM usage per interview in:

```text
runs/<session_id>/llm_usage.jsonl
```

Each entry must include:

```text
provider
model
request purpose
latency
input/output tokens
estimated cost
fallback reason
whether PII redaction ran
```

### Interview economics

Gross per interview:

```text
Rs 550
```

10 percent tax deduction:

```text
Rs 55
```

Cash received:

```text
Rs 495
```

Target automation cost:

```text
Rs 0 to Rs 5 per interview
```

Avoid paid cloud browsers for v1.

---

## 19. Testing Matrix

Run these tests in order.

```text
1. python main.py health
2. python main.py llm-test --provider lmstudio
3. python main.py llm-test --provider openrouter
4. python main.py llm-failover-test
5. python main.py listen-test --seconds 60
6. python main.py browser-scan
7. python main.py join --candidate "Test Candidate" --dry-run
8. python main.py extract-questions
9. python main.py enable-code-editor --question 9 --language Python
10. python main.py evaluate --session runs/test_session
9. python main.py simulate-interview --session runs/test_session
10. python main.py fill-feedback --session runs/test_session
11. python main.py timer-demo --minutes 1
12. python main.py ui
13. python main.py tts-test --text "Hello, let us start."
```

Do not move to the next test until the current one passes.

---

## 20. Live Interview Runbook

### Before interview

```bash
python main.py health
python main.py browser-scan
python main.py llm-test
```

Open LM Studio and preload chosen model.

Recommended:

```text
google/gemma-4-12b
```

If using deeper local verdict:

```text
qwen/qwen3.6-35b-a3b
```

### Start interview

```bash
python main.py start-session --candidate "Candidate Name" --minutes 25
```

Expected actions:

```text
Create runs/<session_id>
Start browser controller
Start transcript listener
Start timer
Wait for user approval to join
```

### During interview

The dashboard should show:

```text
Current question
Live transcript
Suggested follow-up
Score so far
Coding flag
Timer warnings
```

### Coding question

```bash
python main.py enable-code-editor --question <id> --language Python
```

Or click UI button.

### After interview

```bash
python main.py generate-final-verdict --session runs/<session_id>
python main.py fill-feedback --session runs/<session_id>
```

Then manually approve:

```text
FINISH
```

---

## 21. Data Saved Per Session

Each interview should save:

```text
runs/<session_id>/
├── metadata.json
├── questions.json
├── transcript.json
├── transcript.txt
├── evaluation.json
├── feedback_preview.md
├── final_verdict.json
├── action_log.jsonl
└── screenshots/
    ├── dashboard.png
    ├── joined.png
    ├── questions.png
    ├── code_editor.png
    ├── feedback_before.png
    └── feedback_after.png
```

Do not store unnecessary candidate personal data beyond what is needed for
the interview record.

---

## 22. Prompt Requirements

### Follow-up generation prompt should require:

- One concise follow-up only.
- Based only on candidate answer.
- No answer leakage.
- Match role level.
- Stop asking if time is low.

### Scoring prompt should require:

- Score 1 to 5.
- Evidence from transcript.
- Missing concepts.
- Practical feedback.
- Confidence.
- No invented claims.

### Final verdict prompt should require:

- `Strong Hire`, `Hire`, `Borderline`, or `No Hire`.
- Summary of strengths.
- Summary of risks.
- Coding assessment.
- Communication assessment.
- AI/ML depth assessment.
- Feedback suitable for FloCareer.
- Confidence.

---

## 23. Common Failure Cases

### Candidate audio not transcribed

Check:

```text
System Settings -> Privacy and Security -> Screen and System Audio Recording
Chrome audio is audible through system output
Apple Speech helper can run
```

### LM Studio slow

Try:

```text
Use google/gemma-4-12b instead of qwen/qwen3.6-35b-a3b
Reduce context length
Preload model before interview
Use OpenRouter fallback for final verdict
```

### LLM JSON malformed

Use:

```text
JSON parser
JSON repair
Retry strict prompt
Fallback cloud model
```

### Browser selectors break

Use:

```text
Screenshots
Playwright locator debugging
Text-based locators
Optional Stagehand for AI-assisted extraction/clicking
```

### Code editor not visible to candidate

Check toggle text:

```text
SHOW CODE EDITOR TO CANDIDATE = hidden right now, click it
HIDE CODE EDITOR TO CANDIDATE = visible right now, correct
```

### Feedback filled in wrong box

Stop automation. Use screenshot and question id mapping. Never click finish.

---

## 24. First Build Sprint

The first sprint should implement only these:

```text
1. Project skeleton
2. Health check
3. LLM provider interface
4. LM Studio evaluator test
5. OpenRouter evaluator test
6. LM Studio-to-OpenRouter failover test
7. Apple Speech listen-test
8. Browser scan dry run
```

Definition of done:

```text
python main.py health passes
python main.py llm-test --provider lmstudio returns valid score JSON
python main.py llm-test --provider openrouter returns the same schema
python main.py llm-failover-test proves guarded provider failover
python main.py listen-test captures Chrome/system audio
python main.py browser-scan lists FloCareer interviews
```

After that, build join/question/code-editor/fill-feedback.

---

## 25. Handoff Prompt For A New Chat Session

Use this prompt in another session:

```text
We are building a supervised FloCareer interview automation copilot.
Workspace:
/path/to/Flocarrer_Interview_Automation

Existing transcription app:
/path/to/Meeting_transcriber_with_LLM

Use local Playwright for browser automation.
Use Computer Use only as a guarded visual fallback when Playwright fails.
Use LM Studio local OpenAI-compatible API at:
http://127.0.0.1:1234/v1

Use a provider-agnostic LLM router:
- Primary provider: LM Studio
- Fallback provider: OpenRouter
- Redact candidate PII before cloud calls
- Log provider, model, latency, token usage, and estimated cost
- Do not send candidate data to the cloud unless configuration explicitly
  allows it

Use the existing Apple Speech transcriber from Meeting_transcriber_with_LLM.
For FloCareer candidate audio use system audio on and microphone off.

Follow the plan in:
FLOCAREER_AUTOMATION_PLAN.md

Start with Milestone 1 to Milestone 5 only:
health, LM Studio test, OpenRouter test, failover test, listen-test,
browser-scan, join dry-run.
Do not implement voice cloning yet.
Do not auto-click FINISH without explicit approval.
```

---

## 26. Immediate Next Action

Create the skeleton and implement:

```bash
python main.py health
```

Then test it.

Only after health passes, implement:

```bash
python main.py llm-test --provider lmstudio
python main.py llm-test --provider openrouter
python main.py llm-failover-test
python main.py listen-test --seconds 60
python main.py browser-scan
```
