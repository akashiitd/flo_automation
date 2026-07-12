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
- Uses local Qwen3-TTS as the cloned runtime interviewer voice from the greeting onward,
  including standard questions, generated follow-ups, coding instructions, and
  the closing statement.
- Suggests follow-up questions.
- Drafts and fills feedback.
- Stops before final submit unless explicitly approved.

Implementation order and runtime order are different. During development,
Qwen3-TTS is tested independently and integrated after transcription and LLM
responses are stable. In the finished runtime, Qwen speaks first and is the
synthetic interviewer voice throughout the interview. Supertonic remains an
optional fixed-voice alternative, not the cloned-voice runtime.

---

## Current Implementation Status (2026-07-12)

This section is the execution checkpoint for a new development session. The
sections below it remain the detailed target specification. When this status
section and a later milestone section disagree about what is already built,
this status section is authoritative.

The former Supertonic section below has been replaced with the implemented
Qwen3-TTS approach. The routing, consent, and echo-suppression constraints
apply equally to the cloned-voice runtime.

### Repository and operating baseline

```text
Repository: https://github.com/akashiitd/flo_automation
Default branch: main
Python environment: uv
FloCareer starting URL: https://app.flocareer.com/
Authentication: manual in a persistent local Playwright profile
Local LLM: ornith-1.0-35b through LM Studio
Cloud fallback: OpenRouter, only after explicit configuration opt-in
Transcription: existing Meeting_transcriber_with_LLM Apple Speech backend
```

Start every new session by reading:

```text
README.md
FLOCAREER_AUTOMATION_PLAN.md
git status -sb
git log -3 --oneline
```

Never request or paste FloCareer credentials, OTPs, or API keys into chat,
source files, logs, screenshots, tests, or commits. Authentication is performed
by the human in the local persistent browser.

### Verified quality baseline

The following gates passed before this checkpoint was written:

```text
uv run pytest
    -> 44 passed

uv run ruff check .
    -> passed

uv run ruff format --check .
    -> passed

uvx ty check app browser evaluator llm transcriber main.py
    -> passed

uv run python main.py health
    -> Overall: READY_FOR_BROWSER_SCAN
```

The Qwen3-TTS cloned-voice worker is implemented as an optional loopback-only
service. It keeps its MLX model loaded, provides complete-WAV and streamed-PCM
speech endpoints, and has a tested LM Studio sentence-streaming bridge. Local
Loopback playback and candidate-only no-echo isolation are validated; live
FloCareer wiring and barge-in remain unimplemented.

### Milestone status matrix

| Milestone | Status | Evidence |
| --- | --- | --- |
| Project skeleton and configuration | Complete | `.env.example`, typed settings, safe config dump, ignored runtime/secrets |
| 1. Health check | Complete | Python, runs directory, transcriber, Apple Speech, LM Studio, OpenRouter, Playwright, optional Qwen and Supertonic |
| 2. Provider router and evaluator | Complete | LM Studio and OpenRouter return the same validated schema; JSON repair, strict retry, timeout, low-confidence routing, PII redaction, usage logging |
| 3. Apple Speech system audio | Local external-worktree capture and Qwen no-echo isolation validated; portable dependency pending | The dirty external Apple Speech helper worktree selects the exact `CANDIDATE_ONLY` input via an AVFoundation capture session. A 45-second check captured 11 Chrome speech segments as `system` while a distinct Qwen phrase on `INTERVIEWER_TO_CALL` was absent; the external change must be cleaned and committed separately. |
| 4. Browser dashboard scan | Complete | Persistent manual login, delayed-auth protection, loading protection, card/table extraction, screenshot, no launch action |
| 5. Launch, join, and candidate arrival | Implemented; watched validation pending | `--live` handles optional consent and separate Join approval, then records `LAUNCHED → INTERVIEWER_IN_ROOM → WAITING_FOR_CANDIDATE → CANDIDATE_CONNECTED` while keeping the browser open; hang-up and FINISH remain blocked |
| 6. Question extraction | Implemented; watched revalidation pending | `questions-scan` reached 17 sequential cards; supplied `.clFESingleSugDet` HTML and automated coverage preserve multiline text; semantic coding detection and optional reversible Code Editor tab capture have fixture coverage |
| 7. Code editor automation | Persistent-session integration implemented; watched validation pending | The verified card-scoped switch selector, exact question scoping, tab/state revalidation, and candidate-and-question approval are available only through `join --live --enable-code-editor-question`; language and code remain untouched |
| 8. Offline session evaluation | Partial: per-question evaluation and preview implemented; final verdict/strengths/risks pending | `evaluate --session` validates saved questions plus candidate-only `source: system` transcript segments with explicit `question_id`, writes local `evaluation.json` and a non-submitted feedback preview, and never guesses turn boundaries. |
| 9. Supervised interview controller | Partial pure state controller and simulation implemented; LangGraph scope and live wiring pending | `simulate-interview --session` stops at the first human approval by default; its explicit offline-only approval-assumption option records introduction/question/listen/evaluate/follow-up/next-question transitions. It has no browser or TTS side effects. |
| 10. Feedback autofill | Not started | Must remain behind preview and approval gates |
| 11. Interview timer | Partial: pure warnings implemented; controller reactions and live integration pending | `timer-demo --minutes 25` deterministically emits the 15-, 10-, 5-, 2-, and 1-minute warnings plus the deadline without starting a call. |
| 12. Local dashboard | Not started | Depends on stable backend operations |
| 13. Qwen cloned live voice | Local Loopback playback, LM-to-playback, and source isolation validated; room audio and barge-in pending | Persistent local Qwen worker, private reference voice, health probe, text-to-WAV and streamed-PCM clients, sentence-streaming LM bridge, cancellable 24 kHz mono → 48 kHz stereo PCM playback, exact Loopback-device diagnostics, and LM sentence-to-playback wiring are implemented. Qwen health passed; direct Qwen playback wrote 11 chunks, LM Studio → Qwen wrote 28 chunks, and a 45-second Chrome capture had 11 candidate-only segments with no distinct Qwen phrase. |

### Next-session execution plan

The immediate goal is **not** to make an unattended interviewer. It is to
connect the already working local components into a supervised, testable voice
loop with clear audio boundaries.

1. **Cleanly deliver selected-device support.** The local
   `Meeting_transcriber_with_LLM` worktree has exact `CANDIDATE_ONLY` support
   while generic capture remains unchanged, but it is mixed with unrelated user
   work. Separate and commit that dependency first.
2. **Validate a supervised non-production call.** Select
   `INTERVIEWER_TO_CALL` as the FloCareer microphone only for the watched test.
   Confirm candidate-only transcription and no Qwen echo before using any live
   interview.
3. **Wire and validate barge-in in the live loop.** The cancellable playback
   session now has a thread-safe `PlaybackBargeInController` that cancels only
   on a non-empty `source: system` segment. Connect it to the selected-device
   callback only after the Qwen echo-isolation test passes, then validate it in
   a disclosed test call.
4. **Wire the supervised interview controller to the live loop.** The pure
   state machine now models the sequence
   `introduction → ordered question → candidate turn → transcript → rubric
   evaluation → optional follow-up → next question`; connect its explicit
   question boundaries to the transcript adapter, while preserving a human
   approval boundary for candidate-visible prompts and feedback.
5. **Only then add session aggregation, timer, and feedback preview.** Keep
   feedback submission and `FINISH` outside automation.

For a clean continuation, see the private temporary handoff file:

```text
/private/tmp/FLOCAREER_NEXT_SESSION_HANDOFF.md
```

### What is implemented now

#### Configuration and health

```text
main.py config-dump
main.py health
app/config.py
app/health.py
```

- Loads defaults, `.env`, and environment overrides without printing secrets.
- Uses `https://app.flocareer.com/` as the browser entry point.
- Uses `../Meeting_transcriber_with_LLM` as the portable default transcriber
  path; `.env` may override it with an absolute path.
- Creates and verifies the `runs/` directory.
- Launches and closes Playwright Chromium during health validation.
- Validates OpenRouter only when cloud processing is explicitly enabled.

#### LLM evaluation and provider routing

```text
llm/provider.py
llm/lmstudio_provider.py
llm/openrouter_provider.py
llm/provider_router.py
llm/schemas.py
llm/json_repair.py
llm/privacy_redactor.py
llm/usage_tracker.py
evaluator/scoring.py
```

- `ornith-1.0-35b` is the tested local fast and deep model.
- Live local structured evaluation passed in approximately 3.5 to 4.1 seconds.
- Live OpenRouter evaluation passed in approximately 3.7 seconds.
- Both providers return score, rating, evidence, one follow-up, feedback, and
  confidence through the same schema.
- Fast and deep calls have hard wall-clock deadlines.
- Local timeout, invalid output, or low confidence can route to OpenRouter only
  when an API key and cloud-data consent are present.
- OpenRouter requests always pass through deterministic PII redaction.
- Usage records include provider, model, purpose, latency, tokens, estimated
  cost, fallback reason, and whether redaction ran.

#### Apple Speech system-audio capture

```text
transcriber/apple_speech_adapter.py
transcriber/transcript_store.py
main.py listen-test
```

- Reuses the external Meeting Transcriber project; no transcription engine was
  rebuilt.
- Starts Apple Speech with system audio enabled and microphone disabled.
- Refuses to start `listen-test` when microphone capture is enabled.
- Persists callbacks immediately and atomically into per-session JSON and text.
- A real 60-second test captured nine system-audio segments with zero
  microphone segments.

#### Read-only FloCareer dashboard scan

```text
browser/playwright_controller.py
browser/flocareer_page.py
browser/selectors.py
browser/screenshots.py
main.py browser-scan
```

- Opens the root FloCareer application URL in headed persistent Chromium.
- Waits through delayed logged-out dialogs and loading placeholders.
- Requires dashboard readiness to remain stable before scanning.
- Supports both table rows and the actual visible scheduled-interview card
  layout.
- Extracts candidate, role, company, date, and time without opening menus.
- Saves a dashboard screenshot under the session directory.
- Live validation extracted five scheduled cards and launched no interview.

### Commands available now

```bash
uv run python main.py config-dump
uv run python main.py health
uv run python main.py llm-test --provider lmstudio
uv run python main.py llm-test --provider openrouter
uv run python main.py llm-failover-test
uv run python main.py listen-test --seconds 60
uv run python main.py browser-scan --login-timeout 180
uv run python main.py join --candidate "Exact Candidate Name" --dry-run
uv run python main.py join --candidate "Exact Candidate Name" --live
```

Do not assume commands described later in this plan exist until they appear in
`uv run python main.py --help`.

### Current safety boundaries

These restrictions must remain true while implementing later milestones:

```text
Allowed now:
- Read configuration and health state.
- Run local/cloud evaluator tests under configured privacy policy.
- Capture system audio with microphone off.
- Open FloCareer and read scheduled-interview cards.
- Find one exact candidate, open that card's menu, and verify that dry-run
  blocks `Launch Video Interview`.
- Launch and click pre-call Join only after distinct candidate-bound,
  single-use approvals in `--live` mode. If the verified Interviewer Consent
  Form appears, its scoped `OK` requires a third approval; otherwise no consent
  action is attempted.
- After the exact candidate is connected in that same persistent session, show
  one exact code editor only after its separate candidate-and-question-bound
  approval. The workflow never changes language or editor code.
- Save local screenshots, transcripts, and action/usage logs.

Not implemented or not authorized by default:
- Launch or Join without the matching stage approval.
- Enable a code editor outside the guarded persistent `join --live` session,
  without candidate connection, or without its exact question-bound approval.
- Fill feedback fields.
- Click hang-up.
- Click FINISH.
- Run an unattended or undisclosed synthetic interviewer.
```

Runtime candidate data, transcripts, screenshots, browser profiles, `.env`, and
API keys must remain excluded from Git. Public tests and documentation must use
fictional candidate and company names.

### Completed validation milestone: guarded join dry-run

The dry-run half of Milestone 5 was live-validated against a visible scheduled
interview. It found the exact candidate in FloCareer's Material UI grid, opened
only that candidate's menu, found one launch control, and recorded a `BLOCK`
decision without navigating away from the dashboard:

```bash
uv run python main.py join --candidate "Candidate Name" --dry-run
```

#### Goal

Prove that the automation can identify exactly one scheduled candidate and the
candidate-scoped launch controls while making it impossible for dry-run mode to
launch or join an interview.

#### Proposed files

```text
browser/action_guard.py
browser/action_router.py
browser/join_workflow.py
browser/flocareer_page.py       # add candidate-card lookup/menu locators
browser/playwright_controller.py # reuse persistent context/session handling
main.py                         # add join command
tests/test_action_guard.py
tests/test_join_workflow.py
```

Keep the browser modules deep: the CLI should ask one workflow operation to run
and should not contain candidate-card selectors or click policy.

#### Required action vocabulary

Use an explicit enum or equivalent typed vocabulary rather than raw click names:

```text
OPEN_DASHBOARD
FIND_CANDIDATE
OPEN_CANDIDATE_MENU
LAUNCH_INTERVIEW
CLICK_CONSENT_OK
CLICK_JOIN
HANG_UP
FILL_FEEDBACK
FINISH_INTERVIEW
```

The action guard must apply this policy:

| Action | Dry run | Later approved live join |
| --- | --- | --- |
| Open dashboard | Allow | Allow |
| Find/read candidate card | Allow | Allow |
| Open candidate three-dot menu | Allow, then screenshot | Allow |
| Read `Launch Video Interview` option | Allow | Allow |
| Click `Launch Video Interview` | **Block** | Allow only after explicit join approval |
| Click consent-form `OK` | **Block** | Allow only after separate explicit consent approval |
| Click pre-call `Join` | **Block** | Allow only after separate explicit join approval |
| Hang up | **Always block** | **Always block in automation** |
| Fill feedback | Block | Block until Milestone 10 approval flow |
| Click `FINISH` | **Always block** | Allow only through a future separate approval token |

Opening a menu is reversible and permitted in dry-run. Clicking the launch,
join, hang-up, or finish controls is not.

#### Detailed dry-run flow

```text
1. Load configuration and health prerequisites.
2. Open the persistent Playwright profile at https://app.flocareer.com/.
3. Wait through login/loading using the existing stable-readiness logic.
4. Extract scheduled cards using the existing card/table parser.
5. Normalize the requested name for comparison only:
   - trim whitespace
   - collapse repeated spaces
   - case-insensitive comparison
   - do not use fuzzy matching for a live target
6. Require exactly one exact normalized match.
7. If there are zero matches, list sanitized available names and exit nonzero.
8. If there are multiple matches, stop and require a stronger selector such as
   candidate plus date/time; never choose the first match silently.
9. Scope all subsequent locators to the matched candidate card.
10. Save a `candidate_found.png` screenshot.
11. Locate and open only that card's three-dot menu.
12. Confirm exactly one visible `Launch Video Interview` menu item.
13. Record that the launch control exists, but route the attempted launch
    action through the guard and verify it is blocked by dry-run policy.
14. Save a `join_dry_run.png` screenshot with the menu visible.
15. Close the context without launching or joining.
16. Write `runs/<session_id>/action_log.jsonl` with requested action, guard
    decision, candidate identifier, timestamp, and screenshot path.
17. Print `Validation passed: launch control found and blocked by dry run`.
```

Do not log credentials, authentication tokens, full page HTML, or unrelated
candidate details.

#### Automated test plan

Develop in vertical TDD slices at the public workflow and guard seams:

```text
1. Candidate lookup finds one exact fictional candidate card.
2. Lookup is case-insensitive but does not fuzzy-match a different name.
3. Missing candidate returns a visible nonzero result.
4. Duplicate candidate names stop as ambiguous.
5. Candidate-scoped menu lookup does not select a neighboring card's menu.
6. Dry-run guard allows opening the menu.
7. Dry-run guard blocks LAUNCH_INTERVIEW.
8. Dry-run guard blocks CLICK_JOIN, HANG_UP, and FINISH_INTERVIEW.
9. Workflow writes an action log and screenshots.
10. A spy/fake page proves no launch or join click occurred.
```

Use fictional candidates and companies in every fixture.

#### Live validation plan

Run only when at least one scheduled interview card is visible:

```bash
uv run python main.py browser-scan --login-timeout 180
uv run python main.py join --candidate "Exact Visible Name" --dry-run
```

The human should watch the first live dry-run. The run passes only when:

```text
- The correct card is identified.
- The correct card's three-dot menu opens.
- `Launch Video Interview` is visible.
- No interview page or pre-call page opens.
- No Join, hang-up, feedback, or FINISH control is clicked.
- Candidate-found and dry-run screenshots are saved.
- action_log.jsonl records a BLOCK decision for LAUNCH_INTERVIEW.
```

If the FloCareer selector structure cannot be determined safely, save a
screenshot and stop. Do not introduce coordinate clicks or Computer Use in this
milestone merely to force progress.

#### Definition of done for the next session

```text
uv run pytest                    -> all tests pass
uv run ruff check .              -> pass
uv run ruff format --check .     -> pass
uvx ty check ...                 -> pass
uv run python main.py health     -> READY_FOR_BROWSER_SCAN
browser-scan                     -> scheduled cards found
join --dry-run                   -> correct candidate, menu found, launch blocked
git status -sb                   -> understood and intentional
```

Do not implement real launch/join in the same change unless the dry-run has
passed and the user starts a separate request explicitly authorizing it.

### Remaining execution order after join dry-run

The detailed requirements remain in Milestones 5 through 13 below. Continue in
this order and validate before advancing:

1. **Approved real join validation:** run one watched future interview through
   the separate Launch and pre-call Join approvals, plus Consent OK when the
   form appears; keep hang-up blocked.
2. **Question extraction:** capture question text, rubric, ideal answer, coding
   flag, and field locator hints into `questions.json`.
3. **Code editor:** enable only the requested coding card and verify candidate
   visibility from the toggle text.
4. **Offline session evaluation:** map saved transcript segments to questions,
   generate evidence-grounded per-question scores and final verdict files.
5. **LangGraph controller:** add explicit state transitions and human approval
   gates; start with simulation fixtures, not a live interview.
6. **Timer:** implement pure timer events and orchestration reactions before
   wiring it into a live session.
7. **Feedback autofill:** preview first, fill only after approval, and keep
   `FINISH` behind a separate final approval.
8. **Local dashboard:** expose already-tested backend actions without moving
   safety policy into UI button handlers.
9. **Qwen cloned voice:** test the local worker/voice quality independently, then click-to-speak,
   then interruptible duplex routing; never build a hidden clone mode.

### Required human inputs in later sessions

The human may need to provide or perform only the following:

- Sign in manually in the persistent FloCareer Chromium profile.
- Identify an exact scheduled candidate for a dry-run.
- Approve any future real launch and join action separately.
- Confirm organizational and candidate consent before synthetic voice use.
- Approve cloud candidate-data processing before OpenRouter is enabled.
- Review feedback evidence and explicitly approve any future final submission.

Credentials, OTPs, API keys, and private interview data should never be pasted
into the development conversation.

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
ornith-1.0-35b
google/gemma-4-12b
qwen/qwen3.6-35b-a3b
google/gemma-4-31b
google/gemma-4-31b-qat
nvidia/nemotron-3-nano-4b
tinyllama-1.1b-chat-v1.0
cohere-transcribe-03-2026-mlx
qwen3-asr-1.7b
```

Current tested local LLM policy:

```text
1. ornith-1.0-35b for fast follow-ups and feedback drafts
2. ornith-1.0-35b for deeper verdicts until session-level verdict tests exist
3. OpenRouter fallback on timeout, invalid output, or low confidence when allowed
```

Gemma and Qwen remain installed alternatives, but they are not the configured
default at this checkpoint.

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

Use the implemented local Qwen3-TTS loopback worker for cloned voice. Validate
its streamed PCM output independently, then connect it to a supervised call
only after browser control, candidate-only transcription, and answer evaluation
are reliable.

Expected Qwen local server:

```text
http://127.0.0.1:7789
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
    +-> Qwen3-TTS
    |       +-> Approved question/follow-up converted to cloned-voice PCM
    |       +-> Audio queue sends output to the selected virtual microphone
    |
    +-> Human approval
            +-> Review final recommendation and evidence
            +-> Approve feedback autofill
            +-> Approve final FINISH action
```

### Runtime conversation loop

Qwen3-TTS is the text-to-speech layer. It does not decide what to ask. The LLM
and LangGraph decide the next text; Qwen turns that text into streamed PCM
using the authorised private reference voice.

```text
1. Preflight
   -> Start LM Studio and configure OpenRouter fallback
   -> Start Apple Speech system-audio listener
   -> Start the local Qwen worker with the private reference configuration
   -> Start Playwright and join FloCareer
   -> Pre-generate the greeting and standard questions

2. Greeting
   -> LangGraph selects the approved introduction text
   -> Qwen synthesizes it as streamed cloned-voice PCM
   -> Audio queue plays it through the virtual microphone into FloCareer

3. Candidate introduction
   -> Qwen asks the candidate to introduce themselves
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
   -> Qwen converts next_spoken_text into cloned-voice PCM
   -> Audio is played through the virtual microphone
   -> Loop returns to LISTENING

6. Coding question
   -> Browser action router opens the code editor and selects the language
   -> Qwen reads the coding question and instructions
   -> Apple Speech captures the candidate's explanation
   -> Browser controller records editor state/code where technically available
   -> LLM evaluates explanation, approach, correctness, and complexity

7. Closing
   -> Qwen asks whether the candidate has questions
   -> Qwen plays the approved closing statement
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
generated voice as a candidate answer. Route Qwen directly to a virtual
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
│   ├── qwen_service.py
│   ├── qwen_client.py
│   ├── speech_bridge.py
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
LMSTUDIO_FAST_MODEL=ornith-1.0-35b
LMSTUDIO_DEEP_MODEL=ornith-1.0-35b

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
FLOCAREER_URL=https://app.flocareer.com/

# Local Qwen cloned voice (private reference paths are configured locally)
QWEN_TTS_BASE_URL=http://127.0.0.1:7789

# Runtime
RUNS_DIR=runs
DEFAULT_INTERVIEW_MINUTES=25
REQUIRE_APPROVAL_BEFORE_FINISH=true
```

Test after creating `.env`:

```bash
uv run python main.py config-dump
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
- Qwen is optional and reported as available/unavailable.

### Command

```bash
uv run python main.py health
```

### Expected output

```text
Health check
[OK] Python
[OK] Runs directory writable
[OK] Meeting_transcriber_with_LLM path found
[OK] Apple Speech adapter found
[OK] LM Studio reachable
[OK] Local models found: ornith-1.0-35b
[OK] LLM primary provider: lmstudio
[WARN] OpenRouter fallback disabled because cloud candidate data is not allowed
[OK] Playwright browser launch
[WARN] Qwen TTS worker not running

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
  "model": "ornith-1.0-35b",
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
uv run python main.py llm-test --provider lmstudio
uv run python main.py llm-test --provider openrouter
uv run python main.py llm-failover-test
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
uv run python main.py listen-test --seconds 60
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
uv run python main.py browser-scan --login-timeout 180
```

### Expected behavior

1. Open local Chrome/Chromium browser.
2. Navigate to:

```text
https://app.flocareer.com/
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
3. Request candidate-bound Launch approval and click `Launch Video Interview`.
4. Wait for either `Interviewer Consent Form` or a verified pre-call page.
5. If the form appears, save a screenshot, request a separate Consent approval,
   and click only the form's scoped `OK` button. If FloCareer has remembered a
   prior acknowledgement, do not attempt a consent click.
6. Wait for the `Joining as ...` pre-call page.
7. Request a separate Join approval and click `Join`.

Pre-call page:

```text
Joining as AS971
Join button
```

### Command

```bash
uv run python main.py join --candidate "Candidate Name" --live
```

Add dry-run support:

```bash
uv run python main.py join --candidate "Candidate Name" --dry-run
```

### Pass criteria

- Candidate row is found.
- Menu opens.
- `Launch Video Interview` is clicked only after its approval.
- The Interviewer Consent Form is verified and `OK` is clicked only after its
  separate approval.
- Join page is reached.
- `Join` button is clicked only after its separate approval.
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
uv run python main.py questions-scan --candidate "Exact Candidate Name"
```

### Expected output

```text
Extracted questions: 17
Coding question IDs: pending real coding-card HTML fixture
Questions JSON: runs/questions_scan_<timestamp>/questions.json
```

### Pass criteria

- Question numbers match UI.
- Question text is not truncated if expandable text is available.
- Coding-question detection remains pending the real coding-card HTML fixture.
- Screenshot saved for extraction debugging.

---

## 10. Milestone 7: Code Editor Automation

### Objective

Enable the FloCareer code editor for coding questions.

### Flow from screenshots

For a coding question:

1. Scroll to question card.
2. Click `Code Editor` tab.
3. Check toggle text without changing language or code.
4. If it says `SHOW CODE EDITOR TO CANDIDATE`, click it only after the
   candidate-and-question-bound approval.
5. Pass when it says `HIDE CODE EDITOR TO CANDIDATE`.

Meaning:

```text
SHOW CODE EDITOR TO CANDIDATE -> currently hidden, click to show
HIDE CODE EDITOR TO CANDIDATE -> currently visible, correct state
```

### Command

```bash
uv run python main.py join --candidate "Exact Candidate Name" --live \
  --enable-code-editor-question 9
```

### Expected output

```text
Candidate connected
Question 9 found
Code Editor tab opened
Candidate visibility: enabled
Screenshot saved
```

### Pass criteria

- Code editor tab is active.
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
uv run python main.py evaluate --session runs/test_session
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
uv run python main.py simulate-interview --session runs/test_session
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
uv run python main.py fill-feedback --session runs/<session_id>
```

Approval mode:

```bash
uv run python main.py fill-feedback --session runs/<session_id> --approve
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
uv run python main.py timer-demo --minutes 1
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
uv run python main.py ui
```

### Pass criteria

- Dashboard opens locally.
- Live transcript updates.
- Evaluation appears.
- Buttons call backend actions.
- No final submit without approval.

---

## 16. Milestone 13: Qwen Cloned Voice and Live Audio Routing

### Objective

Use the implemented local Qwen3-TTS worker as the interviewer voice for the
introduction, questions, follow-ups, coding instructions, time notices, and
closing statement. The worker conditions each request on the private,
authorised reference recording and exact reference transcript; it is not a
permanent fine-tune.

### Implemented voice path

```text
Ornith in LM Studio → completed sentence → local Qwen worker → PCM chunks
```

The Qwen worker binds only to `127.0.0.1` and exposes complete-WAV and
streamed-PCM endpoints. A streaming test has measured first Qwen PCM at about
half a second after receiving a complete sentence. The end-to-end first-audio
time also includes the time Ornith needs to produce its first sentence.

### Verify before any call integration

```bash
uv run python main.py health
uv run python main.py qwen-tts-stream-test \
  --text "Please explain your approach."
uv run python main.py llm-speak-stream-test \
  --prompt "Ask one concise Python question." --model-class fast
```

The current commands assemble the received PCM into a private WAV artifact for
listening and verification. They do not inject audio into FloCareer.

### Required live-audio architecture

```text
Qwen PCM → local playback adapter → INTERVIEWER_TO_CALL virtual microphone
candidate call audio → CANDIDATE_ONLY virtual loopback → Apple Speech
```

The two directions must remain separate. If Apple Speech receives the Qwen
output, it can transcribe the interviewer as the candidate and cause false
follow-ups. If the call microphone receives candidate/system audio, it can
create feedback. Installation and selection of a virtual audio device (for
example BlackHole or Loopback) are manual user-approved macOS setup steps.

### Pass criteria for live routing

- Qwen PCM is heard by the candidate through the selected call microphone.
- Apple Speech records candidate-only audio and no Qwen output.
- A local recording/WAV is available for troubleshooting without exposing it
  to Git.
- Starting candidate speech stops or cancels current Qwen playback.
- A failed device, unavailable stream, or ambiguous route fails closed and
  leaves the human able to continue the interview.
- Voice use follows required disclosure, consent, and platform policy.

Do not enable live audio injection until these tests pass in a supervised call.

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
Qwen speaks through the selected output.
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
Voice: local Qwen3-TTS with supervised routing later
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

Run tests in order and do not advance past a failed gate. The first six live
gates are complete at this checkpoint; later commands are planned and do not
exist yet.

```text
COMPLETED
1. uv run python main.py health
2. uv run python main.py llm-test --provider lmstudio
3. uv run python main.py llm-test --provider openrouter
4. uv run python main.py llm-failover-test
5. uv run python main.py listen-test --seconds 60
6. uv run python main.py browser-scan --login-timeout 180

NEXT
7. uv run python main.py join --candidate "Candidate Name" --dry-run

PLANNED AFTER JOIN DRY-RUN
8. uv run python main.py join --candidate "Candidate Name"
9. uv run python main.py extract-questions
10. uv run python main.py enable-code-editor --question 9 --language Python
11. uv run python main.py evaluate --session runs/test_session
12. uv run python main.py simulate-interview --session runs/test_session
13. uv run python main.py timer-demo --minutes 1
14. uv run python main.py fill-feedback --session runs/test_session
15. uv run python main.py ui
16. uv run python main.py qwen-tts-stream-test --text "Hello, let us start."
```

Do not move to the next test until the current one passes.

---

## 20. Live Interview Runbook

### Before interview

```bash
uv run python main.py health
uv run python main.py browser-scan --login-timeout 180
uv run python main.py llm-test --provider lmstudio
```

Open LM Studio and preload chosen model.

Recommended:

```text
ornith-1.0-35b
```

The same model is currently configured for deeper local verdict work:

```text
ornith-1.0-35b
```

The rest of this live runbook is a target workflow, not an available production
procedure. Do not use it until each referenced command has been implemented and
validated in the testing matrix.

### Start interview

```bash
uv run python main.py start-session --candidate "Candidate Name" --minutes 25
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
uv run python main.py enable-code-editor --question <id> --language Python
```

Or click UI button.

### After interview

```bash
uv run python main.py generate-final-verdict --session runs/<session_id>
uv run python main.py fill-feedback --session runs/<session_id>
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
Preload ornith-1.0-35b before the interview
Reduce context length
Use a smaller installed local model only after it passes the evaluator schema test
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

The first sprint is complete:

```text
[DONE] 1. Project skeleton
[DONE] 2. Health check
[DONE] 3. LLM provider interface
[DONE] 4. LM Studio evaluator test
[DONE] 5. OpenRouter evaluator test
[DONE] 6. LM Studio-to-OpenRouter failover test
[DONE] 7. Apple Speech listen-test
[DONE] 8. Browser scan dry run
```

Verified definition of done:

```text
health passes with READY_FOR_BROWSER_SCAN
LM Studio returns valid structured score JSON through ornith-1.0-35b
OpenRouter returns the same schema after mandatory PII redaction
failover test proves cloud-disabled blocking and allowed fallback routing
listen-test captures Chrome/system audio with microphone off
browser-scan lists scheduled FloCareer cards without launching interviews
44 automated tests, lint, formatting, type-checking, and compilation pass
```

The second sprint's join dry-run implementation is complete. Automated tests,
lint, formatting, type-checking, CLI help, and runtime health pass. A watched
live validation also passed against the real Material UI dashboard: the exact
candidate menu opened, the page did not navigate, and the action log recorded
`LAUNCH_INTERVIEW` as `BLOCK`.

The Launch-only `questions-scan` workflow is implemented and watched live
validation reached all 17 normal question records with sequential IDs, ideal
answers, and four rating guidelines each. Review then found multiline question
text needed a dedicated binding; the supplied `.clFESingleSugDet` HTML is now
used and covered by automated tests, pending the next watched revalidation. It
stores the result in `questions.json`. Coding-question detection awaits the
real coding-card HTML fixture and is not yet live-validated. It does not click
Join or change the code-editor visibility toggle, evaluation fields, hang-up,
or `FINISH`.

---

## 25. Handoff Prompt For A New Chat Session

Use this prompt in another session:

```text
Continue the supervised FloCareer interview copilot in:
https://github.com/akashiitd/flo_automation

First read README.md and the "Current Implementation Status (2026-07-12)"
section at the top of FLOCAREER_AUTOMATION_PLAN.md. Inspect git status and the
latest commits before editing.

Milestones 1-4 are complete and validated: health/config, LM Studio plus
OpenRouter structured evaluation, Apple Speech system-audio capture, and a
read-only persistent Playwright dashboard scan. The configured local model is
ornith-1.0-35b. Do not rebuild these components.

The guarded join dry-run is implemented and live-validated with typed action
policy, exact candidate matching, candidate-scoped menu selection, screenshots,
and an audit log. FloCareer's unlabelled Material UI cards and numeric menu IDs
have regression coverage. The live run opened the correct menu, stayed on the
dashboard, and logged LAUNCH_INTERVIEW as BLOCK.

The approved real Launch and Join workflow is implemented but has not been run
against a real scheduled interview. `join --candidate "Exact Name" --live`
requires separate candidate-bound, single-use approval phrases immediately
before Launch and Join, plus consent-form OK when the form appears. After Join,
it keeps the browser open until the human manually ends the interview and
confirms that it ended. Hang-up and FINISH are always blocked by the automation.

Use fictional candidate data in tests. Do not request credentials, OTPs, API
keys, or real interview content. Do not use coordinate clicks or Computer Use
to force selectors. Do not run live validation against an ended interview.
Select a future scheduled candidate and have the human watch the first run.
```

---

## 26. Immediate Next Action

Confirm coding-question detection on a future scheduled interview—not the
ended Adik Behera interview—without enabling the editor for the candidate:

```bash
uv run python main.py questions-scan --candidate "Exact Future Candidate"
```

Confirm Launch only when the correct scoped menu is visible. Verify that coding
question IDs are reported from the read-only `Code Editor` tab marker. The scan
must not click Join, switch the tab, or change the code-editor toggle. The
current automated baseline is:

```bash
git status -sb
uv run pytest
uv run ruff check .
uvx ty check app browser evaluator llm transcriber main.py
uv run python main.py health
```

Do not use an ended interview for validation. Automation must never click
hang-up or `FINISH`.
