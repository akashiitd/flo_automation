# FloCareer Interview Copilot

A local-first, supervised copilot for FloCareer interviewers on macOS. The
project combines read-only browser scanning, Apple Speech system-audio
transcription, structured LLM evaluation, guarded cloud fallback, and
per-session audit files.

The copilot is intentionally human-controlled. Launch is available only after
an exact candidate-bound approval. The implementation never clicks hang-up,
fills feedback, changes ratings, or submits `FINISH`.

The complete roadmap and safety constraints are documented in
[`FLOCAREER_AUTOMATION_PLAN.md`](FLOCAREER_AUTOMATION_PLAN.md).

## Current status

| Capability | Status | Command |
| --- | --- | --- |
| Configuration validation | Implemented | `uv run python main.py config-dump` |
| Local environment health check | Implemented | `uv run python main.py health` |
| LM Studio structured evaluation | Implemented | `uv run python main.py llm-test --provider lmstudio` |
| OpenRouter structured evaluation | Implemented, explicit opt-in | `uv run python main.py llm-test --provider openrouter` |
| Guarded provider failover | Implemented | `uv run python main.py llm-failover-test` |
| Apple Speech generic system-audio capture | Historically validated; candidate-only mode is blocked safely | — |
| Persistent local cloned voice service | Implemented | `uv run python main.py qwen-tts-test --text "..."` |
| Local LM Studio → Qwen speech bridge | Implemented | `uv run python main.py llm-speak-test --prompt "..."` |
| Streamed Qwen PCM generation | Implemented and live-validated | `uv run python main.py qwen-tts-stream-test --text "..."` |
| Local Qwen PCM playback / Loopback bus diagnostics | Implemented and live-validated | `uv run python main.py qwen-tts-playback-test --text "..."` |
| Candidate-only Apple Speech capture | Blocked safely pending transcriber device-selection support | — |
| Interview turn controller and barge-in | Not implemented | — |
| Read-only FloCareer dashboard scan | Implemented | `uv run python main.py browser-scan` |
| Guarded candidate join discovery | Implemented and live-validated | `uv run python main.py join --candidate "Exact Name" --dry-run` |
| Approved real Launch, Join, and candidate wait | Implemented; live validation pending | `uv run python main.py join --candidate "Exact Name" --live` |
| Approved normal-question extraction | Implemented; 17-card live scan completed, final multiline fix pending revalidation | `uv run python main.py questions-scan --candidate "Exact Name"` |
| Coding-question detection and DOM capture | Implemented against semantic fixtures; watched revalidation pending | `uv run python main.py questions-scan --candidate "Exact Name"` |
| Guarded code-editor visibility | Integrated with the persistent live Join session; watched validation pending | `uv run python main.py join --candidate "Exact Name" --live --enable-code-editor-question 9` |
| Feedback fill and final submit | Not implemented | — |

## Safety model

- FloCareer authentication is completed manually in a persistent local browser.
- The scanner reads visible dashboard content and saves a screenshot; it does
  not open candidate menus or launch interviews.
- Candidate-only capture must use the manually configured `CANDIDATE_ONLY`
  Loopback input. Generic system-audio capture is blocked while the external
  Apple Speech helper lacks selected-device support.
- LM Studio is the local primary provider.
- OpenRouter is blocked unless both an API key and explicit cloud-data consent
  are configured.
- Candidate PII is redacted before every permitted OpenRouter generation.
- Runtime transcripts, screenshots, browser profiles, and `.env` are ignored by
  Git.
- A human remains responsible for interview questions, feedback, hiring
  decisions, and final submission.

## Requirements

- macOS on Apple Silicon
- Python 3.12 or newer
- [`uv`](https://docs.astral.sh/uv/)
- LM Studio with its local API server available at `http://127.0.0.1:1234/v1`
- The existing `Meeting_transcriber_with_LLM` project with its Apple Speech
  helper
- macOS permissions for Speech Recognition and Screen & System Audio Recording
- Optional: an OpenRouter API key for guarded cloud fallback

The current tested local model is `ornith-1.0-35b`.

## Installation

```bash
git clone https://github.com/akashiitd/flo_automation.git
cd flo_automation

uv sync
uv run playwright install chromium
cp .env.example .env
```

Edit `.env` locally. At minimum, verify the path to the existing transcription
project:

```env
MEETING_TRANSCRIBER_PATH=../Meeting_transcriber_with_LLM
```

If it is not a sibling directory, use an absolute path. Never commit `.env`.

## Configuration

Important settings in `.env`:

```env
# Local provider
LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1
LMSTUDIO_FAST_MODEL=ornith-1.0-35b
LMSTUDIO_DEEP_MODEL=ornith-1.0-35b

# Cloud fallback: disabled until explicitly permitted
OPENROUTER_API_KEY=
LLM_ALLOW_CLOUD_CANDIDATE_DATA=false

# Existing Apple Speech project
MEETING_TRANSCRIBER_PATH=../Meeting_transcriber_with_LLM
TRANSCRIPTION_BACKEND=apple-speech
TRANSCRIBE_SYSTEM_AUDIO=true
TRANSCRIBE_MICROPHONE=false

# FloCareer and safety
FLOCAREER_URL=https://app.flocareer.com/
REQUIRE_APPROVAL_BEFORE_FINISH=true

# Local Qwen cloned-voice worker
QWEN_TTS_BASE_URL=http://127.0.0.1:7789
QWEN_TTS_TIMEOUT_SECONDS=45

# Exact manual Loopback device names. The application never changes macOS defaults.
INTERVIEWER_AUDIO_OUTPUT_DEVICE=INTERVIEWER_TO_CALL
CANDIDATE_AUDIO_INPUT_DEVICE=CANDIDATE_ONLY
```

To allow OpenRouter, set both values locally:

```env
OPENROUTER_API_KEY=your-key
LLM_ALLOW_CLOUD_CANDIDATE_DATA=true
```

The application never prints the key. Redaction is deterministic but should
not be treated as a substitute for organizational approval to use cloud
processing.

## Usage

Run commands from the repository root.

### 1. Validate configuration

```bash
uv run python main.py config-dump
```

The output reports whether secrets are configured without displaying them.

### 2. Check local readiness

Start the LM Studio API server, load the configured model, then run:

```bash
uv run python main.py health
```

Expected final status:

```text
Overall: READY_FOR_BROWSER_SCAN
```

Voice services are optional for browser-only work. Qwen is the cloned-voice
service; Supertonic may appear as an unavailable optional warning.

### 3. Test local evaluation

```bash
uv run python main.py llm-test --provider lmstudio
```

The command returns a validated score, rating, transcript evidence, follow-up,
feedback, confidence, latency, tokens, and estimated cost.

### 4. Test OpenRouter

Only after cloud processing has been approved and enabled:

```bash
uv run python main.py llm-test --provider openrouter
uv run python main.py llm-failover-test
```

Provider usage is appended to:

```text
runs/llm_tests/llm_usage.jsonl
```

### 5. Start the persistent local cloned-voice worker

Qwen is the cloned-voice engine. It runs in its own MLX-Audio Python
environment and keeps the model loaded between requests. Keep the reference
recording and its exact transcript private and outside Git.

Save the transcript in a private local text file, then start the loopback-only
service from the repository root:

```bash
export QWEN_TTS_REFERENCE_AUDIO=/absolute/private/reference.wav
export QWEN_TTS_REFERENCE_TEXT_FILE=/absolute/private/reference.txt

HF_HUB_OFFLINE=1 HF_HUB_DISABLE_XET=1 \
  "$HOME/.local/share/uv/tools/mlx-audio/bin/python" \
  -m tts.qwen_service --host 127.0.0.1 --port 7789
```

`HF_HUB_OFFLINE=1` ensures the service uses the already downloaded local
model. The worker binds only to `127.0.0.1`; do not expose it to the network.

Verify the worker and synthesize supplied text:

```bash
uv run python main.py health
uv run python main.py qwen-tts-test --text "Please explain your approach."
uv run python main.py qwen-tts-stream-test --text "Please explain your approach."
```

The returned WAV is saved under `runs/qwen_tts_<timestamp>/`. The service
receives only text to speak; it does not receive FloCareer controls, browser
data, or credentials.

### 6. Test the local LM Studio → Qwen path

This command streams local LM Studio text, sends each completed sentence to
Qwen, and writes the returned WAV chunks privately:

```bash
uv run python main.py llm-speak-test \
  --prompt "Ask one concise Python question." --model-class fast
```

```text
Ornith in LM Studio → completed sentence → Qwen local service → WAV chunk
```

On the current Mac, a warm one-sentence smoke test completed in about 17
seconds end-to-end for 8.24 seconds of generated speech. The first request
after starting either service can take longer. This bridge creates audio only;
routing it into FloCareer and handling candidate barge-in remain separate work.

For low-latency playback, use the PCM streaming commands instead. Qwen emits
small audio chunks while it is still synthesizing the sentence:

```bash
uv run python main.py qwen-tts-stream-test --text "Please explain your approach."
uv run python main.py llm-speak-stream-test \
  --prompt "Ask one concise Python question." --model-class fast
```

On the current Mac, Qwen emitted its first PCM chunk in about 0.5 seconds after
it received a sentence. A local Ornith → Qwen smoke test reached first audio in
about 15 seconds, dominated by Ornith reaching its first sentence boundary.
The commands also assemble a WAV artifact for inspection; a future audio player
or virtual-microphone route can play the PCM chunks as they arrive.

### 7. Verify the manual Loopback buses and play local Qwen PCM

Loopback is the selected, manually configured routing layer. The verified
devices are both 48 kHz stereo:

```text
Qwen player → INTERVIEWER_TO_CALL (Pass-Thru only) → FloCareer microphone
Google Chrome for Testing → CANDIDATE_ONLY (no Pass-Thru) → candidate capture
```

The diagnostics command only reads CoreAudio device state. It never changes
macOS sound settings:

```bash
uv run python main.py audio-devices
```

After the Qwen worker is running, this supervised local smoke command plays
streamed PCM to `INTERVIEWER_TO_CALL` while retaining a private WAV artifact:

```bash
uv run python main.py qwen-tts-playback-test \
  --text "Please explain your approach."
```

The direct Qwen playback smoke test has passed with eight PCM chunks written to
`INTERVIEWER_TO_CALL`. Do not select the device in FloCareer or inject audio
into a real interview until a supervised test call has passed. The current
external Meeting Transcriber does not yet accept a selected
`system_audio_device`; `listen-test` fails closed rather than capture ambiguous
generic system audio. Update that external component to accept and use the
exact `CANDIDATE_ONLY` device before candidate-only transcription is enabled.

### 8. Candidate-only Apple Speech capture

```bash
uv run python main.py listen-test --seconds 60
```

This command is intentionally blocked until the external transcriber accepts
the selected `CANDIDATE_ONLY` device. Once that support is implemented, play
English speech from Google Chrome for Testing. The command passes only when at
least one `system` segment
is captured and no `microphone` segment is present. Press `Control+C` to stop
early.

Output is saved under:

```text
runs/listen_<timestamp>/transcript.json
runs/listen_<timestamp>/transcript.txt
```

### 6. Scan the FloCareer dashboard

```bash
uv run python main.py browser-scan --login-timeout 180
```

The scanner initially opens `https://app.flocareer.com/` in a persistent local
Chromium profile. Complete authentication manually in that browser. FloCareer
may redirect to `/interviewer/` after login.

The command extracts scheduled interview cards and saves:

```text
runs/browser_scan_<timestamp>/screenshots/dashboard.png
```

It does not click the three-dot menu or launch an interview.

### 7. Validate a candidate's launch control safely

After confirming the exact visible candidate name with `browser-scan`, run:

```bash
uv run python main.py join --candidate "Exact Candidate Name" --dry-run
```

The command requires `--dry-run`. It selects exactly one case-insensitive exact
name match, opens only that candidate card's menu, confirms one visible
`Launch Video Interview` control, and proves the launch action is blocked. It
never clicks Launch or Join. Missing names do not fuzzy-match and duplicate
names stop as ambiguous.

Artifacts are saved under `runs/join_<timestamp>/`:

```text
screenshots/candidate_found.png
screenshots/join_dry_run.png
action_log.jsonl
```

### 8. Launch, accept consent, and Join with separate approvals

Use only for a future scheduled interview while watching the browser:

```bash
uv run python main.py join --candidate "Exact Candidate Name" --live
```

The command pauses before Launch and prints a candidate-bound approval phrase.
After Launch it safely accepts either FloCareer path: when the `Interviewer
Consent Form` appears, it verifies and screenshots the form and requires a
separate approval before clicking its scoped `OK`; when consent was previously
acknowledged, FloCareer may open the verified `Joining as ...` pre-call page
directly and no consent click is attempted. Join always requires its own
approval. Each phrase is single-use and is never written to the action log.
After Join, the browser remains open until you manually end the interview and
type the displayed end confirmation. Automation never clicks hang-up or
`FINISH`.

Live-session artifacts are saved under `runs/join_live_<timestamp>/`:

```text
screenshots/candidate_found.png
screenshots/launch_approval.png
screenshots/consent.png
screenshots/pre_call.png
screenshots/joined.png
action_log.jsonl
```

### 9. Read and expand questions without joining

Run this while watching the browser:

```bash
uv run python main.py questions-scan --candidate "Exact Candidate Name"
```

The command requires the candidate-bound Launch approval and, only when shown,
a separate Consent OK approval. It then expands visible question cards and
saves their full text, ideal answers, rating guidelines, and locator hints.
A watched run reached all 17 normal cards. Full multiline extraction now binds
to FloCareer's supplied `.clFESingleSugDet` structure and has automated
coverage; that final correction awaits the next watched revalidation.
Semantic coding-question detection and automatic read-only DOM capture have
fixture coverage and await watched revalidation. The command never selects a
language, opens or changes an editor, or enables `SHOW CODE EDITOR TO
CANDIDATE`. It never clicks Join, feedback, ratings, `Mark as`, hang-up, or
`FINISH`.

Artifacts are saved under `runs/questions_scan_<timestamp>/`:

```text
questions.json
code_editor_dom.json
screenshots/questions_expanded.png
action_log.jsonl
```

`code_editor_dom.json` is generated automatically and read-only. For every
question exposing a semantic `Code Editor` tab, it records the question-number
evidence, exact tab count, SHOW/HIDE labels, switch-like control candidates,
their rendered state, the nearest control wrapper, and the encompassing card
structure. Each observation is classified as `unique`, `none`, or `ambiguous`.
Hidden-but-mounted controls are captured without opening the tab. Neither the
tab nor any candidate control is clicked during discovery.

To capture the post-tab visibility state without changing candidate visibility,
opt in to reversible scoped navigation:

```bash
uv run python main.py questions-scan --candidate "Exact Candidate Name" \
  --inspect-code-editor-tabs
```

Only exact coding-question `Code Editor` tabs are opened; the `Question` tab is
restored after capture. Browser-only guarded commands do not require LM Studio
to be running.

Structural HTML is allowlist-redacted and capped at 50,000 characters per
snapshot, with truncation and SHA-256 metadata. Unknown number layouts remain
`unresolved` rather than being guessed. The file is written with owner-only
permissions.

This private diagnostic can contain interview content and DOM attributes. Keep
it under ignored `runs/`; do not publish it or paste it into issues or chat.

### 10. Guarded code-editor visibility module

The isolated `browser/code_editor_workflow.py` guard/state module is implemented
for later stitching into the persistent live-session controller. It requires
the candidate binding established only after the exact-match joined-room
transition succeeds, scopes navigation to one exact question-number element,
opens one exact `Code Editor` tab, and interprets the visible state label
conservatively:

```text
SHOW CODE EDITOR TO CANDIDATE -> currently hidden
HIDE CODE EDITOR TO CANDIDATE -> currently visible; do not click
```

Showing a hidden editor requires a single-use approval bound to the candidate
identifier and question ID. The state is revalidated after the operator pause,
and the action passes only after the same card remains stably in the `HIDE...`
state. Ambiguous cards, labels, tabs, or semantic switch controls fail closed.
The module does not guess an unlabelled visual control, use coordinates, select
a language, type code, click hang-up, or click `FINISH`.

The verified card-scoped switch contract is
`input[type="checkbox"][name^="codeSwitch-"]`; it is always scoped to the
already-verified question card. Candidate-visible actions are available only
inside the persistent `join --live` session, after the candidate is connected:

```bash
uv run python main.py join --candidate "Exact Candidate Name" --live \
  --enable-code-editor-question 9
```

Launch, optional consent, Join, and showing the editor each require their own
fresh approval. The live session records `LAUNCHED`, `INTERVIEWER_IN_ROOM`,
`WAITING_FOR_CANDIDATE`, and `CANDIDATE_CONNECTED` transitions in
`room_state_log.jsonl`. The workflow remains watched-validation only until it
has passed against a future scheduled interview.

## Validation and development

```bash
uv run pytest
uv run ruff check .
uv run ruff format --check .
uvx ty check app browser evaluator llm transcriber main.py
```

The project currently has unit/integration coverage for configuration, health
checks, structured schemas, JSON repair, provider behavior, PII redaction,
failover policy, transcript persistence, Apple Speech integration, delayed
authentication states, and FloCareer table/card extraction.

## Session data and privacy

Runtime files live under `runs/<session_id>/`. Depending on the command, a
session may contain transcripts, LLM usage records, and dashboard screenshots.
These files can contain candidate or interview information and are excluded
from Git by default.

Before sharing logs, redact candidate names, contact details, resume fields,
account identifiers, and unnecessary interview content.

## Troubleshooting

### Apple Speech captures nothing

In **System Settings → Privacy & Security**, enable:

- Speech Recognition
- Screen & System Audio Recording
- Microphone only if a future workflow explicitly requires it

Then play audible speech from another Mac application during `listen-test`.

### LM Studio is unreachable or slow

- Start the LM Studio Developer/API server.
- Confirm `LMSTUDIO_BASE_URL=http://127.0.0.1:1234/v1`.
- Load the configured model before a live session.
- Reduce context length if latency is too high.

### OpenRouter is blocked

Both `OPENROUTER_API_KEY` and
`LLM_ALLOW_CLOUD_CANDIDATE_DATA=true` are required. Leaving cloud processing
disabled is a supported local-only configuration.

### FloCareer reports that you are logged out

Run `browser-scan` in headed mode and complete login manually in the opened
Chromium window. The session is stored in `.browser-profile/` for later scans.
Do not send credentials or OTPs through logs or chat.

### Dashboard loads but no candidates are extracted

Confirm that scheduled interview cards are visibly present. The command reports
validation as incomplete when no cards are visible, rather than treating an
empty or loading shell as a pass.

## Roadmap

The next guarded milestones are:

1. Revalidate the guarded Join, question scan, and code-editor flow while a
   human watches a scheduled interview.
2. Build a local audio-output adapter that plays Qwen PCM as it arrives and
   routes it to a deliberately selected virtual microphone for the call.
3. Build a separate candidate-only loopback input for Apple Speech, so Qwen's
   own voice is not transcribed as a candidate answer.
4. Add a stateful interview controller: introduction, ordered questions,
   candidate turn, transcript, rubric evaluation, follow-up, and next question.
5. Add pause/cancel behaviour when the candidate starts speaking, then test
   this full-duplex behaviour in a real call.
6. Add timer and full-session evaluation, followed by human-approved feedback
   preview/autofill.

The detailed next-session handoff is stored outside Git at
`/private/tmp/FLOCAREER_NEXT_SESSION_HANDOFF.md`. Qwen is the cloned-voice
runtime. Supertonic is optional and is not required for the current plan.

Final submission remains outside unattended automation.
