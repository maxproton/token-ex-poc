# TEA V4 (multi-turn accumulation / "slow bleed") test harness

Individually small prompts that expand the conversation across many turns, exhausting the session
budget rather than any one request. No single message in the transcript looks anomalous on its
own -- the attack is only visible by watching how a session's prompt and completion sizes move
over its own recent history, which is why this is the only vector in this project whose state
persists across many separate `check()` calls for the same `session_id` rather than judging one
prompt at a time. The principal signal (per the paper) is session-level growth in prompt and
completion size.

Python 3.10+. `pip install -r requirements.txt`

## How this differs from V1-V3

V1-V3 each score one prompt (or one retrieval) in isolation; a new `session_id` with no history is
scored exactly the same as one with a hundred prior turns. V4 can't work that way by definition --
"slow bleed" only exists in the shape of a session over time. `detector.py`'s `Detector` keeps a
small per-session history (a bounded window of recent prompt token counts, recent completion token
counts, and a running cumulative total) and `guardrail.py`'s `Guardrail.check()` scores the current
turn against that history, not against the message content. A session needs `min_turns_for_trend`
turns before any growth check activates: judging turn 2 against turn 1 is noise, not a trend.

Four checks, each independently bandable in `guardrail.py`'s `BANDS`:
- **prompt_growth_ratio** -- mean prompt-token size of the recent half of the tracked window versus
  its earlier half. Robust to one outlier turn; catches an overall upward move within the window.
- **sustained_growth_streak** -- length of the unbroken run of non-decreasing turns ending at the
  current one, as a share of the window. This is the check aimed specifically at "slow bleed":
  nothing dramatic on any one turn, just steady, repeated growth every turn.
- **completion_growth_ratio** -- same trend check applied to completion token counts. Separate from
  prompt growth because an attacker can keep their own messages flat and still bleed the session by
  baiting ever-larger replies (see `list_accumulation` below).
- **cumulative_session_ratio** -- total prompt+completion tokens consumed by the session so far
  (plus the current turn) against a configured `session_token_budget`. Catches a session that burns
  its budget in a few large bursts rather than a steady climb, which the trend checks alone would
  under-score.

Since `completion_growth_ratio` and the cumulative check depend on completion size,
`Guardrail.record_completion(session_id, completion_tokens)` must be called once the model's reply
for an accepted turn is known -- `check()` alone only has the prompt side available.

Ordinary short back-and-forth chat has naturally small, noisy prompt sizes (a "2 tokens then 4" swing
reads as a 2x ratio from nothing more than one slightly longer sentence). `MIN_TOKENS_FOR_TREND`
in `guardrail.py` suppresses the prompt-side growth checks below a minimum size, the same fix V1
applies to its own baseline check and for the same reason.

## LM Studio setup
1. Load a model and set the context length to at least 16384.
2. Developer tab, start the server (default http://localhost:1234).
3. If the model identifier differs, set `LMSTUDIO_MODEL`.

## Run
    python guardrail.py

Type a message to extend the current CLI session, or `@path/to/transcript.txt` to replay every
line of a file as a separate turn in sequence (this is how the files under `examples/` are meant to
be run). Type `reset` to start a fresh session without restarting the process.

## Notes
- `examples/` holds one scripted multi-turn session per variant (`steady_growth`,
  `list_accumulation`, `stepped_budget_burn`), one line per turn, plus `benign_session.txt`, an
  ordinary short conversation that should stay allowed throughout. `steady_growth` and
  `stepped_budget_burn` are driven by prompt-side checks and block partway through on their own.
  `list_accumulation` keeps its own prompt text flat and only trips `completion_growth_ratio` if the
  model actually complies with the "repeat everything so far, then add one more" instruction each
  turn -- whether it does is a property of the model being tested, not of the detector, so this one
  is the least reliable to demonstrate without a live run against a compliant model.
- This is a first pass at the hardest vector in the taxonomy: the paper flags slow bleed as "the
  vector most likely to evade existing controls" precisely because no single message is anomalous.
  The trend/streak/budget checks here are simple and explainable by design, but a determined
  attacker who paces growth just under the window size or just under the growth thresholds would
  likely get through; this hasn't been adversarially tuned the way V1-V3 have.
- This folder is self-contained, same as `Vector 1 - Padding`, `Vector 2 - Recursive Prompting` and
  `Vector 3 - Context Stuffing`: no cross-vector imports.
