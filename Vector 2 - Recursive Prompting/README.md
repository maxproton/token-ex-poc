# TEA V2 (recursive prompting) test harness

Prompts that force the model to process its own output, or to expand without bound, before it
answers. Unlike V1's padding, the attack here isn't bulk filler the detector can measure by shape --
it's a short, structurally sound instruction that asks the model to loop, critique-and-revise, or
chain steps off its own prior output indefinitely. The principal signals (per the paper) are
instruction nesting depth and runaway generation.

Python 3.10+. `pip install -r requirements.txt`

## Two-sided guard

- **Input side** (`detector.py` + `guardrail.py`): scores the prompt itself before the model is
  called, for stacked/nested recursive directives (`nesting_depth`), unbounded-continuation
  language (`runaway_phrases`), and references to the model's own prior output
  (`self_reference`). V1's detector already covers repetition, compression, unicode anomalies and
  session baseline (shared input-layer checks in the paper's Table 2) -- this one only adds what's
  specific to V2.
- **Output side** (`client.py`'s `chat_guarded`): input inspection alone can't predict what the
  model will actually do with an accepted prompt, so the reply is streamed and re-checked at fixed
  token checkpoints (`checkpoint_tokens`). The stream is aborted mid-generation if: completion
  tokens exceed `predicted_ceiling_tokens` by `ceiling_block_mult`; generated text degenerates into
  repetition or becomes highly compressible; or tokens/second spikes anomalously. `<think>...</think>`
  reasoning content is tracked separately from the visible answer, since a runaway reasoning trace
  and a runaway answer are distinct failure patterns.

## LM Studio setup
1. Load a model and set the context length to at least 16384.
2. Developer tab, start the server (default http://localhost:1234).
3. If the model identifier differs, set `LMSTUDIO_MODEL`.

## Run
    python guardrail.py                      # interactive CLI: scores a prompt, then streams the guarded reply

Dry run without the model: `python mock_lmstudio.py` in a second terminal (streams a normal short
reply, or an endless degenerate reply for any prompt containing the word "loop"), then run the
command above with `LMSTUDIO_URL=http://127.0.0.1:1234/v1`.

## Notes
- `examples/` holds one generated attack per variant (`nested_instruction`, `self_feedback_loop`,
  `unbounded_expansion`, `chained_directives`), each wrapping a real question from `tasks.py`. All
  four score above the reject threshold; the bare question underneath does not.
- Default tokenizer is `approx` (chars / 4). For real numbers set
  `TEA_TOKENIZER=hf:<local path or repo of the model's tokenizer>`.
- This folder is self-contained, same as `Vector 1 - Padding`: no cross-vector imports.
