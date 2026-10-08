# TEA V1 (volume and padding) test harness
## Low hanging fruit type

Python 3.10+. `pip install -r requirements.txt`

## LM Studio setup
1. Load mistral-small-3.2-24b-instruct-2506-mlx and set the context length to at least 16384 in the model load settings.
2. Developer tab, start the server (default http://localhost:1234).
3. If the model identifier differs, set `LMSTUDIO_MODEL`.

## Run
    python -m tea_v1.run_attack                               # cost curve, single large requests, no cap
    python -m tea_v1.run_attack --mode distributed --n 10     # many requests just under the 2000 token cap
    python -m tea_v1.evaluate --results results/<run>         # detector, with and without the baseline check

Dry run without the model: `python mock_lmstudio.py` in a second terminal, then run the commands above.

## Notes
- Cost is always the server-reported `usage`. The local tokenizer is only used for the victim cap and the detector.
- Default tokenizer is `approx` (chars / 4), which disables the tokens_per_char check. For real numbers set
  `TEA_TOKENIZER=hf:<local path or repo of the model's tokenizer>` and re-run evaluate.
- Padding sizes are calibrated per variant from the server's own token counts before the sweep.
- `natural` padding uses seeded word salad unless you put .txt files in `corpus/padding/`.
- The built-in legitimate set is a smoke test. Add real long inputs as `corpus/legit/<category>/*.txt`.
- Flagged or blocked input never updates a session baseline.
