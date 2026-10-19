"""Replay saved attack prompts and legitimate samples through the detector.

No model calls are needed: detection runs on the prompt text, and cost savings come from the token counts
the server reported during the attack run.

  python -m tea_v1.evaluate --results results/<run>
"""
import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

from .classifier import CLASSIFIER_MODEL, ClassifierUnavailable, classify_coherence, loaded_classifier
from .config import Config
from .detector import ALLOW, BLOCK, FLAG, Detector, DetectorConfig
from .guardrail import read_prompt
from .legit_samples import WARMUP_PROMPTS, load
from .tokcount import make_counter


def run_config(name, det_cfg, count, reliable, attacks, legit, tokens_by_id):
    rows = []

    def fresh_session(text):
        d = Detector(count, reliable, det_cfg)
        sid = "s"
        for w in WARMUP_PROMPTS:
            d.inspect(w, sid)
        return d.inspect(text, sid)

    for a in attacks:
        dec = fresh_session(a["prompt"])
        rows.append(dict(config=name, kind="attack", group=f"{a['variant']}", size=a["size"], id=a["id"],
                         level=dec.level, fired=";".join(dec.fired()),
                         tokens=tokens_by_id.get(a["id"]) or dec.tokens))
    for cat, texts in legit.items():
        for i, t in enumerate(texts):
            dec = fresh_session(t)
            rows.append(dict(config=name, kind="legit", group=cat, size=dec.tokens, id=f"{cat}-{i}",
                             level=dec.level, fired=";".join(dec.fired()), tokens=dec.tokens))
    return rows


def report(rows):
    for cfg_name in sorted({r["config"] for r in rows}):
        sub = [r for r in rows if r["config"] == cfg_name]
        print(f"\n=== {cfg_name} ===")
        att = defaultdict(list)
        for r in sub:
            if r["kind"] == "attack" and r["group"] != "none":
                att[(r["group"], r["size"])].append(r)
        print(f"{'attack variant':<16}{'size':>7}{'n':>4}{'flag+':>8}{'block':>8}")
        for (g, s), rs in sorted(att.items()):
            n = len(rs)
            print(f"{g:<16}{s:>7}{n:>4}{sum(r['level'] >= FLAG for r in rs) / n:>8.2f}"
                  f"{sum(r['level'] == BLOCK for r in rs) / n:>8.2f}")
        ba = [r for r in sub if r["kind"] == "attack" and r["group"] != "none"]
        if ba:
            total = sum(r["tokens"] for r in ba) or 1
            blocked = sum(r["tokens"] for r in ba if r["level"] == BLOCK)
            print(f"attack input tokens blocked: {blocked}/{total} ({blocked / total:.0%})")
        leg = defaultdict(list)
        for r in sub:
            if r["kind"] == "legit":
                leg[r["group"]].append(r)
        print(f"\n{'legit category':<16}{'n':>4}{'flag+':>8}{'block':>8}  fired")
        for g, rs in sorted(leg.items()):
            n = len(rs)
            fired = sorted({f.split("=")[0] for r in rs for f in r["fired"].split(";") if f})
            print(f"{g:<16}{n:>4}{sum(r['level'] >= FLAG for r in rs) / n:>8.2f}"
                  f"{sum(r['level'] == BLOCK for r in rs) / n:>8.2f}  {','.join(fired)}")


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default=None)
    ap.add_argument("--legit-dir", default="corpus/legit")
    ap.add_argument("--classify", default=None,
                     help="Run the LLM coherence classifier on one prompt and exit, skipping the "
                          "--results pipeline. Use @path/to/file.txt to load a long prompt from a file.")
    ap.add_argument("--classifier-model", default=CLASSIFIER_MODEL,
                     help="Model id loaded in LM Studio to use as the coherence classifier.")
    args = ap.parse_args()

    if args.classify is not None:
        text = read_prompt(args.classify)
        try:
            with loaded_classifier(args.classifier_model, cfg) as client:
                result = classify_coherence(client, text)
        except ClassifierUnavailable as e:
            raise SystemExit(f"Classifier unavailable: {e}")
        if result["error"]:
            raise SystemExit(f"Classifier call failed: {result['error']}")
        print(f"verdict: {result['label']}  (raw reply: {result['raw']!r})")
        return

    if not args.results:
        raise SystemExit("--results is required unless --classify is given")

    res_dir = Path(args.results)
    attacks = [json.loads(l) for l in open(res_dir / "prompts.jsonl")]
    results = [json.loads(l) for l in open(res_dir / "results.jsonl")]
    tokens_by_id = {r["id"]: r["prompt_tokens"] or r["est_tokens"] for r in results}
    legit = load(args.legit_dir)
    count, reliable = make_counter(cfg.tokenizer)
    if not reliable:
        print("Note: tokenizer is 'approx', so the tokens_per_char check is disabled.")

    rows = []
    rows += run_config("stateless", DetectorConfig(enable_baseline=False), count, reliable, attacks, legit, tokens_by_id)
    rows += run_config("full", DetectorConfig(enable_baseline=True), count, reliable, attacks, legit, tokens_by_id)
    report(rows)
    with open(res_dir / "evaluation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nPer-item decisions: {res_dir / 'evaluation.csv'}")


if __name__ == "__main__":
    main()
