"""Attack runner. Sends padded prompts to the victim and records what the server reports.

Examples
  python -m tea_v1.run_attack                                   # single large requests, no cap
  python -m tea_v1.run_attack --sizes 500 1000 2000 4000 --positions prefix suffix
  python -m tea_v1.run_attack --mode distributed --n 10         # many requests just under the cap
"""
import argparse
import itertools
import json
import random
import statistics
import time
from datetime import datetime
from pathlib import Path

from . import padding
from .client import LMStudioClient, Victim
from .config import Config
from .tasks import SYSTEM_PROMPT, TASKS, check_answer
from .tokcount import make_counter


def calibrate(client, task, variants, probe_chars=1200):
    """Measure real tokens per character for each variant using the server's own usage numbers."""
    base = client.chat(SYSTEM_PROMPT, task.question)
    if base["status"] != "ok":
        raise SystemExit(f"Cannot reach the model: {base.get('error')}")
    tpc = {}
    for v in variants:
        pad = padding.generate(v, probe_chars, random.Random(0), "")
        r = client.chat(SYSTEM_PROMPT, padding.embed(task.question, pad, "prefix"))
        if r["status"] != "ok":
            raise SystemExit(f"Calibration failed for {v}: {r.get('error')}")
        tpc[v] = max((r["prompt_tokens"] - base["prompt_tokens"]) / len(pad), 0.01)
    return base["prompt_tokens"], tpc


def build_jobs(args, cap):
    tasks = TASKS[: args.tasks]
    jobs = [dict(variant="none", size=0, position="none", task=t, rep=r)
            for t in tasks for r in range(args.repeats)]
    if args.mode == "single":
        for v, s, p, t, r in itertools.product(args.variants, args.sizes, args.positions, tasks, range(args.repeats)):
            jobs.append(dict(variant=v, size=s, position=p, task=t, rep=r))
    else:
        size = int(cap * 0.9)
        for v in args.variants:
            for i in range(args.n):
                jobs.append(dict(variant=v, size=size, position="prefix", task=tasks[i % len(tasks)], rep=i))
    return jobs


def summarise(records):
    groups = {}
    for r in records:
        groups.setdefault((r["variant"], r["size"]), []).append(r)
    base = [r["prompt_tokens"] for r in groups.get(("none", 0), []) if r["prompt_tokens"]]
    base_tok = statistics.mean(base) if base else None
    base_lat = statistics.mean([r["latency_s"] for r in groups.get(("none", 0), [])] or [0]) or None
    print(f"\n{'variant':<14}{'target':>7}{'n':>4}{'ok':>4}{'rej':>4}{'tokens':>9}{'x base':>8}{'lat s':>8}{'x base':>8}{'acc':>6}")
    for (v, s), rs in sorted(groups.items()):
        ok = [r for r in rs if r["status"] == "ok"]
        rej = sum(1 for r in rs if r["status"] == "rejected_by_cap")
        toks = statistics.mean([r["prompt_tokens"] for r in ok]) if ok else float("nan")
        lat = statistics.mean([r["latency_s"] for r in ok]) if ok else float("nan")
        acc = statistics.mean([1.0 if r["correct"] else 0.0 for r in ok]) if ok else float("nan")
        xt = toks / base_tok if base_tok and ok else float("nan")
        xl = lat / base_lat if base_lat and ok else float("nan")
        print(f"{v:<14}{s:>7}{len(rs):>4}{len(ok):>4}{rej:>4}{toks:>9.0f}{xt:>8.1f}{lat:>8.2f}{xl:>8.1f}{acc:>6.2f}")


def main():
    cfg = Config()
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["single", "distributed"], default="single")
    ap.add_argument("--variants", nargs="+", default=padding.VARIANTS, choices=padding.VARIANTS)
    ap.add_argument("--sizes", nargs="+", type=int, default=[500, 2000, 8000], help="target padding size in tokens")
    ap.add_argument("--positions", nargs="+", default=["prefix"], choices=padding.POSITIONS)
    ap.add_argument("--tasks", type=int, default=2)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--n", type=int, default=10, help="requests per variant in distributed mode")
    ap.add_argument("--cap", type=int, default=None, help="victim input cap in tokens, 0 disables")
    ap.add_argument("--corpus-dir", default="corpus/padding")
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cap = args.cap if args.cap is not None else (cfg.victim_cap_tokens if args.mode == "distributed" else 0)
    cap = cap or None
    count, _ = make_counter(cfg.tokenizer)
    client = LMStudioClient(cfg)
    victim = Victim(client, count, SYSTEM_PROMPT, cap_tokens=cap)

    out = Path(args.out or Path(cfg.results_root) / datetime.now().strftime("%Y%m%d-%H%M%S"))
    out.mkdir(parents=True, exist_ok=True)
    print(f"Output: {out}  mode={args.mode}  cap={cap}  model={cfg.model}")

    base_tokens, tpc = calibrate(client, TASKS[0], args.variants)
    print(f"Calibration: base prompt {base_tokens} tokens; tokens/char = " + ", ".join(f"{k}:{v:.2f}" for k, v in tpc.items()))
    (out / "calibration.json").write_text(json.dumps({"base_tokens": base_tokens, "tokens_per_char": tpc}, indent=2))

    jobs = build_jobs(args, cap or cfg.victim_cap_tokens)
    records = []
    with open(out / "results.jsonl", "w") as fr, open(out / "prompts.jsonl", "w") as fp:
        for i, j in enumerate(jobs):
            jid = f"{i:04d}"
            rng = random.Random(f"{cfg.seed}-{jid}")
            pad = ""
            if j["variant"] != "none":
                pad = padding.generate(j["variant"], int(j["size"] / tpc[j["variant"]]), rng, args.corpus_dir)
            prompt = padding.embed(j["task"].question, pad, j["position"])
            res = victim.handle(prompt)
            rec = dict(id=jid, mode=args.mode, variant=j["variant"], size=j["size"], position=j["position"],
                       task=j["task"].id, chars=len(prompt), est_tokens=res.get("est_tokens"),
                       status=res["status"], prompt_tokens=res.get("prompt_tokens"),
                       completion_tokens=res.get("completion_tokens"), latency_s=round(res["latency_s"], 3),
                       correct=check_answer(j["task"], res.get("text", "")) if res["status"] == "ok" else None,
                       error=res.get("error"), ts=time.time())
            records.append(rec)
            fr.write(json.dumps(rec) + "\n"); fr.flush()
            fp.write(json.dumps({"id": jid, "variant": j["variant"], "size": j["size"], "position": j["position"],
                                 "prompt": prompt}) + "\n"); fp.flush()
            print(f"[{i + 1}/{len(jobs)}] {j['variant']:<13} size={j['size']:<6} {rec['status']:<16} "
                  f"prompt_tokens={rec['prompt_tokens']} lat={rec['latency_s']}s")
    summarise(records)


if __name__ == "__main__":
    main()
