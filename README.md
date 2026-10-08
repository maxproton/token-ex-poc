# Token Exhaustion Attack (TEA)

Research code and paper for the Token Exhaustion Attack: the deliberate consumption of tokens to cause economic or operational harm to an LLM-backed system or its legitimate users.

Author: Sebastian Wallace, [Breaking Bytes](https://breakingbytes.org)

The paper is in this directory: [`Token Exhaustion Attacks - Research Paper.pdf`]. It covers the concept, threat model, vector taxonomy and defence architecture. This repository holds the experiments that test it.

---

## The idea

LLM systems bill, throttle and fail in tokens. That makes the token a security-relevant resource, yet token consumption is still mostly handled as a cost governance problem: budgets, alerts and per-request caps.

TEA argues that this is not enough. An attacker does not need an exploit or a policy violation. They need a cheap input that provokes an expensive computation, and a defender who has only limited each request in isolation. Each prompt can be benign in content and each request can sit inside its limits while the system is still being drained.

Three properties of LLM systems make per-request limits insufficient:

1. **Cost accumulates across a session.** Conversation history is resent every turn, so total input grows roughly with the square of the number of turns.
2. **Context is expensive.** Attention cost grows with sequence length, so inflated context compounds.
3. **Cost is not proportional to prompt length.** A short prompt can trigger long reasoning or output, and a long prompt can be cheap.

The same shape as a classic denial of service attack applies, expressed in a currency organisations currently manage as a budget line. TEA proposes treating it with the same engineering discipline: designed in, tested, monitored.

### What TEA is and is not

TEA is **not** a claim that resource exhaustion on LLMs was undiscovered. The foundations are well established in the literature (see [Related work](#related-work)). The contribution is:

- A security-first definition and threat model.
- A taxonomy of seven vectors organised by the mechanism of token consumption.
- A layered defence architecture covering input, session, output, model routing and observability.
- Experiments that measure each vector and test whether the proposed detection works, including its false positive rate against legitimate traffic.

---

## Threat model

- **Attacker:** can submit prompts through any legitimate interface: a public chat, an API, a document upload feeding retrieval, or content that a tool-using agent ingests. No access to weights or infrastructure.
- **Attacker advantage:** asymmetry. A small, cheap input causes a large, expensive computation.
- **Defender difficulty:** attribution. Adversarial exhaustion and legitimate high-cost use can look identical, so a defence that cannot separate them either misses attacks or penalises customers.
- **Harms:** inflated cost, degraded availability, displaced legitimate users, pressure on shared infrastructure.

---

## The seven vectors

| # | Folder | Vector | Mechanism | Principal detection signal |
|---|---|---|---|---|
| 1 | [`Vector1`](./Vector1) | Volume and padding | Filler or copy-paste flooding inside the prompt | Repetition, low information density, size against baseline |
| 2 | [`Vector2`](./Vector2) | Recursive prompting | Prompts that make the model process its own output or expand without bound | Instruction nesting depth, runaway generation |
| 3 | [`Vector3`](./Vector3) | Context window stuffing | Duplicated, oversized or irrelevant content pushed in through retrieval or long-context features | Abnormal or duplicate retrieval chunks |
| 4 | [`Vector4`](./Vector4) | Multi-turn accumulation (slow bleed) | Small prompts that grow the conversation over many turns | Session-level growth in prompt and completion size |
| 5 | [`Vector5`](./Vector5) | Complexity inflation | Short prompts engineered to force expensive processing or extended reasoning | Mismatch between prompt length and reasoning effort |
| 6 | [`Vector6`](./Vector6) | Output shape manipulation | Instructions that force exhaustive or repeated output without changing the reasoning | Completion length against a predicted ceiling |
| 7 | [`Vector7`](./Vector7) | Selector gaming | Manipulating perceived complexity to force repeated, costly model switches | Switch frequency, oscillating complexity score |

Vectors 1 to 6 follow the structure of an LLM request. Vector 7 only exists once a defender introduces adaptive model routing, and is included because defences change the attack surface they protect.

---

## The defence architecture

Defence is layered, with observability running through all of it.

| Layer | Representative controls | Vectors addressed |
|---|---|---|
| Input | Historical baseline, repetition and entropy checks, unicode anomaly checks, instruction nesting depth, retrieval chunk checks | 1, 2, 3 |
| Session | Session size accumulation, session completion budgets | 4 |
| Output | Predicted ceiling comparison, streaming checkpoint budgets, output repetition scoring, reasoning-to-answer ratio | 2, 5, 6 |
| Model selector | Complexity scoring, length versus complexity cross-check, switch rate limiting | 5, 7 |
| Observability | Logging at a minimum of input and output | all |

**The Automated Model Selector (AMS)** is a deliberately small, lightweight classifier that scores prompt complexity and emits a single number. Its narrow output is a security decision: a large model reasoning its way to a routing decision would put a second unbounded cost in front of the first, and would itself be exposed to complexity inflation. The routing decision is made downstream by the production team, not by the AMS.

**Observability** is a requirement, not an extra. Systems designed against TEA must log enough to support detection, and the same logs support audit, legal discovery and compliance.

---

## Method

Every vector folder follows the same pattern so results are comparable:

1. **Attack harness.** A fixed task with a machine-checkable answer, with one variable changed at a time. Cost is always the **server-reported token usage**, never a local estimate.
2. **Victim.** A thin wrapper that mimics a typical deployment, including a per-request input cap, so defences are compared against realistic current controls and not an empty target.
3. **Mitigation proof of concept.** A detector for that vector returning allow, flag or block with reasons.
4. **Evaluation.** Detection rate per variant and size, **false positive rate against legitimate traffic**, detector overhead, and an adaptive evasion test where the attacker knows the thresholds.

The false positive rate is treated as a primary result, not a footnote. A detector that cannot tell an attack from a legitimate long input has not solved the problem.

**Reference environment:** LM Studio serving `mistral-small-3.2-24b-instruct-2506-mlx` locally, Python 3.10+.

---

## Status

| Vector | Harness | Detector POC | Evaluation |
|---|---|---|---|
| 1 Volume and padding | First pass written, dry-run against a mock server | First pass written | Not yet run against the model |
| 2 to 6 | Planned | Planned | Planned |
| 7 Selector gaming | Planned | Planned | Identified analytically, not yet validated experimentally |

Nothing in this repository should be read as a validated result until the relevant evaluation has been run and written up.

---

## Repository layout

```
.
├── README.md
├── TEA_Paper_1_Draft.docx
├── Vector1/    volume and padding
├── Vector2/    recursive prompting
├── Vector3/    context window stuffing
├── Vector4/    multi-turn accumulation
├── Vector5/    complexity inflation
├── Vector6/    output shape manipulation
└── Vector7/    selector gaming
```

Each `VectorN` folder has its own README with setup, commands and results.

---

## Responsible use

This is defensive research. The harnesses are designed to measure and mitigate resource exhaustion.

- Run them only against systems you own or have explicit written permission to test.
- Padded and recursive prompts burn real money on paid APIs. If you point any harness at a hosted provider, set a hard spend cap first. The default target is a local model for this reason.
- Do not use this code against production services, shared infrastructure or third parties.

---

## Related work

TEA builds on, and does not replace, existing work:

- Shumailov et al. (2021), [Sponge Examples: Energy-Latency Attacks on Neural Networks](https://arxiv.org/abs/2006.03463).
- Geiping et al. (2024), [Coercing LLMs to do and reveal (almost) anything](https://arxiv.org/abs/2402.14020).
- Gao et al. (2024), [Denial-of-Service Poisoning Attacks against Large Language Models](https://arxiv.org/abs/2410.10760).
- Kumar et al. (2025), [OverThink: Slowdown Attacks on Reasoning LLMs](https://arxiv.org/abs/2502.02542).
- OWASP, Top 10 for LLM Applications 2025, LLM10 Unbounded Consumption.

The full reference list is in the paper.

---

## Citing

If you use this work, please cite the paper. Citation details will be added here once the paper is published.

## Licence

This work is licensed under [Creative Commons Attribution 4.0 International (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/). You may share and adapt it, including commercially, as long as you credit Breaking Bytes (breakingbytes.org).