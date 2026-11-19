# InferStack auto-improve harness: design (2026-11-19)

Status: design, nothing built. Goal: for any model on any hardware, the system finds and keeps the best
serving setup against a stated SLO, and keeps improving the engine itself, with an LLM agent (DeepSeek or
OpenRouter API, model configurable) doing the planning that a human plus a coding assistant did by hand
in P9-P13. Humans set the SLO, the budget and the guardrails; the loop runs continuously.

## 1. What the manual loop was (P9-P13, 2026-11-18/19)

Measure (open-loop goodput, engine counters) -> find the limit (profiler, counters) -> read rivals and
papers -> propose one change -> implement behind a flag -> tests + greedy-equality -> A/B on the GPU ->
keep or reject -> write the result and evidence. 13 experiments, ~$12, 5 kept (v1.30..v1.36), 6 rejected,
2 stopped at a cheap gate. The harness automates exactly this loop, and keeps the record it produced.

## 2. Prior art (ideas only; licenses checked before reuse)

| system | idea we take |
|---|---|
| Darwin Gödel Machine (Sakana/UBC, ICLR 2026) | keep an archive of variants, not a single best; self-edits judged only by benchmarks |
| ShinkaEvolve (Sakana, Apache-2.0) | sample-efficient program evolution: LLM ensemble as mutation operators, novelty filter |
| SICA, Red Queen Gödel Machine | a reviewer agent that co-evolves with the coder catches weak changes |
| karpathy/autoresearch, RightNow-AI/autokernel | fixed harness the agent cannot edit; one file allowlisted; keep/revert per experiment |
| SCOOT (Ant Group, WWW 2025) | Bayesian optimisation of engine knobs against SLOs; learns hidden constraints (crash regions) with a random forest |
| AIConfigurator (NVIDIA, 2026) | a CPU-side performance model prunes configs before any GPU time |
| Novita autotuner | SLO violations as exponential penalties in the score |
| InferenceBench (2026), KernelBench-Verified | public benchmarks for agent-driven inference optimisation; reward hacking is the known failure mode |

## 3. Architecture

```
            +-------------------- knowledge ---------------------+
            | telemetry DB | experiment archive | research inbox |
            +--------------------------+-------------------------+
                                       | context pack
   SLO + budget  --->  PLANNER (LLM)  --->  experiment spec (typed JSON)
                                       |
                 +---------------------+----------------------+
                 v                     v                      v
         settings search        code change (branch)    engine choice / install
         (Layer 1, no LLM)      (Layer 2, LLM edits)    (vLLM / SGLang / InferStack)
                 +---------------------+----------------------+
                                       v
                 EVALUATOR (fixed, agent cannot edit): tests, correctness,
                 goodput A/B x3, noise model, too-good breaker, cost gate
                                       v
                 DECIDER: keep / reject / escalate  --->  archive + telemetry
```

### 3.1 Telemetry DB (SQLite first, one file per deployment, synced to the repo's evidence)

- `hardware`: fingerprint (GPU name, SMs, memory, measured bandwidth and bf16 TFLOPs from a 30 s probe,
  driver, CUDA, provider).
- `model`: repo@revision, architecture (layer types: GDN/attention/MoE), size, MTP heads.
- `profile`: full TOML + hash; parent profile; who proposed it (tuner, agent, human).
- `run`: profile x hardware x model x workload x SLO, commit, cost, raw evidence path.
- `step`: per rate step: good fraction, TTFT/TPOT percentiles, engine counters (rows per step, spec
  acceptance, prefills, host vs GPU ms), so the planner sees *why* a run missed.
- `experiment`: hypothesis, spec, outcome, decision, reason, links (the P9-P13 doc becomes the first rows).
- `proposal`: research-inbox items (source URL, license, summary, status).

### 3.2 Inputs to the planner (context pack, bounded size)

SLO and budget; current best per (model, hardware, workload); the worst SLO miss with its engine counters
and latest profile (`decode_step_profile.py`, `gemm_bw_bench.py`, `piecewise_probe.py`); the experiment
archive (including rejected ideas and why, so it does not retry them); research-inbox items; hardware
facts. The planner returns one spec: kind (settings | code | engine), hypothesis, expected effect, cost
estimate, gate to stop early (like P13 S0).

### 3.3 Executors

- **Layer 1, settings (no LLM, unattended):** `slo_tune.py` v2: objective = open-loop goodput@99 and
  $/1k requests at the SLO; constraints learned from crashes (SCOOT); search space generated from the
  hardware fingerprint (slots from free memory, graph buckets, pacing K/M, pipeline, lazy banks, drafts,
  overlap, prefill budget); a cheap pre-filter (analytic model from probe numbers, AIConfigurator idea).
- **Layer 2, code (LLM edits):** branch per experiment; allowlisted files; must add a flag (off by
  default) and a test; works only through the repo's tools.
- **Engine choice:** install and run pinned vLLM / SGLang / InferStack builds in the same harness; the
  served engine for a (model, hardware, workload) is whichever wins the A/B. "Best setup" includes
  serving with a rival when it wins (today: vLLM for interactive/shared_prefix at 12 req/s).

### 3.4 Evaluator (the part that makes it trustworthy)

Fixed code the agent cannot change (hash-checked before each run):
1. unit + GPU tests (`--tests`), greedy-equality vs the reference path on real prompts;
2. goodput A/B against the current best, >= 3 runs each; keep only if the gain exceeds the measured
   run-to-run spread (P10/P11: ~10 points of good-fraction at the knee);
3. too-good breaker: a gain beyond a physical bound (e.g. step time under the weight-read floor, tok/s over
   the roofline) or a correctness drift stops the experiment and flags it;
4. adversarial inputs for kernels (autokernel style: peaked attention, poisoned pages);
5. cost gate: per-run and per-day spend caps; ask before any campaign over $10.

## 4. Autonomy and guardrails

| action | autonomy |
|---|---|
| settings search, engine selection, profile promotion after the evaluator passes | automatic |
| research ingestion, planning, running experiments within the spend cap | automatic |
| code changes | branch + PR with evidence (current rule: never auto-shipped) |
| spend above cap, new providers, secrets | human |

API keys live in a cloud secret store, never in code or logs. License filter on every proposal: ELv2
(e.g. deepseek-ultrafast) never copied, AGPL/NOASSERTION flagged, Apache/MIT ideas OK with notices.

## 5. Milestones

| M | build | exit test |
|---|---|---|
| M1 | telemetry DB + backfill from P1-P13 evidence; `harness status` CLI | every past experiment queryable with its decision |
| M2 | Layer 1 tuner v2 (goodput objective, new knobs, hardware-aware bounds); run on L4, then T4/A100 (Colab) | finds v1.36-or-better on L4 without hand edits; a T4/A100 profile from scratch |
| M3 | planner + evaluator loop, settings-only actions, LLM via OpenRouter/DeepSeek | runs N experiments unattended, archive grows, no rule broken |
| M4 | Layer 2 code proposals -> PRs | catches a planted cheat; one PR accepted on evidence |
| M5 | research inbox (arXiv/GitHub/rival changelogs, user notes) feeding the planner | one idea from the inbox tested end to end |
| M6 | engine choice across vLLM / SGLang / InferStack per workload | the served engine per cell is the measured winner |
| M7 | shared core with Model Foundry and FactoryOS (same DB schema and loop) | one loop drives two projects |

## 6. Open decisions

- Code changes: keep PR-only (current rule) or allow auto-merge to a canary after the evaluator passes?
- Hardware: Colab only (L4, T4, A100) for now, or add other providers for "any infra"?
- Planner model and monthly API budget.
