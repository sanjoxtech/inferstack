# Auto-improve harness: connecting the engine work to the existing harness (2026-11-19)

Status: gap analysis and build order, nothing built yet. Goal (user, 2026-11-19): for any model on any
hardware, the system finds the best serving setup against a stated SLO and keeps improving the engine
itself, continuously, with an LLM agent (DeepSeek or OpenRouter API, model configurable) doing the
planning a human plus a coding assistant did by hand in P9-P13. Humans set SLO, budget and guardrails.

Decisions (user, 2026-11-19):

- Code changes by the agent: **PR only** (branch + evidence, never auto-shipped). Settings tuning, engine
  choice, experiments and profile promotion after the evaluator passes: automatic.
- Hardware: any. Colab L4 today; paid providers later behind the same executor interface (per-provider
  spend caps; new providers need the user's account and keys).

## 1. What already exists (do not rebuild)

The product loop is defined in [INFERENCE_HARNESS_PLAN.md](INFERENCE_HARNESS_PLAN.md) (2026-11-09) and
reviewed in [INFERENCE_HARNESS_REVIEW.md](INFERENCE_HARNESS_REVIEW.md). Implemented in source (H0-H3 step 2):

| need | existing module |
|---|---|
| durable records, revisions, artifacts | `storage/` (SQLite/PostgreSQL, migrations), `control/` (API, worker, leases) |
| LLM planner (DeepSeek/OpenRouter) with typed plans, read-only tools | `optimization/planner.py` (`ProviderPlanner`, `ToolPlanner`) |
| trusted comparison, repeated interleaved trials, numerical checks | `optimization/experiment.py` (`EndpointVerifier`, repeats=3, logit agreement) |
| controller-owned campaign: diagnosis, stop reasons, lineage | `optimization/campaign.py` |
| cross-campaign memory with applicability | `memory/experience.py`, `research/techniques.py` |
| research intake with provenance, untrusted-content envelope | `research/evidence.py`, `research/fetch.py` |
| bounded implementer handoff (immutable patch artifacts) | H3 step 1 (`c460793`) |
| FactoryOS feedback intake, Colab/Kaggle manifest round trip | H2 (`44c9ac9`), E1.2 (`3516693`) |

## 2. What P9-P13 added outside the harness

Open-loop goodput and $/1k requests at the SLO (`scripts/goodput.py`, SLO class limits), engine counters per
rate step, profilers (`decode_step_profile.py`, `gemm_bw_bench.py`, `piecewise_probe.py`), the unattended
Colab driver loop (`colab_fast_check.py` with retries, own-session cleanup), new engine knobs (wait-for-slot,
48 slots, pacing K/M, pipeline, lazy banks, drafts, BLAS library) and 13 recorded experiments with decisions.

## 3. Gaps, in build order

| G | gap | build | exit test |
|---|---|---|---|
| G1 | the verifier scores closed-loop endpoint trials; the headline is open-loop goodput | goodput objective in the verifier (rate sweep, SLO class limits, every-SLO-at-once rule, run-to-run noise model: keep only gains beyond the measured spread) | v1.36 vs v1.33 re-judged by the harness gives the same decision as P9f/P9g |
| G2 | GPU runs are launched by hand | executor backend that drives `colab_fast_check.py` (own sessions only, retries, cost recorded); provider interface so paid hosts plug in later | a campaign runs a candidate on Colab L4 end to end with no human step |

**G2 status (2026-11-19):** proven on Colab L4. Run `ifs-g2-proof-k3b` ($0.22, 17 min, all gates passed) replayed P10b with no human step: goodput@99 tied at 8 req/s, but the 12 req/s good fraction fell from 64-86% to 17-21%, so `slo_attainment` (mean good fraction over the rate grid) was added as a second objective; the re-judged verdict is `reject (material regression)`, stored in memory. The first run (`ifs-g2-proof-k3`, $0.09) found a quoting bug in the remote step (fixed). `execution/gpu_backend.py` (LocalBackend, ColabBackend over `colab_fast_check.py` with `ifs-*` sessions only, retry on a reclaimed VM, $10 spend gate, cost from wall time), `scripts/remote_ab.py` (on-host AB/BA goodput trials, fresh server per trial, greedy-agreement check), `optimization/remote_verify.py` (judged by `evaluate`, stored as a completed job that `memory.experience` reads). Open: native-vs-rival A/B on the host (rival venvs are only set up by the scorecard path), and wiring `RemoteABVerifier` into the planner loop (G5/G7).
**G4 status (2026-11-20):** built and run on Colab L4 ($0.31, sessions `ifs-g4-knee`, `ifs-g4-kernels`). Sweep steps record miss reasons and, for the native engine, counter deltas from `/v1/native/status`; `evaluation/bottleneck.py` labels the knee and groups decode-step profiler kernels into families (repo Triton kernels mapped by exact name); measurements are stored per deployment, shown in the context pack and a planner tool, and a plan that does not name the measured bottleneck is sent back. v1.36 on L4, interactive: knee at 12 req/s (8 met), misses split TTFT 39 / per-token 45, TTFT split queue 274 ms vs prefill 42 ms, lane 31.6 rows/step at 37.6 ms/step, acceptance 0.96, GPU busy 98%; repeats disagree on decode vs queue_admission (agreement 0.5). Kernel shares at 48 rows: GEMM 55%, GDN 34% (GDN grows with rows: 7% at 8, 24% at 32). Open: a saturated GPU (busy >= 95%) with mixed misses should read as `capacity` (compute-bound), not as a coin flip between its two symptoms.
| G3 | memory lacks the P9-P13 results | backfill the 13 experiments (hypothesis, profile, evidence digest, keep/reject, reason) as harness records | a fresh plan for "interactive 12 req/s" is shown K=3, drafts=2 and the mixed step as known failures on L4 |
| G4 | planner sees settings, not why a run missed | bounded measurement actions: profiler runs and engine counters in the context pack | the planner cites the measured bottleneck (e.g. GDN kernel share) in its hypothesis |
| G5 | settings search is a separate script on an old objective | `slo_tune.py` v2 as a campaign action: goodput objective, crash-region learning (SCOOT idea), space from a hardware probe (memory, bandwidth, TFLOPs) | on a new GPU type (Colab T4/A100) it produces a tuned profile without hand edits |
| G6 | improvement starts only when a human asks | SLO watchdog: serving telemetry below the SLO, a new model/hardware, a new rival release, or a research item opens a campaign within the spend cap | a planted SLO regression opens and closes a campaign unattended |
| G7 | code ideas stop at a plan | H3 step 3: the coding agent (OpenRouter/DeepSeek) works on a branch in an allowlist, adds a flag and a test; the fixed evaluator (hash-checked, agent cannot edit) runs tests, greedy-equality, 3-run goodput A/B and a too-good breaker; result = PR with evidence | catches a planted cheat; one PR accepted on evidence |
| G8 | engine choice per workload is manual | campaigns compare InferStack / vLLM / SGLang per (model, hardware, workload) and serve the measured winner | the served engine per cell equals the measured winner |
| G9 | the harness does not improve itself | H5: candidate planner/diagnoser vs incumbent on frozen tasks and budgets | better outcomes or turnaround with unchanged tests and limits |

**G5 status (2026-11-21):** built; proven on Colab L4 only. `optimization/tune.py` (space from a hardware probe and the measured bottleneck, crash region, sampler: one knob at a time from the baseline's real values, then the best winner per knob combined, then Optuna TPE) and `scripts/remote_tune.py`; the pick is confirmed by the interleaved A/B. L4 rerun (`ifs-g5-tune-l4b`, $0.67 with the A/B): best single moves batch 64 and cuBLAS, combined 0.607 vs 0.546 attainment in tuning, A/B inconclusive (below the noise floor), so v1.36 stays: settings are near their limit on L4 and further gains need code (G7). Colab cost is now billed from the VM's own start/stop markers. New-GPU proof blocked: A100 creation timed out, H100 rejected (no entitlement on this account), both $0 and recorded `infra_failed`, never `inconclusive`. Open: a rented-GPU backend (RunPod/Modal) for the new-hardware exit test.

**G6 status (2026-11-22):** built and exit-tested on the fake engine. `optimization/watchdog.py`: a `ServingSLO` (rate the deployment must carry at >= 99% good requests, per-class TTFT/per-token limits), observers (`probe_observer`: short open-loop run at the SLO rate against the live endpoint, any engine; `telemetry_observer`: normalized Prometheus p99s), and a per-deployment state machine stored as `watch` records with `incident` records. N consecutive breached windows open one incident and run the settings campaign (G5 tune + A/B; no model calls, no code). A kept patch that carries the SLO rate is applied by an optional `apply` hook and re-verified on the live endpoint before the incident closes `resolved`; without the hook it waits as `fix_pending` for promotion. No fix closes `unresolved` and escalates (no new campaign until the deployment revision changes: the next step is G7 or a human); `infra_failed` cools down and retries; a campaign that would pass the daily cap is `blocked_budget`; windows with too few requests never count. Exit test `test_planted_regression_opens_and_closes_a_campaign_unattended`: planted 16-slot profile breaches TTFT, the watchdog tunes, A/B keeps 32-48 slots, applies, re-measures, closes resolved, $0. Open: the other triggers (new model/GPU with no tune on record, rival release, research item), and a long-running service entry point.

**G7 status (2026-11-22):** built and exit-tested on a fake engine whose speed lives in patchable source. `implementation/coder.py` (`ModelImplementer`: any OpenAI-compatible provider, e.g. OpenRouter/DeepSeek; it sees the handoff and the allowed files at the pinned revision plus the last rejection reason, answers whole files; the diff is computed by the controller), `optimization/code_change.py` (`CodeChange`: handoff -> journaled attempts validated by the desk -> GPU A/B of the diff via `RemoteABVerifier(source_diff=...)` / `remote_ab.py --candidate-diff-b64`, which applies it to a pristine export with tests and the evaluator restored -> too-good breaker (goodput@99 more than `max_gain` times, default x1.5, or from zero = `suspicious`) -> PR bundle `patch.diff`, `PR.md`, `pr.json`). The evaluator files are fingerprinted at start (optionally pinned) and re-checked before a bundle. `publish_pr` (branch, push, draft PR) refuses without `allow_publish=True`. Exit test `test_planted_cheats_are_caught_and_an_honest_change_becomes_a_pr`: a test edit is refused by the desk, an output-changing speedup fails the greedy-equality gate, a no-work speedup trips the breaker, and the honest two-lane change is kept and bundled. Open: a real run (L4, OpenRouter/DeepSeek key, one bounded proposal such as a GDN or GEMM kernel idea from G4), wiring the watchdog's `unresolved` escalation to a `CodeChange`, and a fresh-seed rerun for `suspicious` results.

## 4. Prior art used (ideas only, licenses checked)

Darwin Gödel Machine (archive of variants judged only by benchmarks), ShinkaEvolve (Apache-2.0;
sample-efficient LLM-driven program evolution), SICA and the Red Queen Gödel Machine (reviewer agent that
co-evolves with the coder), karpathy/autoresearch and RightNow-AI/autokernel (fixed harness, allowlisted
edits, keep/revert), SCOOT (Bayesian tuning against SLOs with learned crash constraints), NVIDIA
AIConfigurator (CPU-side performance model prunes configs before GPU time), Novita autotuner (SLO
violations as penalties), InferenceBench and KernelBench-Verified (agent benchmarks; reward hacking is the
known failure mode, hence the fixed evaluator and the too-good breaker).

## 5. Guardrails that stay

Spend caps per run and per day; ask before any campaign over $10. Never stop or reuse sessions the harness
did not create. API keys in a secret store, never in code or logs. Research content is untrusted data and
never grants execution authority. License filter on every proposal (ELv2 never copied; AGPL/NOASSERTION
flagged; Apache/MIT ideas with notices). Push to GitHub only when the user says so.
