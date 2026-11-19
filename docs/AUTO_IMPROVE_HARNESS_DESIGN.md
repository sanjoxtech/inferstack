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
| G3 | memory lacks the P9-P13 results | backfill the 13 experiments (hypothesis, profile, evidence digest, keep/reject, reason) as harness records | a fresh plan for "interactive 12 req/s" is shown K=3, drafts=2 and the mixed step as known failures on L4 |
| G4 | planner sees settings, not why a run missed | bounded measurement actions: profiler runs and engine counters in the context pack | the planner cites the measured bottleneck (e.g. GDN kernel share) in its hypothesis |
| G5 | settings search is a separate script on an old objective | `slo_tune.py` v2 as a campaign action: goodput objective, crash-region learning (SCOOT idea), space from a hardware probe (memory, bandwidth, TFLOPs) | on a new GPU type (Colab T4/A100) it produces a tuned profile without hand edits |
| G6 | improvement starts only when a human asks | SLO watchdog: serving telemetry below the SLO, a new model/hardware, a new rival release, or a research item opens a campaign within the spend cap | a planted SLO regression opens and closes a campaign unattended |
| G7 | code ideas stop at a plan | H3 step 3: the coding agent (OpenRouter/DeepSeek) works on a branch in an allowlist, adds a flag and a test; the fixed evaluator (hash-checked, agent cannot edit) runs tests, greedy-equality, 3-run goodput A/B and a too-good breaker; result = PR with evidence | catches a planted cheat; one PR accepted on evidence |
| G8 | engine choice per workload is manual | campaigns compare InferStack / vLLM / SGLang per (model, hardware, workload) and serve the measured winner | the served engine per cell equals the measured winner |
| G9 | the harness does not improve itself | H5: candidate planner/diagnoser vs incumbent on frozen tasks and budgets | better outcomes or turnaround with unchanged tests and limits |

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
