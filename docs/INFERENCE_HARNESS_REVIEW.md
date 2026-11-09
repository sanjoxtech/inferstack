# InferStack implementation and harness-plan review

Reviewed 9 November 2026 against HEAD `5a06e3ef28920ad70fa53f07509872f25838d157` and the working tree. This review reads [ARCHITECTURE.md](ARCHITECTURE.md), [V2_AGENT_HANDOFF.md](V2_AGENT_HANDOFF.md), [CONTROL_RUNBOOK.md](CONTROL_RUNBOOK.md), the cloud plan's dated implementation status, [INFERENCE_HARNESS_PLAN.md](INFERENCE_HARNESS_PLAN.md), and their source/test contracts. It compares the implementation with the intended research → implementation → verification → deployment → feedback loop.

Concurrent changes exist in execution isolation, local/remote infrastructure and earlier review probes. Those changes were preserved. Test observations below apply to the working tree, not a clean or immutable release. No production code, credentials, live database, cluster configuration or paid provider was changed by this review.

## Assessment

InferStack already has substantial serving, experiment execution and verification infrastructure. The earlier cloud P0–P5 packages have implementations; the future-tense specification above their dated status table should not be interpreted as a list of absent modules. Their GPU and production-cluster qualifications remain incomplete.

The larger product described by the user is not yet connected end to end. In particular, the current agent can propose an experiment and execute a supplied source patch, but cannot itself implement a new research technique. Its experiment-history tool sees the current job, not a reusable cross-campaign knowledge base. Targeted bottleneck profiling, continuing refinement after a first successful candidate, real FactoryOS task qualification and independent harness-improvement evaluation need explicit work.

The harness plan has the right ownership and multi-engine direction. It should be implemented by extending the current durable worker, repository, executor and verifier, with a separate candidate-implementation role. Replacing the existing serving/control system would discard working foundations.

## Findings requiring correction

### R1 — P1: a failed trial can be reported as proven compatibility

**Location:** `src/inferstack/research/techniques.py:102–110`.

`assess()` resolves `proven_by` trials and compares only deployment engine and model names. It does not require a successful gate decision, confirm that the trial tested this technique, or establish applicable engine/model/tokenizer/build/precision/hardware/workload identities. The proof loop also does not read and verify the raw trial artifact. A trial with `errors = ['quality gate failed']`, recorded on a different engine version and dtype, reproduced `compatible_proven` for the target deployment.

**Consequence:** planner ranking and reported knowledge can treat invalid or unrelated evidence as a proven technique. This does not by itself bypass the independent final verifier, but it corrupts the knowledge loop the product relies on.

**Implement in H0:** resolve a successful authoritative verdict and exact treatment lineage; verify evidence integrity and applicability. Keep broader transfer evidence as a claim/hypothesis. A trial reference alone is not proof of success. Add regressions for failed gates, unrelated treatment, stale/missing artifacts and mismatched identities.

### R2 — P1: source-change plans are accepted without an executable implementation

**Locations:** `optimization/planner.py:79–83`, `optimization/dag.py:181–194`, `execution/executor.py:127–131` under `src/inferstack/`.

The provider prompt asks for a `source_patch` with a technique ID, files and description. Validation accepts that description-only action. The executor immediately fails it unless `patch_sha256` references an already stored diff. Planner tools are read-only and no intervening coding/materialization stage creates the diff. The review reproduced successful plan validation followed by `BuildFailed: source_patch has no stored patch (patch_sha256)`.

**Consequence:** the central “give it a paper and let it implement a new kernel/technique” path is missing. The existing patch executor is useful, but it handles externally supplied implementations.

**Implement:** make non-executable proposals explicit, and require a validated immutable artifact before execution. In H3 add the bounded implementer handoff described in the refined plan: pinned source, scoped write authority, stored diff/build identity, candidate tests, logs/costs, independent trusted verification and durable recovery. Do not give the research planner unrestricted write access or let generated tests replace authoritative gates.

### R3 — P2: planner and executor disagree about supported profile changes

**Locations:** `optimization/dag.py:30–37`, `execution/executor.py:44–75,279`.

Two contradictions reproduced:

- The planner accepts native `decode_priority` profile changes, but `with_flags()` rejects every `inferstack_native` profile patch as unsupported.
- The executor supports vLLM `load_format=runai_streamer`, but the planner's profile-key allowlist rejects `load_format`.

**Consequence:** a registered feature may be unusable by the optimizer, or a validated experiment can fail before it runs. Manually configuring a loader or using a pre-registered deployment is a different supported path; this finding concerns generated profile-patch experiments.

**Implement in H0:** share a version-aware profile schema/materialization contract between planner and executor. Implement native mapping or reject it before scheduling. Validate effective launch options and observed identity, not only metadata. Allow a loader only for an appropriate pinned image/dependency/model format. Do not enable every upstream flag indiscriminately.

### R4 — P2: grouped content events can manufacture zero ITL

**Location:** `src/inferstack/serving/evidence.py:99–108`.

The live stream parser divides the time between first and last client content events by aggregate completion-token count minus one. A single content event containing two reported tokens produces `itl_ms=0.0`, despite no observation of when those tokens were generated. Network/proxy buffering, speculative decoding and grouped output can change event boundaries independently of engine token timing. Percentiles over per-request means are also not percentiles over individual token gaps.

**Consequence:** ITL-based comparisons or gates can look better when event grouping changes. Existing client content-gap/stall signals remain useful, and this does not invalidate all gateway telemetry.

**Implement in H0:** record measurement provenance and semantics. Keep client first-content, E2E and content-event gaps separate from engine timing. Mark insufficiently observed decode timing unavailable; a grouped event cannot prove zero engine ITL. If a gate needs token-level tail timing, require appropriate evidence rather than a percentile of client-derived per-request averages.

### R5 — P1: synchronous cleanup can starve the worker's lease renewal

**Locations:** `optimization/experiment.py:164–169,191–194`, `control/worker.py:42–47,127–129`, `infrastructure/local.py:190–224` under `src/inferstack/`.

The async executor cleanup invokes synchronous `provider.stop()` directly. Local process termination can poll and sleep while the same event loop must renew job/worker leases. On an expired lease, fenced publication correctly refuses to complete, but the worker can return leaving the job `running` and its experiment step unfinished.

The foundation suite observed this in `test_crash_at_every_transition_then_restart_recovers[step_started:iter0.plan]`. Its preserved temporary database records successful verification at `1794200372.939`, job lease expiry at `1794200373.722`, and allocation cleanup at `1794200374.132`, with no completed experiment step. An isolated rerun passed. A separate deterministic probe then exercised the real `EndpointExecutor.cleanup` and `OptimizationWorker` with a fake provider taking 650ms to stop and a supported 300ms lease: the worker returned `running` with no terminal result. This establishes the blocking-cleanup failure independently of the intermittent timing of the process test.

**Implement in H0:** keep blocking provider stop/reconciliation and filesystem cleanup off the heartbeat event loop, preserve fenced allocation ownership and bounded cancellation/cleanup, and prove renewal throughout a slow stop. Audit other synchronous provider operations reachable from the async worker. Do not hide the defect merely by extending a test sleep or increasing leases. The related recovery must also prove that an interrupted worker cannot publish after takeover.

## What exists and what still needs work

“Implemented” below describes source behavior and test coverage. It does not imply GPU speed, scalability or production qualification.

| Area | Existing implementation | Remaining work |
|---|---|---|
| Engine independence | Native, vLLM, SGLang and generic OpenAI-compatible adapters; capability/conformance machinery | Qualify exact model/engine/hardware intersections; reconcile declared vs tested vs observed support; fix R3. Native research paths remain model-specific. |
| Durable execution | SQLite/PostgreSQL repository, typed records, revisions/artifacts, leases, step journaling, budgets and resource ownership | Extend the same durable lifecycle to implementation and diagnosis artifacts; independently finish qualification of concurrent isolation changes. |
| Research intake | Hashed text/HTML evidence, provenance, deduplication, SSRF-safe URL fetching and technique manifests | Unified research/problem-to-campaign intake, applicability fixes and campaign lineage. PDF is not in the current fetcher's content-type allowlist; accept extracted text initially or add bounded PDF extraction with source/page provenance. |
| Planning and memory | Provider-backed planner, bounded read-only tools, typed validated action DAG; current-job failures/history | Versioned full deployment/workload context and relevant cross-job retrieval. The database retaining records does not mean the planner can retrieve and correctly reuse them. |
| Candidate implementation | Configuration/plugin materialization, stored source-diff application, isolated candidate builds/tests | Explicit coding stage for a new technique, artifact handoff, scoped third-party-engine source builds; R2. Avoid weakening trusted verification boundaries. |
| Diagnosis and refinement | Telemetry queries, trial reasons and replanning after unsuccessful iterations | Targeted profiler actions, structured diagnosis with confidence/alternatives, parent-child hypotheses and controller-owned campaign policy. Worker currently stops on first keep and uses its configured iteration limit; declared `stop_rule` does not govern the complete loop. |
| Numerical/task verification | Repeated interleaved endpoint trials, fixed policy, exact/reference checks, quality/numerical/logprob and serving gates where supported | Representative FactoryOS task fixtures and adapter-specific capability qualification; protect held-out inputs from candidate changes. Unsupported evidence must remain inconclusive. |
| Release and recovery | Immutable releases, attestation, progressive rollout, rollback, restart/replacement/retirement and durable live evidence | Bind the entire new campaign to eligibility and rollout policy; exercise real target-engine and FactoryOS behavior. A kept experiment is intentionally separate from a live release. |
| Cloud serving P1–P5 | Model-cache volumes and immutable staging; loader option; release-scoped pools/admission; manual/advisory/KEDA ownership; shadow traffic and cost attribution | GPU model larger than 2Gi, cold/warm startup, actual GAIE/EPP routing and KEDA/Prometheus with GPU replicas, real provider-bill reconciliation. Manifest generation is not a running cache-aware router. |
| FactoryOS feedback | Versioned idempotent issue intake, bounded triggers, durable outcome outbox/callbacks | Real task runner and release/deployment/trace/evaluator binding across the complete workflow. Latency/throughput/capacity/cost can trigger jobs; quality/error/unknown currently record for triage. Add diagnosis before extending automation. |
| Harness improving itself | Product intent now documented; serving source patches restricted to kernel/runtime roots | Separate harness-candidate build scope and incumbent-vs-candidate evaluation on frozen optimization tasks/budgets. Do not relax serving patch roots globally or let the candidate edit its own evaluator. |

Automatic engine alternatives do not require rebuilding all adapters. Registered candidate deployments can already use another supported engine, and comparability checks use model/hardware identity rather than requiring equal engine names. The missing part is context-aware candidate discovery, qualification and a campaign that selects alternatives fairly.

## Implementation sequence and concrete completion criteria

1. **H0 — Make evidence and executable contracts trustworthy.** Fix R1, R3, R4 and R5; make R2's description-only state explicit. Reconcile capabilities and manifests against a pinned candidate. Preserve previous independent probes and add failure-focused regressions. Acceptance: no failed trial is proven, unsupported native patches do not schedule, loader flags agree across layers, grouped streams cannot claim measured zero engine ITL, and slow cleanup preserves leases and a correct terminal job state.
2. **H1 — Give a new campaign reliable memory.** Build a versioned context snapshot and bounded cross-campaign query over existing relational records/artifacts. Include exact implementation, workload, prior results, failures, policy and missing context. Acceptance: a fresh job finds a relevant failed treatment and does not confuse an incompatible GPU/model/version result with proof.
3. **H2 — Turn inputs into diagnosed problems.** Route submitted research and FactoryOS issues through one typed lineage; add bounded measurement/profile requests and a diagnosis result. Acceptance: a latency symptom can be attributed to queue/cache/kernel/network evidence or remain uncertain; a paper does not grant execution authority or install dependencies.
4. **H3 — Implement and refine a candidate.** First deliver immutable patch/build handoff, then bounded coding-agent integration. Add controller-owned first-improvement/continued-refinement policy, linked hypotheses and recovery/accounting for each implementation attempt. Acceptance: an actual source idea yields a stored diff, independent verification and a traceable next hypothesis after an induced bottleneck; cancel/crash leaves no unowned resource or silently substituted artifact.
5. **H4 — Complete the operational feedback loop.** Connect representative FactoryOS tests, dataset checks, release eligibility, rollout policy and actual stable/candidate endpoint results. Acceptance: both a FactoryOS issue and a submitted research technique complete the workflow, including a failing canary and restart recovery. A justified reject/inconclusive result is valid; a speedup is never fabricated to pass a demonstration.
6. **H5 — Improve the harness itself.** Compare a candidate planner/retriever/diagnoser against the incumbent under fixed optimization tasks and resource budgets; add safe setup reuse. Acceptance: verified optimization outcomes or turnaround improve without changing held-out tests, acceptance rules or spending limits. Persistent experimental workers require separately proven state reset and fresh final measurement.

Run real-engine qualification alongside software integration once a reproducible candidate exists. The first A100 run should record the exact A100 memory size, model/tokenizer and engine/build revisions, precision, workload/cache conditions, raw outputs, policy and cost/resource use. T4 FP16 is a separate narrower profile. Neither a CPU fixture nor a T4 result establishes A100/BF16 performance.

Do not prioritize multi-node serving, prefill/decode disaggregation, LoRA, spot orchestration, a universal compiler, a new vector database or continuous X monitoring before this loop works. Promote one when a measured workload need and compatible implementation justify it.

## Verification performed for this review

Focused suite command:

```sh
.venv/bin/python -m pytest tests/test_planner_dag.py tests/test_research.py tests/test_worker.py tests/test_wiring.py tests/test_quality_phases.py tests/test_integrations.py tests/test_evaluation.py tests/test_live_evidence.py tests/test_autoscale.py tests/test_routing_pools.py tests/test_model_cache.py -q --junitxml=results/harness-review-2026-11-09.xml
```

Observed: **135 passed, 1 skipped**, 30.049 seconds in the JUnit suite. This covers focused planner/research/worker/verification/integration and serving-feature contracts. It is not a full suite or GPU qualification.

Independent gap reproductions:

```sh
.venv/bin/python docs/verification/harness_gap_probes.py
```

All six described observations reproduced: failed-trial proof reuse; accepted source description followed by missing-patch execution failure; rejected loader profile despite executable flag support; accepted native profile followed by executor rejection; one grouped content event reporting zero ITL; synchronous cleanup leaving the job running after lease expiry. [The probe script](verification/harness_gap_probes.py) uses temporary SQLite/artifacts and no remote services. These are gap reproductions, not passing product acceptance tests. Raw output is at `results/harness-gap-probes-2026-11-09.json`.

Additional foundation/real-process lifecycle suite:

```sh
.venv/bin/python -m pytest tests/test_contracts.py tests/test_control_store.py tests/test_repository.py tests/test_execution_store.py tests/test_durable_execution.py tests/test_engines.py tests/test_attestation.py tests/test_releases.py tests/test_release_runtime.py tests/test_rollout.py tests/test_integrated_serving.py -k 'not postgres' -q --junitxml=results/harness-review-foundation-2026-11-09.xml
```

Observed: **77 passed, 1 failed**, 118.144 seconds. The failure was the crash-recovery case in R5. An isolated rerun passed in 5.017 seconds (`results/harness-review-crash-recheck-2026-11-09.xml`); that does not erase the original failure or the deterministic cleanup reproduction. The foundation suite excludes PostgreSQL parametrizations and does not substitute for hosted PostgreSQL, container, GPU or real inference-router/scaler tests. Logs and JUnit files are retained under `results/harness-review-*2026-11-09.*`.

The existing `results/compatibility-local.json` names older commit `663439a` with `dirty: true`, no GPU/CUDA/vLLM/SGLang, and no Gateway API, InferencePool, KEDA or LeaderWorkerSet CRDs in kind. That manifest cannot establish qualification of HEAD, the current dirty tree, GPU inference or an installed cache-aware serving stack. Earlier larger pass counts in the handoff remain dated, repository-reported evidence; this review does not re-certify them.

## Files for the next implementer

- [INFERENCE_HARNESS_PLAN.md](INFERENCE_HARNESS_PLAN.md): intended product and refined H0–H5 contracts, including the previously missing implementation handoff.
- This review: concrete source gaps, reproduced counterexamples, implementation order and verification limits.
- [ARCHITECTURE.md](ARCHITECTURE.md): ownership and architecture contracts.
- [CLOUD_SERVING_RESEARCH_PLAN.md](CLOUD_SERVING_RESEARCH_PLAN.md): P0–P5 specifications plus dated implementation status; read both.
- [V2_AGENT_HANDOFF.md](V2_AGENT_HANDOFF.md) and [CONTROL_RUNBOOK.md](CONTROL_RUNBOOK.md): implementation history and operating procedures.
- [FACTORYOS_CONTRACT.md](FACTORYOS_CONTRACT.md) and [COLAB_V2_QUALIFICATION.md](COLAB_V2_QUALIFICATION.md): current feedback boundaries and target-hardware qualification procedure.

Start implementation at H0, preserve concurrent file ownership, and qualify a committed candidate before reporting an integrated release.
