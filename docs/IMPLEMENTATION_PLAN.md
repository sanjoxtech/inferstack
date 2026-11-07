# Implementation plan: InferStack serving and evolution

Status: proposed implementation sequence, ready for review.
Prepared November 7, 2026. Source audit:
`33f897e39f484447036867e7c7e4c296e1b50806`.

This planning change edits documentation only. Release labels below describe
future scope; they do not change the current package version or claim delivery.
See [architecture](ARCHITECTURE.md) and [source audit](CURRENT_STATE_AUDIT.md).

## Outcome to demonstrate

An operator identifies a hosted deployment and asks InferStack to optimize it,
or supplies a technique, or sends a FactoryOS inference issue. InferStack retrieves
real evidence, researches applicable alternatives, generates a plan, executes an
isolated candidate, verifies the change and records the outcome. A verified
release can deploy to a configured target, with monitoring and rollback.

The first complete demonstration uses a configuration change on an attached
worker. Research-driven code/plugin changes and cloud provisioning follow on
the same contracts.

## Release sequence

| Phase | Target release | Deliverable | Completion evidence |
| --- | --- | --- | --- |
| 0 | Planning baseline | Architecture, repository audit and release gates | These three linked documents; no implementation claim |
| 1 | v1.6.1 | Trustworthy measurements and deterministic judge | Protocol/metric fixtures, incomplete-result rejection, repeated-trial report and A100 validation |
| 2 | v1.7.0 | Persistent deployment/evidence foundation and managed engines | Restart-safe records; real Native/vLLM/SGLang adapter checks; normalized telemetry |
| 3 | v1.8.0 | First complete planner-executor-verifier loop | Real provider plan, isolated config candidate, measured verdict, durable history and resume |
| 4 | v1.9.0 | Research-driven code and backend/plugin experiments | Supplied source -> provenance -> compatible implementation -> independent tests -> verdict |
| 5 | v1.10.0 | Hardware/cloud lifecycle | One real configured provider target supports deploy/inspect/scale/release, resource/cost limits and cleanup |
| 6 | v1.11.0 | Verified release rollout and FactoryOS integration | Canary/drain/rollback with correlated FactoryOS feedback and actual serving measurements |
| 7 | v2.0.0 | Continuous operation and broader model/hardware coverage | Several supported deployments, new-model intake, bounded recurring jobs and regression-aware operation |

Phase numbers are dependencies, not mandatory separate GPU sessions.
Keep Colab work consolidated into resumable release-validation runs.

## Phase 1: repair the measurement and decision foundation

Preserve the v1.6 native candidate; first make evaluation trustworthy.

- Extract reusable benchmark/metric models from ad hoc scripts.
- Fix output-token accounting, first-content timing, grouped chunks, empty events,
  errors, truncated streams and authoritative usage handling.
- Preserve chunk/transport timing separately from token-level timing.
- Reject missing/invalid values and distinguish failed, incomplete and unsupported evidence.
- Add complete model/tokenizer/workload/hardware/software/build fingerprints and
  effective-feature evidence. Resolve mutable source revisions for experiments.
- Replace "matrix pass means optimization success" with independent correctness,
  quality, health, noise, SLO, performance and promotion-eligibility decisions.
- Recover or implement the earlier CV/MAD, repeated-trial, epsilon-Pareto and
  objective-ranking machinery with verified semantics. Do not assume it is imported.
- Fix required-operator accounting and retain raw request/trace artifacts.
- Add a locked, reproducible environment path for benchmark variants. Each engine
  runs in its own environment/image so dependency conflicts are explicit.

Acceptance: synthetic stream grouping cannot change the counted token throughput;
unknown latency cannot win; a slower candidate can complete execution successfully
but be rejected for the chosen objective; real repeated GPU measurements are
traceable to immutable source identities.

Tests target protocol edge cases and decision invariants. They must include
regression scenarios that the existing scripts would incorrectly pass.
Do cheap checks locally and run a consolidated A100 matrix after repairs.
A short smoke run may support development; it cannot establish an unsupported p99 SLA.

## Phase 2: establish deployment identity, storage and engine lifecycle

- Introduce typed domain models and versioned persistence schemas.
- Add SQLite/local and PostgreSQL/shared metadata modes with migrations;
  local/object-store artifact interfaces with checksums.
- Record ModelArtifact, HardwareEnvironment, Deployment, WorkloadSpec, Objective,
  Evidence, Technique, OptimizationRequest, Plan, Experiment, Trial and Release.
- Start with an attached-resource provider: inspect and lease an existing GPU
  worker or remote endpoint. No cloud provisioning required for this stage.
- Wrap the current native engine. Implement real vLLM and SGLang process/endpoint
  adapters with capabilities, attach/start, readiness, streaming, cancellation
  semantics, metrics and stop. Unsupported cancellation/profile features are explicit.
- Extend model routing beyond the current single-Qwen CLI path. Keep gateway
  admission separate from each engine's internal token scheduler.
- Add deployment-correlated request events, normalized metrics and profile imports.
  Collect from engine metrics where available; add native metric export.
- Isolate per-deployment processes to protect module-global patch/counter state.
- Implement a control API/CLI plus bounded query tools for current state and evidence.

Acceptance: restart the control service and retain deployment/evidence records;
attach a real serving engine; route to two independently identified supported
deployments; one unhealthy deployment does not invalidate unrelated healthy routing.
Prove each shipped engine adapter against an actual supported engine runtime.
If GPU capacity requires sequential tests, do not claim concurrent hosting from them.

## Phase 3: ship one full optimization loop

Initial scope: an existing supported deployment, a configuration change and an
attached experiment worker. User text and FactoryOS/internal problem records use
the same OptimizationRequest. Text-supplied techniques can be considered here;
automated internet research is implemented in Phase 4.

- Build the durable workflow controller with planner, executor and verifier steps.
- Add the first real optimizer provider adapter: DeepSeek or the user's configured
  OpenRouter route. Keep the provider contract separate from serving engine adapters.
- Give the planner read tools for identity, metrics/logs/profiles, code/config,
  previous experiments, technique evidence and resource availability.
- Validate structured plans, evidence references, supported actions, compatibility,
  budgets, required baseline, workload and rollback target before execution.
- Execute baseline and candidate in isolated environments with exclusive GPU leases.
- Run the deterministic judge from Phase 1. An LLM verifier can explain results
  and request a bounded revision but cannot overwrite measured gate failures.
- Persist every attempt and artifact. Support status, resume, cancel and cleanup.
  Record keep, reject, retest and inconclusive states.
- Demonstrate a real accepted or rejected candidate; a real rejection is a valid
  functioning-loop result. Do not fabricate an improvement for the demonstration.

Acceptance: issue "optimize deployment D for objective O"; the planner retrieves
actual evidence, proposes a configuration experiment, the executor runs it, and
the verifier produces a replayable decision. Restart mid-workflow and recover
without duplicate candidate actions or leaked GPU resources.

## Phase 4: add research and implementation experiments

- Add text/URL ingestion and research adapters for public web/GitHub/paper sources.
  Preserve the supplied text, resolved source, revision, retrieval time and hash.
- Track claims separately from local measurements. Treat source content as evidence,
  not executable instructions. Keep source/license provenance with copied code.
- Add technique cards and version-aware compatibility checks.
- Support three implementation strategies: configuration, pinned plugin/backend,
  and code/kernel change. Create an experiment branch/worktree and diff for changes.
- Add build/test/profile tools for disposable containers or dedicated GPU VMs.
- Allow planner revision based on compile failures, numerical failures and measured
  regressions, bounded by attempt and cost limits.
- Keep evaluator/workload versions authoritative outside the candidate patch.
- Store negative results and suppress identical retries unless the fingerprint or
  hypothesis materially changes.

Acceptance: supplied technique text/link leads to cited research and a scoped plan;
the executor really changes/configures the candidate; independent correctness and
performance gates accept, reject or mark it inapplicable. A plugin that is unsupported
on the active GPU/model/version is rejected before expensive execution.

This release must show genuine code/plugin execution, not only a natural-language plan.

## Phase 5: integrate hardware and a cloud target

- Define infrastructure APIs for inventory, attach, allocate, inspect, deploy,
  scale, stop and release with idempotency keys.
- Configure one concrete provider/account/region/GPU target before provisioning.
  Build one working adapter before broad provider coverage.
- Bind resources to leases and jobs; collect hardware topology, device compatibility,
  CPU/RAM, storage and network constraints.
- Build image/artifact publication and remote worker execution, startup/warmup,
  health, termination and failed-launch cleanup.
- Record priced estimates separately from measured/billed costs. Enforce configured
  GPU-hour, concurrency, spend and maximum-replica limits.
- Implement GPU placement and replica control; schedule optimization experiments
  away from serving resources or inside an explicit maintenance allocation.
- Evaluate Ray Serve/Kubernetes as deployment adapters when justified by required
  scaling; do not make them dependencies of the first local loop.

Acceptance: deploy a supported model on the configured real GPU target, collect
evidence, restart/scale within available capacity and cleanly release resources.
Inject a launch failure and prove cleanup and budget accounting.

No provider is selected or provisioned by this planning change.

## Phase 6: connect verification to rollout and FactoryOS

- Produce immutable release manifests binding source/model/tokenizer/image/plugin/
  profile revisions and verification evidence.
- Separate candidate creation, verification and live promotion eligibility.
- Deploy a replacement worker, warm it, send representative/shadow traffic where
  appropriate, canary a bounded traffic share and monitor objectives.
- Drain existing requests; preserve a known-good release; implement automated rollback
  under the configured policy. Quality and longer-duration reliability evidence
  remain required beyond kernel/microbenchmark correctness.
- Define a versioned InferenceIssue contract with request/deployment identity,
  observations, desired SLO and workload references.
- Add issue acknowledgements, progress and outcome APIs. FactoryOS supplies a
  problem and consumes results; InferStack owns the optimization workflow.
- Route confirmed training/model-quality issues to Model Foundry and harness/tool
  issues to FactoryOS, carrying the relevant evidence.
- Include appropriate endpoint authentication, scoped credentials and resource
  isolation before exposing cloud-facing control/serving operations.

Acceptance: FactoryOS submits an inference issue; InferStack independently runs the
loop; a verified release enters a real canary; an injected serving regression causes
a rollback. Repeated callbacks or deploy retries do not duplicate live changes.

## Phase 7: broaden coverage and make the loop continuous

- Implement capability-driven new-model intake with correct baseline/fallbacks.
- Cover several supported model families and hardware environments with a measured
  engine/plugin compatibility matrix.
- Trigger bounded optimization from user requests, drift, SLO regressions, deployment
  changes and ingested research. Add source watchers only as configured services.
- Extend placement to heterogeneous/multi-node resources and distributed engine
  features when their adapters and actual hardware support them.
- Use experiment history to improve search ranking and avoid repeated failures.
- Permit a hosted optimizer model behind the same optimizer-provider interface;
  ensure its GPU allocation does not contaminate the target experiments.
- Validate control-plane failure behavior while the serving plane continues.

Acceptance: onboard another supported model and reuse relevant experience; handle
a new technique and a new inference issue through the same workflow; survive
control-service restarts and maintain an auditable release/rollback history.

A learned inference policy can be explored later after there is enough trustworthy
data. No policy-model training is required to deliver this product.

## Suggested package boundaries

Existing packages stay while interfaces are extracted incrementally.

| Package | Intended responsibility |
| --- | --- |
| `api/`, `serving/` | Serving gateway, routing and engine-independent request behavior |
| `runtime/`, `kernels/` | Native execution implementations |
| `engines/` | Engine adapter protocol and Native/vLLM/SGLang/remote implementations |
| `foundry/` | Model-artifact integration and identity resolution |
| `domain/`, `storage/` | Typed contracts, migrations, metadata and artifact repositories |
| `telemetry/`, `benchmark/`, `evaluation/` | Normalized evidence, workloads, repeated trials and deterministic decisions |
| `control/` | Durable workflow state and control API |
| `evolution/intake/`, `evolution/research/` | Problem/source normalization and research tools |
| `evolution/planning/`, `evolution/providers/` | Structured plans and optimizer model adapters |
| `evolution/execution/`, `evolution/verification/` | Candidate workspaces/builds and verification orchestration |
| `plugins/` | Technique manifests, compatibility and installation/build integration |
| `infrastructure/`, `deployment/` | Resource providers, workers, releases, traffic and rollback |
| `integrations/factoryos/` | Versioned inference feedback contract |

These directories should appear when they contain functioning implementations.
Do not add empty packages or mock production adapters to suggest progress.

## Tool and contract examples

Proposed agent tools (not existing CLI commands):

- `inspect_deployment(deployment_id)`
- `query_metrics(deployment_id, window)`
- `query_logs(deployment_id, trace_ids, window)`
- `read_profile(artifact_id)`
- `find_prior_experiments(fingerprint, technique)`
- `research_technique(source_id, question)`
- `create_candidate(plan_id)`
- `run_experiment(candidate_id, workload_id)`
- `evaluate_experiment(experiment_id, policy_id)`
- `prepare_release(experiment_id, target_id)`
- `rollout_release(release_id, rollout_policy_id)`

They operate on validated IDs and bounded evidence queries. Persistent records
include schema version, idempotency keys, timestamps and parent/attempt lineage.

An OptimizationPlan must bind target deployment, hypothesis, supporting evidence,
candidate treatment, compatible resources, action DAG, baseline/workload,
required gates, iteration/cost budget and rollback/cleanup behavior.
An ExperimentResult includes raw evidence and each gate outcome; narrative
model conclusions cannot substitute for missing measurements.

## First implementation task after review

Implement Phase 1 / v1.6.1:

1. Repair the streaming benchmark's token/timing/completion semantics.
2. Introduce validated measurement records and incomplete-evidence rejection.
3. Make the matrix distinguish execution success from optimization success.
4. Restore/reimplement repeated-trial and constrained comparison decisions.
5. Package one resumable A100 verification run; retain its real evidence.

Phase 2 database/contracts work can proceed without occupying a GPU while that
validation is pending. No architecture work depends on repeatedly repackaging ZIPs;
all source and plans live in the GitHub repository.
