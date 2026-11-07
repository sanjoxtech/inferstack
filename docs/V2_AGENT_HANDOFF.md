# InferStack v2.0 — implementation handoff

Planning date: 7 November 2026. Owner of this handoff: planning agent.
Delivery: one integrated v2.0 candidate; internal commits and iterations are welcome.
Do not require the user to install or benchmark each intermediate version.
This document supersedes release-by-release delivery in IMPLEMENTATION_PLAN.md.
ARCHITECTURE.md remains the product architecture; this document defines execution.

## 1. Goal and project boundaries

InferStack hosts supported models and improves measured serving goodput under
latency, quality, reliability, memory and cost constraints. It owns an inference
optimization harness: research/planner → isolated executor → independent verifier
→ bounded revision loop → verified release → monitored rollout.

FactoryOS owns application agents, tools and harness engineering. Model Foundry
owns training/adaptation and source model artifacts. InferStack owns inference
engines, serving representations, infrastructure integration and inference
optimization. Improvements to training or application harnesses must be routed
back to their owner with evidence, not silently implemented here.

Required entry points share one workflow:
- Optimize an existing deployment against explicit objectives.
- Investigate a supplied technique (text first, accessible URL also supported).
- Diagnose inference feedback from FactoryOS or internal telemetry.

The optimizer provider is interchangeable with the target serving model. Support
DeepSeek and OpenRouter first, with an explicit capability-based provider contract.
Do not require a particular provider model ID or embed credentials in code.

## 2. Verified starting point and limits

Committed code baseline: `ed4586914628e2ffa99c88fee7f5e63fb163d55e` in
`sanjoxtech/inferstack`. Package version still 1.6.0. Prior implementation session
reported 57 passing CPU tests at this commit. Reproduce that result in a clean
environment; it is not evidence of live provider, GPU or cloud validation.

| Area | Available at baseline | Still needed |
| --- | --- | --- |
| Native runtime | Qwen batching/cache/scheduler/kernel paths | Hardware qualification, effective-feature evidence |
| Engine integration | HTTP adapter, explicit local subprocess supervision | Versioned capabilities, resource placement, runtime attestation |
| Measurements | Usage accounting, client first-content timing, native emission TPOT | Phase traces, goodput/cost, workload/tail sample policy |
| Persistence | SQLite leases/events, CAS records, hashed artifacts | Migrations, shared-host DB, recovery and side-effect reconciliation |
| Optimization | JSON planner selecting registered candidates; bounded worker | Tool-driven research, generated actions, code/plugin execution |
| Verification | Repeated endpoint trials; exact-match suite; mean E2E judge | Numerical/semantic/load gates and held-out evaluator isolation |
| Releases | Frozen records, weighted routes, manual promotion/rollback | Frozen runtime artifacts, rollout monitoring, actual draining |
| Integration | General job submission with source label | Versioned FactoryOS issues and Foundry artifact contracts |

Local draft work after that commit added research text ingestion, technique
compatibility and planner context, but was NOT pushed or validated when the user
switched to planning. Treat it as optional draft input, not delivered functionality.
Files touched: research/__init__.py, research/evidence.py, optimization/planner.py,
control/api.py, tests/test_research.py, and local documentation additions. A clean
GitHub checkout will not contain these changes. Do not depend on their presence.

## 3. Non-negotiable implementation rules

1. Use real adapters and execution. Test doubles belong in tests; unsupported
   capabilities return explicit unsupported/inconclusive states.
2. Keep serving independent of the optimizer. Separate process/container per
   deployment; native global patches must not leak between model instances.
3. Pin model/tokenizer/config revisions, engine/container/plugin versions, workload,
   evaluation policy and hardware identity. A URL or mutable record is not a runtime identity.
4. An LLM explanation cannot satisfy a correctness or release gate. Candidate code
   cannot alter trusted evaluators, held-out fixtures or baseline artifacts.
5. Missing metrics never become zero. SSE events never become generated token counts.
   Distinguish client content timing, engine emission timing and GPU kernel timing.
6. Reserve resources for experiments. Do not compare candidates under uncontrolled
   optimizer/serving contention and call the difference an optimization.
7. Every external side effect needs an identity, ownership, budget and cleanup path.
   A lease fences DB writes; it does not stop an orphan process or release a cloud GPU.
8. Do not overwrite concurrent repository changes. Work from a fresh checkout,
   follow applicable AGENTS.md instructions, preserve existing changes, and push
   reviewable commits. Avoid large refactors unrelated to the work package.

## 4. Execution order and work packages

Each package must add functioning behavior, tests and runbook updates. These are
internal work packages, not user-facing releases. Record implementation status and
validation evidence per package; do not assign a fictional completion percentage.

### A. Baseline reconciliation and contracts

Inspect the baseline source and reproduce the tests. Review uncommitted drafts
if available. Convert loosely structured records into versioned schemas for model,
deployment, hardware, workload, objective, evidence, technique, plan, trial,
release, rollout, resource allocation and feedback.

Define canonical IDs, immutable snapshots, CAS revisions, timestamp units,
metric definitions, schema migration rules and redaction/retention settings.
Add a capability matrix with supported/unsupported/unknown evidence for each
engine and operation. Unknown must not be treated as supported.

Acceptance: existing tests pass; old records migrate without losing history;
invalid cross-record references fail; incompatible schema versions fail clearly.
Primary areas: storage/, control/, engines/, core/.

### B. Durable execution and resource ownership

Extend jobs to persist step state, attempts, retry eligibility, typed errors,
absolute deadlines and budgets (provider tokens, wall time, GPU time, spend).
Use SQLite for local use and PostgreSQL for shared control-plane operation;
provide tested migrations and equivalent lease/CAS semantics.

Introduce worker/allocation IDs, heartbeats and resource leases. Persist side-effect
intent before launch, attach the identity to processes/containers/cloud resources,
and reconcile ambiguous launches after restart. Cancel/timeout must stop work,
collect partial evidence and release resources. Resume only idempotent steps.
Add an operator-visible orphan state and recovery action.

Acceptance: process-kill/restart tests at every transition; stale worker cannot
publish; repeated requests do not create duplicate resources; orphan cleanup is
verified; budget exhaustion stops new steps while retaining results.
Primary areas: control/worker.py, control/store.py, storage/, new infrastructure/.
Depends on A.

### C. Engine lifecycle, telemetry and model onboarding

Implement native, vLLM and SGLang adapters against pinned supported versions.
Cover launch/attach, ready, stream/complete, cancel, metrics, graceful drain and
stop. Report actual backend/feature activation. Add per-model health and admission
so an unhealthy deployment does not globally block healthy models.

Inventory GPU UUID/type/VRAM/topology, CPU/RAM, driver/runtime/library versions.
Capture request/trace identity and queue, tokenization, prefill, decode, cache,
sampling and transport observations where available. Missing remote telemetry
must remain unavailable. Store raw observations with normalized mappings.

Onboarding resolves immutable Foundry/Hugging Face artifacts, checks task/operation
support, selects a reference path and produces a compatible execution profile.
Use immutable revisions; never silently accept main as a reproducible identity.

Acceptance: real multi-model adapter conformance on declared environments;
stream disconnect and overload tests; cache isolation; unsupported architecture
fails clearly; capability assertions match runtime observations.
Depends on A/B. Primary areas: engines/, serving/, api/, foundry/, telemetry/.

### D. Research ingestion and technique registry

Implement text ingestion with source attribution, retrieval timestamp where
applicable, content hashes, provenance type and verification status. Deduplicate
content without losing distinct source lineage. Record pinned code revisions and
license metadata for candidate dependencies.

Add bounded URL retrieval/search adapters with explicit timeout, maximum size,
redirect policy, content-type handling and private-network restrictions. Prevent
SSRF including redirects/DNS rebinding; record blocked/unavailable sources. No
login-wall bypass is required. Supplied text must work without search access.
Treat source content as untrusted evidence, never as execution authority.

Technique manifests declare engine/runtime versions, model operations, dtype,
shapes, hardware requirements, dependencies and expected benefit/risks. Separate
compatibility claims from compatibility proven by tests. A claim becomes a
candidate experiment, not an accepted optimization.

Acceptance: corrupt evidence fails integrity checks; citations resolve to stored
snapshots; incompatible techniques are excluded; unavailable URLs preserve a
usable job outcome; hostile content cannot request privileged tool actions.
Depends on A; planner integration in E. Primary areas: research/, plugins/.

### E. Tool-driven planner and provider adapters

Replace candidate selection as the only planning action with a typed experiment
DAG: hypothesis, baseline snapshot, evidence references, treatment, prerequisites,
resource/cost estimate, action types, expected measurements, gates and stop rule.
Actions include configuration changes, pinned plugin installation and source patch
experiments. Validate the DAG and policy before execution.

Expose narrow tools for deployment/history lookup, bounded telemetry queries,
profiles, code inspection, research and compatibility checks. Validate tool
arguments and results; cap tool calls/context/provider consumption. Normalize
provider errors, structured output and optional tool support; preserve raw outputs
for audit with secrets removed. Use failed-attempt history to avoid identical retries.

Acceptance: both configured provider adapters execute a planning smoke test;
malformed outputs, invented citations and unsupported actions are rejected;
planner cannot write directly to production or declare its own verdict.
Depends on A/D and telemetry contracts in C. Primary areas: optimization/, providers/.

### F. Isolated executor for configuration, plugins and code

Materialize a candidate in an isolated checkout/worktree and container with a
pinned environment. Apply validated config or a patch, install pinned compatible
plugins, build, run preflight checks and produce immutable build artifacts.
Keep source-diff/build-log lineage and failed attempts. Use supported third-party
extension points or explicit versioned forks.

Constrain filesystem mounts, network, secrets, process/GPU resources and time.
Never mount the container-engine socket into candidate code. Trusted evaluator
runs separately. Execute argument arrays through allowlisted action handlers;
arbitrary text from research is not a shell command. Candidate may not rewrite
baseline, trusted fixtures or the active deployment.

Acceptance: three real paths (config/plugin/source) create distinct reproducible
artifacts; failed build/cancel/timeout cleans resources; evaluator tampering is
blocked; active serving remains unchanged during an experiment.
Depends on B/C/E. Primary areas: optimization/, sandbox/, plugins/, artifacts/.

### G. Independent verification and selection

Build versioned verification policies appropriate to the change:
- Kernel/backend: numerical oracle with justified dtype/shape tolerances.
- Quantization/approximation: task-quality budget and representative held-out evals.
- Serving: completeness, cancellation, cache isolation, liveness and overload.
- Performance: matched workload/environment, warmup, balanced repeated trials,
  concurrency/arrival patterns, cold/warm cache strata, noise and uncertainty.

Measure TTFT, TPOT/ITL with explicit definitions, E2E, throughput, SLO goodput,
memory, failure rate and cost where authoritative data exists. Set sample-adequacy
rules before collecting tail metrics. Apply hard constraints before epsilon-Pareto
comparison and ranking. Preserve keep/reject/retest/inconclusive with reasons;
retests consume budget. Independent evaluation must not trust candidate-generated
pass flags. Repair legacy analyzer paths that default missing metrics to zero or
mark a matrix passed without the full required gates.

Acceptance: deliberately wrong/slow/noisy/incomplete candidates fail or remain
inconclusive correctly; correct faster candidate passes only its declared policy;
raw evidence reproduces the verdict; unavailable cost cannot satisfy a cost SLO.
Depends on C/F and B budgets. Primary areas: evaluation/, scripts/, tests/.

### H. Infrastructure and deployment controller

Provide an attached-host adapter (including ephemeral Colab experiment workers)
and one concrete cloud execution target. Planning default: an existing GPU-enabled
Kubernetes cluster, portable across cloud providers. This is deployment/resource
management, not automatic cloud-account or VM-cluster creation. If raw VM
provisioning is required, implement a named provider adapter as additional scope.

The adapter must inventory, allocate, launch, inspect, scale, drain, stop and release
resources, with allocation identities, limits and cost observations. Pin container
digests; qualify storage/model download and GPU runtime compatibility. Add resource
quotas and reject budget violations. Use external credentials supplied at runtime.
A real account/cluster, region, GPU quota and spend ceiling are required for live
validation, but not to finish code, manifests and contract tests.

Acceptance: one real attached-host lifecycle and one real cluster lifecycle;
retry/restart does not duplicate allocations; deletion/cancellation reconciles
resources; unavailable price/quota is explicit. No unbounded allocation loops.
Depends on B/C/F. Primary areas: infrastructure/, deployment/, deployment manifests.

### I. Releases, staged rollout and integrations

Bind release identity to model/tokenizer revision, container/build digest, plugins,
profile, placement requirements, workload and verification-policy/evidence hashes.
Attest worker runtime against the manifest; frozen records alone are insufficient.

Launch and warm replacements, establish health, shift traffic in configured
stages, monitor latency/errors/quality signals, promote or roll back automatically
under policy. Drain in-flight streams with a deadline before stopping old workers.
Persist rollout progress and recover without losing the previous stable release.

Define versioned FactoryOS InferenceIssue and outcome contracts with trace IDs,
idempotency and deduplication. Use a durable outbox for authorized callbacks.
Route model/harness/inference issues with evidence and an unknown category.
Add explicit periodic/SLO triggers with cooldown, concurrency limits and budgets;
manual requests and internal triggers enter the same pipeline.

Acceptance: induced canary regression rolls back; restart mid-rollout is safe;
streams drain; duplicate feedback creates one workflow; callbacks retry without
duplicate effects; continuous triggers cannot create unbounded jobs.
Depends on G/H, with integration contracts from A. Primary areas: releases/,
serving/gateway.py, integrations/, control/.

### J. Consolidated qualification and handoff

Produce one installation/runbook and one Colab notebook with GPU detection,
preflight checks, pinned revision checkout, capability-aware test matrix and
artifact export. Test native BF16 reference first; include quantized/compiled/
third-party variants only where supported. T4 and A100 are separate qualification
profiles, not interchangeable benchmark evidence.

Include a CPU smoke path, authenticated configuration examples, database backup/
restore/migration guidance, dependency lock strategy, CI, container builds and
failure recovery instructions. CI should distinguish CPU conformance from GPU,
provider and cloud integration evidence. Skips must name the missing requirement.

Acceptance: a new checkout follows the runbook successfully; no secrets required
for CPU checks; user performs one consolidated environment run; report explicitly
separates implemented, CPU-tested, GPU-tested, provider-tested and cloud-tested.
Only then tag a v2.0 release candidate; do not label it production-qualified until
the advertised live environments and operational checks pass.
Depends on all packages.

## 5. Dependency map

A → B; A → D; A/B → C; C/D → E; B/C/E → F;
C/F → G; B/C/F → H; G/H → I; all → J.

Implement sequentially where contracts change. Parallel work is optional only
when explicitly assigned by the project owner; this plan does not spawn agents.
Use internal integration commits frequently. No intermediate user test request is
required merely because a work package finished.

## 6. End-to-end acceptance scenarios

| Scenario | Required observable outcome |
| --- | --- |
| Optimize hosted model | Reads real telemetry, creates candidate, measures matched trials, stores independent verdict |
| Supplied text technique | Stores provenance, checks compatibility, researches if configured, executes isolated experiment |
| FactoryOS inference issue | Correlates traces, deduplicates, runs workflow, returns versioned outcome |
| Incorrect faster kernel | Numerical gate rejects despite speed improvement |
| Lower-quality quantization | Quality budget controls acceptance; no claim of numerical identity |
| Missing/noisy metrics | Inconclusive or bounded retest; never implicit pass |
| Worker/control-plane crash | Recovery fences stale writes and reconciles external resources |
| Bad canary | Automatic rollback preserves stable service and drains streams |
| New unsupported model | Explicit capability gap and onboarding outcome, no fabricated deployment |
| Budget exhausted | No new costly action; retained evidence and cleaned resources |

All three entry points must be demonstrated through the same real workflow, not
three unrelated demos. One pass of mock HTTP tests does not satisfy live adapter
or GPU qualification.

## 7. Agent start instruction

Start from the GitHub baseline, read this handoff and ARCHITECTURE.md, inspect the
current branch and applicable AGENTS.md, reproduce tests, and record the observed
state. Execute A through J without asking the user to approve routine reversible
implementation decisions. Finish all code and offline checks before requesting
missing environment details for live qualification. Do not invent credentials,
cloud limits or a successful GPU run. Update this document's progress ledger in
each implementation commit with files changed, checks run, evidence and remaining
limitations. Keep the original native engine functional throughout.

Progress at handoff: A–J are planned; partial foundations exist as described in
section 2. No package is certified complete by this planning document.

## 8. Progress ledger

Each entry records what exists, how it was checked and what remains. Status words:
implemented, CPU-tested, GPU-tested, provider-tested, cloud-tested.

### Baseline reproduction — 7 November 2026

- Commit `6e306777da6d1a0607d1e6ab277adec96766e581`, Colab A100-SXM4-40GB
  (`GPU-9f3e5751-d767-8e95-f6e0-6dece69bfc96`), torch 2.14.1+cu130, CUDA 13.0,
  Triton 3.8.0, transformers 5.x.
- `pytest`: 57 passed, 0 failed, 0 skipped (count taken from result markers; the
  doubled `-q` suppressed pytest's summary line).
- `scripts/v160_kernel_equivalence.py` reported `pass: false` (add_rmsnorm norm
  max_abs 0.03125 vs 3e-2; silu_mul 0.0625 vs 3e-2). An FP32-oracle diagnostic over
  three seeds showed the fused kernels at max 0.5 ULP / mean ~0.25 ULP with no
  element above 1 ULP, while the BF16 references reached ~1.37 ULP max (double
  rounding). Verdict: kernels correctly rounded; the script's fixed absolute
  tolerance against a BF16 reference is the defect. Repair belongs to Package G
  (numerical oracle: FP32 reference, ULP bound). No performance claim was made; the
  fusion matrix was not run.

### Package A — contracts, schemas, migrations, capabilities

Status: implemented, CPU-tested.

- `contracts/records.py`: versioned pydantic schemas for model, hardware,
  deployment, workload, objective, evidence, technique, plan, trial, release,
  route, rollout, allocation and feedback (InferenceIssue). Canonical ID pattern,
  `id`/`id@N` references, pinned references for plans/trials, immutable model
  revisions, registered-metric enforcement, `None` for missing metrics.
- `contracts/metrics.py`: canonical metric definitions (unit, source, aggregation),
  separating client content timing, engine emission timing and sampled GPU memory.
- `contracts/redaction.py`: literal-credential rejection, redaction, retention
  settings contract (TTL enforcement deferred to B/J).
- `storage/migrations.py`: numbered, transactional migrations with ledger table;
  v1.6 databases upgrade in place; newer databases fail clearly.
- `storage/repository.py`: immutable `record_history`, typed `put_record` with
  reference resolution and `record_refs`, `read(revision=)`, `history`,
  `read_typed`, `referrers`. Untyped `put` retained for internal callers.
- `engines/capabilities.py`: supported/unsupported/unknown matrix with basis and
  evidence per engine/operation; `require()` refuses unknown.
- Control API: typed record writes (422 invalid/incompatible/secret, 409
  conflict/dangling), history, capabilities and schemas endpoints.
- `DeploymentSpec.from_record` so schema-v1 deployment fields do not break
  existing launch/route consumers. Releases and routes use the typed path.
- Checks: `pytest` CPU subset 38 passed (27 existing + 11 new in
  `tests/test_contracts.py`); torch-dependent tests not re-run in this
  environment (no code they cover was changed).
- Remaining: PostgreSQL dialect for the same migrations (Package B); TTL
  enforcement; runtime observations refining the capability matrix (Package C);
  schema v2 plan as experiment DAG (Package E).

### Package B — durable execution and resource ownership

Status: implemented, CPU-tested (SQLite and PostgreSQL 16).

- `storage/db.py`: one database layer for SQLite and PostgreSQL; a single writer
  lock per transaction (SQLite `BEGIN IMMEDIATE`, PostgreSQL advisory xact lock)
  makes lease/fencing/CAS semantics identical. Trade-off: no concurrent writers.
- `storage/migrations.py`: dialect-aware DDL; migration 4 adds job controls
  (worker, deadline, budget, usage, max_attempts, backoff, last_error),
  `job_steps`, `workers` and `allocations`.
- `control/store.py`: durable steps (`reuse`/`run`/`ambiguous`), lease-fenced
  usage charges, derived experiment and GPU-second usage, typed failure with
  retry/backoff, deadline-at-claim, cancellation keeping the token so the last
  holder can attach partial evidence once, worker registry with heartbeats.
- `control/errors.py`: typed errors; only explicit transient classes retry;
  messages scrubbed of credential substrings.
- `control/worker.py`: plan (idempotent, reused) and experiment (side-effecting,
  reconciled before re-run) steps; budget check before every step; absolute
  deadline; partial evidence on cancel/deadline/failure.
- `infrastructure/`: allocation intent before launch, idempotency key per
  job/step/attempt/purpose, process identity (host, pid, pgid, kernel start
  time) and environment tag, verified stop before release, reconciler for dead
  workers, `orphaned` state and operator recovery (retry or force with note).
- API/CLI: submission budget/deadline/max_attempts, allocations, recover,
  workers, `--database`, `reconcile` action; worker reconciles on start.
- Checks: 75 CPU tests passed. New: 26 store tests (13 x SQLite + PostgreSQL,
  real temporary cluster), 15 real-process tests — a worker subprocess killed
  (`os._exit(137)`) at each of 10 transitions then a fresh worker resumes, plus
  wedged-worker takeover, cancel, deadline, budget and orphan recovery. Mutation
  checks: reconciler releasing without stopping fails 2 crash tests; removing
  lease fencing from allocation intent fails 3 tests.
- Remaining: process provider is local-host only (containers/Kubernetes in H);
  spend accounting needs authoritative prices (H); provider token charging is
  wired when planners report usage (E); retention TTL enforcement job (J).

### Package D — research ingestion and technique registry

Status: implemented, CPU-tested. Not tested against the public internet (this
environment's egress is allowlisted); retrieval was exercised against a local
HTTP server with an injected address policy and resolver.

- `research/evidence.py`: normalized text ingestion; content-addressed storage
  with one evidence record per (content, source) for deduplicated lineage;
  idempotent re-ingestion; integrity-checked reads; citation resolution to pinned
  snapshots; nonce-delimited untrusted envelope for prompts.
- `research/fetch.py`: SSRF-guarded retrieval — address policy enforced inside the
  transport on every connection and redirect hop, any-bad-address rejection,
  vetted-IP dialing (no rebinding window), redirect/downgrade policy, time/size
  caps (declared and streamed), content-type allowlist, HTML-to-text, proxies off.
- `research/techniques.py`: claim vs proof assessment against deployment engine
  version, dtype, hardware (name, compute capability, memory), capability matrix
  and dependency pinning; incompatible techniques excluded from candidates.
- Schemas: evidence v2 (source key, availability, reason, bytes, content type)
  and technique v2 (shapes, hardware minimums, required capabilities, status,
  proof required for `proven`); v1 records remain valid.
- Worker `research` step (idempotent, unreachable URLs recorded and non-fatal);
  API evidence/text, evidence/url, evidence read, technique candidates; supplied
  `technique_text` ingested on submission; planner renders evidence as untrusted
  data, strips raw text from context and pins cited revisions.
- Checks: 96 CPU tests passed (21 new in `tests/test_research.py`).
- Remaining: search adapters (no search provider configured); license detection
  for fetched code repositories; tool-driven research in the planner (E).
