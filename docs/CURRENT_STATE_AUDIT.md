# Current-state audit

Reviewed November 7, 2026 against commit
`33f897e39f484447036867e7c7e4c296e1b50806`.

Scope: the current private GitHub snapshot and matching local files.
Older conversation claims, feature names and historical benchmark summaries are
not evidence that their implementations or raw measurements exist in this snapshot.

## Capability inventory

| Area | Current evidence | Status and gap |
| --- | --- | --- |
| Native execution | [QwenHybridContinuousEngine](../src/inferstack/runtime/qwen_continuous.py), native kernels and runtime tests | Implemented Qwen-oriented candidate; current GPU correctness/performance not validated in this review |
| Scheduler/cache/sampling | Chunked prefill, live refill, compaction, cancellation, prefix-state cache, sampler and diagnostics | Reusable native foundation; requires real-workload and GPU validation |
| Forward fusion | [qwen_forward_fusion.py](../src/inferstack/kernels/qwen_forward_fusion.py), ablation profiles, equivalence script | Candidate implementations with fallbacks; no new speedup established |
| API | [app.py](../src/inferstack/api/app.py) | Completions/chat/streaming, health/readiness/model/status routes; no common managed remote-engine gateway or public-deployment access controls |
| Model service registry | [service.py](../src/inferstack/serving/service.py) | In-memory multi-service container; not a durable fleet/deployment registry |
| CLI serving path | [cli.py](../src/inferstack/cli.py), lines 9-16 | Constructs one Qwen native engine and service; does not dispatch serving by engine/model capability |
| Third-party engines | [catalog.py](../src/inferstack/engines/catalog.py), lines 4-16 | Descriptions/install checks and command construction for vLLM/SGLang; no common start/attach/health/stop/proxy adapter lifecycle |
| Generic remote API | Catalog entry only | Listed, not a functioning adapter in this snapshot |
| Model Foundry integration | Filesystem manifests and Hugging Face loader in [foundry/](../src/inferstack/foundry/) | Artifact-loading integration, not training or a persistent model lifecycle service |
| Observability | In-memory native counters/status, nvidia-smi polling, runtime/operator profiling scripts | Useful experiment instrumentation; no persistent per-deployment telemetry pipeline, time-series collection or trace index |
| Benchmarking | Streaming client, concurrency sweep, fusion matrix and historical reference JSON | Useful experiments with measurement/decision defects listed below |
| Statistical judge | Search across source/scripts found no CV/MAD/epsilon-Pareto judge implementation | Prior v1.4 behavior must be recovered or reimplemented and verified |
| Knowledge database | No SQL/storage repositories for techniques, jobs, plans or experiments | Missing |
| Optimization harness | No planner/researcher/executor/verifier workflow | Missing |
| Research tools and provider adapters | No optimizer provider or source-ingestion implementation | Missing |
| Technique/backend plugin management | Qwen-specific patch/install functions | No general compatibility/provenance/build/rollback registry |
| Infrastructure/cloud lifecycle | Device strings and local subprocesses | No provider adapters, placement/inventory manager, provisioning, budget enforcement or fleet lifecycle |
| Release promotion | No immutable release registry, canary or traffic controller | Missing |
| FactoryOS issue ingress | No versioned inference issue API | Missing |

## Measurement and decision findings

### A1. Streaming events are counted as tokens

Location: [benchmark_client.py](../scripts/benchmark_client.py), lines 25-43.
Every parsed data event appends a timestamp, and `len(chunks)` becomes
`output_tokens`. Metadata, finish events and multi-token chunks break this
assumption. The client also starts first-token timing before confirming a valid
content event.

Reproduced locally with a synthetic protocol fixture: an empty metadata event,
two content events ("hello", " world"), and a finish event carrying
`usage.completion_tokens = 2`. The client returned `output_tokens = 4`.
This fixture checks protocol accounting, not real GPU behavior.

Impact: cross-engine TPS and TPOT can be wrong. The sweep's tokenizer argument
is currently not used to correct the count.

Required repair: authoritative usage/token IDs, separately named chunk gaps,
explicit estimates if necessary, first-content semantics, stream-completion
validation, and fixtures for grouped/empty/malformed/truncated streams.

### A2. Matrix success is not an optimization verdict

Location: [v160_colab_matrix.py](../scripts/v160_colab_matrix.py), lines 33-40,
79-89 and 103-107.

The summary checks numerical-equivalence status, readiness, request success,
feature probes and runtime profiler status. Output agreement is recorded but
has no required threshold. Improvement against baseline, SLO feasibility and
operator-profile success are not mandatory parts of the final `pass` expression.
Analysis runs after the pass decision and its status is not included in it.

Reproduced the request-acceptance function with a successful row lacking latency
and throughput metrics; it returned `pass: true`.

Required repair: separate execution-complete, correctness, quality, evidence-complete,
performance and promotion-eligible outcomes. Evaluate required artifacts and
analysis failures. A passing experiment execution must not imply a winning candidate.

### A3. Missing metrics become zeros

Location: [v160_analyze_results.py](../scripts/v160_analyze_results.py), lines 15-18.
Missing TTFT/TPOT maps to 0.0. Reproduced `metric({}, 'tpot') == 0.0`.

Impact: a missing result may look like excellent latency in a report.
Missing operator rows can likewise look like eliminated operators; the operator
profiler exports only its top 40 rows.

Required repair: unknown/missing state, finite-value validation, evidence
completeness and complete required-operator counts. Never treat absence from
a truncated top-operations list as proof of removal.

### A4. Reproducibility and tail confidence are incomplete

The matrix uses a fixed order, one sweep per lane and a small request count.
Reports do not bind all artifacts to a complete model/tokenizer/build/workload/
environment fingerprint. Manifests use source revision `main`; the kernel
equivalence script loads a fixed model's current config.

The GPU monitor samples only the first `nvidia-smi` row, making it unsuitable
as a fleet/multi-GPU metric. Sampled peak VRAM is not an exact allocation maximum.

Required repair: resolve revisions, preserve original raw rows, record timing
semantics and cache policy, gather repeat trials with uncertainty, qualify p99
sample adequacy, and identify each physical GPU and deployment. Historical
reference JSON is useful context but insufficient for a new promotion verdict.

### A5. Multi-model isolation and API behavior need platform hardening

The registry supports several services in memory, but the CLI only constructs
one. Aggregate readiness requires all services ready. Native kernel patch state
and fusion counters are module-global. The HTTP models have no numerical bounds
for generation parameters, and finish reason is always reported as `stop`.
Non-streaming disconnect handling is not explicitly wired to native cancellation.

Required repair: per-deployment worker isolation, independent health/routing,
capacity reservations, validated request contracts, accurate finish/usage metadata,
and disconnect/overload/backpressure behavior tests. External credentials,
authentication and tenant/resource controls become required for a cloud-facing endpoint.

### A6. Feature declarations do not prove active capabilities

The native runtime explicitly reports `paged_kv_active: False` and
`actual_cuda_graph_replay_proven: False`. Flags and architecture-routing
labels are not an implemented general model router or evidence of graph replay.
Fallback paths may serve correctly while disabling the intended optimization.

Required repair: expose supported/configured/effective/validated states separately,
and bind capability evidence to each deployed build and model.

## What to preserve

- Native scheduler/cache/request abstractions and their behavioral tests.
- Qwen-specific kernel experiments and per-feature diagnostics.
- Filesystem model-artifact integration, extended through explicit contracts.
- CLI/API compatibility where practical.
- Benchmark/profile artifacts as raw evidence with correct interpretation.

Do not replace these with placeholder implementations. Wrap and incrementally
extract reusable interfaces. Do not claim old v1.4 functionality survived the
snapshot without locating its code and tests.

## Validation limits

Source paths and control flow were inspected directly. The protocol-accounting,
missing-metric and acceptance examples above were reproduced with standard-library
fixtures; no model call or GPU run was made. PyTorch, pytest, FastAPI and httpx
are not installed in this execution environment. The full test suite and A100
equivalence/performance matrix were therefore not run here. The earlier 12/12
static audit is a wiring check, not GPU correctness or production readiness.

The [implementation plan](IMPLEMENTATION_PLAN.md) turns these findings into
release gates.
