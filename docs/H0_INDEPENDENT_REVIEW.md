# H0 independent completion review

Date: 9 November 2026. Initial reviewed commit: `52b9f8d` (H0: trustworthy evidence and executable contracts). Latest recheck: `eb69168`, including recovery fix `63c8644`. Production source was clean at both checks. No production code or earlier tests were changed by this review.

**Latest verdict: H0 is implemented but not accepted as complete.** H0-F3's slow-reconciliation reproduction now passes. H0-F1 (proof scope) and H0-F2 (transport-coalesced timing) still reproduce at `eb69168`. The original source-proposal/executor guards, shared profile schema and normal cleanup offloading are present. The sections below retain the initial findings and identify the resolved item.

The implementation agent reports 361 passed / 17 skipped for its full local suite and earlier probes. That is implementation-reported evidence. The independent results below are a separate check of this commit; GPU qualification remains separate in either case.

## Findings and resolution

### H0-F1 — P1: proof applicability still collapses distinct execution identities

**Location:** `src/inferstack/research/techniques.py:45–65`.

The new proof checks correctly reject failed verdicts, false required gates, recorded errors and missing/tampered artifacts. However, `reuse_key()` reduces a model record to its model revision alone and hardware to a tuple of GPU names. It omits tokenizer/config revision, source identity, image/build identity, runtime versions and hardware properties; workload/evaluator scope is not a key component either.

Starting from the implementation's own successful proof fixture, each independent mutation below still returned `compatible_proven` with `tr@1`:

- Change the target's model record to a different tokenizer revision while preserving model weight revision.
- Set a different target serving-image digest without proof of that image.
- Point the target at a different hardware record with different CUDA/driver/topology details but the same GPU name.

These are not new optimization features. They are remaining gaps in H0's narrowly scoped proof reuse and the plan's “different revision/GPU/dtype/workload does not become proven automatically” requirement.

**Fix:** use an explicit versioned applicability identity retaining model/tokenizer/config, engine/build/plugin, relevant hardware/runtime and treatment/workload/evaluator scope. Missing or differing required identity stays unknown/claimed; an unpinned reference must not silently become evidence about its current record. If a compatibility result intentionally spans environments, represent its supported scope explicitly and establish evidence for that scope. Do not grant global proven status from one trial. Add negative regressions for these three mutations and a positive genuinely matching case.

### H0-F2 — P2: transport-coalesced events still report zero ITL

**Location:** `src/inferstack/serving/evidence.py:97–110`.

The fix rejects one content event reporting two tokens. It still assumes `content-event count == usage-token count` establishes one independently timed event per token. All events parsed from one `feed()` call receive the same `now` timestamp.

The independent probe feeds two valid, separate content SSE events plus authoritative usage of two tokens in one transport chunk. The result is `content_events=2`, `output_tokens=2`, `itl_ms=0.0`. No generation interval was observed. A network read may contain several SSE events even if upstream sent them separately; event count equality also does not identify how tokens map to pieces.

**Fix:** retain client-event timing as client-event evidence, explicitly record its provenance/limitations, and mark token timing unavailable when token boundaries or timestamps are insufficiently observed. In particular, transport grouping must not become a zero token-generation measurement. Preserve the useful first-content/E2E/stall signals and keep engine timing requirements separate. Add a multi-event/single-chunk regression; the existing one-event/two-token test does not cover it.

### H0-F3 — P1: crash reconciliation blocks lease renewal — reproduction resolved

**Latest status at `eb69168`: resolved for the reported reproduction.** Commit `63c8644` offloads `reconcile_allocation()` in `reconcile_step()`. The unchanged independent slow-reconciliation probe now returns `completed`, with a terminal `reject` result and a new candidate executed after resource reconciliation. The initial finding below is retained as history. Broader crash/takeover regressions remain part of qualification.

**Location:** `src/inferstack/control/worker.py:57–67`, especially line 63.

Normal executor cleanup now offloads synchronous provider stops, but `reconcile_step()` still calls `reconcile_allocation()` synchronously on the heartbeat event loop. That helper can invoke a blocking provider stop.

The independent reproduction creates a real interrupted experiment step in temporary SQLite, expires its old lease, and starts a new real `OptimizationWorker` with a supported 300ms lease. A fake provider takes 650ms to stop the prior resource through the real reconciliation helper. The resource is reconciled, but the new worker cannot renew its lease during the stop. It returns `state=running`, `result=null`, and never starts the new candidate.

**Fix:** offload blocking reconciliation/provider operations while preserving fenced ownership and safe handling of cancellation/takeover. Prove both that heartbeats continue during slow reconciliation and that a superseded worker cannot publish. Keep unresolved ownership/resource identity conservative. Increasing test sleeps or lease lengths alone does not resolve the blocked event loop.

## Verification and reproduction

Independent regression command:

```sh
.venv/bin/python -m pytest tests/test_harness_h0.py tests/test_research.py tests/test_planner_dag.py tests/test_isolated_executor.py tests/test_live_evidence.py tests/test_model_cache.py tests/test_worker.py tests/test_durable_execution.py tests/test_release_runtime.py tests/test_wiring.py tests/test_integrations.py tests/test_rollout.py tests/test_attestation.py tests/test_releases.py tests/test_integrated_serving.py -k 'not postgres' -q --junitxml=results/h0-independent-review-2026-11-09.xml
```

Result: **179 passed, 0 failed, 0 skipped**, 153.894 seconds in the JUnit suite, independently run against `52b9f8d`. This is CPU/local-process evidence with PostgreSQL parametrizations excluded; it does not claim hosted-database, GPU or production-cluster qualification. Passing these regressions does not cover the additional boundary cases reproduced below.

Independent boundary-probe command:

```sh
.venv/bin/python docs/verification/h0_followup_probes.py
```

Observed **five reproduced counterexamples across three failure classes**: three identity mutations still receive proven status, transport coalescing returns zero ITL, and interrupted-step reconciliation loses the new worker's lease. The script uses temporary SQLite and fake provider stops, no real GPU/cluster/process resources. It reuses `proof_fixture()` from `tests/test_research.py` to demonstrate that the implementation's accepted positive case remains accepted after incompatible target mutations. These probes report defects; a successful script exit is not product acceptance.

Raw observations: `results/h0-followup-probes-2026-11-09.json`. The [probe script](verification/h0_followup_probes.py) and this report can be given directly to the H0 implementer. Preserve the original probes and add corrected-behavior regression tests; do not weaken independent acceptance to fit the implementation.

## Completion decision

Finish **H0-F1 and H0-F2**, independently rerun their negative and positive cases plus affected regression tests, and record evidence against the resulting committed source. H0-F3's reported reproduction is resolved. H1 planning may continue, but H0 must not be labeled complete or GPU-qualified from the present commit. Notebook environment/model-engine compatibility and actual A100/T4 runs remain separate qualification work.

## Recheck at eb69168

The same boundary script was rerun without changing the reproductions. Raw output: `results/h0-followup-eb69168-2026-11-09.json`.

| Boundary case | Observed | Status |
|---|---|---|
| Changed tokenizer revision | `compatible_proven` | H0-F1 remains open |
| Changed serving image digest | `compatible_proven` | H0-F1 remains open |
| Changed hardware CUDA/driver/topology record | `compatible_proven` | H0-F1 remains open |
| Two content SSE events in one transport read | `itl_ms=0.0` | H0-F2 remains open |
| 650ms interrupted-step reconciliation with a 300ms lease | `completed`, terminal result present, next candidate executed | H0-F3 reproduction resolved |

Affected regression suite: H0 acceptance, research, live evidence, control store, repository, execution store and durable execution, excluding PostgreSQL parametrizations. Independently observed **88 passed, 0 failed, 0 skipped**, 105.182 seconds, on `eb69168`. JUnit/log prefix: `results/h0-recheck-eb69168-2026-11-09`. These regressions pass while the four remaining counterexamples above still reproduce; they do not establish H0-F1/F2 acceptance or GPU qualification.
