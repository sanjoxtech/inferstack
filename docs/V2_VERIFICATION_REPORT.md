# Independent v2 implementation verification — 8 November 2026

**Verdict: changes required before accepting the integrated v2 candidate.** The
repository contains substantial working foundations, and its existing suite
passes here. However, the A–J acceptance criteria are not fulfilled by the
integrated workflow. Several failures can admit an unverified candidate, promote
the wrong release, or declare resources released while work remains alive.
These are implementation failures, separate from pending GPU/provider/cloud
qualification.

Reviewed source: `ff777007e05abcb991e84094ea8e4f68336f627e` (`2.0.0rc1` metadata).
No production implementation was changed during this review. Added this report
and independent acceptance probes; no GitHub writes, commits or external deployments.
Existing `.serena/` contents were preserved.

## Evidence and scope

Read ARCHITECTURE.md, V2_AGENT_HANDOFF.md (including its progress ledger),
CONTROL_RUNBOOK.md and IMPLEMENTATION_PLAN.md before verification. Traced the
control API/CLI, durable worker/store, record migrations, providers/planner/DAG,
research, executor/sandbox, engine adapters, measurements/numerics, infrastructure,
gateway, release/rollout, onboarding and qualification paths. Reviewed relevant
tests and the native runtime/benchmark interfaces. This is not a mathematical
audit of every Triton kernel.

Read the connected private GitHub repository, default-branch commit and package
diff, branches, PR collection, Actions runs and releases. Both `main` and
`v2/integration` point to the reviewed commit. GitHub returned no PRs, no Actions
runs and no releases. There is no newer remote implementation or CI result to
resolve the local findings. See the [reviewed commit](https://github.com/sanjoxtech/inferstack/commit/ff777007e05abcb991e84094ea8e4f68336f627e).

| Check | Observed outcome |
| --- | --- |
| Fresh external Python 3.12 environment, editable `.[dev,postgres]` plus CPU torch | Installed successfully |
| Existing complete pytest collection | **200 passed, 13 skipped**, exit 0, 99.08 seconds |
| Skips | All 13 PostgreSQL variants: no `initdb`/`pg_ctl` or configured test server |
| Independent acceptance probes | **15 failing contract assertions/scenarios**, outside default testpaths |
| Actual CPU-host qualification script | **not_qualified**, exit 2: environment/kernel-equivalence failed, native smoke skipped |
| Native CUDA, real vLLM/SGLang, real providers, Docker/Kubernetes, FactoryOS | Not validated in this environment |

The existing suite count is an observation from this environment, not a
reproduction of the handoff's exact “183 passed, 1 deselected” selection. CPU
torch tests were included here. The installed Intel macOS torch 2.2.2 emits a
NumPy 2 compatibility warning; there is no CUDA evidence from this run. Full
environment details are in `results/v2-verification/environment.json`.

Raw evidence is retained locally under `results/v2-verification/`: `baseline.log`
and `.xml`, `contract-probes.log` and `.xml`, `qualification-runner.log`, and the
qualification runner's `qualification/qualification.json` with per-gate logs.
`results/` is gitignored by the existing repository policy. The real qualification
run again passed its default test selection; no GPU/provider/cluster passes are
inferred from that result.

Reproduce the existing suite and independent probes separately:

```bash
python -m pytest -o addopts='' -q -rs
python -m pytest -o addopts='' docs/verification/v2_contract_probes.py -q --tb=short
```

The second command deliberately asserts the intended acceptance contracts and
currently exits 1. It is outside `testpaths = ["tests"]`, so it does not change the
baseline suite. Injected observations test controller/gate decisions; they are
not live GPU/cloud/performance evidence. Filesystem and process probes use only
temporary sentinels and harmless child processes, with explicit cleanup.

## Findings

P1 means a release-blocking correctness, isolation, lifecycle or integration
failure. P2 means a material recovery, operability or qualification defect.
Each finding identifies the implementation, observable consequence and repair
needed to satisfy the handoff.

### V01 — P1: candidate code can write outside its workspace

Location: `execution/sandbox.py:75`, `execution/executor.py:145` and `:233`.

The sandbox changes the working directory/environment and applies rlimits and
an optional network namespace. It does not restrict filesystem access or change
the candidate's user identity. The probe successfully rewrites a temporary
“trusted evaluator” sentinel outside the candidate workspace. Restoring test
files before running candidate code does not prevent that code from modifying
them, the control database, or other files writable by the same user afterward.
The network namespace does not create a filesystem boundary. Candidate launch
also bypasses `Sandbox.run` and its resource controls.

Fix: run untrusted builds/tests/serving in a real isolation boundary with explicit
mounts and permissions, a separately protected evaluator, and enforced process,
network and GPU limits. Refuse the source/plugin path if that boundary is absent.
Probe: `test_sandbox_cannot_rewrite_files_outside_candidate_workspace`.

### V02 — P1: declared verification policy is silently ignored

Location: `optimization/experiment.py:185`, `optimization/dag.py:132`.

The DAG accepts gates such as `numerical_oracle`, `cancellation` and `overload`,
but EndpointVerifier evaluates only its five hardcoded gates. A plan requesting
all three extra gates receives `keep` without any of them running. Its measure
action's repetitions/workload/objective references and expected measurements are
also not the policy the verifier executes; verifier construction supplies these
independently. A gate named in the plan is not enforced simply because the DAG
validated it.

Fix: bind the executable verification policy to the frozen plan; dispatch every
required gate and mark unsupported/missing gates inconclusive. Release creation
must check coverage of that policy. Probe:
`test_endpoint_cannot_keep_without_requested_numerical_and_serving_gates`.

### V03 — P1: NaN candidate and reference can pass the numerical oracle

Location: `evaluation/numerical.py:53`.

`check('x', [nan], [0.0], dtype='bfloat16', reference=[nan])` returns
`pass=True`, basis `no_worse_than_reference`, with infinite errors. The finite
check on the ordinary bound is bypassed by the fallback's `inf <= inf` comparison.

Fix: validate finite candidate, oracle, reference and resulting error statistics
before either acceptance branch. Probe:
`test_numerical_gate_rejects_nonfinite_candidate_and_reference`.

### V04 — P1: absent identity is treated as proven comparability

Location: `optimization/experiment.py:214`.

Both missing `hardware_ref` values compare equal, so `comparable=True` and a
candidate can be kept. Comparing model display names also does not establish
identical weights/config/tokenizer revisions or actual runtime identity. Even a
shared unpinned hardware reference does not freeze its observed revision. The
runbook acknowledges unrecorded hardware, but this conflicts with the handoff's
pinned-environment and unknown-is-not-pass requirements.

Fix: require complete pinned experiment identities and attested environment
comparability, with intentional treatments recorded explicitly. Missing identity
must be inconclusive. Probe: `test_unknown_hardware_cannot_satisfy_comparability`.

### V05 — P1: production rollout defaults to no attestation

Location: `releases/rollout.py:57`, `control/api.py:181`,
`control/__main__.py:26`, `engines/adapters.py:203`.

Attestation is optional. The control CLI configures the observer but supplies no
attestation function. The authenticated rollout API therefore shifts traffic
with no runtime check; the probe returns HTTP 200. The standalone attestor, even
when supplied, checks only served model name and engine version, not the pinned
weights/tokenizer/build/container/plugin identity required by the release design.

Fix: require a complete attestation before initial routing and revalidate the
replacement runtime against the release manifest. Probe:
`test_rollout_api_refuses_traffic_when_attestation_is_not_configured`.

### V06 — P1: rollout can promote a different candidate than it measured

Location: `releases/rollout.py:106` and `:114`.

After collecting observations, `step()` reads the latest route revision and
promotes whichever candidate is currently there. It does not verify that the
candidate/stable pair still belongs to this rollout. The probe replaces the
route's candidate with `rel-other`; stepping the old rollout promotes
`rel-other` using `rel-cand` observations and marks the old rollout promoted.
Using the newest revision is not sufficient concurrency protection.

Fix: bind route ownership and expected route revision to the rollout, verify both
release IDs before every mutation, and stop/reconcile conflicts. Probe:
`test_rollout_cannot_promote_a_different_concurrent_candidate`.

### V07 — P1: crash between route and rollout writes leaves unrecoverable progress

Location: `releases/rollout.py:63`, `:65`, `:81`.

Route and rollout transitions are separate transactions. An injected interruption
after the first route write leaves live candidate traffic and a `pending` rollout
with `stage_started_at=None`. A fresh controller's `step()` raises TypeError
subtracting None from the current time. The same multi-record design affects
advance/promote transitions. Existing job-worker crash tests do not establish
rollout crash safety.

Fix: transact related metadata together or persist a transition intent with
idempotent reconciliation for each route/rollout boundary. Probe:
`test_rollout_recovers_route_write_before_progress_write`.

### V08 — P1: a kept owned candidate produces a release pointing at a stopped server

Location: `control/worker.py:114`, `releases/manager.py:9`,
`execution/executor.py:239`.

The worker always stops its candidate after verification. Release creation freezes
that candidate's endpoint/command but does not provision or warm a replacement;
rollout only edits routes. A real-process probe runs a candidate, records `keep`,
creates and routes its release successfully, then gets gateway HTTP 502
(`ConnectError`). For isolated builds, cleanup also removes the source/venv that
the derived command references; retained diffs/hashes/logs are not a deployable
immutable build artifact.

Fix: persist a runnable artifact independently of the experiment workspace and
implement replacement launch, warmup, attestation, monitored routing and physical
drain/stop. Probe: `test_kept_owned_candidate_can_be_served_from_its_release`.

### V09 — P1: local cleanup confirms only the process-group leader is gone

Location: `infrastructure/local.py:94`, `infrastructure/reconcile.py:38`.

If a launcher exits with a live child in its process group, `stop()` immediately
returns True because the leader is no longer alive. Reconciliation similarly
marks the allocation released without inspecting descendants. A real-process
probe observes a surviving child while `stop()` reports success. GPU engines
with worker subprocesses can therefore outlive accounting and recovery.

Fix: track and verify the complete allocation/cgroup, preserving PID-reuse
protection, before release. Probe:
`test_local_stop_cleans_children_after_process_leader_has_exited`.

### V10 — P1: Kubernetes deletion is mistaken for completed GPU cleanup

Location: `infrastructure/remote.py:173`, `:181`.

`stop()` deletes with `--wait=false` and checks only whether the Deployment UID
still exists. Deployment removal is not proof its ReplicaSets/Pods terminated.
The injected asynchronous-deletion probe leaves a running Pod while `stop()`
returns True. Discovery also searches only Deployments, so leftover ancillary
resources after partial failure need additional reconciliation.

Fix: use ownership-safe deletion, discover all resources by allocation, and wait
for associated Pods/GPU work to terminate before releasing the allocation. This
probe is a CLI contract simulation, not a live cluster result. Probe:
`test_kubernetes_stop_waits_for_pods_not_just_deployment_object`.

### V11 — P1: local GPU budgets default to zero usage

Location: `optimization/experiment.py:27`, `control/worker.py:105`,
`control/store.py:314`.

Local allocation creation reads `context.get('gpu_count', 0)`. The worker's
context contains only the job and step key; neither CLI nor executor resolves
GPU count/placement from a reservation. A native launch is recorded with zero
GPUs and therefore zero GPU seconds/spend regardless of duration. Recording an
allocation row also does not reserve exclusive devices or prevent experiment
contention with serving.

Fix: require an explicit validated resource reservation before launch, propagate
its device identity/count, and enforce budget/quota/admission against it.
Unknown GPU allocation must not become zero. Probe:
`test_native_local_gpu_launch_does_not_default_gpu_usage_to_zero`.

### V12 — P1: isolated builds block lease renewal and cancellation

Location: `execution/executor.py:209`, `execution/sandbox.py:84`.

Async `execute()` calls synchronous build methods and `Popen.communicate` on
the worker event loop. A harmless 0.4-second build blocks a 10 ms heartbeat loop
for approximately 0.47 seconds. Real pip/build/test commands can last minutes,
exceeding the default 30-second job lease; cancellation/deadline tasks cannot run
during that interval. Build processes also precede the serving allocation intent.

Fix: use supervised asynchronous subprocesses with allocation/attempt ownership,
ongoing heartbeats and cancellation that kills the process group and waits for
confirmed cleanup. Probe: `test_isolated_build_yields_to_worker_heartbeats`.

### V13 — P2: resume at an exhausted budget drops previous results

Location: `control/worker.py:69`, `:133`, `:158`.

History starts empty on a fresh worker. Budget checks happen before durable
successful steps are reused. If the prior experiment consumed the last allowed
experiment, restart short-circuits before restoring it and returns
`experiments=[]`, `decision=inconclusive`. Original step/event rows survive, but
the terminal outcome no longer carries the required retained history.

Fix: rehydrate durable history before checking whether new work may start; permit
reading reused results without spending additional budget. Probe:
`test_restart_at_exhausted_budget_preserves_prior_experiment_results`.

### V14 — P2: a latency-gated 100% stage cannot obtain a stable comparison

Location: `releases/rollout.py:81`, `:102`, `:111`.

The stage window resets on advancement. At 100% candidate routing there are no
new stable requests in that window once older requests finish. With
`max_latency_regression` configured, the controller waits forever for stable
latency. Existing successful-promotion tests supply fixed stable observations
even at 100%, masking this ordinary traffic pattern.

Fix: define a comparison policy for the final stage: retained qualified baseline,
shadow/control probes, or a suitable earlier comparison followed by candidate
absolute SLO gates. Probe:
`test_latency_gated_rollout_can_finish_100_percent_stage`.

### V15 — P2: qualification accepts a failed kernel process if stdout claims pass

Location: `scripts/v2_qualify.py:70`.

The kernel gate ignores subprocess exit status once it parses a report. The
probe supplies exit 1 with `pass:true`; the gate says `passed` and the injected
qualification run says `qualified`. It also accepts an empty check collection.
The test-suite gate records skips but does not distinguish missing required
coverage, and the vLLM “conformance” branch always skips even if vLLM is installed.
The script therefore does not establish all package acceptance criteria.

Fix: require successful process exit, a complete expected check manifest, and
valid finite results; define deployment-specific required integration gates and
classify their skips explicitly. Probe:
`test_qualification_does_not_accept_nonzero_kernel_process_exit`.

### V16 — P1: implemented components are not connected in the supported worker path

Location: `control/__main__.py:76`, `:79`, `:83`,
`optimization/experiment.py:49`, `control/worker.py:124`.

Source inspection confirms the CLI always constructs EndpointExecutor. It never
selects IsolatedExecutor, so validated profile-patch/plugin/source plans fail as
unsupported. The worker is constructed without a fetcher, so submitted technique
URLs become “retrieval not configured” evidence despite the runbook describing
worker retrieval. ToolPlanner receives no live telemetry callback. No remote
image-build path connects locally patched sources/plugins to the pinned image
launched by Docker/Kubernetes. These components exist, but the three required
entry points cannot demonstrate the complete specified workflow through this CLI.

Fix: explicitly configure and wire executor selection/build identity, safe
retrieval and adapter-backed telemetry; verify config/plugin/source requests
through the actual control API and worker entrypoint. Evidence here is source
tracing rather than an additional live-provider test.

### V17 — P2: runtime and package versions disagree

Location: `src/inferstack/__init__.py:1`, `pyproject.toml:7`.

Package metadata says `2.0.0rc1`; `inferstack.__version__`, CLI `--version`, native
health and native status still say `1.6.0`. A deployment expecting rc1 will fail
the version comparison; reports cannot consistently identify the candidate.

Fix: use one version source and check package/CLI/API agreement. Verified directly
with `importlib.metadata.version('inferstack')` and `inferstack.__version__`.

## Acceptance assessment by work package

| Package | Evidence that works here | Acceptance still missing or contradicted |
| --- | --- | --- |
| A — contracts/migrations | Typed records, reference rejection, SQLite migration/history/CAS tests pass | Mandatory pinned runtime identity is not enforced by verification/release eligibility; PostgreSQL not revalidated here |
| B — durable execution | Existing fencing, step, cancellation and transition-crash tests pass | V09/V11/V12/V13; cleanup, budget and retained-result guarantees are incomplete |
| C — engines/telemetry/onboarding | CPU adapter processes, gateway isolation/admission/disconnect and normalization tests pass | Real native/vLLM/SGLang conformance, complete phase telemetry, cache isolation evidence and immutable runtime attestation |
| D — research | Provenance/integrity, local retrieval and SSRF regression tests pass | CLI URL wiring (V16), search adapter and dependency/repository license work remain |
| E — planner/providers | Mocked provider protocols, bounded DAG validation and read tools pass existing tests | Live provider smoke, actual telemetry wiring, binding executable gates and measurement policy (V02/V16) |
| F — executor | Existing patch filtering, local source/plugin/config tests pass | V01/V12/V16; real isolation, immutable runnable artifacts, remote candidate image build, native profile changes and patch generation |
| G — verifier | CPU ULP/math and aggregate judge tests pass | V02/V03/V04; no integrated numerical/semantic/load/tail/goodput/cost policy proving the handoff's wrong/slow/incomplete-candidate scenarios |
| H — infrastructure | CLI-double lifecycle and price-accounting tests pass | V10/V11/V16; live cluster, resource reservations/quotas, scale/drain lifecycle and image build pipeline |
| I — rollout/integrations | Existing canary regression/rollback and feedback deduplication tests pass | V05–V08/V14; replacement deployment/physical drain, crash/concurrency safety, durable callback outbox, bounded periodic/SLO triggers and FactoryOS end-to-end |
| J — qualification | Consolidated runner and Colab instructions exist | V15/V17; no CI workflows/runs, dependency lock or container build pipeline in the reviewed tree; deployment-specific live acceptance remains open |

FactoryOS intake/status is useful partial integration. It is not the specified
end-to-end feedback/outcome/callback workflow: there is no durable outbox or
periodic/SLO trigger service in the reviewed source. This is missing planned
scope, not merely a missing external account. Likewise, waiting for an A100 run
will not repair the control-plane defects above.

## Recommended repair order

1. Close unsafe acceptance paths first: V01–V05. Freeze an executable evaluation
   policy and complete identities; unsupported checks must fail closed.
2. Make resource ownership real: V09–V12, allocation reservations and cleanup
   across child processes, Pods and build steps.
3. Produce immutable runnable artifacts, wire the supported worker path, and
   implement replacement launch/warmup/attestation before route changes (V08/V16).
4. Fix rollout ownership, transaction recovery and final-stage evidence
   (V06/V07/V14), then exercise kill/restart and concurrent operator actions.
5. Repair retained-result, qualification and version semantics (V13/V15/V17),
   reconcile the documentation ledger, and add CI and reproducible builds.
6. Run the three entry points through one workflow, then perform the consolidated
   native GPU, provider and chosen cluster qualification with complete evidence.

Keep the substantial working foundations. Do not interpret green unit/integration
tests or the current “implemented: yes” ledger as certification of A–J acceptance.
The review supports continued integration work; it does not support a v2.0 final
release or a production-qualified claim.

## Remediation — 8 November 2026

Implementation agent response to V01–V17. Every probe in
`docs/verification/v2_contract_probes.py` now passes (15/15). Two probes were
amended, both marked in the probe source with "Amended after review":

- `test_sandbox_cannot_rewrite_files_outside_candidate_workspace` asserted the
  tampering command exits 0. That presumes a copy-on-write view; a deny boundary
  fails the write closed. The amended probe requires the outside write to be
  refused, a write inside the workspace to succeed, and the sentinel to be unchanged.
- `test_kept_owned_candidate_can_be_served_from_its_release` used deployments with
  no hardware/model identity, which V04 (correctly) makes inconclusive. The fixture
  now declares pinned identities. Its cleanup retires the retained runtime.

| Finding | Repair | Evidence |
| --- | --- | --- |
| V01 | Sandbox has a real write boundary: macOS Seatbelt or Linux bubblewrap; writes confined to the workspace, credential paths unreadable, offline steps have no network. Candidate servers launch inside it with rlimits (trusted `limit_exec`, `python -I`). Plugin/source experiments are refused without a boundary. | probe; `test_sandbox_confines_writes_reads_and_network`; isolated end-to-end test asserts the server command is wrapped |
| V02 | Verification policy = mandatory gates + every plan gate + measure params. Cancellation and overload run as probes; other gates are unsupported (None → inconclusive). Unsatisfiable repeats/workload/objective/measurements → inconclusive. Policy hash in evidence; releases require every policy gate True. | probe; `test_release_requires_evidence_covering_its_policy` |
| V03 | Non-finite candidate, oracle, reference or empty comparison never passes (also `check_relative`). | probe |
| V04 | Comparability needs pinned, equal hardware and model identities resolved to revisions; missing → None (inconclusive), different → False. A failed gate rejects even when others are unknown. | probe; `test_missing_identity_is_not_comparable` |
| V05 | Control API refuses rollouts without an attestor; CLI wires `release_attestor`, which checks served model, model/tokenizer revision, engine version, image or build digest and plugins. Unpinned manifest fields are unverified. | probe; `tests/test_attestation.py` |
| V06 | Rollout records route body + revision it wrote; any other writer moves it to `conflict` without touching the route. | probe |
| V07 | Two-phase route transitions (intent persisted, route written, progress recorded); a restarted controller completes or rejects the intent idempotently. Rollout schema v3. | probe; audit-trail test |
| V08 | A kept candidate's process is retained under `serving:<deployment>` ownership; isolated builds keep the workspace as a tree-hashed runnable artifact in the release. `ReleaseRuntime` reuses, replaces (warm + attest), and retires (drain + verified stop) runtimes; rollout uses it before traffic and after promotion/rollback. | probe; `tests/test_release_runtime.py` |
| V09 | Local stop covers the process group and tagged descendants; reconciler checks residual members. | probe |
| V10 | Kubernetes stop waits for the allocation's Pods (owner-chain aware, same-name replacements excluded); unknown API state is never "released". | probe; reconciliation test |
| V11 | GPU reservation resolved from reservation → deployment `resources` → engine minimum (GPU engines ≥ 1, zero refused); exclusive device reservation across live allocations; `CUDA_VISIBLE_DEVICES` set. | probe |
| V12 | Builds run in a worker thread; cancellation kills the build process group and waits; a build allocation (intent + tag) precedes the first build process. | probe; failed-build cleanup test |
| V13 | Durable step results are read before budget checks. | probe |
| V14 | 100% stage compares with the stable latency retained from the last stage with stable traffic; no baseline → hold. | probe |
| V15 | Kernel gate requires exit 0, the full 27-check manifest, finite statistics; test-suite skips classified; `--require` makes integration gates blocking. | probe |
| V16 | `control/wiring.py`: isolated executor selection with pinned clean source revision, SSRF-safe retrieval by default, adapter-backed planner telemetry; container providers refuse locally built candidates (no image pipeline). Planner prompt lists registered metrics. | `tests/test_wiring.py` |
| V17 | One version source (`inferstack.__version__`, dynamic in pyproject). | `tests/test_version.py` |

Live evidence gathered in this pass (not CPU doubles):
- Docker daemon 29.9.2: `DockerProvider` launch → readiness → label discovery →
  observed image digest equals pinned digest → verified stop (`tests/test_docker_live.py`).
- Providers: OpenRouter (`deepseek/deepseek-chat`, tools) produced a validated plan
  (6,875 tokens). DeepSeek (`deepseek-chat`) produced a validated plan (18,794 tokens)
  on one of two runs after the metric-list fix; the other exhausted 8 rounds. Treat
  DeepSeek planning as working but not yet reliable.
- PostgreSQL 18.6 (hosted Neon, direct TLS endpoint): all 13 PostgreSQL store/lease/
  fencing variants passed (26/26 with SQLite, 671 s over the network); the fixture
  now drops every test database (0 left behind).

Still open (unchanged by this pass): GPU qualification (native, vLLM, SGLang,
kernel equivalence on A100), Kubernetes cluster run, candidate image build
pipeline for container providers, logit/semantic quality oracles, FactoryOS
end-to-end, CI workflows and dependency lock.

## Remediation, second pass — 8 November 2026

Closes the residual limits of the first pass and the report's non-numbered scope.

| Item | Change | Evidence |
| --- | --- | --- |
| V05 residual | `launch_observer`: running command line (model/tokenizer revision), Docker image ID/digest, Pod `imageID`, recomputed runnable digest; wired into the control CLI attestor. | `test_live_runtime_is_attested_from_observed_launch` (real process) |
| V08 residual | Gateway `POST /gateway/drain/{deployment}`; `http_drain`; control `serve` builds a `ReleaseRuntime` (launch/warm/attest, drain, verified stop) for rollouts. | `test_retire_drains_in_flight_requests_through_the_gateway` (real gateway, in-flight request completes) |
| V16 residual | `ImageBuilder`: offline `docker build --network none` from the pinned base, recorded wheelhouse and patched source; content-addressed image; optional registry push for Kubernetes. Container providers now run plugin/source candidates instead of refusing. | live: plugin installed in image, container runs exactly the built image ID |
| FactoryOS | Migration 5. Durable outbox row written in the job's terminal transaction; `Idempotency-Key` delivery with backoff, dead-letter and re-arm; HTTPS-only callbacks; `outbox-deliver` CLI. | lost-ack redelivery yields one effect; 503s retried; 400 dead-lettered |
| Triggers | Periodic and SLO triggers: mandatory budget, cooldown, daily cap, one job in flight per trigger, global cap, window-idempotent submission; missing telemetry never fires. `triggers-tick` CLI, API. | `tests/test_integrations.py` |
| Kubernetes lifecycle | `scale` (waits for exact ready/pod count), graceful `drain` (preStop sleep + grace period), namespace `ResourceQuota`/`LimitRange`, non-root UID; control-plane quotas (`max_live`, `max_gpus`) and GPU-seconds integrated over scale history. | live kind v1.37 run: quota blocks the 3rd replica, drain to 0, verified stop |
| CI / lock / containers | `uv.lock` + hashed `requirements*.lock` (`requires-python >=3.11`: transformers 5.22 needs it); `Dockerfile` (pinned base, `--require-hashes`, non-root, bubblewrap); `.github/workflows/ci.yml` (lock check, CPU+PostgreSQL matrix, probes, container, Docker live, kind, manual provider smoke). | local: lock check, hashed install on clean 3.11, image builds and reports the version. Not yet run on GitHub Actions. |
| Phase telemetry | Streaming pass (client TTFT, SSE-event gaps), engine histogram deltas (queue/prefill/decode/TTFT/TPOT), authoritative `output_tokens_per_s`, p95 only with >= 200 samples, metric source recorded per metric. | `tests/test_quality_phases.py` |
| Quality | Graded cases (token F1 / contains) with a declared `quality_budget`; `logit_agreement` gate from engine logprobs; unsupported → inconclusive. | same file |
| Qualification | `engine_conformance.py` (vLLM/SGLang live launch-and-check), `postgresql`/`docker`/`kubernetes` gates, `gpu_profile` per run. | runner checked against an emulated engine (not GPU evidence) |
