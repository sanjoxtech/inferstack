# Independent second review — 8 November 2026

**Verdict: changes required.** The remediation fixes the original probe cases,
but the claim that all 17 findings and remaining v2 scope are closed is broader
than the behavior verified here. Fifteen additional acceptance probes reproduce
remaining defects and regressions. Several affect serving availability, trusted
measurements, resource ownership or credential isolation.

Reviewed local commit: `faa5cda47023ed54d6c5d364351f3f304e6059de`.
Reviewed changes: the five commits after `ff777007`, including executor/sandbox,
resource ownership, release runtime and attestation, rollout recovery, verification
phases/quality, outbox/triggers, qualification, CI and dependency/build files.
No production implementation was changed in this review. Added only this report
and `docs/verification/v2_second_pass_probes.py`; existing `.serena/` was preserved.

## Source and validation evidence

When checked through the connected GitHub API, both `main` and `v2/integration`
still pointed to `ff777007e05abcb991e84094ea8e4f68336f627e`. The local branch was
five commits ahead. Thus the remediation reviewed here is local; a fresh GitHub
checkout still gets the old implementation. No remote writes or pushes were made.

| Check | Result |
| --- | --- |
| Revised repository suite in default environment | 233 passed, 14 skipped: 13 PostgreSQL variants and 1 Kubernetes test |
| Earlier acceptance probes, including the documented fixture amendments | 15 passed |
| Combined suite + earlier probes | 248 passed, 14 skipped, exit 0, 158.34 seconds |
| Live Docker lifecycle and plugin image build | Both passed within the repository suite, Docker 29.9.2 |
| Separate live Kubernetes test, existing local `kind-inferstack` context | 1 passed, 45.64 seconds; quota/scale/drain/stop, CPU only |
| `uv lock --check` | Passed |
| Independent second-pass probes | 15 reproduced failures, no collection errors |
| Separate hosted PostgreSQL variants, configured `.env` endpoint | 13 passed, 0 skipped, 701.18 seconds; 0 `test_` databases left afterward |

The `.env` connection remains slow from this machine: four connection attempts
had median setup 2415.7 ms; twenty `SELECT 1` round trips had median 319.1 ms.
Client TLS was active. Those measurements do not establish a slow database engine:
they include network, connection negotiation and the configured service endpoint.
The earlier medians were 2108.6 ms setup and 326.2 ms per query.

The independent hosted PostgreSQL run completed successfully, including lease,
CAS, fencing, budget, cancellation and worker-registry variants. A read-only
cleanup check confirmed zero `test_` databases remaining. Database connectivity
and these store semantics are verified; the failures below are not caused by a
missing PostgreSQL service.

No live provider calls were made during this second review. The previous
implementation agent's provider evidence is not a new independent result.
Native CUDA, vLLM/SGLang GPU conformance and GPU scheduling remain unverified
here. The kind lifecycle test is not GPU-cluster qualification.

The two amended original probes are reasonable: a deny-based filesystem boundary
should refuse an outside write, and a release fixture must supply identities now
required for comparability. Their passing results do not establish the broader
contracts exercised below.

## Findings

P1 identifies a release-blocking failure of a stated correctness, isolation,
ownership or serving guarantee. P2 identifies a material integration or
qualification defect. Every numbered finding has a corresponding executable
probe in `docs/verification/v2_second_pass_probes.py`.

### R01 — P1: the sandbox can still read host-project credentials

Location: `execution/sandbox.py:76` (`seatbelt_profile`) and `:131` (`wrap`).

The new write restriction works, but reads are allowed across the host except
for a fixed list of common credential directories. Project `.env` files are not
excluded. A confined process successfully read a fake secret in a temporary
host-project `.env` outside its workspace. The actual repository `.env` was not
read by the probe. The Linux read-only root bind has the same read-access issue.
Serving candidates run with network access, so environment scrubbing alone does
not protect credentials reachable through files.

Repair: expose only necessary readable mounts, hide project/config credentials
and process credentials, and validate both read and write isolation for builds
and serving. Do not rely on a blacklist of known home directories.
Probe: `test_candidate_cannot_read_host_project_dotenv`.

### R02 — P1: a declared measurement may be absent while the verdict is keep

Location: `optimization/experiment.py:277` (`policy`) and `:496` (`complete`).

The verifier checks that `expected_measurements` names are supported, but does
not require those measurements to exist in the collected trials. Completeness
checks only configured objective metrics. A plan expecting both `e2e_ms` and
`output_tokens_per_s` gets `keep` with every output-throughput observation absent
when only E2E is configured as an objective. This leaves V02 partially open.

Repair: enforce finite observations and source/coverage requirements for all
measurements required by the frozen policy, or return inconclusive. Include the
effective objectives, quality thresholds and fixture binding in that policy.
Probe: `test_missing_declared_measurement_cannot_keep`.

### R03 — P1: streaming TTFT counts metadata as generated content

Location: `optimization/experiment.py:388` (`phases`).

The new streaming pass splits reads on `data:` and treats every event except
`[DONE]` as content without parsing JSON. An assistant-role event with an empty
content field at 1 ms followed by actual content at 100 ms reports client TTFT
of 1 ms. Usage/finish events also affect chunk gaps, and fragmented SSE events
are not buffered. An objective can therefore reward protocol packaging changes
instead of faster generated content. This regresses the established measurement
contract, despite the older `StreamMeasurement` implementation handling content.

Repair: reuse a buffered SSE parser and the existing content-aware measurement
logic; preserve transport gaps separately and handle incomplete/error streams.
Probe: `test_ttft_ignores_role_and_usage_metadata`.

### R04 — P1: logprob agreement accepts incomplete candidate sequences

Location: `optimization/experiment.py:434` (`logit_agreement`).

Both cases and tokens are compared using `zip`; extra baseline cases/tokens are
silently omitted. Candidate `['the']` with one matching logprob passes against
baseline `['the','cat','sat']` with three logprobs, reporting zero divergence.
No sequence-length or complete-coverage check protects this acceptance branch.

Repair: require complete matching case/token structure, validate logprob lengths
and finiteness, and record coverage. Distinguish missing evidence from genuine
token disagreement. Greedy selected-token logprobs should retain that narrow
scope; they are not a full-distribution or teacher-forced numerical oracle.
Probe: `test_logprob_gate_cannot_accept_a_truncated_matching_prefix`.

### R05 — P1: a frozen release's expected model revision remains mutable

Location: `releases/manager.py:30` and `releases/attestation.py:79`.

Verification resolves identity references to revisions, but release creation
copies the deployment's original possibly-unpinned `model_ref`. Attestation later
resolves that reference against the latest model record. The probe creates an
actual kept-verdict release via ReleaseManager, updates the model record using
CAS, and observes the old release's expected weights revision change from
`bbbb…` to `cccc…`. This changes the identity expected of a previously verified
release, without changing its manifest digest.

Repair: freeze resolved model/tokenizer/config/hardware/plugin snapshots in the
release and use those exact revisions for deployment and attestation. Reject
unresolved mutable references at release creation.
Probe: `test_release_model_identity_does_not_follow_later_record_updates`.

### R06 — P1: route conflict can still stop the currently serving stable runtime

Location: `releases/rollout.py:175` and `:218`.

The two-phase transition correctly detects a route conflict, but the final-stage
caller ignores the state it returns and unconditionally retires the old stable.
The probe changes the route back to `rel-base` during observation. The stored
rollout becomes `conflict`, but `step()` reports `promoted` and calls
`retire('rel-base')`—the release still serving the live route. This can cause an
outage despite the new route-ownership checks.

Repair: honor the returned transition state and ownership before every physical
side effect; conflict must never trigger retirement. Re-check ownership across
the observation awaits and record retirement intent with the owned transition.
Probe: `test_rollout_route_conflict_does_not_retire_live_stable`.

### R07 — P1: runtime retirement is not recoverable after a terminal route transition

Location: `releases/rollout.py:169`, `:219` and `:155`.

The rollout becomes `promoted`/`rolled_back` before retirement runs. An interruption
or exception at retirement leaves that side effect unfinished. A restarted
controller sees the terminal state and returns immediately, without retrying it.
The probe injects the first retirement failure and observes exactly one call
after another `step()`. Serving-owned allocations are exempt from normal stale
worker reconciliation, so this can leave persistent charged resources.

Repair: persist retirement/drain steps and their ownership/status; terminal
success must include their confirmed completion or a durable cleanup obligation
that recovery actually resumes. Probe:
`test_terminal_rollout_retries_interrupted_runtime_retirement`.

### R08 — P1: post-promotion rollback routes to a stopped previous runtime

Location: `releases/rollout.py:220`, `control/api.py` route rollback handler,
`releases/runtime.py:112`.

Promotion now physically stops the previous stable. The existing rollback API
still changes only route metadata and never calls `ReleaseRuntime.ensure` to
restore it. A real-process probe starts both runtimes, promotes the candidate,
calls the authenticated rollback API (HTTP 200), then gets gateway HTTP 502 from
the restored route. With gateway draining configured, the old deployment's
admission flag also remains draining; no release lifecycle path undrains it.

Repair: warm and attest the rollback target before changing traffic, restore
admission, and retire the losing runtime only after the safe transition. Define
how rollback artifacts/resources are retained and recovered.
Probe: `test_post_promotion_rollback_restores_a_live_previous_runtime`.

### R09 — P1: the control server cannot manage container release runtimes

Location: `control/__main__.py:55`, `releases/runtime.py:54`, `:57`, `:85`, `:119`.

The control server always constructs ReleaseRuntime with the local process
provider, even though attestation knows about Docker/Kubernetes providers.
ReleaseRuntime does not dispatch by allocation provider. Retiring a retained
Docker allocation calls LocalProcessProvider and raises `KeyError: 'pid'`.
Container deployments also have empty local commands, so an unavailable container
is treated as an attached endpoint rather than relaunched through `launch`.
Live candidate image-build tests do not cover this production release path.

Repair: dispatch readiness, launch, identity and stop through the release's actual
provider/context; persist placement and remote container specification. Test
retention, replacement, promotion and rollback through the configured control CLI.
Probe: `test_control_runtime_dispatches_container_retirement_to_its_provider`.

### R10 — P1: build allocation rows are not linked to actual build processes

Location: `execution/executor.py:203`, `:250`, `:254`; `execution/sandbox.py:174`.

A build intent row is written, but its ID is never passed to Sandbox.run as
`INFERSTACK_ALLOCATION_ID`. No external identity or active state is recorded for
the build. A real confined subprocess prints `MISSING` for its allocation tag;
the row is later released with `external=None`. After a worker crash the
reconciler cannot find these untagged processes and may report `never_launched`.
The image-build subprocess path similarly is not owned by Sandbox's live-process
set, and its build allocation has already been released.

Repair: give each build/image step a durable identity, tag/record every subprocess
and external build resource, and keep ownership live through verified teardown.
Exercise hard-kill/restart during pip, preflight and image build, not just normal
build failure. Probe: `test_build_process_carries_its_persisted_allocation_identity`.

### R11 — P1: process-group cleanup regresses protection against PID/PGID reuse

Location: `infrastructure/local.py:137` (`members`) and `:161` (`stop`).

The new descendant cleanup includes every process with the recorded PGID even
when the leader's start token no longer matches and none has the allocation tag.
Once an old group disappears, its ID may later be reused by an unrelated group.
The probe supplies that stale identity and intercepts signals: the provider sends
both SIGTERM and SIGKILL to the unrelated reused group. No real process receives
those intercepted signals in the test.

Repair: verify ownership of each member/group using durable start/namespace/tag
identity; do not infer current ownership from a historical numeric PGID alone.
Uncertain ownership must become an operator-visible unresolved state.
Probe: `test_stale_process_group_identity_never_signals_reused_group`.

### R12 — P1: implicit GPU reservations still allow uncontrolled device sharing

Location: `optimization/experiment.py:16`, `:65`, `infrastructure/allocations.py:64`.

The engine-minimum fallback fixes zero GPU accounting, but returns no device
identity. Exclusivity is enforced only when `spec.gpu_devices` exists; the worker
does not supply a reservation by default. Two GPU-engine launches both proceed
with count 1, without CUDA_VISIBLE_DEVICES or identified device reservations.
They may contend on the same default device and contaminate serving/benchmarks.
Replacement serving allocations additionally bypass this reservation path.

Repair: resolve actual hardware/device placement and reserve it atomically before
launch; fail closed on unknown placement. The count is an accounting quantity,
not an exclusive GPU lease. Probe:
`test_two_implicit_gpu_reservations_do_not_launch_on_unidentified_devices`.

### R13 — P2: concurrent trigger ticks can exceed the global/per-trigger cap

Location: `integrations/triggers.py:118` and `:133`.

Admission checks happen in one transaction, then job submission happens in
another. Two identical tickers straddling a normal interval boundary both pass
the checks before either submits; their window-specific idempotency keys differ.
The probe creates two pending jobs for the same trigger despite `max_active=1`
and the advertised one-unfinished-job-per-trigger rule.

Repair: transact admission, deduplication, job insertion and firing-state updates
together, or use a durable fenced reservation/unique active-trigger constraint.
Probe: `test_trigger_global_admission_is_atomic_across_tickers`.

### R14 — P2: cancellation omits the FactoryOS terminal outcome callback

Location: `control/store.py:191` (`cancel`); compare `finish`/`fail` outbox writes.

The new durable outbox is written for ordinary completion/failure but not for
cancellation. A callback-enabled job becomes terminal with no outbox row. The
deadline-before-claim branch similarly bypasses the normal failure enqueue path.
FactoryOS receivers waiting for the promised terminal outcome can wait forever.

Repair: route every terminal transition through an atomic, idempotent outcome
enqueue path and define how later partial cancellation evidence updates it.
Probe: `test_cancelled_factoryos_job_enqueues_terminal_callback`.

### R15 — P2: optional integration skips prevent ordinary GPU qualification

Location: `scripts/v2_qualify.py:100` (`suite_gate`).

On a GPU host the suite gate treats every skip except PostgreSQL-related skips
as missing required coverage. However, the same script explicitly makes Docker
and Kubernetes optional unless requested with `--require`. A Colab GPU run with
no Docker daemon or Kubernetes context therefore fails `test_suite` just because
those optional tests skip, even when neither integration is required. The probe
reproduces this contradictory gate result.

Repair: classify skips by the selected qualification profile and required gates.
Optional integration tests should be excluded or recorded separately; requested
integration skips must remain blocking. Probe:
`test_optional_container_skips_do_not_fail_gpu_suite_gate`.

## Assessment of the previous remediation

Confirmed improvements include the outside-write boundary, finite numerical
rejection, missing-hardware inconclusive behavior, required API attestation,
metadata route-transition recovery, retained verified candidate, descendant/Pod
cleanup, nonzero GPU accounting, offloaded builds, exhausted-budget result reuse,
100% stage baseline retention, kernel-process exit validation, worker wiring,
version consistency, dependency lock and real container lifecycle paths.

Those improvements are worth retaining. The new probes show that V01, V02,
V04–V09, V11, V12 and V16 still have material residuals or regressions. V03,
V13, V14 and V17's original cases passed here; V10's revised CPU/live kind
scenarios passed; V15's original kernel-report problem passed but its expanded
qualification logic has R15. These are scoped assessments, not certification
that every possible edge case of the other findings is resolved.

The current local code is meaningfully further along than the GitHub baseline,
but passing the original fifteen probes does not establish completion of A–J.
The highest-priority repairs are R06/R08 (serving outage), R01 (credential reads),
R02–R05 (verification/identity), then resource ownership and durable cleanup.
After fixes, run old and new probes, provider-aware rollout/rollback tests,
hard-crash build/retirement tests, concurrent trigger tests, and the selected GPU
qualification. Push the reviewed source before relying on a fresh GitHub/Colab
checkout or GitHub CI evidence.

## Reproduction and artifacts

```bash
python -m pytest -o addopts='' -q -rs tests docs/verification/v2_contract_probes.py
python -m pytest -o addopts='' -q --tb=short docs/verification/v2_second_pass_probes.py
```

The second command currently fails all fifteen contract probes; these are
deliberately outside the default repository testpaths. They use temporary fake
credentials, local processes and controlled provider/clock observations. No real
keys are used or disclosed. The live hosted database tests create and drop their
own test databases, and the kind test creates/deletes its own namespace.

Raw evidence is retained locally in `results/v2-second-review/` (gitignored):
suite/previous-probe JUnit XML and log, second-pass probe XML/log, hosted PostgreSQL
and local kind results, and a small machine-readable summary. Production source
and the implementation agent's remediation ledger were left unchanged.

## Remediation — 8 November 2026 (implementation agent)

All fifteen findings were fixed. The fifteen second-pass probes are unchanged and
pass; the fifteen original probes pass, with one amendment noted below. CI runs
both probe files. New regression tests are in `tests/test_second_review.py`.

| Finding | Fix |
| --- | --- |
| R01 | Sandbox reads are an allowlist: system roots, this Python and its site-packages, the inferstack package, the HF model cache, declared extras (local wheel index, command program dirs) and the writable paths. Seatbelt denies `file-read-data` elsewhere and keychain mach lookups. bwrap binds only those roots, with its own PID namespace. Verified on macOS, and on Linux in a privileged container: host `.env` unreadable or invisible, other PIDs hidden, venv and pip still work. |
| R02 | The policy binds measurements, objectives, quality budget, logprob limit and fixture digest. A declared measurement that is not observed, finite, on every trial makes `complete` None (inconclusive). |
| R03 | Buffered SSE parser feeding `StreamMeasurement`. Only non-empty content counts; role, usage and finish events are metadata. A fragmented event is timed when complete. A stream with an error or no terminal event yields no TTFT. |
| R04 | Complete case/token coverage is required. Malformed or missing evidence gives None, and a different greedy sequence (including length) gives False. Coverage is recorded. |
| R05 | `ReleaseManager.freeze_identity` stores resolved pins (model/hardware `id@rev`, weights/tokenizer revisions) in the release, and an unresolved mutable reference is rejected. Attestation uses the frozen identity. Without one, an unpinned reference is unverified. |
| R06 | Ownership is re-checked after the observation awaits. A transition ending in `conflict` never triggers retirement, and retirement skips any release still on the live route. |
| R07 | The terminal transition writes `retirement: pending` atomically. It is marked done only after the runtime confirms the stop, and `step`/`settle` resume a pending obligation. |
| R08 | The rollback API warms (relaunching if stopped), attests and undrains the target before moving traffic, then retires the loser. Gateway: `POST /gateway/undrain/{id}`. |
| R09 | `ReleaseRuntime` dispatches launch/alive/stop through the allocation's recorded provider and launches container releases through their placement. The container spec is persisted on the allocation. The control server passes every configured provider, and an unconfigured provider marks the allocation orphaned instead of raising. |
| R10 | The build allocation is active with a tag-only identity. Every sandbox step, and the docker build CLI (plus an image label), carries `INFERSTACK_ALLOCATION_ID`. It is released only after no tagged process remains, and image build now happens inside that ownership. A hard-kill reconcile test is included. |
| R11 | Ownership comes from the live leader (start token) or the tag. A historical PGID with its leader gone is never signalled, and untagged members are reported uncertain, so stop returns False and the allocation becomes orphaned. |
| R12 | `place_devices` reserves identified free devices atomically under the writer lock, for experiments and for serving replacements. The child gets `CUDA_VISIBLE_DEVICES`. Device UUIDs come from the inventory (or `infrastructure.gpu_devices` / `INFERSTACK_GPU_DEVICES`). An unidentified GPU launch is refused. |
| R13 | Trigger admission (re-read row, cooldown, in-flight, global, daily), job insert and firing update run in one writer transaction (`Store.submit_in`). |
| R14 | Cancellation and deadline-before-claim enqueue the outcome callback in their terminal transaction. The callback is idempotent per job. |
| R15 | `suite_gate` classifies skips by owning integration. Optional ones that were not required are recorded as `skipped_optional`; a required integration's skip blocks; any other skip blocks on a GPU host. |

Original probe amended: `test_native_local_gpu_launch_does_not_default_gpu_usage_to_zero`
now supplies one identified device. Under R12 a launch without one is refused,
and the probe still checks that GPU accounting is nonzero. Some test fixtures were
also adjusted. Fake streams now end with `[DONE]`, attestation fixtures pin
`model_ref`, and the isolated executor fixture declares fake device IDs.

Local qualification after the fixes (`--require postgresql --require docker
--require kubernetes`): verdict `not_qualified` on CPU, blocked only by
environment, kernel_equivalence and native_smoke (no GPU). test_suite: 256 passed,
0 skipped. Hosted PostgreSQL: 26 passed. Docker: 2 passed. kind Kubernetes: 1 passed.
Both probe files: 30 passed.

Still unverified here: GPU device placement on real NVIDIA hardware, vLLM/SGLang
conformance, Kubernetes GPU scheduling, container rollback through the live
control CLI against a gateway process, and a GitHub Actions run (commits not pushed).
