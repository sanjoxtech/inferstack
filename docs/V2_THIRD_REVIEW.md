# Independent third review — 8 November 2026

**Verdict: the previous probe cases are fixed, but release-blocking residuals
remain.** Six additional probes reproduce five issues in replacement execution,
launch isolation, retirement recovery and container placement.

Reviewed commit: `f50aab1eb92e03dce154461dd832d3112ae29fd8`, following `faa5cda`.
Inspected the remediation diff, its regression tests and the complete affected
runtime/sandbox/ownership paths. Production implementation and the implementation
agent's ledger were not changed. Added this report and
`docs/verification/v2_third_pass_probes.py`; existing `.serena/` was preserved.

## Validation

| Check | Independent result here |
| --- | --- |
| Revised repository suite | 242 passed, 14 skipped (13 PostgreSQL, 1 Kubernetes) |
| Both earlier probe files | All 30 passed |
| Combined suite and earlier probes | 272 passed, 14 skipped, 160.28 seconds |
| Targeted hosted PostgreSQL store checks | 4 passed: restart/idempotency, stale-worker fencing, competing claims, cancellation; 198.41 seconds; zero test databases left |
| Live Docker lifecycle and candidate image build | Both passed within the repository suite |
| Additional live Docker ReleaseRuntime lifecycle | Passed: launch, healthy reuse, replacement after runtime death, retirement; every allocation released |
| Dependency lock | `uv lock --check` passed |
| New third-pass probes | 6 reproduced failures, no collection errors |

The targeted PostgreSQL selection exercises the changed `Store.submit_in` entry
path and cancellation/fencing behavior. The entire thirteen-variant hosted suite
was not repeated at this commit; it passed at the preceding reviewed revision.
No provider API calls or new Kubernetes/GPU qualification were performed here.
The live Docker release check explicitly selected the Docker provider and used
CPU only. It does not establish the Kubernetes placement cases below.

When checked through the connected GitHub API, `main` and `v2/integration` still
pointed to `ff777007e05abcb991e84094ea8e4f68336f627e`; local `main` was six commits
ahead. The repaired implementation is therefore not yet what a fresh GitHub
checkout or its CI would verify. No remote writes were made during this review.

## Findings

### T01 — P1: replacement runtimes lose the verified launch environment and inherit control-process secrets

Location: `src/inferstack/releases/runtime.py:150`,
`src/inferstack/execution/executor.py:288`,
`src/inferstack/optimization/experiment.py:94`.

IsolatedExecutor launches patched source with a scrubbed environment containing
its workspace `PYTHONPATH`; plugin launches also depend on the selected environment.
The release stores the command and runnable tree digest, but not that safe launch
environment. ReleaseRuntime reconstructs only CUDA_VISIBLE_DEVICES and calls
`provider.spawn` with its default `inherit_env=True`.

Two real-process probes confirm the consequences:

1. A sandboxed candidate module launches successfully from a workspace using
   PYTHONPATH. Its retained runnable tree still matches the frozen digest. After
   injected runtime death, replacement fails with `No module named
   audit_candidate_server` and never becomes ready because PYTHONPATH was lost.
2. A sandboxed replacement can read an invented `AUDIT_ONLY_FAKE_TOKEN` from the
   control process's environment. It writes only `visible`/`hidden` to a temporary
   marker; no real credential was inspected. Filesystem restrictions do not
   remove inherited environment variables. Real control/provider credentials
   present in that environment would cross the same boundary.

Repair: freeze an allowlisted, credential-free execution specification alongside
the runnable artifact, including the required Python/environment/module paths and
working directory. Reconstruct it for replacements and launch with
`inherit_env=False`, updating only deliberate runtime variables such as reserved
devices. Reuse the original isolation and environment construction path.

Probes:
`test_retained_runnable_replacement_preserves_verified_pythonpath` and
`test_sandboxed_replacement_does_not_inherit_control_process_secrets`.

### T02 — P1: launching an absolute script grants access to sibling credentials

Location: `src/inferstack/execution/sandbox.py:232`.

The revised read allowlist correctly blocks the previous direct host `.env`
probe. However, `Sandbox.launcher` automatically adds the parent directory of
every absolute file argument to readable roots. An absolute server script in a
host project consequently grants reads to that whole project, including its
sibling `.env`.

The new probe executes a temporary host-project `server.py` through the actual
candidate launcher. It successfully reads its neighboring fake `.env`, despite
the candidate workspace being a different directory. No real host secret was
read. This is a launch-path residual of R01, not a failure of the direct
`Sandbox.run` test that now passes.

Repair: expose specific program files and explicit dependency/package roots, or
copy/materialize them into the candidate artifact. Do not silently broaden file
arguments into directory-wide access. Test the complete serving launcher as well
as individual build subprocesses.

Probe: `test_candidate_launcher_does_not_grant_read_access_to_script_sibling_secrets`.

### T03 — P1: failed retirement becomes “done” without stopping the orphan

Location: `src/inferstack/releases/runtime.py:49` and `:186`,
`src/inferstack/releases/rollout.py:194`.

The new durable retirement obligation handles an exception followed by a
successful retry. It fails when the provider returns `False` instead of raising.
The first retirement marks the allocation `orphaned` and leaves the obligation
pending. On the next step, `serving_allocations` selects only active allocations,
so it excludes that orphan and returns no resources. The controller interprets
the empty result as complete and marks the obligation `done` with no error.

The probe observes one provider stop call across two steps; the resource's row
remains orphaned while its retirement becomes done. Operator recovery still
exists, but the automatic obligation now falsely claims completion and stops
retrying. This can leave serving resources charged indefinitely.

Repair: retirement must consider all unresolved allocations of the release,
including orphaned/releasing/intended resources. Mark completion only after every
owned resource is proven released, and preserve unresolved provider/ownership
evidence for subsequent retries or explicit operator intervention.

Probe: `test_orphaned_retirement_remains_pending_until_resource_is_stopped`.

### T04 — P1: replacement can change a Kubernetes release into a Docker deployment

Location: `src/inferstack/releases/runtime.py:81` and `:101`,
`src/inferstack/optimization/experiment.py:112`,
`src/inferstack/execution/executor.py:343`.

The runtime now dispatches retirement through the allocation's recorded provider.
Replacement placement is still inferred afresh from `profile.container.provider`
or the currently configured provider list. Ordinary experiment launch/container
materialization does not stamp that provider into the deployment/release.

When both Docker and Kubernetes are configured (the control CLI normally adds
Docker), a dead retained Kubernetes allocation whose container profile omits
`provider` is replaced using Docker. The probe records one Docker launch and no
Kubernetes launch despite the retained allocation's Kubernetes identity and
persisted container spec.

Repair: freeze provider, cluster/context, namespace and placement in the release
at creation from its actual allocation. Replacements must replay that placement;
ambiguous legacy manifests should require explicit resolution rather than
defaulting to another provider.

Probe: `test_replacement_preserves_retained_kubernetes_placement`.

### T05 — P1: Kubernetes GPU replacements require an impossible local-device reservation

Location: `src/inferstack/releases/runtime.py:120` and `:136`,
`src/inferstack/infrastructure/allocations.py:38`.

Every replacement passes through the host-device `place_devices` function. A
normal Kubernetes container spec uses `gpus: 1`, which KubernetesProvider maps to
its scheduler's GPU request/limit. The replacement path supplies neither an
explicit host GPU UUID nor a local device pool, so it always fails before calling
the Kubernetes provider: `no exclusive GPU placement ... requested without
identified devices`.

This is independent of whether a real cluster has an available GPU: the code
rejects the valid count-based Kubernetes specification before contacting it.
Using a list in `gpus` is not a viable workaround; the Kubernetes manifest code
expects a numeric quantity there. The probe uses a configured injected cluster
provider and confirms that its launch is never reached. It is a contract test,
not evidence of real GPU scheduling.

Repair: make reservation provider-specific. Attached hosts need exclusive known
devices; Kubernetes needs an atomic bounded allocation with scheduler requests,
namespace/resource quotas and observed node/device identity after placement.
Do not apply a control-host UUID reservation as a prerequisite for remote Pod
scheduling.

Probe: `test_gpu_kubernetes_replacement_uses_cluster_scheduler`.

## Assessment and next checks

The remediation materially improves the earlier cases: measurement coverage,
SSE parsing, logprob lengths, frozen model identity, post-observation route
conflicts, exception recovery for retirement, ordinary local rollback,
provider-based retirement, build tags, conservative PID/PGID ownership, identified
local GPU reservations, atomic trigger admission, terminal callbacks and skip
classification all pass the existing review probes. Keep those changes.

The remaining problems concentrate in the transition from a verified running
candidate to durable serving: its environment and placement must be frozen and
replayable, and cleanup must account for unresolved resources. Passing the first
startup and ordinary rollback path is insufficient to establish those recovery
guarantees.

Prioritize T01/T02 for secret isolation and executable identity, then T03 cleanup
accounting and T04/T05 provider-specific replacement. Validate actual source/plugin
replacement, provider failure followed by recovery, and Kubernetes release
replacement in addition to the existing happy-path lifecycle tests. GPU and
provider/cloud qualification claims must remain scoped to live evidence.

## Reproduction and evidence

```bash
python -m pytest -o addopts='' -q -rs tests \
  docs/verification/v2_contract_probes.py \
  docs/verification/v2_second_pass_probes.py
python -m pytest -o addopts='' -q --tb=short \
  docs/verification/v2_third_pass_probes.py
```

The second command currently fails six probes. They are outside the default
testpaths and use fake secrets, temporary local processes and controlled provider
observations; no real credentials or cloud GPUs are accessed. Test processes are
retired in finally blocks.

Local raw evidence is in `results/v2-third-review/` (gitignored): combined suite
and prior-probe log/XML, third-pass log/XML, targeted hosted PostgreSQL log/XML,
live Docker release-runtime result, and a machine-readable summary. The hosting
test fixtures create and drop their own databases. No changes were pushed.

## Remediation — 8 November 2026 (implementation agent)

All five findings were fixed. The six third-pass probes pass unchanged, and so do
all thirty earlier probes. CI runs all three probe files. Regression tests were
added to `tests/test_second_review.py` (third-review section).

| Finding | Fix |
| --- | --- |
| T01 | The executor records the scrubbed launch environment (`launch_env`) on the candidate allocation, and `ReleaseManager.create` freezes it into the release as `execution.env`. A replacement launches with exactly that environment, re-checked through `scrubbed_env` (credential-like keys refused), and with `inherit_env=False`. Fallback order: the release's frozen env, then the env recorded on its earlier serving allocation, then a minimal allowlisted env. Reserved devices are the only variables added. |
| T02 | `Sandbox.launcher` exposes each absolute program file as a single readable file (Seatbelt `literal`, bwrap single-file bind), never its directory. A script's siblings stay unreadable. |
| T03 | Retirement walks every unresolved serving allocation of the release (intent, active, releasing and orphaned) and retries orphans. The obligation stays pending until all of them are proven released. A launch with no recorded identity stays orphaned unless provably never started. |
| T04 | The release freezes `placement` (provider, resource type, namespace, context and host) from its serving allocation. Replacement order: frozen placement, then profile `provider`, then the provider of the recorded serving allocation. With several remote providers and none of these, the replacement is refused instead of guessing one. |
| T05 | Reservation is provider-specific. Providers whose scheduler places GPUs (`schedules_devices`, i.e. Kubernetes) get count-based accounting with no host-UUID prerequisite. Local container runtimes with a count reserve identified host devices and launch with `--gpus device=<reserved>`. |

Not established here: real Kubernetes GPU scheduling, observed node/device
identity after Pod placement (not yet recorded), and quota-bounded GPU
replacement on a live cluster. All of these need a GPU node pool.

Local qualification after these fixes (`--require postgresql --require docker
--require kubernetes`): `not_qualified` on CPU, blocked only by environment,
kernel_equivalence and native_smoke (no GPU). test_suite: 260 passed, 0 skipped.
Hosted PostgreSQL: 26 passed. Docker: 2 passed. kind: 1 passed. All three probe
files: 36 passed.
