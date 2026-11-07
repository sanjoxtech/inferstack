# Integrated v2.0 build

The user's November 7 direction supersedes intermediate release delivery in
IMPLEMENTATION_PLAN.md. Build one integrated v2.0 candidate in successive internal
implementation passes. The user performs one consolidated setup/validation run
when the whole candidate is ready. Run automated CPU/integration checks during
implementation; consolidate hardware validation and report untested capabilities.

## Dependency-ordered backlog

- [ ] Trustworthy streaming measurement, numerical/quality/performance judge.
- [ ] Persistent domain records, artifacts, migrations and durable workflows.
- [ ] Native/vLLM/SGLang engine lifecycle, multi-model routing and telemetry.
- [ ] Optimizer provider adapters and tool-driven planner.
- [ ] Research ingestion and technique provenance/compatibility.
- [ ] Isolated executor for configuration, plugins and source changes.
- [ ] Independent verifier, bounded revision loop and experiment history.
- [ ] Infrastructure inventory, worker leases and a concrete cloud adapter.
- [ ] Release manifest, canary, draining and rollback.
- [ ] FactoryOS feedback integration and new-model onboarding.
- [ ] Continuous triggers, resource budgets and restart recovery.
- [ ] Consolidated installation, integration tests and hardware validation notebook.

## Implementation checkpoint

The SQLite job-store primitive now provides idempotent submission, exclusive
claims, expiring leases, stale-worker fencing, cancellation, terminal results and
transactional event history. This is one foundation, not a completed workflow
engine or v2.0 release. Worker side effects still require their own idempotency
and resource cleanup; a database lease alone cannot stop an expired process.

Remaining backlog stays open until functioning code and appropriate evidence exist.
The package remains at its existing version until the integrated release is ready.

## November 7 implementation pass: executable optimization path

Implemented (not a v2.0 completion claim):

- Content-aware streaming measurements, authoritative usage, native token-emission
  TPOT, missing-data handling, repeated-trial CV/MAD and constrained epsilon judge.
- Revision-checked domain records and SHA-256 evidence artifacts.
- Authenticated control API; HTTP engine adapters with local subprocess supervision.
- Lease-renewing worker, cancellation, bounded planner/executor/verifier iterations,
  append-only stage events and per-attempt candidate cleanup.
- Configured JSON provider planner, registered candidate execution, repeated paired
  endpoint trials with exact-match quality gates and recorded evidence.
- Verified release manifests, route CAS, weighted canary selection, promotion and
  rollback; multi-model HTTP gateway pins each request to a release specification.
- Native streaming usage/finish reasons, generation bounds and Unicode buffering.

Scope still open: research fetching/tool-driven planning, arbitrary code/plugin
experiments in sandboxes, GPU numerical/semantic verifiers, infrastructure/cloud
allocation, continuous trigger policies, operational migrations and hard-crash
process reconciliation. Release manifests freeze configuration records; remote
endpoint contents must still be pinned and attested by a deployment controller.
The current endpoint verifier covers only the supplied exact-match workload and
mean end-to-end latency. It does not certify numerical parity or general quality.

See CONTROL_RUNBOOK.md for the executable path and validation boundaries.
