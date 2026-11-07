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
