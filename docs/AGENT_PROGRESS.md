# Agent progress log (live)

Shared, append-only status for every agent working in this repository. Update at least
every ~5 minutes while working and at every milestone (newest entry at the top of the
log). Each entry: time (UTC), agent, current task, state, files touched, next step.
Check `git log origin/main` for the authoritative history; this file is coordination.

## Current state (summary)

- Reviewed HEAD `main`: c460793 (H3 step 1). CI: five successful jobs; manual provider smoke skipped.
- Harness plan: H0 accepted; H1/H2 implemented; H3 step 1 implemented (bounded implementer handoff, immutable validated patch and resume). H3 refinement/stop policy/demo and H4/H5 remain.
- Independent focused check on c460793: 39 H1/H2/H3 tests passed, no failures/skips (PostgreSQL parametrizations excluded).
- GPU: Colab A100 run at d33b470 **qualified** (with same-commit CI sandbox evidence,
  docs/verification/COLAB_A100_2026-11-09.md). Speed numbers single-run only.
- Qualification scope: the recorded A100 qualification covers the older d33b470 native/Qwen3.5-2B-Base environment, not latest H3, other engines, T4 or production GPU Kubernetes. Repeated speed/task-quality validation and final integrated qualification remain.
- In-progress work observed: uncommitted CI workflow changes for hosted PostgreSQL/provider checks; preserve the implementation agent's edits. No result from the running hosted PostgreSQL verification is claimed here.

## Log

### 2026-11-09 11:10 UTC — Claude Code (implementation agent)
- Task: H3 step 2 (campaign control) + "investigate this claim" action. In progress, uncommitted.
- Built: optimization/campaign.py (diagnose_outcome, CampaignState, targets); worker loop now plan -> execute -> verify -> diagnose -> replan with lineage; stops on target_met / kept / iteration_limit / budget_exhausted / stagnation / needs_implementation; failed builds are diagnosed outcomes; best verified candidate kept; ProblemDesk.investigate + POST /control/problems/{id}/investigate.
- Evidence so far: tests/test_harness_h3_campaign.py 15 passed.
- Files I am editing: control/worker.py, optimization/{campaign,planner,tools}.py, diagnosis/intake.py, control/api.py.
- CI note: provider smoke is manual-only again (owner decision: paid calls).

### 2026-11-09 10:40 UTC — Claude Code (implementation agent)
- CI fixes. Neon job: run only the PostgreSQL test modules (torch was not installed there); Neon passed 15/15 in CI.
- Live provider smoke failed in CI (model returned non-JSON / invalid plans). Planner: tolerant JSON extraction, explicit stop_rule values, `profile_options` (real allowed keys per engine) in the context, clearer config_change error. Local live runs: OpenRouter and DeepSeek both plan_validated with 0 corrections.
- Flaky crash test: durable-execution test lease 1.0 s -> 2.0 s (a 1.6 s CI stall lost the lease before the crash point).
- Not mine, left uncommitted: docs/INFERENCE_HARNESS_PLAN.md native-engine paragraph (another agent).

### 2026-11-09 10:27 UTC — Claude Code (implementation agent)
- Task: CI uses hosted services. GitHub secrets set: NEON_DATABASE_URL, OPENROUTER_API_KEY, DEEPSEEK_API_KEY (values never printed).
- ci.yml: new job "PostgreSQL tests on hosted Neon" (push/manual; -k postgres; each test creates/drops its own DB); provider smoke now runs on push to main too.
- Note: Neon is us-east-2; from India each query is ~250 ms, so run Neon tests in CI, not locally.
- H3 step 1 CI (c460793): green.

### 2026-11-09 10:21 UTC — Codex (independent progress review)
- Reviewed c460793; production source unchanged by this review. Read H1 memory/context, H2 intake/diagnosis, H3 handoff contracts and recorded A100 qualification scope.
- Independently ran tests/test_harness_h1.py, test_harness_h2.py and test_harness_h3.py with PostgreSQL parametrizations excluded: 39 passed, 0 failed, 0 skipped, 27.628s. JUnit/log: results/progress-check-h1-h3-c460793.*. This is focused CPU/local sandbox verification, not a full source audit or GPU rerun.
- Confirmed exact-commit CI: five successful jobs and one skipped manual provider job (run 42598807688).
- A100 evidence: d33b470 has 27 passing kernel-equivalence checks and native smoke; original report blocked on sandbox skips, amended report qualified using same-commit CI under the recorded policy. All performance lanes remain inconclusive_single_run. Raw archives were user downloads, not independently rerun here.
- Next: finish H3 verifier-driven refinement, controller stop policy and induced-bottleneck demonstration; then H4 FactoryOS-to-release integration and qualification of the final integrated GPU candidate. H5 remains later.

### 2026-11-09 10:09 UTC — Claude Code (implementation agent)
- Task: H3 step 1 (implementer handoff) complete and committed.
- Evidence: tests/test_harness_h3.py 18 passed; full suite 380 passed / 18 skipped; probes 36 passed (macOS, CPU).
- Added: control API /control/implementations/{id} (GET, /patch, /revise, /cancel, /resume); CLI `implementation --id [--diff --source-root]`; serve uses INFERSTACK_SOURCE_ROOT for validation.
- Next: wait for CI, then H3 step 2 (verifier-driven parent/child hypotheses + controller stop policy).

### 2026-11-09 10:03 UTC — Claude Code (implementation agent)
- Task: H3 step 1, implementer handoff (uncommitted, in progress).
- Done so far: new `implementation` record + `src/inferstack/implementation/handoff.py` (ImplementationDesk: controller-owned handoff, journaled attempts, isolated validation, immutable content-addressed candidate; CommandImplementer = coding-agent path in the sandbox). Worker opens a handoff for a needs_implementation plan and resumes from `implementation_ids` with the exact patch. dag binds `implementation_id` to the validated patch. Executor and desk share `materialize_source`. Sandbox.run gained `readable_files`.
- Evidence: tests/test_harness_h3.py 17 passed (macOS Seatbelt).
- Files I am editing (please avoid): implementation/, control/worker.py, optimization/dag.py, execution/{patches,executor,sandbox}.py, contracts/records.py.
- Next: control API + CLI for implementations, full suite, plan status, commit, CI.

### 2026-11-09 09:53 UTC — Claude Code (implementation agent)
- Task: H2 verified on CI (8dfea02 green: py3.11, py3.12, Docker image, kind, lock).
- State: H2 complete. Waiting for owner go-ahead on H3.
- Next: H3 step 1 = implementer handoff contract (stored patch artifact + build manifest, journaled attempts), before any autonomous coding.

### 2026-11-09 09:48 UTC — Claude Code (implementation agent)
- Task: H2 committed (44c9ac9): diagnosis/ intake + problem records; created this file.
- State: verifying CI for 44c9ac9.
- Files: src/inferstack/diagnosis/*, contracts/records.py (Problem), control/api.py, tests/test_harness_h2.py.
- Next: confirm CI green, then start H3 design (implementer handoff contract first).
