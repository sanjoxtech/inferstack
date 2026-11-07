# Control-plane development runbook

This is an integrated development checkpoint, not a production-ready v2.0 release.
The existing native serving command remains available. No cloud resource is
created by installation or the tests.

## Install and validate

```bash
python -m pip install -e '.[dev]'
# CPU-only development tests (install an appropriate GPU torch wheel on Colab):
python -m pip install torch --index-url https://download.pytorch.org/whl/cpu
python -m pytest
```

## Run control API

Set `INFERSTACK_CONTROL_TOKEN` in the environment to a secret of your choosing.
Do not put credentials in deployment records, source files or commit messages.

```bash
python -m inferstack.control serve --state .inferstack --port 8010
```

The authenticated OpenAPI UI at `/docs` describes registration, jobs and routes
(the UI itself is public; operations require the bearer token). Register baseline
and candidate through `PUT /control/records/deployment/{id}` with a JSON body:

```json
{"body":{"id":"baseline","engine":"vllm","model":"your-model-id","base_url":"http://127.0.0.1:8000"}}
```

Register candidate under a different ID and endpoint/port. Root URLs must not
include `/v1`; the adapter appends it. `api_key_env` references an environment
variable when the serving endpoint requires authentication. An optional `command`
argument list launches a local candidate process without a shell. Operators must
configure separate ports, GPU allocation and environment; the planner cannot
invent or execute commands. Replacing a record requires `expected_revision`.

Submit via `POST /control/requests`:

```json
{"request_key":"unique-request-1","deployment_id":"baseline","goal":"Reduce mean request latency without failing our exact-match suite","source":"user","technique_text":null}
```

FactoryOS can submit through the same authenticated contract with
`source="factoryos"`. Text suggestions can be provided as `technique_text`.
External research retrieval and automatic issue classification are not yet wired.

## Run one job

Create an operator configuration JSON (values below are placeholders):

```json
{
  "max_iterations": 3,
  "planner": {
    "base_url": "https://api.deepseek.com",
    "model": "YOUR_PROVIDER_MODEL_ID",
    "api_key_env": "DEEPSEEK_API_KEY",
    "candidate_ids": ["candidate"],
    "evidence_ids": []
  },
  "verifier": {
    "baseline_id": "baseline",
    "cases": [{"prompt":"2+2 =", "expected":"4", "max_tokens":4}],
    "objectives": [{"metric":"e2e_ms", "direction":"min", "epsilon":0.02}],
    "repeats": 3
  }
}
```

The expected text is exact, including whitespace: use your real task fixtures.
The only metric this endpoint verifier produces is mean `e2e_ms`. Unsupported
metrics remain inconclusive. The planner endpoint is Chat Completions compatible;
OpenRouter can use its `/api/v1` base and a separately configured credential.
Provider-specific JSON support must be checked with the selected model.

```bash
python -m inferstack.control worker-once --state .inferstack --config worker.json
```

The worker claims one pending job, renews its lease, plans against stored evidence,
starts/attaches a registered candidate, runs warmup and alternating paired trials,
judges evidence, and cleans up its own candidate process. Failed and noisy results
can replan up to the iteration budget. A cancelled job loses its lease and the
worker cancels its active coroutine on the next heartbeat. A hard process crash
requires external process reconciliation; database fencing alone does not kill
or reclaim an orphan GPU worker.

The provider receives request text, supplied evidence, candidate engine/model
identities and prior experiment history. Do not submit sensitive prompts or logs
unless your chosen provider is approved to process them. Credentials and launch
commands are excluded from the planner's deployment context.

## Records, schemas and migrations

Public record kinds written through `PUT /control/records/{kind}/{id}` are validated
against versioned schemas (`src/inferstack/contracts/records.py`): `deployment`,
`model`, `hardware`, `workload`, `objective`, `technique`, `evidence`, `feedback`.
`release`, `route`, `plan`, `trial`, `rollout` and `allocation` are written only by
the control plane. Pass `schema_version` to target a specific version; an
unsupported version returns 422.

- Reference fields name another record as `id` or `id@N` (pinned revision). A
  dangling reference returns 409. Plans and trials must pin revisions.
- Literal credentials (keys such as `api_key`, `token`, `authorization`, or values
  that look like `sk-…`, `ghp_…`, `Bearer …`) are rejected with 422. Reference
  secrets through `*_env` fields holding an environment-variable name.
- Model artifacts must use immutable revisions: Hugging Face requires a 40-hex
  commit; `main`, `latest`, `HEAD` and similar are rejected.
- Every write appends an immutable row to `record_history`. Read an earlier
  snapshot with `GET /control/records/{kind}/{id}?revision=N` and the full lineage
  with `GET /control/records/{kind}/{id}/history`.
- `GET /control/schemas` lists record schema versions, the canonical metric
  definitions (unit, source, aggregation) and the database schema version.
  Timestamps are UTC epoch seconds; durations carry `_ms`/`_s` suffixes.
- `GET /control/capabilities[?engine=…]` returns the engine capability matrix with
  `supported`/`unsupported`/`unknown` and the basis (`declared`, `tested`,
  `observed`). Unknown is never treated as supported.

The database migrates automatically on open (`schema_migrations` table). Back up
`.inferstack/control.db` before upgrading. v1.6 databases keep all jobs, events and
records; because v1.6 overwrote revisions in place, only the latest revision of
each record exists and its backfilled history row is labeled
`migrated_from_v1_6:prior_revisions_unrecoverable`. Records whose body does not
satisfy the current schema are kept verbatim with `schema_version` NULL; re-write
them through the API to validate them. Opening a database written by a newer
InferStack fails with `IncompatibleSchema` rather than downgrading it.

## Durable execution, budgets and resource ownership

**Database.** `--database` selects the backend; default is SQLite at
`STATE/control.db`. For a shared control plane use PostgreSQL:

```bash
python -m pip install -e '.[postgres]'
python -m inferstack.control worker-once --database postgresql://user@host/inferstack --config worker.json
```

Both backends run the same numbered migrations and serialize writes through one
writer lock (SQLite `BEGIN IMMEDIATE`, PostgreSQL a transaction advisory lock),
so lease/fencing/CAS behavior is identical. Back up before upgrading: SQLite
`sqlite3 control.db ".backup control.bak"`; PostgreSQL `pg_dump`.

**Submission controls.** `POST /control/requests` accepts `budget`
(`wall_seconds`, `gpu_seconds`, `provider_tokens`, `spend_usd`, `experiments`),
an absolute `deadline_at` (UTC epoch seconds) and `max_attempts` (1–10).
Budgets are checked before every step; exhaustion stops new steps and completes
the job with `stop_reason` and all experiments gathered so far. `experiments` and
`gpu_seconds` are derived from durable step and allocation records (GPU seconds
= active time × GPU count), not from separately charged counters.

**Steps and resume.** Each iteration is `iter{i}.plan` (idempotent; a succeeded
plan is reused after restart) and `iter{i}.experiment` (side-effecting: launch,
verify, clean up). An interrupted experiment is re-run only after its
allocations are reconciled. Job detail (`GET /control/jobs/{id}`) includes
`steps`, `usage`, `budget` and `last_error`.

**Errors and retries.** Errors are stored typed (`type`, `code`, `retryable`,
scrubbed `message`). Only transient conditions (network/timeouts, HTTP 408/429/5xx,
`ProviderUnavailable`, `ResourceUnavailable`) are retryable, and only while
`attempt < max_attempts`, with linear backoff. Unknown exceptions are permanent.

**Resource ownership.** Every launched process has an allocation row written
*before* the launch (`intent`), carries `INFERSTACK_ALLOCATION_ID` in its
environment, runs in its own process group, and is identified by host, PID,
process group and kernel start time (PID reuse cannot be mistaken for it). An
allocation is released only after the process is verified gone.

**Workers and reconciliation.** `worker-once` registers a worker with a heartbeat
lease and reconciles allocations of dead workers before claiming work. To run
reconciliation alone: `python -m inferstack.control reconcile --state .inferstack`.
Outcomes: `never_launched`, `unrecorded_launch_stopped` (found by tag),
`orphan_stopped`, `already_exited`, or `orphaned` when a stop cannot be confirmed.

**Operator recovery.** `GET /control/allocations?state=orphaned` lists unresolved
resources; `GET /control/workers` lists workers and heartbeats.
`POST /control/allocations/{id}/recover` with `{}` retries the stop; after manual
cleanup, `{"force_note": "what you did"}` force-releases with an audit note.

**Cancel and deadline.** Cancellation fences the worker immediately; on its next
heartbeat it stops work, stops and releases its processes, and attaches partial
evidence once (`partial_evidence` event). A deadline does the same and fails the
job with a non-retryable `DeadlineExceeded` carrying the partial history.

Scope: the local process provider covers attached hosts (including Colab
workers). Containers and Kubernetes resources use the same allocation contract
in Package H. Reconciliation stops processes on the host where it runs; it does
not reach other machines.

## Release and route

A kept experiment can create a manifest via
`POST /control/jobs/{job_id}/release`. This rejects changed deployment records or
unverified evidence. Configure a route through `PUT /control/routes/{model}`
(query parameters: `stable_release`, `expected_revision`, optional
`candidate_release`, `weight`). Promotion and rollback each require the current
route revision. Existing requests retain their selected backend during changes.

```bash
python -m inferstack.control gateway --state .inferstack --port 8008
```

Gateway clients use the same bearer token and normal completion/chat endpoints.
A release is a frozen specification and evidence reference, not a provisioned
cloud resource. Remote endpoint runtime identity is not attested yet. Automatic
canary-health decisions, physical worker draining and cloud provisioning remain
open. Never use the narrow exact-match verifier as proof of general model quality.

## Evidence and tests

Tests cover storage, lease competition/cancellation, planner schema/allowlist,
HTTP adapter behavior, incorrect candidate rejection, evidence integrity, release
CAS/rollback, native stream accounting and the existing CPU scheduler/kernel
references. HTTP provider/engine integration uses mock transports. No live LLM
provider, vLLM/SGLang GPU process, CUDA performance or cloud deployment has been
validated in this environment.

Provider protocol reference: https://api-docs.deepseek.com/guides/json_mode/
