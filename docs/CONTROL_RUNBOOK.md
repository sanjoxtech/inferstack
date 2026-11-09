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

## Research evidence and techniques

**Supplied text** needs no network access. `technique_text` on a request is
stored as evidence (`source_key=request:<request_key>`, provenance
`user_supplied`) and its ID is added to the job payload. Operators can add text
directly with `POST /control/evidence/text` (`text`, optional `title`,
`source_key`, `license`).

**URLs.** `technique_urls` (up to 10) on a request are fetched by the worker in a
durable `research` step before planning; `POST /control/evidence/url` fetches one
immediately. Retrieval is bounded (15 s total, 2 MB, 3 redirects, ports 80/443,
text/HTML/Markdown/JSON only) and SSRF-guarded: every connection, including each
redirect hop, resolves the host and is refused if any resolved address is not
globally routable (loopback, private, link-local/metadata, reserved, IPv4-mapped
forms). The vetted IP is the one dialed, so DNS rebinding cannot swap it. https to
http redirects are refused. Environment proxies are ignored by design. Blocked,
unavailable (including login walls), unsupported and oversized sources are stored
as evidence with `availability` and a reason; the job continues.

**Integrity and lineage.** Content is stored once by SHA-256; each distinct
source gets its own evidence record, so identical text from a paper and a post
keeps both lineages. `GET /control/evidence/{id}` verifies the stored content
against the record hash and returns 409 if it was altered. Plans pin cited
evidence as `id@revision`; a citation that does not resolve is rejected.

**Untrusted content.** Evidence reaches the optimizer provider only inside a
nonce-delimited `UNTRUSTED_EVIDENCE` envelope, and the raw `technique_text` is
removed from the request context. Model output fields beyond the plan schema are
rejected. Nothing derives commands, tool calls or further URLs from content.

**Techniques.** Register manifests with `PUT /control/records/technique/{id}`
(schema v2: engines with version specifiers, operations, dtypes, shapes, hardware,
`min_compute_capability`, `min_gpu_memory_bytes`, `requires_capabilities`,
dependencies with license, claims, `proven_by` trial refs, `status`).
`GET /control/techniques/candidates?deployment_id=…` returns non-incompatible
techniques for that deployment as `compatible_proven`, `compatible_claimed` or
`unknown`, with blockers/unknowns and `executable` (false until every dependency
is pinned with `==` and has a license). A claim is a candidate experiment, never
an accepted optimization; `proven` requires referenced trials.

## Engines, telemetry and onboarding

**Onboard a model** with `POST /control/onboard` (`model_id`, `repo_id`,
`revision`, optional `config`). A mutable Hugging Face ref (`main`) is resolved
to a commit and config is read at that commit (requires `.[hf]` and network);
otherwise pass a 40-hex commit and the config. Outcomes: `supported` (native
reference path and an execution profile), `needs_engine_probe` (vLLM/SGLang
may support it; prove it with a launch on the pinned engine version), or
`unsupported` with the gap (non-generative task, remote code without
`allow_remote_code`). Onboarding writes a `model` record for usable artifacts and
never creates a deployment. Foundry manifests that pin `revision: main` must be
re-pinned to an immutable release.

**Launch plans.** Adapters build pinned commands (`--revision`,
`--tokenizer-revision`, host, port, dtype); mutable revisions are refused. Use
them in a deployment record's `command` with a separate port per deployment.
Declared supported ranges: vLLM `>=0.10,<1.0`, SGLang `>=0.4,<1.0` — not yet
qualified on hardware.

**Attestation.** `GET /control/deployments/{id}/attestation` asks the running
server what it is (vLLM `/version`, SGLang `/get_server_info`, native
`/v1/native/status`) and compares the served model and engine version with the
record: `attested`, `mismatch`, or `unverified` when something cannot be
checked. `features` reports observed activation with its evidence (for example
SGLang radix cache is "enabled_by_config", not proof of hits).

**Telemetry.** `GET /control/deployments/{id}/telemetry` scrapes `/metrics` and
normalizes it per engine mapping (engine TTFT/TPOT/E2E/queue/prefill/decode as
means plus bucket-interpolated p50/p99 estimates, running/waiting, KV usage,
token counters). Absent families are `null` with a reason; counter resets inside
a window give `null`. The raw scrape is stored as an artifact. Mapping status is
`declared` until a qualification run marks the engine version `tested`.

**Inventory.** `POST /control/inventory/{hardware_id}` records every GPU from
`nvidia-smi` (UUID, name, memory, compute capability), topology for multi-GPU
hosts, CPU, RAM, driver and library versions. Without `nvidia-smi` the GPU list
is empty and the reason is returned.

**Gateway admission.** Each deployment has its own circuit breaker (default 5
consecutive upstream failures, 10 s cooldown, one half-open probe), in-flight cap
(default 64; excess gets 429 with Retry-After) and drain flag (503 while
draining). Upstream 4xx are passed through; upstream 5xx and connection errors
return 502 and count against that deployment only. `GET /gateway/status` shows
breaker state and counters. `X-Request-Id` (if well-formed) and W3C
`traceparent` are forwarded upstream and returned. A client disconnect closes the
upstream stream, which is how vLLM/SGLang cancel generation.

## Tool-driven planner and optimizer providers

Add a `provider` block to the worker config to use the DAG planner (the legacy
`planner`-only config keeps the candidate-selection planner):

```json
{
  "provider": {"kind": "deepseek", "model": "YOUR_MODEL_ID", "api_key_env": "DEEPSEEK_API_KEY"},
  "planner": {"candidate_ids": ["candidate"], "max_tokens": 60000, "max_tool_calls": 12},
  "verifier": {"baseline_id": "baseline", "cases": [...], "objectives": [...], "repeats": 3}
}
```

`kind` is `deepseek` (tools on by default), `openrouter` (set `"tools": true` only
if the routed model supports tool calls; errors inside 200 bodies are handled) or
`openai_compatible` (requires `base_url`). Base URLs must be https. Credentials are
read from the named environment variable only.

The planner returns an experiment DAG: hypothesis, pinned `baseline_ref`, cited
`evidence_ids`, actions (`config_change`, `plugin_install`, `source_patch`,
exactly one terminal `measure`), expected registered metrics, gates (must include
correctness, quality, health, comparable, complete), estimate and stop rule.
Validation rejects schema violations, cycles, unknown citations, unregistered
candidates, any action on the baseline or a production deployment, profile keys
outside the per-engine allowlist, patches outside `src/inferstack/kernels|runtime`,
plugins that are incompatible or not pinned, estimates over remaining budget and
identical retries of a treatment that already failed. Rejections are sent back to
the model for up to `max_corrections` fixes.

Tools (when the provider supports them) are read-only: deployment summaries
without endpoints/commands, experiment history, telemetry, source reading under
fixed roots, evidence search (untrusted snippets), technique listing and
compatibility. There are no write tools and no verdict field. Rounds, tool calls
and total provider tokens are capped; tokens are charged to the job's
`provider_tokens` budget. Each planning transcript (redacted) is stored as an
artifact referenced from `plan.planner.audit_artifact`.

`plugin_install` and `source_patch` actions validate now but execute only once the
isolated executor (Package F) is present; until then they fail as unsupported.

**Live provider check:** `python scripts/provider_smoke.py --kind deepseek
--model <id> --key-env DEEPSEEK_API_KEY` (and the same with `--kind openrouter`)
runs one real planning call and validation without contacting any serving
endpoint.

## Isolated execution (profile changes, plugins, source patches)

`IsolatedExecutor` runs plans whose actions go beyond swapping in a registered
candidate. `EndpointExecutor` refuses such plans with a clear error.

- **profile_patch** (vLLM/SGLang): engine flags are mapped from the allowlisted
  keys and replace existing values; the baseline command must contain `--port`
  (a fresh port is assigned). Native profile patches are not yet supported.
- **plugin_install**: `pip download --only-binary` of the exact pin into a
  per-experiment wheelhouse (the only networked step), SHA-256 of every wheel
  recorded, then offline install into a venv layered on the base environment
  (`--system-site-packages`), and `pip freeze` captured. Point `pip_index_args`
  at a mirror or `--no-index --find-links DIR` for air-gapped hosts.
- **source_patch**: the plan must reference a stored diff (`patch_sha256`). It is
  parsed before use: only regular source files under `src/inferstack/kernels/` or
  `src/inferstack/runtime/` that the plan declared; no traversal, absolute paths,
  symlinks, mode changes, binary hunks or renames out of bounds; tests, evaluator,
  optimizer, control and execution code are never patchable. The diff is applied
  to an export of the pinned `source_revision`; `tests/` and the evaluator are
  then restored from that revision; touched files must compile and the declared
  `tests` must pass, offline, before anything launches.
- **Candidate launch**: a new deployment record snapshot (`<experiment>-candidate`)
  with plugin refs and profile, a scrubbed environment (allowlist only; any
  `*KEY*`, `*TOKEN*`, `*SECRET*`, … variables are dropped and refused), venv
  first on PATH, patched source on PYTHONPATH, allocation tag and identity.
- **Sandbox**: build/test steps run with rlimits (CPU, files, file size,
  optional address space), wall timeout, and a private network namespace when
  `unshare -n` is permitted (Linux as root, e.g. Colab). The isolation report on
  each candidate says which controls applied; `Sandbox(require_network_isolation=True)`
  refuses to run without the namespace. This is process-level isolation on an
  attached host, not a VM/container boundary.
- **Artifacts and cleanup**: diff, test log, wheel hashes, freeze and build log
  are stored by hash and returned on the candidate; workspaces are deleted on
  cleanup and on every build failure (nothing is launched and no record is
  written for a failed build).

## Verification gates and numerical equivalence

The endpoint verifier computes every gate from evidence: `correctness`/`quality`
(exact match on every case and repeat), `health` (both endpoints ready before
and after), `complete` (every scored trial has finite values for every objective
metric and every case returned one choice), `comparable` (unchanged record
revisions, same model, same `hardware_ref`; both unrecorded is allowed but
reported as `hardware: unrecorded`). Any failed gate rejects; any unknown gate is
inconclusive. An improvement must exceed both epsilon and twice the larger
median absolute deviation. Each scored trial is stored as a `trial` record pinned
to exact deployment revisions (usable as technique `proven_by` evidence), and the
evidence artifact records the verifier code hash and workload fixture hash.

`scripts/v160_kernel_equivalence.py` (GPU) now compares fused kernels with FP32
oracles on the same BF16 inputs and reports error in ULPs: pass at <=1 ULP max and
<=0.5 ULP mean, or no worse than the BF16 reference; grouped GDN output allows 2
ULP and FP32 state uses a 1e-3 relative bound. `v160_analyze_results.py` keeps
missing metrics as `null` and gives each lane a judge verdict (three runs per
side required; single runs are `inconclusive_single_run`). The Colab matrix
reports `liveness_pass` only and states that it makes no performance claim.

## Infrastructure providers and spend

Choose where candidates run in the worker config:

```json
"infrastructure": {
  "provider": "kubernetes",
  "kubernetes": {"namespace": "inferstack-exp", "context": "gke-a100", "egress_cidrs": ["10.20.0.0/16"]},
  "gpu_name": "NVIDIA A100-SXM4-40GB",
  "prices": {"kubernetes/NVIDIA A100-SXM4-40GB": {"usd_per_gpu_hour": 3.67, "source": "GCP on-demand quote", "as_of": "2026-11-01"}}
}
```

`provider` is `local` (default), `docker` or `kubernetes`. Remote candidates
declare `profile.container` on their deployment record: `image` pinned by digest
(`repo@sha256:…`), `container_port`, `host_port` (Docker), `gpus`, `args`, and
optionally `memory`, `cpu`, `node_selector`, `read_only_mounts`. The
deployment's `base_url` must point at the published port or the Service. The
executor launches through the provider, then attaches and waits for readiness
(default 900 s); cleanup deletes the container or the Deployment, Service and
NetworkPolicy and releases the allocation.

Hardening applied by default — Docker: read-only root, `--cap-drop ALL`,
no-new-privileges, PID and memory limits, explicit GPU devices, port bound to
127.0.0.1. Kubernetes: non-root, no privilege escalation, read-only root, no
service-account token, GPU requests/limits, NetworkPolicy with DNS-only egress
plus `egress_cidrs` (pre-stage weights or allow your model cache). Reconciliation
finds resources by the `inferstack.allocation` label and only deletes objects
whose UID matches the recorded identity, so a recreated object with the same
name is never removed.

Spend: prices are operator configuration with a source and date, stamped on each
allocation. Spend = active time x GPUs x price and counts against `spend_usd`. A
job with a spend budget whose allocation has no known price is treated as
exhausted. Verify current provider prices before relying on them.

## Model artifact cache and startup (P2)

Weights live in an immutable cache on an operator-provided volume (a PVC; never
an arbitrary hostPath), separate from the 2Gi scratch. A model record's pinned
`checksums` define the entry; the key covers source, repo, weights/tokenizer/
config revisions and every file digest, so revisions never share an entry.

- Publish: a one-shot downloader Job (`KubernetesProvider.cache_job_manifest`) is
  the only workload holding artifact credentials (`envFrom` Secret) and write
  access. It fetches into staging, verifies every pinned checksum, writes the
  manifest last and publishes with one atomic rename. Locally:
  `python -m inferstack.infrastructure.model_cache publish --root DIR
  --source huggingface|DIR --model-json '<pinned model record>'`.
- Serve: set `profile.container.model_cache = {claim, mount_path, model,
  verify_image?}`. The PVC is mounted read-only; with `verify_image` an
  init-container re-hashes the entry before the engine starts, so a missing,
  incomplete or corrupt entry fails readiness. `compile_cache_size` adds a
  bounded writable engine-cache volume (`XDG_CACHE_HOME`, `VLLM_CACHE_ROOT`).
- Eviction is LRU within `--max-bytes` and never removes an entry leased by a
  live runtime.
- Loader: `profile_patch.load_format` (`auto`, `safetensors`,
  `runai_streamer`) maps to vLLM `--load-format`. `runai_streamer` needs its
  optional dependency in the image and Safetensors weights; qualify it per
  vLLM/image combination.
- Startup evidence: replacements record `ready_s` and, on Kubernetes, per-pod
  `scheduled_s`, `initialized_s` (includes cache verification), `started_s`
  (image pull inside) and `ready_s` (engine load and compile inside) on the
  allocation. Speedup claims require measured cold, warm and restart profiles.

Not verified here: a real >2Gi model on a GPU cluster, the streamer loader, and
storage-class access modes or zone binding for multi-replica topologies.

## Release-scoped routing (P3)

InferStack keeps the only stable/candidate split (per request, in its gateway).
Within a release, replica choice belongs to the llm-d / Gateway API Inference
Extension endpoint picker; InferStack does not implement a cache scheduler.

- Each immutable release gets an `InferencePool` (`pool-r-<hash>`) and an
  `HTTPRoute` on the shared inference Gateway that matches the
  `x-inferstack-release` header (`routing/pools.py`). Set
  `profile.routing.release_header` on the deployment (its `base_url` is the
  shared Gateway) and the InferStack gateway adds the header upstream, so the
  picker only chooses among that release's pods. The endpoint the picker chose is
  recorded per request when the data plane reports it
  (`x-gateway-destination-endpoint`, configurable via `endpoint_headers`).
- Membership is identity-gated: the pool selects `inferstack.io/admitted=true`,
  which no pod template sets. `ReleaseRuntime(routing=PoolRouting(...))` applies
  the pool and is ready only after `admit` enumerates every pod and labels those
  whose observed image digest and `--revision` match the release's frozen
  identity. Mismatches get `inferstack.io/quarantine=<reason>`; unready or
  unreported pods stay out. Call `ReleaseRuntime.admit(release_id)` periodically
  so replacement and scale-out pods are admitted only after their own check.
- Retirement deletes the pool and route before stopping pods. A previous stable
  keeps its pool (warm rollback) until its release is retired.
- Trust level: kubelet imageID and API-reported container arguments show what was
  launched, not the tensors resident in GPU memory.

Verified here: object generation, admission logic and pool lifecycle (unit
tests), and admission labelling of real pods on kind. Not verified: an installed
llm-d / GAIE data plane, measured cache-affinity gains, and router restart or
draining behaviour on a GPU cluster.

## Progressive rollout and FactoryOS feedback

Run the control server with `INFERSTACK_GATEWAY_URL` (and
`INFERSTACK_GATEWAY_TOKEN` if the gateway uses a different token), or with
`INFERSTACK_ROLLOUT_EVIDENCE=store` when gateways and control share the same
database. Start with `POST /control/rollouts` and call
`POST /control/rollouts/{id}/step` periodically; `pause`, `resume` (restarts the
stage window) and `abort` are available. Weights are non-decreasing and end at 100.

Live evidence (since 9 November 2026). Every gateway request writes one durable
row to `serving_events` (migration 6) with separate phases: upstream first byte,
first generated content (streams; role/usage/finish events never count), full
completion, longest content gap, engine token count and inter-token latency
only when usage is reported. Outcomes are ok, error, cancelled, timeout and
rejected (with source: draining, unhealthy, capacity). Windows are read from the
store over a closed `[since, until)` interval and attributed by release ID, so a
restart or several gateway workers keep complete accounting
(`GET /gateway/observe?release_id=...&since=...&until=...`). `GET /metrics`
exports Prometheus text labelled by release and deployment only (no prompts).
Event retention follows `event_ttl_days`.

Stage gates (all optional, frozen with the rollout; its `policy_sha256` is
recorded): `max_error_rate`, `max_cancel_rate`, `max_empty_stream_rate`,
absolute `ttft_p95_ms`, `ttft_p99_ms`, `e2e_p95_ms`, `itl_p95_ms`,
`stall_p95_ms`, relative `ttft_p95_regression`, `e2e_p95_regression`,
`itl_p95_regression`, legacy `max_latency_regression` (median first byte),
`min_samples` and `max_duration_s`. A p95 exists only from 200 samples, a p99
from 1000; until then the stage holds. A stage that reaches `max_duration_s`
without sufficient evidence ends `inconclusive` and rolls back; it never passes
on a timer. Telemetry loss holds. At 100% the relative gates compare against the
stable evidence retained from the last stage with stable traffic, and record that
basis. Each decisive step stores its closed-window evidence (summaries, checks,
policy digest) as an immutable artifact listed in `evidence_refs`.

Evaluation binding: a release created from verification evidence carries
`evaluation` (workload fixture digest, verifier code digest, evidence artifact).
Through the API, `evaluation.required` defaults to true: before any stage
advances the bound evaluation must pass. Missing or corrupt evidence holds; a
recorded quality regression rolls back. `suite_sha256` / `evaluator_sha256` may
be pinned in the request. Candidate code cannot change the suite or evaluator.

FactoryOS integration is specified in `docs/FACTORYOS_CONTRACT.md`
(`POST /control/feedback`, `GET /control/feedback/{issue_id}`): actionable
categories create one optimization job per idempotency key; a kept candidate is
never routed automatically.

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
