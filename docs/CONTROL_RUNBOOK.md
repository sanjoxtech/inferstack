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
