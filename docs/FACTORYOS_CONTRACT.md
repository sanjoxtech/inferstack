# FactoryOS ↔ InferStack contract

Contract version = InferStack `feedback` record schema version (currently 1),
returned as `contract_version` on every intake response. InferStack never changes
FactoryOS state; FactoryOS never changes InferStack deployments or routes.

## 1. Report an inference issue

`POST /control/feedback` (Bearer control token)

```json
{
  "issue_id": "fos-2026-11-08-0042",
  "idempotency_key": "fos-2026-11-08-0042",
  "source": "factoryos",
  "trace_ids": ["4bf92f3577b34da6a3ce929d0e0e4736"],
  "deployment_ref": "qwen35-prod@7",
  "category": "latency",
  "observations": {"p95_client_ttft_ms": 1840, "window": "2026-11-08T03:00Z/04:00Z", "concurrency": 48},
  "objective_ref": null,
  "reported_at": 1794100000.0
}
```

- `category`: `latency`, `throughput`, `capacity`, `cost` start an optimization
  job; `error`, `quality`, `unknown` are recorded for operators only (quality and
  errors need human triage, not automatic optimization).
- `deployment_ref` must name an existing deployment record (`id` or `id@revision`).
- `trace_ids` should be the `X-Request-Id`/W3C trace IDs the gateway returned.
- Unknown fields are rejected (422). Re-sending the same `issue_id` with identical
  content is idempotent; different content is a 409. Jobs are keyed by
  `idempotency_key`, so retries never start a second job.

Response: `{"issue_id", "contract_version", "job_id" | null, "status": "queued" | "recorded_not_actionable"}`.

## 2. Track the outcome

`GET /control/feedback/{issue_id}` → `{"status": pending|running|completed|failed|cancelled,
"decision": keep|reject|retest|inconclusive|null, ...}`.

A `keep` decision means a candidate passed verification; it is **not** live.
Traffic only moves through an explicit progressive rollout (below), which an
operator or FactoryOS automation with operator credentials starts.

## 3. Progressive rollout (operator)

`POST /control/rollouts` `{"rollout_id", "model", "release_id", "stages": [{"weight": 5,
"min_duration_s": 600, "max_error_rate": 0.01, "max_latency_regression": 0.1}, ...,
{"weight": 100, ...}], "min_requests": 50}` then `POST /control/rollouts/{id}/step`
periodically (or `pause`, `resume`, `abort`). Requires the control server to run
with `INFERSTACK_GATEWAY_URL` (and `INFERSTACK_GATEWAY_TOKEN` if different).
