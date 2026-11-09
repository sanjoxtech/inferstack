"""Multi-model gateway pinned to immutable release specifications per request.

Per deployment: admission (capacity, circuit breaker, drain) so one unhealthy or
overloaded model never blocks others. Per request: a request ID (client-supplied
if well-formed, else generated) and W3C traceparent are propagated upstream and
returned to the client. Upstream failures keep their meaning: an upstream 4xx is
returned as-is, 5xx/connection errors count against the deployment's breaker and
return 502. Streams are primed before the response starts so an upstream error is
never reported as a 200. A client disconnect closes the upstream stream, which is
how engines receive cancellation.

Every request leaves one durable evidence record (serving/evidence.py): outcome
(ok, error, cancelled, timeout, rejected + source) and separate phases (upstream
first byte, first generated content, full completion, longest content gap, engine
token count and inter-token latency when usage is reported). `/gateway/observe`
summarises a window from the shared store, so restarts and multiple gateway
workers keep complete accounting; `/metrics` exports Prometheus text.
"""
import asyncio
import hmac
import re
import time
import uuid

import httpx
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, StreamingResponse

from inferstack.engines.managed import DeploymentSpec, ManagedEngine
from inferstack.releases.manager import ReleaseManager
from inferstack.serving.evidence import EvidenceStore, LiveStream, Metrics, RequestRecord
from inferstack.serving.health import AdmissionController, Rejected

_REQUEST_ID = re.compile(r'^[A-Za-z0-9._:-]{1,128}$')
_TRACEPARENT = re.compile(r'^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$')


def create_gateway(repository, token, engine_factory=ManagedEngine, admission=None, evidence=None):
    if not token:
        raise ValueError('gateway token required')
    manager = ReleaseManager(repository)
    admission = admission or AdmissionController()
    evidence = evidence or EvidenceStore(repository)
    metrics = Metrics()
    shadow_inflight, shadow_tasks = {}, set()

    def shadow_target(model, chosen, body, rid):
        """The release to mirror this request to, or None. Sampled by request ID, bounded by
        in-flight copies, a durable total budget (shared by all gateway workers) and expiry."""
        try:
            record = repository.read('shadow', model)['body']
        except KeyError:
            return None
        import hashlib as _h
        if not record.get('enabled') or time.time() >= record['expires_at'] or record['release'] == chosen:
            return None
        if body.get('tools') or body.get('functions') or body.get('tool_choice'):
            return None  # tool calls may have side effects downstream: never mirrored
        bucket = int(_h.sha256(('shadow:' + rid).encode()).hexdigest()[:8], 16) / 0xFFFFFFFF
        if bucket >= record['sample_rate'] or shadow_inflight.get(model, 0) >= record['max_inflight']:
            return None
        used = len(evidence.rows(release_id=record['release'], since=record['started_at'], shadow=True))
        if used + shadow_inflight.get(model, 0) >= record['budget_requests']:
            return None
        return record['release']

    async def mirror(model, release_id, body, chat, upstream):
        shadow_inflight[model] = shadow_inflight.get(model, 0) + 1
        spec_body = repository.read('release', release_id)['body']['deployment']['body']
        engine = engine_factory(DeploymentSpec.from_record(spec_body))
        started = time.perf_counter()
        headers = {**upstream, 'X-InferStack-Shadow': '1'}
        try:
            if body.get('stream'):
                live = LiveStream(started)
                outcome = 'ok'
                try:
                    async for data in engine.stream(body, chat=chat, headers=headers):
                        live.feed(data, time.perf_counter())
                    outcome = 'error' if live.error or not live.done else 'ok'
                except Exception as exc:  # noqa: BLE001 - recorded, never surfaced to the client
                    outcome = 'timeout' if isinstance(exc, httpx.TimeoutException) else 'error'
                await record(RequestRecord(release_id, spec_body['id'], outcome, True, shadow=True,
                                           **live.phases(time.perf_counter())))
            else:
                try:
                    result = await engine.complete(body, chat=chat, headers=headers)
                    usage = (result.get('usage') or {}) if isinstance(result, dict) else {}
                    await record(RequestRecord(release_id, spec_body['id'], 'ok', False, shadow=True,
                                               e2e_ms=(time.perf_counter() - started) * 1000,
                                               output_tokens=usage.get('completion_tokens'),
                                               input_tokens=usage.get('prompt_tokens')))
                except Exception as exc:  # noqa: BLE001
                    await record(RequestRecord(release_id, spec_body['id'],
                                               'timeout' if isinstance(exc, httpx.TimeoutException) else 'error',
                                               False, shadow=True))
        finally:
            shadow_inflight[model] -= 1
            await engine.close()

    async def record(rec):
        metrics.observe(rec)
        try:
            await asyncio.to_thread(evidence.record, rec)
        except Exception:  # noqa: BLE001 - evidence loss is reported by the window's sample counts
            metrics.observe(RequestRecord(rec.release_id, rec.deployment_id, 'error', rec.stream,
                                          rejection_source='evidence_write_failed'))

    async def authorize(authorization: str | None = Header(default=None)):
        if not authorization or not hmac.compare_digest(authorization.encode(), ('Bearer ' + token).encode()):
            raise HTTPException(401, 'invalid credential')
    app = FastAPI(title='InferStack release gateway', dependencies=[Depends(authorize)])
    app.state.admission = admission
    app.state.evidence = evidence

    @app.get('/v1/models')
    def models():
        return {'object': 'list', 'data': [{'id': r['id'], 'object': 'model'} for r in repository.list_records('route')]}

    @app.get('/gateway/status')
    def status():
        return {'deployments': {key: admission.status(key) for key in sorted(admission.states)}}

    @app.get('/gateway/observe')
    def observe(deployment_id: str | None = None, since: float = 0.0, release_id: str | None = None,
                until: float | None = None):
        """Closed-window evidence summary from the durable store (all gateway workers)."""
        if deployment_id is None and release_id is None:
            raise HTTPException(422, 'deployment_id or release_id required')
        return evidence.window(deployment_id=deployment_id, release_id=release_id, since=since, until=until)

    @app.get('/metrics')
    def prometheus():
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(metrics.render({k: s.inflight for k, s in admission.states.items()}),
                                 media_type='text/plain; version=0.0.4')

    @app.post('/gateway/drain/{deployment_id}')
    async def drain(deployment_id: str, deadline_seconds: float = 30.0):
        """Stop admitting new requests to a deployment and wait (bounded) for in-flight ones."""
        if not 0 < deadline_seconds <= 3600:
            raise HTTPException(422, 'deadline_seconds must be in (0, 3600]')
        return await admission.drain(deployment_id, deadline_seconds)

    @app.post('/gateway/undrain/{deployment_id}')
    def undrain(deployment_id: str):
        """Admit requests again (a rollback target restored to service)."""
        admission.undrain(deployment_id)
        return admission.status(deployment_id)

    def rejected(exc: Rejected, headers):
        extra = {'Retry-After': str(exc.retry_after)} if exc.retry_after else {}
        return JSONResponse({'error': {'message': exc.reason, 'type': 'admission'}}, status_code=exc.status,
                            headers={**headers, **extra})

    async def forward(request: Request, chat=False):
        body = await request.json()
        if not isinstance(body, dict) or not isinstance(body.get('model'), str):
            raise HTTPException(422, 'model required')
        supplied = request.headers.get('x-request-id', '')
        rid = supplied if _REQUEST_ID.match(supplied) else uuid.uuid4().hex
        try:
            release = manager.select(body['model'], rid)
        except KeyError:
            raise HTTPException(404, 'model has no active release')
        spec_body = release['body']['deployment']['body']
        key = spec_body['id']
        headers = {'X-InferStack-Release': release['id'], 'X-Request-Id': rid, 'X-InferStack-Deployment': key}
        upstream = {'X-Request-Id': rid}
        # Release-scoped pool routing: the shared inference Gateway matches this header to the
        # release's own InferencePool, so the picker only chooses among that release's pods.
        routing = ((spec_body.get('profile') or {}).get('routing') or {})
        if routing.get('release_header'):
            from inferstack.routing.pools import RELEASE_HEADER
            upstream[RELEASE_HEADER] = routing['release_header']

        def endpoint_of(engine):
            got = getattr(engine, 'last_headers', None) or {}
            for name in routing.get('endpoint_headers') or ('x-gateway-destination-endpoint',):
                if got.get(name):
                    return str(got[name])[:200]
            return None
        trace = request.headers.get('traceparent', '')
        if _TRACEPARENT.match(trace):
            upstream['traceparent'] = headers['traceparent'] = trace
        stream_requested = bool(body.get('stream'))
        target = shadow_target(body['model'], release['id'], body, rid)
        if target is not None:
            task = asyncio.create_task(mirror(body['model'], target, dict(body), chat, dict(upstream)))
            shadow_tasks.add(task)
            task.add_done_callback(shadow_tasks.discard)
        try:
            probe = admission.admit(key)
        except Rejected as exc:
            source = {'deployment draining': 'draining', 'deployment unhealthy': 'unhealthy',
                      'deployment at capacity': 'capacity'}.get(exc.reason, 'admission')
            await record(RequestRecord(release['id'], key, 'rejected', stream_requested, status=exc.status,
                                       rejection_source=source))
            return rejected(exc, headers)
        engine = engine_factory(DeploymentSpec.from_record(spec_body))
        ok = False
        started = time.perf_counter()
        ttfb = lambda: (time.perf_counter() - started) * 1000

        def failure(exc):
            return 'timeout' if isinstance(exc, httpx.TimeoutException) else 'error'

        def upstream_error(exc):
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                return True, JSONResponse({'error': {'message': 'upstream rejected request',
                                                     'upstream_status': exc.response.status_code}},
                                          status_code=exc.response.status_code, headers=headers)
            return False, JSONResponse({'error': {'message': 'upstream engine failure', 'type': type(exc).__name__}},
                                       status_code=502, headers=headers)
        if stream_requested:
            chunks = engine.stream(body, chat=chat, headers=upstream)
            live = LiveStream(started)
            try:
                first = await chunks.__anext__()
            except StopAsyncIteration:
                first = b''
            except (httpx.HTTPError, OSError) as exc:
                ok, response = upstream_error(exc)
                admission.release(key, ok=ok, probe=probe)
                await engine.close()
                await record(RequestRecord(release['id'], key, 'ok' if ok else failure(exc), True,
                                           status=response.status_code, e2e_ms=ttfb()))
                return response
            live.feed(first, time.perf_counter())
            first_byte_ms = ttfb()

            async def generate():
                outcome = 'ok'
                try:
                    if first:
                        yield first
                    async for data in chunks:
                        live.feed(data, time.perf_counter())
                        yield data
                except (GeneratorExit, asyncio.CancelledError):
                    outcome = 'cancelled'
                    raise  # client disconnected: closing the upstream stream cancels the engine request
                except Exception as exc:
                    outcome = failure(exc)
                    raise
                finally:
                    ended = time.perf_counter()
                    await chunks.aclose()
                    await engine.close()
                    if outcome == 'ok' and (live.error or not live.done):
                        outcome = 'error'  # engine error event or stream ended without its terminal event
                    admission.release(key, ok=outcome in ('ok', 'cancelled'), probe=probe, ttfb_ms=first_byte_ms)
                    await record(RequestRecord(release['id'], key, outcome, True, status=200,
                                               endpoint=endpoint_of(engine), **live.phases(ended)))
            # Existing streams remain pinned during route changes; no worker is killed.
            return StreamingResponse(generate(), media_type='text/event-stream', headers=headers)
        outcome, status, tokens, prompt_tokens = 'error', 502, None, None
        try:
            result = await engine.complete(body, chat=chat, headers=upstream)
            ok = True
            latency = ttfb()
            outcome, status = 'ok', 200
            usage = (result.get('usage') or {}) if isinstance(result, dict) else {}
            count = lambda v: v if isinstance(v, int) and not isinstance(v, bool) and v >= 0 else None
            tokens, prompt_tokens = count(usage.get('completion_tokens')), count(usage.get('prompt_tokens'))
            return JSONResponse(result, headers=headers)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            ok, response = upstream_error(exc)
            latency = None
            outcome, status = ('ok' if ok else failure(exc)), response.status_code
            return response
        except asyncio.CancelledError:
            outcome, latency = 'cancelled', None
            raise
        finally:
            admission.release(key, ok=ok, probe=probe, ttfb_ms=latency if ok else None)
            await engine.close()
            # Non-streamed: no first-content phase exists; only full completion latency.
            await record(RequestRecord(release['id'], key, outcome, False, status=status, endpoint=endpoint_of(engine),
                                       e2e_ms=latency if outcome == 'ok' else None, output_tokens=tokens,
                                       input_tokens=prompt_tokens))

    @app.post('/v1/completions')
    async def completions(request: Request):
        return await forward(request)

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        return await forward(request, True)
    return app
