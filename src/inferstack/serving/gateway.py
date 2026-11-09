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
        trace = request.headers.get('traceparent', '')
        if _TRACEPARENT.match(trace):
            upstream['traceparent'] = headers['traceparent'] = trace
        stream_requested = bool(body.get('stream'))
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
                    await record(RequestRecord(release['id'], key, outcome, True, status=200, **live.phases(ended)))
            # Existing streams remain pinned during route changes; no worker is killed.
            return StreamingResponse(generate(), media_type='text/event-stream', headers=headers)
        outcome, status, tokens = 'error', 502, None
        try:
            result = await engine.complete(body, chat=chat, headers=upstream)
            ok = True
            latency = ttfb()
            outcome, status = 'ok', 200
            usage = (result.get('usage') or {}).get('completion_tokens') if isinstance(result, dict) else None
            tokens = usage if isinstance(usage, int) and not isinstance(usage, bool) else None
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
            await record(RequestRecord(release['id'], key, outcome, False, status=status,
                                       e2e_ms=latency if outcome == 'ok' else None, output_tokens=tokens))

    @app.post('/v1/completions')
    async def completions(request: Request):
        return await forward(request)

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        return await forward(request, True)
    return app
