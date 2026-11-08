"""Multi-model gateway pinned to immutable release specifications per request.

Per deployment: admission (capacity, circuit breaker, drain) so one unhealthy or
overloaded model never blocks others. Per request: a request ID (client-supplied
if well-formed, else generated) and W3C traceparent are propagated upstream and
returned to the client. Upstream failures keep their meaning: an upstream 4xx is
returned as-is, 5xx/connection errors count against the deployment's breaker and
return 502. Streams are primed before the response starts so an upstream error is
never reported as a 200. A client disconnect closes the upstream stream, which is
how engines receive cancellation.
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
from inferstack.serving.health import AdmissionController, Rejected

_REQUEST_ID = re.compile(r'^[A-Za-z0-9._:-]{1,128}$')
_TRACEPARENT = re.compile(r'^[0-9a-f]{2}-[0-9a-f]{32}-[0-9a-f]{16}-[0-9a-f]{2}$')


def create_gateway(repository, token, engine_factory=ManagedEngine, admission=None):
    if not token:
        raise ValueError('gateway token required')
    manager = ReleaseManager(repository)
    admission = admission or AdmissionController()

    async def authorize(authorization: str | None = Header(default=None)):
        if not authorization or not hmac.compare_digest(authorization.encode(), ('Bearer ' + token).encode()):
            raise HTTPException(401, 'invalid credential')
    app = FastAPI(title='InferStack release gateway', dependencies=[Depends(authorize)])
    app.state.admission = admission

    @app.get('/v1/models')
    def models():
        return {'object': 'list', 'data': [{'id': r['id'], 'object': 'model'} for r in repository.list_records('route')]}

    @app.get('/gateway/status')
    def status():
        return {'deployments': {key: admission.status(key) for key in sorted(admission.states)}}

    @app.get('/gateway/observe')
    def observe(deployment_id: str, since: float):
        return admission.observe(deployment_id, since)

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
        try:
            probe = admission.admit(key)
        except Rejected as exc:
            return rejected(exc, headers)
        engine = engine_factory(DeploymentSpec.from_record(spec_body))
        ok = False
        started = time.perf_counter()
        ttfb = lambda: (time.perf_counter() - started) * 1000

        def upstream_error(exc):
            if isinstance(exc, httpx.HTTPStatusError) and exc.response.status_code < 500:
                return True, JSONResponse({'error': {'message': 'upstream rejected request',
                                                     'upstream_status': exc.response.status_code}},
                                          status_code=exc.response.status_code, headers=headers)
            return False, JSONResponse({'error': {'message': 'upstream engine failure', 'type': type(exc).__name__}},
                                       status_code=502, headers=headers)
        if body.get('stream'):
            chunks = engine.stream(body, chat=chat, headers=upstream)
            try:
                first = await chunks.__anext__()
            except StopAsyncIteration:
                first = b''
            except (httpx.HTTPError, OSError) as exc:
                ok, response = upstream_error(exc)
                admission.release(key, ok=ok, probe=probe)
                await engine.close()
                return response
            first_byte_ms = ttfb()

            async def generate():
                upstream_failed = False
                try:
                    if first:
                        yield first
                    async for data in chunks:
                        yield data
                except (GeneratorExit, asyncio.CancelledError):
                    raise  # client disconnected: closing the upstream stream cancels the engine request
                except Exception:
                    upstream_failed = True
                    raise
                finally:
                    await chunks.aclose()
                    await engine.close()
                    admission.release(key, ok=not upstream_failed, probe=probe, ttfb_ms=first_byte_ms)
            # Existing streams remain pinned during route changes; no worker is killed.
            return StreamingResponse(generate(), media_type='text/event-stream', headers=headers)
        try:
            result = await engine.complete(body, chat=chat, headers=upstream)
            ok = True
            latency = ttfb()
            return JSONResponse(result, headers=headers)
        except (httpx.HTTPError, OSError, ValueError) as exc:
            ok, response = upstream_error(exc)
            latency = None
            return response
        finally:
            admission.release(key, ok=ok, probe=probe, ttfb_ms=latency if ok else None)
            await engine.close()

    @app.post('/v1/completions')
    async def completions(request: Request):
        return await forward(request)

    @app.post('/v1/chat/completions')
    async def chat(request: Request):
        return await forward(request, True)
    return app
