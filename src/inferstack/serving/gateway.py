"""Multi-model gateway pinned to immutable release specifications per request."""
import uuid
from fastapi import FastAPI,HTTPException,Request,Depends,Header
from fastapi.responses import StreamingResponse
from inferstack.engines.managed import DeploymentSpec,ManagedEngine
from inferstack.releases.manager import ReleaseManager
import hmac


def create_gateway(repository,token,engine_factory=ManagedEngine):
    if not token:raise ValueError('gateway token required')
    manager=ReleaseManager(repository)
    async def authorize(authorization:str|None=Header(default=None)):
        if not authorization or not hmac.compare_digest(authorization.encode(),('Bearer '+token).encode()):
            raise HTTPException(401,'invalid credential')
    app=FastAPI(title='InferStack release gateway',dependencies=[Depends(authorize)])

    @app.get('/v1/models')
    def models():
        return {'object':'list','data':[{'id':r['id'],'object':'model'} for r in repository.list_records('route')]}

    async def forward(request:Request,chat=False):
        body=await request.json()
        if not isinstance(body,dict) or not isinstance(body.get('model'),str):raise HTTPException(422,'model required')
        rid=uuid.uuid4().hex
        try:release=manager.select(body['model'],rid)
        except KeyError:raise HTTPException(404,'model has no active release')
        engine=engine_factory(DeploymentSpec(**release['body']['deployment']['body']))
        headers={'X-InferStack-Release':release['id'],'X-Request-Id':rid}
        if body.get('stream'):
            async def generate():
                try:
                    async for data in engine.stream(body,chat=chat):yield data
                finally:await engine.close()
            # Existing streams remain pinned during route changes; no worker is killed.
            return StreamingResponse(generate(),media_type='text/event-stream',headers=headers)
        try:
            from fastapi.responses import JSONResponse
            return JSONResponse(await engine.complete(body,chat=chat),headers=headers)
        finally:await engine.close()

    @app.post('/v1/completions')
    async def completions(request:Request):return await forward(request)
    @app.post('/v1/chat/completions')
    async def chat(request:Request):return await forward(request,True)
    return app
