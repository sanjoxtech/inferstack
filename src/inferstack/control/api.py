from __future__ import annotations
import hmac
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from inferstack.storage.repository import Repository


class Submission(BaseModel):
    request_key:str=Field(min_length=1,max_length=200)
    deployment_id:str=Field(min_length=1)
    goal:str=Field(min_length=1,max_length=20000)
    source:str='user'
    technique_text:str|None=Field(default=None,max_length=100000)


class RecordBody(BaseModel):
    body:dict
    expected_revision:int|None=None


def create_control_app(repository:Repository,token:str):
    if not token:raise ValueError('control API token required')
    app=FastAPI(title='InferStack control plane')
    async def authorize(authorization:str|None=Header(default=None)):
        if not authorization or not hmac.compare_digest(authorization.encode(),('Bearer '+token).encode()):
            raise HTTPException(401,'invalid control credential')
    from fastapi import Depends
    auth=[Depends(authorize)]

    @app.get('/health')
    def health():return {'status':'ok'}

    @app.post('/control/requests',dependencies=auth)
    def submit(body:Submission):
        try:repository.read('deployment',body.deployment_id)
        except KeyError:raise HTTPException(404,'unknown deployment')
        try:job=repository.submit(body.request_key,body.model_dump())
        except ValueError as exc:raise HTTPException(409,str(exc))
        return {'job_id':job,'job':repository.get(job)}

    @app.get('/control/jobs/{job_id}',dependencies=auth)
    def job(job_id:str):
        try:return repository.get(job_id)
        except KeyError:raise HTTPException(404,'unknown job')

    @app.post('/control/jobs/{job_id}/cancel',dependencies=auth)
    def cancel(job_id:str):
        try:repository.get(job_id)
        except KeyError:raise HTTPException(404,'unknown job')
        return {'cancelled':repository.cancel(job_id)}

    @app.put('/control/records/{kind}/{identifier}',dependencies=auth)
    def put(kind:str,identifier:str,record:RecordBody):
        if kind not in {'deployment','model','hardware','workload','objective','technique','evidence'}:
            raise HTTPException(422,'unsupported public record type')
        if kind=='deployment':
            from inferstack.engines.managed import DeploymentSpec
            try:
                spec=DeploymentSpec(**record.body)
                if spec.id != identifier:raise ValueError('deployment id must match record identifier')
            except (TypeError,ValueError) as exc:raise HTTPException(422,str(exc))
        try:revision=repository.put(kind,identifier,record.body,record.expected_revision)
        except ValueError as exc:raise HTTPException(409,str(exc))
        return {'revision':revision}

    @app.get('/control/records/{kind}/{identifier}',dependencies=auth)
    def read(kind:str,identifier:str):
        try:return repository.read(kind,identifier)
        except KeyError:raise HTTPException(404,'unknown record')
    from inferstack.releases.manager import ReleaseManager
    releases=ReleaseManager(repository)

    @app.post('/control/jobs/{job_id}/release',dependencies=auth)
    def release(job_id:str):
        try:return {'release_id':releases.create(job_id)}
        except KeyError:raise HTTPException(404,'missing job or evidence')
        except ValueError as exc:raise HTTPException(409,str(exc))

    @app.post('/control/routes/{model}/promote',dependencies=auth)
    def promote(model:str,expected_revision:int):
        try:return {'revision':releases.promote(model,expected_revision)}
        except KeyError:raise HTTPException(404,'unknown route')
        except ValueError as exc:raise HTTPException(409,str(exc))

    @app.post('/control/routes/{model}/rollback',dependencies=auth)
    def rollback(model:str,expected_revision:int):
        try:return {'revision':releases.rollback(model,expected_revision)}
        except KeyError:raise HTTPException(404,'unknown route')
        except ValueError as exc:raise HTTPException(409,str(exc))

    @app.put('/control/routes/{model}',dependencies=auth)
    def route(model:str,stable_release:str,expected_revision:int,candidate_release:str|None=None,weight:int=0):
        try:return {'revision':releases.route(model,stable_release,candidate_release=candidate_release,
                                             weight=weight,expected_revision=expected_revision)}
        except KeyError:raise HTTPException(404,'unknown release')
        except ValueError as exc:raise HTTPException(409,str(exc))

    return app
