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
    technique_urls:list[str]=Field(default_factory=list,max_length=10)
    budget:dict|None=None
    deadline_at:float|None=None
    max_attempts:int=Field(default=1,ge=1,le=10)


class TextEvidence(BaseModel):
    text:str=Field(min_length=1,max_length=1000000)
    title:str|None=Field(default=None,max_length=300)
    source_key:str|None=Field(default=None,max_length=500)
    license:str|None=None


class UrlEvidence(BaseModel):
    url:str=Field(min_length=1,max_length=2000)


class Recovery(BaseModel):
    force_note:str|None=Field(default=None,max_length=500)


class RecordBody(BaseModel):
    body:dict
    expected_revision:int|None=None
    schema_version:int|None=None


PUBLIC_KINDS={'deployment','model','hardware','workload','objective','technique','evidence','feedback'}


def create_control_app(repository:Repository,token:str,fetcher=None):
    if not token:raise ValueError('control API token required')
    from inferstack.research import EvidenceStore,EvidenceIntegrityError,SafeFetcher
    evidence_store=EvidenceStore(repository)
    fetcher=fetcher or SafeFetcher()
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
        payload=body.model_dump(exclude={'budget','deadline_at','max_attempts'})
        for url in body.technique_urls:
            problem=fetcher.check_url(url)
            if problem:raise HTTPException(422,f'technique url rejected: {problem}')
        payload['evidence_ids']=[]
        if body.technique_text:
            # Supplied text is attributed evidence; it works without any search access.
            payload['evidence_ids'].append(evidence_store.ingest_text(body.technique_text,
                source_key='request:'+body.request_key,provenance='user_supplied',title='technique text from '+body.source))
        try:job=repository.submit(body.request_key,payload,budget=body.budget,deadline_at=body.deadline_at,
                                  max_attempts=body.max_attempts)
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

    @app.post('/control/evidence/text',dependencies=auth)
    def evidence_text(body:TextEvidence):
        import hashlib
        key=body.source_key or 'operator:'+hashlib.sha256(body.text.encode()).hexdigest()[:16]
        try:identifier=evidence_store.ingest_text(body.text,source_key=key,title=body.title,license=body.license)
        except ValueError as exc:raise HTTPException(422,str(exc))
        return {'evidence_id':identifier,'lineage':evidence_store.lineage(identifier)}

    @app.post('/control/evidence/url',dependencies=auth)
    async def evidence_url(body:UrlEvidence):
        return await evidence_store.ingest_url(body.url,fetcher)

    @app.get('/control/evidence/{identifier}',dependencies=auth)
    def evidence_read(identifier:str,revision:int|None=None):
        try:return evidence_store.read(identifier,revision)
        except KeyError:raise HTTPException(404,'unknown evidence')
        except EvidenceIntegrityError as exc:raise HTTPException(409,str(exc))

    @app.get('/control/techniques/candidates',dependencies=auth)
    def technique_candidates(deployment_id:str):
        from inferstack.research import candidates
        try:return {'candidates':candidates(repository,deployment_id)}
        except KeyError:raise HTTPException(404,'unknown deployment')

    @app.get('/control/allocations',dependencies=auth)
    def allocations(state:str|None=None,job_id:str|None=None):
        from inferstack.infrastructure import AllocationManager
        return {'allocations':AllocationManager(repository,'api').list(state=state,job_id=job_id)}

    @app.post('/control/allocations/{allocation_id}/recover',dependencies=auth)
    def recover_allocation(allocation_id:str,body:Recovery):
        from inferstack.infrastructure import LocalProcessProvider,recover
        try:return recover(repository,LocalProcessProvider(),allocation_id,force_note=body.force_note)
        except KeyError:raise HTTPException(404,'unknown allocation')
        except ValueError as exc:raise HTTPException(409,str(exc))

    @app.get('/control/workers',dependencies=auth)
    def workers():return {'workers':repository.workers()}

    @app.put('/control/records/{kind}/{identifier}',dependencies=auth)
    def put(kind:str,identifier:str,record:RecordBody):
        from pydantic import ValidationError
        from inferstack.contracts.records import IncompatibleSchema
        from inferstack.contracts.redaction import SecretInRecord
        if kind not in PUBLIC_KINDS:raise HTTPException(422,'unsupported public record type')
        try:revision=repository.put_record(kind,identifier,record.body,record.expected_revision,record.schema_version)
        except (ValidationError,IncompatibleSchema,SecretInRecord) as exc:raise HTTPException(422,str(exc))
        except ValueError as exc:raise HTTPException(409,str(exc))
        return {'revision':revision}

    @app.get('/control/records/{kind}/{identifier}',dependencies=auth)
    def read(kind:str,identifier:str,revision:int|None=None):
        try:return repository.read(kind,identifier,revision)
        except KeyError:raise HTTPException(404,'unknown record')

    @app.get('/control/records/{kind}/{identifier}/history',dependencies=auth)
    def history(kind:str,identifier:str):
        try:return {'history':repository.history(kind,identifier)}
        except KeyError:raise HTTPException(404,'unknown record')

    @app.get('/control/capabilities',dependencies=auth)
    def capabilities(engine:str|None=None):
        from inferstack.engines.capabilities import matrix
        return {'capabilities':[c.as_dict() for c in matrix() if engine in (None,c.engine)]}

    @app.get('/control/schemas',dependencies=auth)
    def schemas():
        from inferstack.contracts.records import CURRENT_VERSIONS
        from inferstack.contracts.metrics import METRICS,TIMESTAMP_UNIT
        return {'record_schemas':CURRENT_VERSIONS,'timestamp_unit':TIMESTAMP_UNIT,
                'metrics':{n:{'unit':m.unit,'definition':m.definition,'source':m.source,'better':m.better,
                              'aggregation':m.aggregation} for n,m in METRICS.items()},
                'database_schema_version':repository.schema_version}

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
