from __future__ import annotations
import json
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


class Onboarding(BaseModel):
    model_id:str=Field(min_length=1,max_length=200)
    repo_id:str=Field(min_length=1,max_length=300)
    revision:str=Field(min_length=1,max_length=200)
    source:str='huggingface'
    tokenizer_revision:str|None=None
    config:dict|None=None
    allow_remote_code:bool=False


class Recovery(BaseModel):
    force_note:str|None=Field(default=None,max_length=500)


class RecordBody(BaseModel):
    body:dict
    expected_revision:int|None=None
    schema_version:int|None=None


PUBLIC_KINDS={'deployment','model','hardware','workload','objective','technique','evidence','feedback'}


class RolloutStart(BaseModel):
    rollout_id:str=Field(min_length=1,max_length=200)
    model:str=Field(min_length=1,max_length=300)
    release_id:str=Field(min_length=1,max_length=200)
    stages:list[dict]=Field(min_length=1,max_length=10)
    min_requests:int=Field(default=20,ge=1)
    # The release's bound held-out evaluation must pass before any stage advances.
    evaluation:dict=Field(default_factory=lambda:{'required':True})


ACTIONABLE={'latency','throughput','capacity','cost'}


def create_control_app(repository:Repository,token:str,fetcher=None,rollout_observer=None,rollout_attest=None,
                       release_runtime=None,implementations=None):
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

    @app.post('/control/onboard',dependencies=auth)
    def onboard_model(body:Onboarding):
        from inferstack.foundry.onboarding import OnboardingError,onboard
        try:return onboard(repository,model_id=body.model_id,repo_id=body.repo_id,revision=body.revision,
                           source=body.source,tokenizer_revision=body.tokenizer_revision,config=body.config,
                           allow_remote_code=body.allow_remote_code)
        except OnboardingError as exc:raise HTTPException(422,str(exc))
        except Exception as exc:raise HTTPException(502,f'artifact resolution failed: {type(exc).__name__}')

    async def _adapter(deployment_id):
        from inferstack.engines.adapters import adapter_for
        from inferstack.engines.managed import DeploymentSpec,ManagedEngine
        try:record=repository.read('deployment',deployment_id)
        except KeyError:raise HTTPException(404,'unknown deployment')
        engine=ManagedEngine(DeploymentSpec.from_record(record['body']))
        return record,engine,adapter_for(record['body']['engine'],engine.client)

    @app.get('/control/deployments/{deployment_id}/attestation',dependencies=auth)
    async def attestation(deployment_id:str):
        from inferstack.engines.adapters import attest
        record,engine,adapter=await _adapter(deployment_id)
        try:
            result=await attest(record['body'],adapter)
            return {**result,'deployment_revision':record['revision'],
                    'features':await adapter.observed_features(result['identity'])}
        finally:await engine.close()

    @app.get('/control/deployments/{deployment_id}/telemetry',dependencies=auth)
    async def telemetry(deployment_id:str):
        record,engine,adapter=await _adapter(deployment_id)
        try:
            snapshot=await adapter.telemetry(engine_version=record['body'].get('engine_version'))
            raw=snapshot.pop('raw_text',None)
            if raw is not None:snapshot['raw_artifact']=repository.artifact(raw.encode())
            return snapshot
        finally:await engine.close()

    @app.post('/control/inventory/{hardware_id}',dependencies=auth)
    def inventory(hardware_id:str,provider:str='attached',expected_revision:int|None=None):
        from inferstack.infrastructure.inventory import collect
        body,notes=collect(provider=provider)
        try:revision=repository.put_record('hardware',hardware_id,body,expected_revision)
        except ValueError as exc:raise HTTPException(409,str(exc))
        return {'revision':revision,'hardware':body,'notes':notes}

    from inferstack.releases.rollout import RolloutController,RolloutError

    def controller():
        if rollout_observer is None:raise HTTPException(503,'rollout observation is not configured (gateway URL)')
        # Traffic never shifts to an unattested runtime: no attestor, no rollout.
        if rollout_attest is None:raise HTTPException(503,'release attestation is not configured; refusing to route traffic')
        return RolloutController(repository,observe=rollout_observer,attest=rollout_attest,runtime=release_runtime)

    @app.post('/control/rollouts',dependencies=auth)
    async def rollout_start(body:RolloutStart):
        try:return await controller().start(body.rollout_id,body.model,body.release_id,body.stages,
                                            min_requests=body.min_requests,evaluation=body.evaluation)
        except KeyError:raise HTTPException(404,'unknown route or release')
        except (RolloutError,ValueError) as exc:raise HTTPException(409,str(exc))

    @app.post('/control/rollouts/{rollout_id}/{action}',dependencies=auth)
    async def rollout_action(rollout_id:str,action:str):
        c=controller()
        try:
            if action=='step':outcome=await c.step(rollout_id)
            elif action=='pause':outcome=c.pause(rollout_id) or 'paused'
            elif action=='resume':outcome=c.resume(rollout_id) or 'progressing'
            elif action=='abort':outcome=await c.abort(rollout_id)
            else:raise HTTPException(404,'unknown rollout action')
        except KeyError:raise HTTPException(404,'unknown rollout')
        except (RolloutError,ValueError) as exc:raise HTTPException(409,str(exc))
        return {'outcome':outcome,'rollout':repository.read('rollout',rollout_id)}

    @app.post('/control/feedback',dependencies=auth)
    def feedback(body:dict):
        """FactoryOS InferenceIssue intake (contract version = feedback schema version)."""
        from pydantic import ValidationError
        from inferstack.contracts.records import parse_ref,validate
        try:issue=validate('feedback',body)
        except (ValidationError,ValueError) as exc:raise HTTPException(422,str(exc)[:1000])
        try:repository.put_record('feedback',issue.issue_id,body)
        except ValueError as exc:
            if 'expected_revision' not in str(exc):raise HTTPException(409,str(exc))
            if repository.read('feedback',issue.issue_id)['body']!=issue.model_dump(mode='json'):
                raise HTTPException(409,'issue_id reused with different content')
        record=repository.read('feedback',issue.issue_id)
        if issue.deployment_ref:
            try:repository.read('deployment',parse_ref(issue.deployment_ref)[0])
            except KeyError:raise HTTPException(409,'unknown deployment')
        # H2: every issue becomes a typed problem, is diagnosed from evidence (owner,
        # bottleneck, confidence) and only an evidenced inference symptom queues a job.
        callback=None
        if issue.source=='factoryos':
            # Terminal job state writes a durable InferenceOutcome to the outbox (same transaction).
            callback={'issue_id':issue.issue_id,'idempotency_key':issue.idempotency_key,
                      'trace_ids':issue.trace_ids[:100],'topic':'factoryos.outcome'}
        problem=ProblemDesk(repository).from_feedback(issue.issue_id,callback=callback)['body']
        job_id=problem.get('job_id')
        return {'issue_id':issue.issue_id,'contract_version':record['schema_version'],'job_id':job_id,
                'status':'queued' if job_id else 'recorded_not_actionable',
                'problem':{'id':ProblemDesk_id(issue.issue_id),'state':problem['state'],
                           'owner':problem['diagnosis'].get('owner'),'bottleneck':problem['diagnosis'].get('bottleneck'),
                           'confidence':problem['diagnosis'].get('confidence'),
                           'route_to':problem['diagnosis'].get('route_to'),
                           'measurement_requests':problem['diagnosis'].get('measurement_requests')}}

    @app.post('/control/research',dependencies=auth)
    def research_submission(body:dict):
        """Submit a paper/post/technique text: stored as untrusted evidence, recorded as a claim."""
        text=body.get('text')
        if not isinstance(text,str) or not text.strip():raise HTTPException(422,'text required')
        try:
            problem=ProblemDesk(repository).from_research(
                text,source_key=str(body.get('source_key') or body.get('uri') or 'submission')[:300],
                title=body.get('title'),uri=body.get('uri'),deployment_id=body.get('deployment_id'),
                technique_id=body.get('technique_id'))
        except KeyError:raise HTTPException(404,'unknown deployment')
        except ValueError as exc:raise HTTPException(422,str(exc)[:500])
        return problem

    @app.get('/control/problems/{problem_id}',dependencies=auth)
    def read_problem(problem_id:str):
        try:return repository.read('problem',problem_id)
        except KeyError:raise HTTPException(404,'unknown problem')

    @app.post('/control/problems/{problem_id}/observations',dependencies=auth)
    def problem_observations(problem_id:str,body:dict):
        """Attach new numeric measurements and re-diagnose (e.g. after a measurement request)."""
        try:return ProblemDesk(repository).add_observations(problem_id,body.get('observations') or {})
        except KeyError:raise HTTPException(404,'unknown problem')
        except ValueError as exc:raise HTTPException(409,str(exc))

    # --- H3: bounded implementer handoff ------------------------------------------------
    from inferstack.implementation import ArtifactLocked,AttemptLimit,ImplementationDesk
    readonly_desk=ImplementationDesk(repository)

    def _impl(impl_id):
        try:return readonly_desk.read(impl_id)
        except KeyError:raise HTTPException(404,'unknown implementation')

    @app.get('/control/implementations/{impl_id}',dependencies=auth)
    def read_implementation(impl_id:str):
        return _impl(impl_id)

    @app.post('/control/implementations/{impl_id}/patch',dependencies=auth)
    def submit_patch(impl_id:str,body:dict):
        """Human/operator implementer: submit a unified diff (and optional candidate tests).
        It is validated in an isolated workspace before it can become the candidate."""
        _impl(impl_id)
        if implementations is None:raise HTTPException(503,'no implementation validator configured')
        diff=body.get('diff')
        if not isinstance(diff,str):raise HTTPException(422,'diff (unified diff text) required')
        tests=body.get('candidate_tests') or {}
        if not isinstance(tests,dict):raise HTTPException(422,'candidate_tests must be an object')
        try:return implementations.submit(impl_id,diff,implementer={'kind':'human','via':'control_api'},
                                          candidate_tests=tests)
        except (ArtifactLocked,AttemptLimit) as exc:raise HTTPException(409,str(exc))

    @app.post('/control/implementations/{impl_id}/revise',dependencies=auth)
    def revise_implementation(impl_id:str):
        _impl(impl_id)
        return readonly_desk.revise(impl_id)

    @app.post('/control/implementations/{impl_id}/cancel',dependencies=auth)
    def cancel_implementation(impl_id:str):
        _impl(impl_id)
        try:return (implementations or readonly_desk).cancel(impl_id)
        except ArtifactLocked as exc:raise HTTPException(409,str(exc))

    @app.post('/control/implementations/{impl_id}/resume',dependencies=auth)
    def resume_implementation(impl_id:str):
        """Queue an experiment job that executes exactly the validated candidate."""
        record=_impl(impl_id)['body']
        if record['state']!='validated':raise HTTPException(409,f"implementation is {record['state']}, not validated")
        candidate=record['candidate']
        job_id=repository.submit('implementation:'+candidate['candidate_id'],{
            'deployment_id':record['baseline_ref'].split('@')[0],'implementation_ids':[impl_id],
            'goal':f"Measure validated candidate {candidate['candidate_id']} for: {record['handoff']['hypothesis'][:500]}",
            'source':'implementation','evidence_ids':list(record['plan'].get('evidence_ids') or [])})
        return {'job_id':job_id,'candidate_id':candidate['candidate_id']}

    from inferstack.integrations.outbox import Outbox
    from inferstack.integrations.triggers import TriggerService
    outbox=Outbox(repository)

    @app.get('/control/outbox',dependencies=auth)
    def outbox_list(state:str|None=None):
        return {'items':outbox.list(state)}

    @app.post('/control/outbox/{outbox_id}/retry',dependencies=auth)
    def outbox_retry(outbox_id:str):
        try:outbox.retry_dead(outbox_id)
        except ValueError as exc:raise HTTPException(409,str(exc))
        return {'retried':outbox_id}

    @app.post('/control/triggers',dependencies=auth)
    def trigger_create(body:dict):
        try:return TriggerService(repository).create(body.get('trigger_id',''),body.get('kind',''),body.get('spec') or {})
        except ValueError as exc:raise HTTPException(422,str(exc))

    @app.get('/control/triggers',dependencies=auth)
    def trigger_list():return {'items':TriggerService(repository).list()}

    @app.post('/control/triggers/{trigger_id}/{state}',dependencies=auth)
    def trigger_state(trigger_id:str,state:str):
        try:TriggerService(repository).set_state(trigger_id,state)
        except ValueError as exc:raise HTTPException(422,str(exc))
        return TriggerService(repository).get(trigger_id)

    @app.get('/control/feedback/{issue_id}',dependencies=auth)
    def feedback_status(issue_id:str):
        try:record=repository.read('feedback',issue_id)
        except KeyError:raise HTTPException(404,'unknown issue')
        with repository.connection() as db:
            row=db.execute('SELECT id FROM jobs WHERE request_key=?',('feedback:'+record['body']['idempotency_key'],)).fetchone()
        if row is None:return {'issue_id':issue_id,'status':'recorded_not_actionable'}
        job=repository.get(row['id'])
        outcome=(job['result'] or {}).get('decision') if job['state']=='completed' else None
        return {'issue_id':issue_id,'job_id':job['id'],'status':job['state'],'decision':outcome,
                'release':None,'note':'a kept candidate still requires an explicit progressive rollout'}

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

    @app.get('/control/experience/{deployment_id}',dependencies=auth)
    def prior_experience(deployment_id:str,limit:int=20,treatment_signature:str|None=None):
        """Earlier campaigns' outcomes for this deployment's model, classified applicable/transferable."""
        from inferstack.memory import experience
        try:return experience(repository,deployment_id,limit=limit,treatment_signature=treatment_signature)
        except KeyError:raise HTTPException(404,'unknown deployment')
        except ValueError as exc:raise HTTPException(422,str(exc))

    @app.post('/control/context/{deployment_id}',dependencies=auth)
    def deployment_context(deployment_id:str,body:dict|None=None):
        """Build and store a versioned context snapshot (optional body: {workload, objectives})."""
        from inferstack.memory import context_snapshot
        body=body or {}
        try:return context_snapshot(repository,deployment_id,workload=body.get('workload'),
                                    objectives=body.get('objectives'))
        except KeyError:raise HTTPException(404,'unknown deployment')

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
    from inferstack.diagnosis.intake import ProblemDesk, _problem_id
    ProblemDesk_id=lambda issue_id:_problem_id('feedback',issue_id)
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
    async def rollback(model:str,expected_revision:int):
        """Safe rollback: warm (and attest) the target runtime and restore its admission
        before traffic moves; retire the losing runtime only after the route changed."""
        try:
            current=repository.read('route',model)
        except KeyError:raise HTTPException(404,'unknown route')
        if current['revision']!=expected_revision:raise HTTPException(409,'revision conflict')
        body=current['body']
        target=body['stable'] if body.get('candidate') else body.get('previous')
        loser=body.get('candidate') or body.get('stable')
        if target is None:raise HTTPException(409,'no rollback target')
        warmed=None
        if release_runtime is not None:
            try:warmed=await release_runtime.ensure(target)
            except Exception as exc:  # noqa: BLE001 - traffic stays where it is
                raise HTTPException(503,f'rollback target runtime unavailable, route unchanged: {exc}')
            if rollout_attest is not None:
                attested=await rollout_attest(repository.read('release',target))
                if attested.get('status')!='attested':
                    raise HTTPException(503,f"rollback target not attested ({attested.get('status')}), route unchanged")
        try:revision=releases.rollback(model,expected_revision)
        except KeyError:raise HTTPException(404,'unknown route')
        except ValueError as exc:raise HTTPException(409,str(exc))
        retirement=None
        if release_runtime is not None and loser and loser!=target:
            try:retirement=await release_runtime.retire(loser)
            except Exception as exc:  # noqa: BLE001 - route is already safe; report the cleanup failure
                retirement={'error':f'{type(exc).__name__}: {exc}'[:500],'release_id':loser}
        return {'revision':revision,'target':target,'runtime':warmed,'retirement':retirement}

    @app.put('/control/routes/{model}',dependencies=auth)
    def route(model:str,stable_release:str,expected_revision:int,candidate_release:str|None=None,weight:int=0):
        try:return {'revision':releases.route(model,stable_release,candidate_release=candidate_release,
                                             weight=weight,expected_revision=expected_revision)}
        except KeyError:raise HTTPException(404,'unknown release')
        except ValueError as exc:raise HTTPException(409,str(exc))

    return app
