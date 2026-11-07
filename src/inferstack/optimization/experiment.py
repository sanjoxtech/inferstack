from __future__ import annotations
import asyncio
import hashlib
import json
import time
from pathlib import Path
from inferstack.engines.managed import DeploymentSpec, ManagedEngine
from inferstack.evaluation.judge import evaluate, Objective


class EndpointExecutor:
    """Runs registered candidates in separate processes or attaches to test endpoints.

    With `allocations` and `provider`, every launch is preceded by a persisted
    allocation intent, the process is tagged with its allocation identity, and
    the allocation is released only after the process is verified stopped.
    """
    accepts_context = True

    def __init__(self, repository, log_root, engine_factory=ManagedEngine, *, allocations=None, provider=None):
        self.repository=repository
        self.log_root=Path(log_root)
        self.engine_factory=engine_factory
        self.allocations,self.provider=allocations,provider
        if (allocations is None)!=(provider is None):raise ValueError('allocations and provider go together')
        self.active={}
        self.owned={}

    def _spawner(self, context, spec):
        if self.allocations is None or not spec.command or context is None:return None
        async def spawn(argv, log):
            alloc=self.allocations.intent(context['job'], context['step_key'], purpose='candidate:'+spec.id,
                provider=self.provider.name, resource_type=self.provider.resource_type,
                spec={'deployment_id':spec.id,'argv':argv}, gpu_count=context.get('gpu_count',0))
            if alloc['state']!='intent':raise RuntimeError(f"allocation {alloc['id']} already {alloc['state']}")
            process=await self.provider.spawn(alloc['id'], argv, log)
            self.allocations.faults('process_started')
            external=self.provider.external_for(process.pid)
            self.allocations.record_external(alloc['id'], external)
            self.allocations.faults('process_spawned')
            self.allocations.activate(alloc['id'], context['job'], external)
            self.owned[context['namespace']]=(alloc['id'],external)
            return process
        return spawn

    async def execute(self, plan, namespace, context=None):
        record=self.repository.read('deployment',plan['candidate_id'])
        if record['revision']!=plan['candidate_revision']:raise ValueError('candidate changed after planning')
        spec=DeploymentSpec.from_record(record["body"])
        baseline=self.repository.read('deployment',plan['baseline_id'])
        if baseline['body']['base_url'].rstrip('/')==spec.base_url.rstrip('/'):
            raise ValueError('candidate must use a separate endpoint before execution')
        if baseline['body']['model']!=spec.model:raise ValueError('candidate model mismatch')
        engine=self.engine_factory(spec)
        if context is not None:context={**context,'namespace':namespace}
        try:
            await engine.start(self.log_root/(namespace+'.log'),spawn=self._spawner(context,spec))
        except BaseException:
            await engine.close()
            self._release(namespace,'launch_failed')
            raise
        self.active[namespace]=engine
        return {'id':namespace, 'deployment_id':spec.id, 'revision':record['revision'],
                'spec_sha256':hashlib.sha256(self.repository.encode(record['body']).encode()).hexdigest()}

    def _release(self, namespace, note):
        owned=self.owned.pop(namespace,None)
        if owned is None:return
        alloc_id,external=owned
        if self.provider.stop(external):self.allocations.release(alloc_id,note)
        else:self.allocations.mark_orphaned(alloc_id,'stop_failed_during_'+note)

    async def cleanup(self, candidate):
        engine=self.active.pop(candidate['id'],None)
        if engine:await engine.close()
        if self.allocations is not None:self._release(candidate['id'],'cleanup')


class EndpointVerifier:
    """Repeated paired HTTP trials with exact-match task checks.

    Scope: deterministic completion workloads and end-to-end latency. Numerical
    logits, streaming TPOT, task-specific semantic quality need separate verifiers.
    """
    def __init__(self, repository, executor, baseline_id, cases, objectives, *, repeats=3):
        if repeats<3 or not cases:raise ValueError('three repeats and nonempty workload required')
        for case in cases:
            if set(case)!={'prompt','expected','max_tokens'} or not isinstance(case['expected'],str):
                raise ValueError('each case requires prompt, expected, max_tokens')
            if not isinstance(case['max_tokens'],int) or isinstance(case['max_tokens'],bool) or case['max_tokens']<=0:
                raise ValueError('max_tokens must be positive')
        self.repository,self.executor=repository,executor
        self.baseline_id,self.cases,self.repeats=baseline_id,cases,repeats
        self.objectives=[Objective(**o) for o in objectives]

    async def trial(self,engine):
        times=[];correct=True;outputs=[]
        for case in self.cases:
            start=time.perf_counter()
            result=await engine.complete({'prompt':case['prompt'],'max_tokens':case['max_tokens'],'temperature':0})
            times.append((time.perf_counter()-start)*1000)
            choices=result.get('choices',[])
            text=choices[0].get('text') if len(choices)==1 else None
            correct &= text==case['expected']
            outputs.append(text)
        return {'e2e_ms':sum(times)/len(times)},correct,outputs

    async def verify(self,plan,candidate):
        if plan.get('baseline_id')!=self.baseline_id:raise ValueError('job baseline does not match verifier configuration')
        baseline_record=self.repository.read('deployment',self.baseline_id)
        if plan.get('baseline_revision')!=baseline_record['revision']:raise ValueError('baseline changed after planning')
        candidate_record=self.repository.read('deployment',candidate['deployment_id'])
        baseline_spec=DeploymentSpec.from_record(baseline_record["body"])
        candidate_spec=DeploymentSpec.from_record(candidate_record["body"])
        if candidate_record['revision']!=candidate['revision']:raise ValueError('candidate changed during execution')
        if baseline_spec.base_url.rstrip('/')==candidate_spec.base_url.rstrip('/'):
            raise ValueError('baseline and candidate endpoints must be isolated')
        baseline=self.executor.engine_factory(baseline_spec)
        target=self.executor.active[candidate['id']]
        try:
            healthy=await baseline.ready() and await target.ready()
            trials=[[],[]];correct=True;outputs=[[],[]]
            # Warmup excluded from scored trials.
            await self.trial(baseline);await self.trial(target)
            for repeat in range(self.repeats):
                for index in ([0,1] if repeat%2==0 else [1,0]):
                    metrics,valid,texts=await self.trial((baseline,target)[index])
                    trials[index].append(metrics);outputs[index].append(texts);correct &= valid
            healthy &= await baseline.ready() and await target.ready()
            stable=self.repository.read('deployment',self.baseline_id)['revision']==baseline_record['revision']
            stable &= self.repository.read('deployment',candidate['deployment_id'])['revision']==candidate['revision']
            gates={'correctness':correct,'quality':correct,'health':healthy,
                   'complete':True,'comparable':stable and baseline_spec.model==candidate_spec.model}
            verdict=evaluate(*trials,self.objectives,gates=gates)
            evidence={'baseline':baseline_record,'candidate':candidate,'workload':self.cases,
                      'trials':trials,'outputs':outputs,'gates':gates,'verdict':verdict,
                      'scope':'exact-match completion suite; mean end-to-end latency only'}
            artifact=self.repository.artifact(self.repository.encode(evidence).encode())
            return {**verdict,'evidence':artifact,'scope':evidence['scope']}
        finally:
            await baseline.close()
