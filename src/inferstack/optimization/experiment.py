from __future__ import annotations
import asyncio
import hashlib
import math
import json
import time
from pathlib import Path
from inferstack.engines.managed import DeploymentSpec, ManagedEngine
from inferstack.evaluation.judge import evaluate, Objective


GPU_ENGINES = frozenset({'inferstack_native', 'vllm', 'sglang'})


def resolve_gpu_reservation(context, spec, resources=None):
    """GPU reservation for a local launch: (count, devices, basis). Never silently zero.

    Precedence: explicit reservation in the worker context, then the deployment's
    declared `resources`, then the engine's minimum. GPU engines reserve at least
    one device; only engines that do not use a GPU may record zero.
    """
    context, resources = context or {}, resources or {}
    reservation = context.get('reservation') or {}
    for source, basis in ((reservation, 'reservation'), (resources, 'deployment_resources')):
        devices = source.get('gpu_devices')
        if devices:
            if not isinstance(devices, (list, tuple)) or not all(isinstance(d, str) and d for d in devices):
                raise ValueError('gpu_devices must be a list of device identifiers')
            return len(devices), list(devices), basis
        if source.get('gpu_count') is not None:
            count = source['gpu_count']
            if isinstance(count, bool) or not isinstance(count, int) or count < 0:
                raise ValueError('gpu_count must be a non-negative integer')
            if count == 0 and spec.engine in GPU_ENGINES:
                raise ValueError(f'{spec.engine} needs a GPU; a zero-GPU reservation is not valid')
            return count, None, basis
    if context.get('gpu_count'):
        return int(context['gpu_count']), None, 'context'
    if spec.engine in GPU_ENGINES:
        return 1, None, 'engine_minimum'
    return 0, None, 'cpu_engine'


class EndpointExecutor:
    """Runs registered candidates in separate processes or attaches to test endpoints.

    With `allocations` and `provider`, every launch is preceded by a persisted
    allocation intent, the process is tagged with its allocation identity, and
    the allocation is released only after the process is verified stopped.
    """
    accepts_context = True

    def __init__(self, repository, log_root, engine_factory=ManagedEngine, *, allocations=None, provider=None,
                 pricing=None, gpu_name=None, ready_timeout=900):
        self.pricing,self.gpu_name,self.ready_timeout=pricing,gpu_name,ready_timeout
        self.repository=repository
        self.log_root=Path(log_root)
        self.engine_factory=engine_factory
        self.allocations,self.provider=allocations,provider
        if (allocations is None)!=(provider is None):raise ValueError('allocations and provider go together')
        self.active={}
        self.owned={}

    def _spawner(self, context, spec, env=None, resources=None):
        if self.allocations is None or not spec.command or context is None:return None
        async def spawn(argv, log):
            gpus,devices,basis=resolve_gpu_reservation(context,spec,resources)
            alloc=self.allocations.intent(context['job'], context['step_key'], purpose='candidate:'+spec.id,
                provider=self.provider.name, resource_type=self.provider.resource_type,
                spec={'deployment_id':spec.id,'argv':argv,'gpu_basis':basis,
                      **({'gpu_devices':devices} if devices else {}),**self._price(gpus)}, gpu_count=gpus)
            if alloc['state']!='intent':raise RuntimeError(f"allocation {alloc['id']} already {alloc['state']}")
            child_env,inherit=env,env is None
            if devices:child_env={**(env or {}),'CUDA_VISIBLE_DEVICES':','.join(devices)}
            process=await self.provider.spawn(alloc['id'], argv, log, env=child_env, inherit_env=inherit)
            self.allocations.faults('process_started')
            external={**self.provider.external_for(process.pid),'allocation_id':alloc['id']}
            self.allocations.record_external(alloc['id'], external)
            self.allocations.faults('process_spawned')
            self.allocations.activate(alloc['id'], context['job'], external)
            self.owned[context['namespace']]=(alloc['id'],external)
            return process
        return spawn

    async def execute(self, plan, namespace, context=None):
        if any(a.get('type') in ('plugin_install','source_patch') for a in plan.get('actions',[])) or \
                any('profile_patch' in a.get('params',{}) for a in plan.get('actions',[])):
            raise ValueError('this executor only runs registered candidates; use IsolatedExecutor for '
                             'profile_patch, plugin_install and source_patch actions')
        record=self.repository.read('deployment',plan['candidate_id'])
        if record['revision']!=plan['candidate_revision']:raise ValueError('candidate changed after planning')
        return await self.launch(record, plan, namespace, context)

    def _price(self, gpus):
        if self.pricing is None:
            return {'price': None} if gpus else {}
        return self.pricing.stamp(self.provider.name, self.gpu_name or '', gpus)

    async def _launch_remote(self, record, spec, namespace, context, extra):
        """Container/Kubernetes: launch through the provider, then attach and wait for readiness."""
        container=dict((record['body'].get('profile') or {}).get('container') or {})
        if not container:raise ValueError('remote providers need profile.container (image@sha256, ports, args)')
        gpus=int(container.get('gpus') if isinstance(container.get('gpus'),int) else len(container.get('gpus') or []))
        alloc=self.allocations.intent(context['job'],context['step_key'],purpose='candidate:'+spec.id,
            provider=self.provider.name,resource_type=self.provider.resource_type,
            spec={'deployment_id':spec.id,'container':container,**self._price(gpus)},gpu_count=gpus)
        external=self.provider.launch(alloc['id'],container)
        self.allocations.record_external(alloc['id'],external)
        self.owned[namespace]=(alloc['id'],external)
        self.allocations.activate(alloc['id'],context['job'],external)
        engine=self.engine_factory(DeploymentSpec.from_record({**record['body'],'command':[]}))
        deadline=time.monotonic()+self.ready_timeout
        try:
            while not await engine.ready():
                if time.monotonic()>deadline or not self.provider.alive(external):
                    raise RuntimeError(f'{self.provider.name} candidate {spec.id} did not become ready')
                await asyncio.sleep(2)
        except BaseException:
            await engine.close()
            self._release(namespace,'launch_failed')
            raise
        self.active[namespace]=engine
        return {'id':namespace,'deployment_id':spec.id,'revision':record['revision'],'external':external,
                'spec_sha256':hashlib.sha256(self.repository.encode(record['body']).encode()).hexdigest(),**(extra or {})}

    async def launch(self, record, plan, namespace, context, env=None, extra=None):
        spec=DeploymentSpec.from_record(record["body"])
        if context is not None and getattr(self.provider,'launches_remotely',False):
            baseline=self.repository.read('deployment',plan['baseline_id'])
            if baseline['body']['model']!=spec.model:raise ValueError('candidate model mismatch')
            return await self._launch_remote(record,spec,namespace,context,extra)
        baseline=self.repository.read('deployment',plan['baseline_id'])
        if baseline['body']['base_url'].rstrip('/')==spec.base_url.rstrip('/'):
            raise ValueError('candidate must use a separate endpoint before execution')
        if baseline['body']['model']!=spec.model:raise ValueError('candidate model mismatch')
        engine=self.engine_factory(spec)
        if context is not None:context={**context,'namespace':namespace}
        try:
            await engine.start(self.log_root/(namespace+'.log'),spawn=self._spawner(context,spec,env,record['body'].get('resources')))
        except BaseException:
            await engine.close()
            self._release(namespace,'launch_failed')
            raise
        self.active[namespace]=engine
        return {'id':namespace, 'deployment_id':spec.id, 'revision':record['revision'],
                'spec_sha256':hashlib.sha256(self.repository.encode(record['body']).encode()).hexdigest(),
                **(extra or {})}

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


def _code_sha256():
    from pathlib import Path as _P
    digest=hashlib.sha256()
    for path in sorted([_P(__file__),_P(__file__).parents[1]/'evaluation'/'judge.py']):
        digest.update(path.read_bytes())
    return digest.hexdigest()


VERIFIER_CODE_SHA256=_code_sha256()


MAX_REPEATS=20
MEASURED_METRICS=frozenset({'e2e_ms'})
# Gates this verifier can execute. Anything else a plan requests is reported as
# unsupported (None), which makes the verdict inconclusive, never a silent pass.
SERVING_GATES=frozenset({'cancellation','overload'})


class EndpointVerifier:
    """Repeated paired HTTP trials with exact-match task checks, bound to the plan's policy.

    The verification policy is the mandatory gates plus every gate the frozen plan
    declares, with the plan's measure parameters. Gates this verifier cannot run
    (numerical oracle, quality budget, cache isolation, ...) are recorded as
    unsupported and make the verdict inconclusive. Comparability requires pinned,
    equal hardware and model identities on both sides; missing identity is unknown.

    Scope: deterministic completion workloads, end-to-end latency, cancellation and
    overload probes. Numerical logits, streaming TPOT and semantic quality need
    separate verifiers.
    """
    def __init__(self, repository, executor, baseline_id, cases, objectives, *, repeats=3, workload_ref=None,
                 objective_ref=None, overload_concurrency=8):
        if repeats<3 or not cases:raise ValueError('three repeats and nonempty workload required')
        for case in cases:
            if set(case)!={'prompt','expected','max_tokens'} or not isinstance(case['expected'],str):
                raise ValueError('each case requires prompt, expected, max_tokens')
            if not isinstance(case['max_tokens'],int) or isinstance(case['max_tokens'],bool) or case['max_tokens']<=0:
                raise ValueError('max_tokens must be positive')
        self.repository,self.executor=repository,executor
        self.baseline_id,self.cases,self.repeats=baseline_id,cases,repeats
        self.objectives=[Objective(**o) for o in objectives]
        self.workload_ref,self.objective_ref=workload_ref,objective_ref
        self.overload_concurrency=overload_concurrency

    def policy(self,plan):
        """The executable policy for this plan: required gates, repeats and binding problems."""
        required=sorted(set(plan.get('gates') or [])|{'correctness','quality','health','comparable','complete'})
        problems=[]
        measure=next((a for a in plan.get('actions') or [] if a.get('type')=='measure'),None)
        params=(measure or {}).get('params') or {}
        repeats=self.repeats
        if params.get('repeats') is not None:
            asked=params['repeats']
            if not isinstance(asked,int) or isinstance(asked,bool) or asked<1:problems.append('invalid repeats')
            elif asked>MAX_REPEATS:problems.append(f'plan requests {asked} repeats; verifier maximum is {MAX_REPEATS}')
            else:repeats=max(repeats,asked)
        for key,configured in (('workload_ref',self.workload_ref),('objective_ref',self.objective_ref)):
            if params.get(key) is not None and params[key]!=configured:
                problems.append(f'plan {key} {params[key]!r} is not the verifier-configured {configured!r}')
        expected=set((measure or {}).get('expected_measurements') or plan.get('expected_measurements') or [])
        unmeasured=sorted(expected-MEASURED_METRICS)
        if unmeasured:problems.append('measurements not produced by this verifier: '+','.join(unmeasured))
        return {'required':required,'repeats':repeats,'problems':problems,
                'unsupported':sorted(set(required)-{'correctness','quality','health','comparable','complete'}-SERVING_GATES)}

    def comparability(self,baseline_record,candidate_record,stable):
        """True only with pinned, equal hardware and model identities; None when identity is missing."""
        b,c=baseline_record['body'],candidate_record['body']
        detail={'records_stable':stable,'model_name':b['model']==c['model']}
        identity={}
        for key,kind in (('hardware_ref','hardware'),('model_ref','model')):
            pins=[]
            for body in (b,c):
                ref=body.get(key)
                if not ref:pins.append(None);continue
                from inferstack.contracts.records import parse_ref
                try:
                    identifier,revision=parse_ref(ref)
                    record=self.repository.read(kind,identifier,revision)
                    pins.append(f"{identifier}@{record['revision']}")
                except (KeyError,ValueError):pins.append(None)
            identity[key]=pins
            detail[key]='missing' if None in pins else 'same' if pins[0]==pins[1] else 'different'
        if not stable or not detail['model_name'] or 'different' in (detail['hardware_ref'],detail['model_ref']):
            return False,detail,identity
        if 'missing' in (detail['hardware_ref'],detail['model_ref']):return None,detail,identity
        return True,detail,identity

    async def serving_gate(self,name,engine):
        """Cancellation/overload probes on the candidate. None when the engine cannot be probed."""
        case=self.cases[0]
        payload={'prompt':case['prompt'],'max_tokens':case['max_tokens'],'temperature':0}
        if not (hasattr(engine,'complete') and hasattr(engine,'ready')):return None,'engine does not expose completions'
        try:
            if name=='cancellation':
                if not hasattr(engine,'stream'):return None,'engine does not expose streaming'
                stream=engine.stream(payload)
                try:
                    await asyncio.wait_for(stream.__anext__(),30)
                except StopAsyncIteration:pass
                finally:await stream.aclose()
                # After a client disconnect the engine must stay live and still answer correctly.
                result=await engine.complete(payload)
                choices=result.get('choices',[])
                ok=await engine.ready() and len(choices)==1 and choices[0].get('text')==case['expected']
                return ok,'stream cancelled after first chunk; engine live and correct afterwards'
            if name=='overload':
                import httpx as _httpx
                async def one():
                    try:
                        result=await engine.complete(payload)
                        choices=result.get('choices',[])
                        return 'ok' if len(choices)==1 and choices[0].get('text')==case['expected'] else 'wrong'
                    except _httpx.HTTPStatusError as exc:
                        return 'shed' if exc.response.status_code in (429,503) else 'error'
                    except _httpx.HTTPError:return 'error'
                outcomes=await asyncio.gather(*(one() for _ in range(self.overload_concurrency)))
                live=await engine.ready()
                ok=live and 'error' not in outcomes and 'wrong' not in outcomes and 'ok' in outcomes
                return ok,f'{self.overload_concurrency} concurrent requests: '+','.join(sorted(outcomes))+f'; live={live}'
        except Exception as exc:  # noqa: BLE001 - a probe failure is evidence, recorded with its type
            return False,f'{name} probe failed: {type(exc).__name__}'
        return None,f'{name} not supported'

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

    def record_trials(self,plan,candidate,baseline_record,trials,raw_sha256):
        """Persist each scored trial as a typed `trial` record pinned to exact deployment revisions."""
        refs=[]
        now=self.repository.clock()
        pins={'baseline':f"{baseline_record['id']}@{baseline_record['revision']}",
              'candidate':f"{candidate['deployment_id']}@{candidate['revision']}"}
        for role,side in zip(('baseline','candidate'),trials):
            for index,metrics in enumerate(side):
                identifier=f"{candidate['id']}-{role}-{index}"
                try:
                    self.repository.put_record('trial',identifier,{'experiment_id':candidate['id'],
                        'deployment_ref':pins[role],'role':role,'repeat_index':index,'metrics':metrics,
                        'raw_artifact_sha256':raw_sha256,'started_at':now,'ended_at':now})
                except ValueError as exc:
                    if 'expected_revision' not in str(exc):raise
                refs.append(f'{identifier}@1')
        return refs

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
            policy=self.policy(plan)
            trials=[[],[]];correct=True;outputs=[[],[]]
            # Warmup excluded from scored trials.
            await self.trial(baseline);await self.trial(target)
            for repeat in range(policy['repeats']):
                for index in ([0,1] if repeat%2==0 else [1,0]):
                    metrics,valid,texts=await self.trial((baseline,target)[index])
                    trials[index].append(metrics);outputs[index].append(texts);correct &= valid
            healthy &= await baseline.ready() and await target.ready()
            stable=self.repository.read('deployment',self.baseline_id)['revision']==baseline_record['revision']
            stable &= self.repository.read('deployment',candidate['deployment_id'])['revision']==candidate['revision']
            # Gates are computed from evidence, never asserted.
            metric_names={o.metric for o in self.objectives}
            complete=all(len(side)==policy['repeats'] for side in trials) and all(
                isinstance(t.get(m),(int,float)) and math.isfinite(t[m]) for side in trials for t in side for m in metric_names)
            complete&=all(text is not None for side in outputs for run in side for text in run)
            comparable,comparability,identity=self.comparability(baseline_record,candidate_record,stable)
            gates={'correctness':correct,'quality':correct,'health':healthy,'complete':complete,'comparable':comparable}
            gate_notes={}
            for name in policy['required']:
                if name in gates:continue
                if name in SERVING_GATES:gates[name],gate_notes[name]=await self.serving_gate(name,target)
                else:gates[name],gate_notes[name]=None,'not supported by EndpointVerifier'
            if policy['problems']:
                verdict={'decision':'inconclusive','reasons':['verification policy not satisfiable: '+
                                                              '; '.join(policy['problems'])]}
            else:
                verdict=evaluate(*trials,self.objectives,gates=gates,required=policy['required'])
            fixture_sha=hashlib.sha256(self.repository.encode(self.cases).encode()).hexdigest()
            evidence={'baseline':baseline_record,'candidate':candidate,'workload':self.cases,
                      'trials':trials,'outputs':outputs,'gates':gates,'verdict':verdict,
                      'verifier':{'code_sha256':VERIFIER_CODE_SHA256,'fixture_sha256':fixture_sha,
                                  'repeats':policy['repeats'],'order':'interleaved, warmup excluded'},
                      'policy':{**policy,'sha256':hashlib.sha256(self.repository.encode(policy).encode()).hexdigest()},
                      'gate_notes':gate_notes,'comparability':comparability,'identity':identity,
                      'scope':'exact-match completion suite (quality gate = exact match); mean end-to-end latency only'}
            artifact=self.repository.artifact(self.repository.encode(evidence).encode())
            trial_refs=self.record_trials(plan,candidate,baseline_record,trials,artifact['sha256'])
            return {**verdict,'evidence':artifact,'scope':evidence['scope'],'gates':gates,'trial_refs':trial_refs,
                    'policy_sha256':evidence['policy']['sha256']}
        finally:
            await baseline.close()
