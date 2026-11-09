"""Run control API or one durable optimization job from explicit configuration."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from inferstack.storage.repository import Repository


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['serve','gateway','worker-once','reconcile','retention','outbox-deliver',
                                          'triggers-tick','context','experience'])
    parser.add_argument('--deployment',help='context/experience: deployment id')
    parser.add_argument('--limit',type=int,default=20,help='experience: maximum items returned')
    parser.add_argument('--treatment',help='experience: only this treatment signature')
    parser.add_argument('--workload',type=Path,help='context: JSON workload fixture (list of cases)')
    parser.add_argument('--callback-url',help='outbox-deliver: FactoryOS outcome endpoint (https)')
    parser.add_argument('--callback-token-env',default='FACTORYOS_CALLBACK_TOKEN')
    parser.add_argument('--max-active',type=int,default=2,help='triggers-tick: global unfinished triggered jobs')
    parser.add_argument('--event-ttl-days',type=int)
    parser.add_argument('--artifact-ttl-days',type=int)
    parser.add_argument('--dry-run',action='store_true')
    parser.add_argument('--state',default='.inferstack')
    parser.add_argument('--database',help='database URL (sqlite:///path or postgresql://...); default STATE/control.db')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8010)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    state=Path(args.state)
    repo=Repository(args.database or state/'control.db',state/'artifacts')
    if args.action in {'serve','gateway'}:
        import uvicorn
        from inferstack.control.api import create_control_app
        token=os.environ.get('INFERSTACK_CONTROL_TOKEN','')
        from inferstack.serving.gateway import create_gateway
        observer=None
        if args.action=='serve' and os.environ.get('INFERSTACK_GATEWAY_URL'):
            from inferstack.releases.rollout import http_observer
            observer=http_observer(repo,os.environ['INFERSTACK_GATEWAY_URL'],
                                   os.environ.get('INFERSTACK_GATEWAY_TOKEN') or token)
        elif args.action=='serve' and os.environ.get('INFERSTACK_ROLLOUT_EVIDENCE')=='store':
            # Gateways write durable evidence to this same store: read windows directly.
            from inferstack.releases.rollout import evidence_observer
            observer=evidence_observer(repo)
        from inferstack.releases.attestation import launch_observer,release_attestor
        from inferstack.infrastructure import LocalProcessProvider
        providers={'local':LocalProcessProvider()}
        if os.environ.get('INFERSTACK_DOCKER_ATTEST','1')=='1':
            from inferstack.infrastructure.remote import DockerProvider
            providers['docker']=DockerProvider()
        if os.environ.get('INFERSTACK_K8S_NAMESPACE'):
            from inferstack.infrastructure.remote import KubernetesProvider
            providers['kubernetes']=KubernetesProvider(namespace=os.environ['INFERSTACK_K8S_NAMESPACE'])
        attest=release_attestor(repo,observe_launch=launch_observer(repo,providers))
        runtime=None
        if args.action=='serve':
            # Releases are launched/warmed before traffic and drained/stopped after promotion or rollback.
            from inferstack.releases.runtime import ReleaseRuntime
            from inferstack.releases.rollout import http_drain
            from inferstack.releases.rollout import http_undrain
            drain=undrain=None
            if os.environ.get('INFERSTACK_GATEWAY_URL'):
                gateway_token=os.environ.get('INFERSTACK_GATEWAY_TOKEN') or token
                drain=http_drain(os.environ['INFERSTACK_GATEWAY_URL'],gateway_token)
                undrain=http_undrain(os.environ['INFERSTACK_GATEWAY_URL'],gateway_token)
            # Every configured provider: retirement dispatches to the allocation's own provider.
            from inferstack.control.wiring import host_gpu_devices
            gpu_devices=[d for d in os.environ.get('INFERSTACK_GPU_DEVICES','').split(',') if d] or \
                host_gpu_devices({},'local')
            runtime=ReleaseRuntime(repo,providers['local'],providers=providers,log_root=state/'release-logs',
                                   attest=attest,drain=drain,undrain=undrain,gpu_devices=gpu_devices)
        app=create_control_app(repo,token,rollout_observer=observer,rollout_attest=attest,release_runtime=runtime) \
            if args.action=='serve' \
            else create_gateway(repo,token)
        uvicorn.run(app,host=args.host,port=args.port)
    elif args.action=='retention':
        from inferstack.contracts.redaction import RetentionPolicy
        from inferstack.control.maintenance import enforce
        policy=RetentionPolicy(event_ttl_days=args.event_ttl_days,artifact_ttl_days=args.artifact_ttl_days)
        print(json.dumps(enforce(repo,policy,dry_run=args.dry_run),indent=2))
    elif args.action=='outbox-deliver':
        from inferstack.integrations.outbox import Outbox,http_sender
        if not args.callback_url:parser.error('--callback-url required for outbox-deliver')
        print(json.dumps(asyncio.run(Outbox(repo).deliver_due(http_sender(args.callback_url,
                                                                           token_env=args.callback_token_env)))))
    elif args.action=='triggers-tick':
        from inferstack.integrations.triggers import TriggerService
        observe=None
        if os.environ.get('INFERSTACK_GATEWAY_URL'):
            from inferstack.integrations.slo import gateway_slo_observer
            observe=gateway_slo_observer(os.environ['INFERSTACK_GATEWAY_URL'],
                                         os.environ.get('INFERSTACK_GATEWAY_TOKEN') or os.environ.get('INFERSTACK_CONTROL_TOKEN',''))
        print(json.dumps(asyncio.run(TriggerService(repo,observe=observe,max_active=args.max_active).tick()),indent=2))
    elif args.action in ('context','experience'):
        if not args.deployment:parser.error('--deployment required')
        if args.action=='experience':
            from inferstack.memory import experience
            out=experience(repo,args.deployment,limit=args.limit,treatment_signature=args.treatment)
        else:
            from inferstack.memory import context_snapshot
            workload=json.loads(args.workload.read_text()) if args.workload else None
            out=context_snapshot(repo,args.deployment,workload=workload)
        print(json.dumps(out,indent=2,default=str))
    elif args.action=='reconcile':
        from inferstack.infrastructure import LocalProcessProvider,reconcile
        print(json.dumps(reconcile(repo,LocalProcessProvider()),indent=2))
    else:
        if not args.config:parser.error('--config required for worker-once')
        config=json.loads(args.config.read_text())
        async def run():
            from inferstack.optimization.planner import ProviderPlanner
            from inferstack.optimization.experiment import EndpointVerifier
            from inferstack.control.worker import OptimizationWorker
            from inferstack.infrastructure import AllocationManager,LocalProcessProvider,reconcile
            lease=config.get('lease_seconds',30)
            worker_id=repo.register_worker(lease)
            infra=config.get('infrastructure',{})
            kind=infra.get('provider','local')
            if kind=='docker':
                from inferstack.infrastructure.remote import DockerProvider
                provider=DockerProvider(**infra.get('docker',{}))
            elif kind=='kubernetes':
                from inferstack.infrastructure.remote import KubernetesProvider
                provider=KubernetesProvider(**infra['kubernetes'])
            elif kind=='local':
                provider=LocalProcessProvider()
            else:parser.error(f'unknown infrastructure provider {kind}')
            pricing=None
            if infra.get('prices'):
                from inferstack.infrastructure.pricing import PriceTable
                pricing=PriceTable(infra['prices'])
            # Resolve resources left by dead workers before taking new work.
            for action in reconcile(repo,provider):print(json.dumps({'reconciled':action}))
            allocations=AllocationManager(repo,worker_id,quotas=infra.get('quotas'))
            from inferstack.control.wiring import build_executor,build_fetcher,telemetry_source
            if kind!='local':
                # Build steps always run as local processes; reconcile their allocations too.
                for action in reconcile(repo,LocalProcessProvider()):print(json.dumps({'reconciled':action}))
            if 'provider' in config:
                # Tool-driven DAG planner with a configured optimizer provider (DeepSeek/OpenRouter/...).
                from inferstack.optimization.planner import ToolPlanner
                from inferstack.providers import create_provider
                telemetry=telemetry_source(repo) if config.get('planner_telemetry',True) else None
                planner=ToolPlanner(repo,create_provider(config['provider']),telemetry=telemetry,**config['planner'])
            else:
                planner=ProviderPlanner(repo,**config['planner'])
            from inferstack.control.wiring import host_gpu_devices
            executor=build_executor(repo,config,state=state,allocations=allocations,provider=provider,
                                    pricing=pricing,gpu_name=infra.get('gpu_name'),
                                    gpu_devices=host_gpu_devices(config,provider.name))
            verifier=EndpointVerifier(repo,executor,**config['verifier'])
            try:
                result=await OptimizationWorker(repo,planner,executor,verifier,lease_seconds=lease,
                    max_iterations=config.get('max_iterations',3),worker_id=worker_id,
                    allocations=allocations,provider=provider,fetcher=build_fetcher(config)).once()
                print(json.dumps(result,indent=2))
            finally:
                await planner.close()
                repo.retire_worker(worker_id)
        asyncio.run(run())

if __name__=='__main__':main()
