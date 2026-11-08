"""Run control API or one durable optimization job from explicit configuration."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from inferstack.storage.repository import Repository


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['serve','gateway','worker-once','reconcile'])
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
        app=create_control_app(repo,token) if args.action=='serve' else create_gateway(repo,token)
        uvicorn.run(app,host=args.host,port=args.port)
    elif args.action=='reconcile':
        from inferstack.infrastructure import LocalProcessProvider,reconcile
        print(json.dumps(reconcile(repo,LocalProcessProvider()),indent=2))
    else:
        if not args.config:parser.error('--config required for worker-once')
        config=json.loads(args.config.read_text())
        async def run():
            from inferstack.optimization.planner import ProviderPlanner
            from inferstack.optimization.experiment import EndpointExecutor,EndpointVerifier
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
            allocations=AllocationManager(repo,worker_id)
            if 'provider' in config:
                # Tool-driven DAG planner with a configured optimizer provider (DeepSeek/OpenRouter/...).
                from inferstack.optimization.planner import ToolPlanner
                from inferstack.providers import create_provider
                planner=ToolPlanner(repo,create_provider(config['provider']),**config['planner'])
            else:
                planner=ProviderPlanner(repo,**config['planner'])
            executor=EndpointExecutor(repo,state/'logs',allocations=allocations,provider=provider,
                                      pricing=pricing,gpu_name=infra.get('gpu_name'))
            verifier=EndpointVerifier(repo,executor,**config['verifier'])
            try:
                result=await OptimizationWorker(repo,planner,executor,verifier,lease_seconds=lease,
                    max_iterations=config.get('max_iterations',3),worker_id=worker_id,
                    allocations=allocations,provider=provider).once()
                print(json.dumps(result,indent=2))
            finally:
                await planner.close()
                repo.retire_worker(worker_id)
        asyncio.run(run())

if __name__=='__main__':main()
