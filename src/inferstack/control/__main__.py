"""Run control API or one durable optimization job from explicit configuration."""
import argparse
import asyncio
import json
import os
from pathlib import Path
from inferstack.storage.repository import Repository


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['serve','gateway','worker-once'])
    parser.add_argument('--state',default='.inferstack')
    parser.add_argument('--host',default='127.0.0.1')
    parser.add_argument('--port',type=int,default=8010)
    parser.add_argument('--config',type=Path)
    args=parser.parse_args()
    state=Path(args.state)
    repo=Repository(state/'control.db',state/'artifacts')
    if args.action in {'serve','gateway'}:
        import uvicorn
        from inferstack.control.api import create_control_app
        token=os.environ.get('INFERSTACK_CONTROL_TOKEN','')
        from inferstack.serving.gateway import create_gateway
        app=create_control_app(repo,token) if args.action=='serve' else create_gateway(repo,token)
        uvicorn.run(app,host=args.host,port=args.port)
    else:
        if not args.config:parser.error('--config required for worker-once')
        config=json.loads(args.config.read_text())
        async def run():
            from inferstack.optimization.planner import ProviderPlanner
            from inferstack.optimization.experiment import EndpointExecutor,EndpointVerifier
            from inferstack.control.worker import OptimizationWorker
            planner=ProviderPlanner(repo,**config['planner'])
            executor=EndpointExecutor(repo,state/'logs')
            verifier=EndpointVerifier(repo,executor,**config['verifier'])
            try:
                result=await OptimizationWorker(repo,planner,executor,verifier,
                    max_iterations=config.get('max_iterations',3)).once()
                print(json.dumps(result,indent=2))
            finally:await planner.close()
        asyncio.run(run())

if __name__=='__main__':main()
