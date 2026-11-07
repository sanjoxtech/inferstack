import asyncio
import pytest
from inferstack.storage.repository import Repository
from inferstack.control.worker import OptimizationWorker

class Planner:
    async def plan(self, payload, history):
        return {'iteration':len(history), 'deployment_id':payload['deployment_id']}

class Executor:
    def __init__(self): self.cleaned=[]
    async def execute(self, plan, namespace): return {'id':namespace}
    async def cleanup(self, candidate): self.cleaned.append(candidate['id'])

class Verifier:
    async def verify(self, plan, candidate):
        return {'decision':'reject' if plan['iteration']==0 else 'keep'}

@pytest.mark.asyncio
async def test_loop_persists_and_cleans(tmp_path):
    repo=Repository(tmp_path/'db',tmp_path/'artifacts')
    job=repo.submit('one',{'deployment_id':'d'})
    executor=Executor()
    result=await OptimizationWorker(repo,Planner(),executor,Verifier()).once()
    assert result['state']=='completed'
    assert result['result']['decision']=='keep'
    assert len(executor.cleaned)==2
    assert [e['kind'] for e in result['events']].count('verified')==2
    assert Repository(tmp_path/'db',tmp_path/'artifacts').get(job)['result']==result['result']

@pytest.mark.asyncio
async def test_cancellation_stops_work(tmp_path):
    repo=Repository(tmp_path/'db',tmp_path/'artifacts')
    job=repo.submit('one',{'deployment_id':'d'})
    entered=asyncio.Event()
    class SlowVerifier:
        async def verify(self, plan, candidate):
            entered.set()
            await asyncio.Event().wait()
    executor=Executor()
    worker=OptimizationWorker(repo,Planner(),executor,SlowVerifier(),lease_seconds=.3)
    task=asyncio.create_task(worker.once())
    await entered.wait()
    repo.cancel(job)
    result=await asyncio.wait_for(task,2)
    assert result['state']=='cancelled'
    assert len(executor.cleaned)==1
    assert result['result'] is None
