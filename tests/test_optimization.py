import json
import httpx
import pytest
from inferstack.storage.repository import Repository
from inferstack.engines.managed import ManagedEngine
from inferstack.optimization.planner import ProviderPlanner
from inferstack.optimization.experiment import EndpointExecutor,EndpointVerifier
from inferstack.control.worker import OptimizationWorker


def setup(tmp_path):
    repo=Repository(tmp_path/'db',tmp_path/'artifacts')
    for name in ('base','candidate'):
        repo.put('deployment',name,{'id':name,'model':'m','engine':'generic_openai','base_url':f'http://{name}'})
    return repo

@pytest.mark.asyncio
async def test_provider_rejects_unregistered_execution(tmp_path,monkeypatch):
    monkeypatch.setenv('TEST_PLANNER_KEY','test-only')
    repo=setup(tmp_path)
    def handler(request):
        return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({
            'candidate_id':'not-registered','hypothesis':'test','evidence_ids':[]})}}]})
    planner=ProviderPlanner(repo,base_url='https://planner',model='configured-model',api_key_env='TEST_PLANNER_KEY',
        candidate_ids=['candidate'],transport=httpx.MockTransport(handler))
    try:
        with pytest.raises(ValueError,match='unregistered'):
            await planner.plan({'deployment_id':'base'},[])
    finally:await planner.close()

@pytest.mark.asyncio
async def test_real_components_integrate_with_mock_http(tmp_path,monkeypatch):
    monkeypatch.setenv('TEST_PLANNER_KEY','test-only')
    repo=setup(tmp_path)
    def handler(request):
        if request.url.host=='planner':
            return httpx.Response(200,json={'choices':[{'finish_reason':'stop','message':{'content':json.dumps({
                'candidate_id':'candidate','hypothesis':'registered alternate config','evidence_ids':[]})}}]})
        if request.url.path=='/v1/models':return httpx.Response(200,json={'data':[{'id':'m'}]})
        return httpx.Response(200,json={'choices':[{'text':'wrong' if request.url.host=='candidate' else 'four'}]})
    transport=httpx.MockTransport(handler)
    planner=ProviderPlanner(repo,base_url='https://planner',model='configured-model',api_key_env='TEST_PLANNER_KEY',
        candidate_ids=['candidate'],transport=transport)
    executor=EndpointExecutor(repo,tmp_path/'logs',engine_factory=lambda spec:ManagedEngine(spec,transport=transport))
    verifier=EndpointVerifier(repo,executor,'base',[{'prompt':'2+2','expected':'four','max_tokens':4}],
                              [{'metric':'e2e_ms','direction':'min'}])
    repo.submit('integration',{'deployment_id':'base','goal':'reduce latency'})
    try:
        result=await OptimizationWorker(repo,planner,executor,verifier,max_iterations=1).once()
        assert result['state']=='completed'
        assert result['result']['decision']=='reject'
        verdict=result['result']['experiments'][0]['verdict']
        evidence=json.loads(repo.read_artifact(verdict['evidence']['sha256']))
        assert evidence['gates']['quality'] is False
        assert not executor.active
    finally:await planner.close()
