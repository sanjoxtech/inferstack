import asyncio
import httpx
from fastapi.testclient import TestClient
from inferstack.engines.managed import DeploymentSpec, ManagedEngine
from inferstack.control.api import create_control_app
from inferstack.storage.repository import Repository


def test_engine_readiness_and_model_override():
    def handler(request):
        if request.url.path=='/v1/models':return httpx.Response(200,json={'data':[{'id':'actual'}]})
        import json
        assert json.loads(request.content)['model']=='actual'
        return httpx.Response(200,json={'choices':[{'text':'ok'}]})
    async def run():
        engine=ManagedEngine(DeploymentSpec('d','vllm','actual','http://test'),httpx.MockTransport(handler))
        assert await engine.ready()
        assert (await engine.complete({'model':'wrong','prompt':'hi'}))['choices'][0]['text']=='ok'
        await engine.close()
    asyncio.run(run())


def test_control_auth_registration_and_submit(tmp_path):
    repo=Repository(tmp_path/'db',tmp_path/'artifacts')
    client=TestClient(create_control_app(repo,'test-token'))
    assert client.get('/control/jobs/nope').status_code==401
    headers={'Authorization':'Bearer test-token'}
    body={'id':'d','engine':'vllm','model':'actual','base_url':'http://localhost:8000'}
    assert client.put('/control/records/deployment/d',headers=headers,json={'body':body}).status_code==200
    payload={'request_key':'request-1','deployment_id':'d','goal':'reduce latency'}
    a=client.post('/control/requests',headers=headers,json=payload)
    b=client.post('/control/requests',headers=headers,json=payload)
    assert a.status_code==200 and a.json()['job_id']==b.json()['job_id']
    assert client.post('/control/jobs/'+a.json()['job_id']+'/cancel',headers=headers).json()['cancelled']
