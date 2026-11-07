import asyncio
import json
from fastapi.testclient import TestClient
from inferstack.api.app import create_app
from inferstack.core.request import Request, RequestState
from inferstack.evaluation.metrics import StreamMeasurement

class Tokenizer:
    def decode(self,tokens,**kw):return ''.join(chr(t) for t in tokens)
class Service:
    model_id='test'
    tokenizer=Tokenizer()
    def diagnostics(self):return {}
    def decode_token(self,t):return chr(t)
    async def submit(self,prompt,sampling):
        r=Request('id',(1,2),sampling,state=RequestState.FINISHED,
            generated_token_ids=[65,66],token_emission_times=[10,10.1],finish_reason='length')
        q=asyncio.Queue()
        for token in (65,66,None):q.put_nowait(token)
        return r,q
    def cancel(self,rid):pass
class Registry:
    def ready(self):return True
    def get(self,name):return Service()
    def list(self):return [Service()]
    async def start_all(self):pass
    async def close_all(self):pass

def test_native_usage_and_engine_timing():
    with TestClient(create_app(Registry())) as c:
        body={'model':'test','prompt':'hi','max_tokens':2,'stream':True,'stream_options':{'include_usage':True}}
        response=c.post('/v1/completions',json=body)
        measurement=StreamMeasurement(0)
        for line in response.text.splitlines():
            if line.startswith('data:'):measurement.feed(line[5:].strip(),1)
        result=measurement.result(2)
        assert result['output_tokens']==2
        assert result['finish_reason']=='length'
        assert round(result['tpot_ms'])==100
        body['stream']=False
        assert c.post('/v1/completions',json=body).json()['usage']['total_tokens']==4
        body['max_tokens']=0
        assert c.post('/v1/completions',json=body).status_code==422

def test_unicode_split_tokens_match_non_streaming():
    class UnicodeTokenizer:
        def decode(self,tokens,**kw):return '\ufffd' if len(tokens)==1 else 'ह'
    class UnicodeService(Service):tokenizer=UnicodeTokenizer()
    class UnicodeRegistry(Registry):
        def get(self,name):return UnicodeService()
    with TestClient(create_app(UnicodeRegistry())) as c:
        body={'model':'test','prompt':'hi','stream':True,'stream_options':{'include_usage':True}}
        response=c.post('/v1/completions',json=body)
        measurement=StreamMeasurement(0)
        for line in response.text.splitlines():
            if line.startswith('data:'):measurement.feed(line[5:].strip(),1)
        assert measurement.result(2)['output_text']=='ह'
        assert measurement.result(2)['output_tokens']==2
