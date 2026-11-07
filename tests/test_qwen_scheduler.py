import asyncio,time
from types import SimpleNamespace
import torch
from inferstack.config import load_profile
from inferstack.core.request import Request,SamplingParams,RequestState
from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine
from inferstack.serving.service import ModelService,ModelServiceRegistry

class Tok:
    eos_token_id=31;pad_token_id=0
    def encode(self,text,add_special_tokens=False):return [((ord(c)%20)+1) for c in text] or [1]
    def decode(self,ids,skip_special_tokens=True):return ''.join(chr(65+(int(x)%26)) for x in ids)
class Layer:
    def __init__(self,b,t):
        self.keys=torch.zeros((b,1,t,2));self.values=torch.zeros((b,1,t,2));self.conv_states={0:torch.zeros((b,2,3))};self.recurrent_states={0:torch.zeros((b,2,2,2))}
class Cache:
    def __init__(self,b,t):self.layers=[Layer(b,t)]
class Model:
    config=SimpleNamespace(model_type='qwen3_5',max_position_embeddings=4096,text_config=SimpleNamespace(model_type='qwen3_5',layer_types=['linear_attention','full_attention'],max_position_embeddings=4096))
    def __init__(self,sleep=0,fail_after=None):self.calls=[];self.sleep=sleep;self.fail_after=fail_after
    def forward(self,input_ids,attention_mask=None,position_ids=None,past_key_values=None,use_cache=True,**kw):
        if self.sleep:time.sleep(self.sleep)
        if self.fail_after is not None and len(self.calls)>=self.fail_after:raise RuntimeError('boom')
        b,t=input_ids.shape;self.calls.append((b,t,past_key_values is not None))
        if past_key_values is None:past=Cache(b,t)
        else:
            past=past_key_values
            for layer in past.layers:
                add=torch.zeros((b,1,t,2));layer.keys=torch.cat([layer.keys,add],dim=-2);layer.values=torch.cat([layer.values,add],dim=-2)
        logits=torch.zeros((b,t,32));logits[...,5]=10
        return SimpleNamespace(logits=logits,past_key_values=past)
    __call__=forward

def cfg():
    c=load_profile('configs/profiles/production-v1.6.0-baseline.toml');c.native.custom_gdn_triton=False;c.native.prefix_state_cache=False;c.native.batch_wait_ms=1;c.native.prefill_chunk_tokens=2;c.native.prefill_chunks_per_turn=1;c.native.max_requests_per_batch=8;c.native.max_active_cohorts=4;c.kernels.torch_compile=False;c.native.compile_model=False;return c
async def drain(q,timeout=5):
    out=[]
    while True:
        x=await asyncio.wait_for(q.get(),timeout)
        if x is None:return out
        if isinstance(x,BaseException):raise x
        out.append(x)

def test_batched_requests_use_one_decode_forward_batch():
    async def go():
        c=cfg();m=Model(.001);e=QwenHybridContinuousEngine(m,Tok(),c,'cpu');await e.start();r1=Request('a',(1,2,3),SamplingParams(max_new_tokens=5));r2=Request('b',(1,2,3),SamplingParams(max_new_tokens=5));q1=await e.submit(r1);q2=await e.submit(r2);await asyncio.gather(drain(q1),drain(q2));d=e.diagnostics();await e.close();assert d['max_observed_batch']>=2;assert any(b==2 and t==1 for b,t,_ in m.calls)
    asyncio.run(go())

def test_late_request_live_refills_batch():
    async def go():
        c=cfg();m=Model(.002);e=QwenHybridContinuousEngine(m,Tok(),c,'cpu');await e.start();q1=await e.submit(Request('a',(1,2),SamplingParams(max_new_tokens=12)))
        for _ in range(200):
            if e.diagnostics()['active_batches']:break
            await asyncio.sleep(.001)
        q2=await e.submit(Request('b',(1,2,3,4),SamplingParams(max_new_tokens=4)));await asyncio.gather(drain(q1),drain(q2));d=e.diagnostics();await e.close();assert d['mid_generation_admissions']>0;assert d['live_rebatches']>0;assert d['rows_refilled']>0
    asyncio.run(go())

def test_decode_priority_interleaves_prefill_chunks():
    async def go():
        c=cfg();m=Model(.001);e=QwenHybridContinuousEngine(m,Tok(),c,'cpu');await e.start();q1=await e.submit(Request('a',(1,2),SamplingParams(max_new_tokens=12)))
        for _ in range(200):
            if e.diagnostics()['active_batches']:break
            await asyncio.sleep(.001)
        mark=len(m.calls);q2=await e.submit(Request('b',(1,2,3,4,5,6,7,8),SamplingParams(max_new_tokens=2)));await asyncio.gather(drain(q1),drain(q2));calls=m.calls[mark:];await e.close();pref=[i for i,x in enumerate(calls) if x[1]==2];assert len(pref)>=4;assert any(calls[i][1]==1 for i in range(pref[0]+1,pref[1]))
    asyncio.run(go())

def test_threadsafe_queue_delivery_counter():
    async def go():
        e=QwenHybridContinuousEngine(Model(),Tok(),cfg(),'cpu');await e.start();q=await e.submit(Request('a',(1,2),SamplingParams(max_new_tokens=3)));vals=await drain(q);d=e.diagnostics();await e.close();assert len(vals)==3;assert d['threadsafe_queue_emits']>=4
    asyncio.run(go())

def test_cancellation_compacts():
    async def go():
        e=QwenHybridContinuousEngine(Model(.002),Tok(),cfg(),'cpu');await e.start();r=Request('a',(1,2),SamplingParams(max_new_tokens=50));q=await e.submit(r)
        await asyncio.wait_for(q.get(),3);assert e.cancel('a')
        await asyncio.sleep(.1);d=e.diagnostics();await e.close();assert d['cancellations']>=1;assert d['rows_compacted']>=1
    asyncio.run(go())

def test_worker_failure_makes_registry_unready():
    async def go():
        c=cfg();m=Model(fail_after=1);e=QwenHybridContinuousEngine(m,Tok(),c,'cpu');s=ModelService('m',Tok(),e);reg=ModelServiceRegistry();reg.register(s);await reg.start_all();q=await e.submit(Request('x',(1,2,3,4),SamplingParams(max_new_tokens=3)));x=await asyncio.wait_for(q.get(),3);assert isinstance(x,BaseException)
        for _ in range(100):
            if not reg.ready():break
            await asyncio.sleep(.001)
        assert not reg.ready() and e.fatal_error;await reg.close_all()
    asyncio.run(go())

def test_prefix_snapshots_every_aligned_boundary():
    async def go():
        c=cfg();c.native.prefix_state_cache=True;c.native.prefix_cache_min_tokens=2;c.native.prefix_cache_block_tokens=2;c.native.prefix_state_cache_max_entries=10;e=QwenHybridContinuousEngine(Model(),Tok(),c,'cpu');await e.start();q=await e.submit(Request('a',(1,2,3,4,5,6),SamplingParams(max_new_tokens=2)));await drain(q);d=e.diagnostics();await e.close();assert d['prefix_block_snapshots']>=3;assert d['prefix_state_cache']['entries']>=3
    asyncio.run(go())

def test_graph_bucket_pads_to_four():
    c=cfg();c.native.explicit_cuda_graph=True;c.native.cuda_graph_batch_sizes=(1,2,4,8);m=Model();e=QwenHybridContinuousEngine(m,Tok(),c,'cpu');e._loop_ref=asyncio.new_event_loop();rs=[Request(str(i),(1,2),SamplingParams(max_new_tokens=3)) for i in range(3)];qs=[asyncio.Queue() for _ in rs];job=e._prepare_prefill(list(zip(rs,qs)));state=e._advance_prefill(job,10);e._decode(state);assert m.calls[-1][0]==4;assert e.diagnostics()['cuda_graph']['bucket_padded_rows_accounted']>=1;e._loop_ref.close();e._gpu_executor.shutdown(wait=False,cancel_futures=True)

def test_static_graph_lane_keeps_finished_rows_until_batch_finishes():
    c=cfg();c.native.explicit_cuda_graph=True;c.native.graph_static_cache=True
    e=QwenHybridContinuousEngine(Model(),Tok(),c,'cpu');e._loop_ref=asyncio.new_event_loop()
    r1=Request('a',(1,2),SamplingParams(max_new_tokens=1));r2=Request('b',(1,2),SamplingParams(max_new_tokens=3));q1=asyncio.Queue();q2=asyncio.Queue()
    # Manually construct a dynamic fake cache; this test targets static-lane row policy.
    job=e._prepare_prefill([(r1,q1),(r2,q2)]);job.past=Cache(2,2);state=e._advance_prefill(job,10)
    assert state is not None and len(state.requests)==2 and state.requests[0].state==RequestState.FINISHED
    e._compact(state);assert len(state.requests)==2
    e._loop_ref.close();e._gpu_executor.shutdown(wait=False,cancel_futures=True)

def test_static_cache_factory_uses_fixed_capacity(monkeypatch):
    import sys
    class FakeStaticCache:
        def __init__(self,config,max_cache_len,**kw):self.config=config;self.max_cache_len=max_cache_len
        def get_max_length(self):return self.max_cache_len
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(StaticCache=FakeStaticCache))
    c=cfg();c.native.explicit_cuda_graph=True;c.native.graph_static_cache=True;c.native.graph_cache_max_tokens=256;c.native.kv_padding_interval_size=128
    e=QwenHybridContinuousEngine(Model(),Tok(),c,'cpu');cache=e._make_static_cache(70)
    assert cache is not None and cache.get_max_length()==256 and e._graph_static_cache_uses==1
    e._gpu_executor.shutdown(wait=False,cancel_futures=True)

def test_static_graph_prefill_preallocates_fixed_attention_mask(monkeypatch):
    import sys
    class FakeStaticCache:
        def __init__(self,config,max_cache_len,**kw):self.max_cache_len=max_cache_len
        def get_max_length(self):return self.max_cache_len
    monkeypatch.setitem(sys.modules,'transformers',SimpleNamespace(StaticCache=FakeStaticCache))
    c=cfg();c.native.explicit_cuda_graph=True;c.native.graph_static_cache=True;c.native.graph_cache_max_tokens=256;c.native.kv_padding_interval_size=128
    e=QwenHybridContinuousEngine(Model(),Tok(),c,'cpu');r=Request('x',(1,2,3),SamplingParams(max_new_tokens=8));job=e._prepare_prefill([(r,asyncio.Queue())])
    assert job.attention_mask.shape==(1,256)
    assert job.attention_mask[0,:3].sum().item()==3
    assert job.attention_mask[0,3:].sum().item()==0
    e._gpu_executor.shutdown(wait=False,cancel_futures=True)
