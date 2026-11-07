from types import SimpleNamespace
import torch
from inferstack.runtime.hybrid_prefix_cache import *
from inferstack.runtime.sampler import FastSampler
from inferstack.core.request import Request,SamplingParams
class L:
    def __init__(self,b,t):
        self.keys=torch.arange(b*t*2,dtype=torch.float32).reshape(b,1,t,2);self.values=self.keys+100;self.conv_states={0:torch.ones(b,2,3)};self.recurrent_states={0:torch.ones(b,2,2,2)}
class C:
    def __init__(self,b,t):self.layers=[L(b,t)]

def test_cow_clones_mutable_but_shares_kv():
    c=C(1,4);f=fork_cache_cow(c);assert f.layers[0].keys.data_ptr()==c.layers[0].keys.data_ptr();assert f.layers[0].conv_states[0].data_ptr()!=c.layers[0].conv_states[0].data_ptr()

def test_select_rows():
    c=C(3,2);x=select_cache_rows(c,torch.tensor([2,0]));assert x.layers[0].keys.shape[0]==2;assert x.layers[0].recurrent_states[0].shape[0]==2

def test_merge_rows_pads_kv():
    a=C(2,2);b=C(1,4);m=merge_cache_rows([a,b],4);assert m.layers[0].keys.shape==(3,1,4,2);assert m.layers[0].conv_states[0].shape[0]==3

def test_prefix_lru_and_longest():
    pc=HybridPrefixStateCache(max_entries=2,min_tokens=4,block_tokens=4,max_bytes=10_000_000);c=C(1,4);assert pc.put((1,2,3,4),c,4,torch.ones(1,8));assert pc.longest((1,2,3,4,5)).tokens==(1,2,3,4);assert pc.diagnostics()['hits']==1

def test_prefix_rejects_unaligned():
    pc=HybridPrefixStateCache(max_entries=2,min_tokens=4,block_tokens=4,max_bytes=10_000_000);assert not pc.put((1,2,3,4,5),C(1,5),5,torch.ones(1,8))

def req(i,temp=0,seed=None,**kw):return Request(str(i),(1,2),SamplingParams(max_new_tokens=4,temperature=temp,seed=seed,**kw))
def test_greedy_sampler_cpu():
    s=FastSampler('cpu',True);x=torch.tensor([[1.,5.,2.],[3.,1.,4.]]);y=s.sample(x,[req(1),req(2)]);assert y.tolist()==[1,2]
def test_seeded_sampling_reproducible():
    s=FastSampler('cpu',False);x=torch.tensor([[1.,2.,3.,4.]]);r=req(1,temp=.8,seed=42,top_p=.9,top_k=3,min_p=.01);a=s.sample(x,[r],2);b=s.sample(x,[r],2);assert a.item()==b.item()
def test_repetition_penalty_path():
    s=FastSampler('cpu',False);r=req(1,temp=.7,seed=1,repetition_penalty=1.2);r.generated_token_ids=[3];y=s.sample(torch.tensor([[1.,2.,3.,4.]]),[r],1);assert y.numel()==1
