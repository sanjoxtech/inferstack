from __future__ import annotations
from importlib.util import find_spec
_STATS={'calls':0,'triton_calls':0,'fallback_calls':0}
try:
    import triton, triton.language as tl
    _HAS_TRITON=True
    @triton.jit
    def _argmax_stage1(x,vals,idxs,V:tl.constexpr,N:tl.constexpr,CHUNK:tl.constexpr):
        row=tl.program_id(0);chunk=tl.program_id(1);offs=chunk*CHUNK+tl.arange(0,CHUNK);mask=offs<V
        z=tl.load(x+row*V+offs,mask=mask,other=-float('inf')).to(tl.float32);m=tl.max(z,axis=0);j=tl.argmax(z,axis=0);tl.store(vals+row*N+chunk,m);tl.store(idxs+row*N+chunk,chunk*CHUNK+j)
    @triton.jit
    def _argmax_stage2(vals,idxs,out,N:tl.constexpr,BLOCK:tl.constexpr):
        row=tl.program_id(0);offs=tl.arange(0,BLOCK);mask=offs<N;z=tl.load(vals+row*N+offs,mask=mask,other=-float('inf'));j=tl.argmax(z,axis=0);idx=tl.load(idxs+row*N+j);tl.store(out+row,idx)
except Exception:
    triton=None;tl=None;_HAS_TRITON=False

def greedy_argmax(logits):
    import torch
    _STATS['calls']+=1
    if _HAS_TRITON and getattr(logits,'is_cuda',False) and logits.ndim==2:
        try:
            b,v=logits.shape;chunk=1024;n=triton.cdiv(v,chunk);vals=torch.empty((b,n),device=logits.device,dtype=torch.float32);idxs=torch.empty((b,n),device=logits.device,dtype=torch.int64);out=torch.empty((b,),device=logits.device,dtype=torch.int64)
            _argmax_stage1[(b,n)](logits,vals,idxs,V=v,N=n,CHUNK=chunk,num_warps=4);block=triton.next_power_of_2(n);_argmax_stage2[(b,)](vals,idxs,out,N=n,BLOCK=block,num_warps=4);_STATS['triton_calls']+=1;return out
        except Exception:_STATS['fallback_calls']+=1
    else:_STATS['fallback_calls']+=1
    return torch.argmax(logits,dim=-1)

def diagnostics():return dict(_STATS,triton_installed=_HAS_TRITON and find_spec('triton') is not None,algorithm='two_stage_tiled_argmax',chunk=1024)
