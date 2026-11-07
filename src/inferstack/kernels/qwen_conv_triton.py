from __future__ import annotations
from importlib.util import find_spec
_STATS={'patched':False,'calls':0,'triton_calls':0,'fallback_calls':0,'patch_error':None}
_ORIGINAL=None
try:
    import triton, triton.language as tl
    _HAS_TRITON=True
    @triton.jit
    def _conv_update(h,s,w,bias,out,C:tl.constexpr,K:tl.constexpr,HAS_BIAS:tl.constexpr,SILU:tl.constexpr,BLOCK:tl.constexpr):
        row=tl.program_id(0);tile=tl.program_id(1);offs=tile*BLOCK+tl.arange(0,BLOCK);mask=offs<C
        # one program handles channels for a batch row, looping the tiny convolution K
        x=tl.load(h+row*C+offs,mask=mask,other=0.0).to(tl.float32)
        acc=tl.zeros((BLOCK,),tl.float32)
        for j in range(K-1):
            old=tl.load(s+(row*C+offs)*K+(j+1),mask=mask,other=0.0).to(tl.float32)
            tl.store(s+(row*C+offs)*K+j,old,mask=mask)
            acc+=old*tl.load(w+offs*K+j,mask=mask,other=0.0).to(tl.float32)
        tl.store(s+(row*C+offs)*K+(K-1),x,mask=mask)
        acc+=x*tl.load(w+offs*K+(K-1),mask=mask,other=0.0).to(tl.float32)
        if HAS_BIAS: acc+=tl.load(bias+offs,mask=mask,other=0.0).to(tl.float32)
        if SILU: acc=acc*tl.sigmoid(acc)
        tl.store(out+row*C+offs,acc,mask=mask)
except Exception:
    triton=None;tl=None;_HAS_TRITON=False

def torch_conv_update_reference(hidden_states,conv_state,weight,bias=None,activation=None):
    import torch
    if hidden_states.ndim!=3 or hidden_states.shape[-1]!=1:raise ValueError('single-token update only')
    conv_state[..., :-1].copy_(conv_state[..., 1:].clone());conv_state[..., -1].copy_(hidden_states[...,0])
    out=(conv_state.float()*weight.float().unsqueeze(0)).sum(-1)
    if bias is not None:out=out+bias.float().unsqueeze(0)
    if (activation or '').lower() in {'silu','swish'}:out=torch.nn.functional.silu(out)
    return out.to(hidden_states.dtype).unsqueeze(-1)

def _triton(hidden_states,conv_state,weight,bias=None,activation=None):
    import torch
    if not _HAS_TRITON or not hidden_states.is_cuda:raise RuntimeError('CUDA Triton required')
    if hidden_states.ndim!=3 or hidden_states.shape[-1]!=1:raise ValueError('single-token only')
    b,c,_=hidden_states.shape;k=conv_state.shape[-1];out=torch.empty((b,c),device=hidden_states.device,dtype=hidden_states.dtype);block=256
    _conv_update[(b,triton.cdiv(c,block))](hidden_states[...,0].contiguous(),conv_state,weight.contiguous(),bias if bias is not None else hidden_states[...,0],out,C=c,K=k,HAS_BIAS=bias is not None,SILU=(activation or '').lower() in {'silu','swish'},BLOCK=block,num_warps=4)
    return out.unsqueeze(-1)

def install_qwen35_conv_kernel(enabled=True):
    global _ORIGINAL
    if not enabled:return diagnostics()
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
        if not _STATS['patched']:
            _ORIGINAL=mod.causal_conv1d_update
            def wrapper(hidden_states,conv_state,weight,bias=None,activation=None,*args,**kwargs):
                _STATS['calls']+=1
                try:
                    out=_triton(hidden_states,conv_state,weight,bias,activation);_STATS['triton_calls']+=1;return out
                except Exception:
                    _STATS['fallback_calls']+=1;return _ORIGINAL(hidden_states,conv_state,weight,bias,activation,*args,**kwargs)
            mod.causal_conv1d_update=wrapper;_STATS['patched']=True;_STATS['patch_error']=None
    except Exception as exc:_STATS['patch_error']=f'{type(exc).__name__}: {exc}'
    return diagnostics()

def diagnostics():return dict(_STATS,triton_installed=_HAS_TRITON and find_spec('triton') is not None)
