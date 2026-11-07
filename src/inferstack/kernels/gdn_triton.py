from __future__ import annotations
from importlib.util import find_spec
_STATS={'patched':False,'calls':0,'triton_calls':0,'fallback_calls':0,'chunk_calls':0,'chunk_triton_calls':0,'chunk_fallback_calls':0,'patch_error':None}
_ORIGINAL=None;_ORIGINAL_CHUNK=None
try:
    import triton,triton.language as tl
    _HAS_TRITON=True
    @triton.jit
    def _gdn_decode(q,k,v,g,beta,state,out,K:tl.constexpr,V:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
        bh=tl.program_id(0);vb=tl.program_id(1);vo=vb*BV+tl.arange(0,BV);vm=vo<V
        decay=tl.exp(tl.load(g+bh));bt=tl.load(beta+bh);mem=tl.zeros((BV,),tl.float32)
        for k0 in range(0,K,BK):
            ko=k0+tl.arange(0,BK);km=ko<K;kv=tl.load(k+bh*K+ko,mask=km,other=0.).to(tl.float32);so=bh*K*V+ko[:,None]*V+vo[None,:];m=km[:,None]&vm[None,:]
            sv=tl.load(state+so,mask=m,other=0.).to(tl.float32)*decay;tl.store(state+so,sv,mask=m);mem+=tl.sum(sv*kv[:,None],axis=0)
        vv=tl.load(v+bh*V+vo,mask=vm,other=0.).to(tl.float32);delta=(vv-mem)*bt;acc=tl.zeros((BV,),tl.float32)
        for k0 in range(0,K,BK):
            ko=k0+tl.arange(0,BK);km=ko<K;kv=tl.load(k+bh*K+ko,mask=km,other=0.).to(tl.float32);qv=tl.load(q+bh*K+ko,mask=km,other=0.).to(tl.float32);so=bh*K*V+ko[:,None]*V+vo[None,:];m=km[:,None]&vm[None,:]
            sv=tl.load(state+so,mask=m,other=0.)+kv[:,None]*delta[None,:];tl.store(state+so,sv,mask=m);acc+=tl.sum(sv*qv[:,None],axis=0)
        tl.store(out+bh*V+vo,acc,mask=vm)
except Exception:
    triton=None;tl=None;_HAS_TRITON=False

def _l2(x):
    import torch
    return x*torch.rsqrt((x*x).sum(-1,keepdim=True)+1e-6)

def torch_gdn_decode_reference(query,key,value,g,beta,initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=False,**kwargs):
    import torch
    if query.shape[1]!=1:raise ValueError('T=1 only')
    dtype=query.dtype;q,k,v,b,d=[x.transpose(1,2).contiguous().float() for x in (query,key,value,beta,g)]
    if use_qk_l2norm_in_kernel:q=_l2(q);k=_l2(k)
    q=q/(q.shape[-1]**.5);B,H,_,K=k.shape;V=v.shape[-1];state=torch.zeros((B,H,K,V),device=q.device) if initial_state is None else initial_state.float().clone()
    state*=d[:,:,0].exp()[...,None,None];kk=k[:,:,0];vv=v[:,:,0];mem=(state*kk.unsqueeze(-1)).sum(-2);delta=(vv-mem)*b[:,:,0].unsqueeze(-1);state+=kk.unsqueeze(-1)*delta.unsqueeze(-2);out=(state*q[:,:,0].unsqueeze(-1)).sum(-2).unsqueeze(2).transpose(1,2).contiguous().to(dtype)
    return out,(state if output_final_state else None)

def torch_gdn_chunk_reference(query,key,value,g,beta,initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=False,**kwargs):
    import torch
    state=initial_state;outs=[]
    for t in range(query.shape[1]):
        o,state=torch_gdn_decode_reference(query[:,t:t+1],key[:,t:t+1],value[:,t:t+1],g[:,t:t+1],beta[:,t:t+1],state,True,use_qk_l2norm_in_kernel);outs.append(o)
    return torch.cat(outs,1),(state if output_final_state else None)

def _triton_decode(query,key,value,g,beta,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=False,**kwargs):
    import torch
    if not _HAS_TRITON or not query.is_cuda or query.shape[1]!=1:raise RuntimeError('CUDA T=1 required')
    dtype=query.dtype;q,k,v,b,d=[x.transpose(1,2).contiguous().float() for x in (query,key,value,beta,g)]
    if use_qk_l2norm_in_kernel:q=_l2(q);k=_l2(k)
    q=q/(q.shape[-1]**.5);B,H,_,K=k.shape;V=v.shape[-1];state=torch.zeros((B,H,K,V),device=q.device) if initial_state is None else initial_state.float().contiguous();out=torch.empty((B,H,V),device=q.device,dtype=torch.float32)
    _gdn_decode[(B*H,triton.cdiv(V,32))](q[:,:,0].contiguous(),k[:,:,0].contiguous(),v[:,:,0].contiguous(),d[:,:,0].contiguous(),b[:,:,0].contiguous(),state,out,K=K,V=V,BK=32,BV=32)
    return out.unsqueeze(2).transpose(1,2).contiguous().to(dtype),(state if output_final_state else None)

def install_qwen35_gdn_kernel(enabled=True,patch_chunk=True):
    global _ORIGINAL,_ORIGINAL_CHUNK
    if not enabled:return diagnostics()
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
        if not _STATS['patched']:
            _ORIGINAL=mod.torch_recurrent_gated_delta_rule
            def recurrent(*args,**kwargs):
                _STATS['calls']+=1
                try:o=_triton_decode(*args,**kwargs);_STATS['triton_calls']+=1;return o
                except Exception:_STATS['fallback_calls']+=1;return _ORIGINAL(*args,**kwargs)
            mod.torch_recurrent_gated_delta_rule=recurrent
            # Short-chunk first-party reference path is intentionally correctness-first.
            # A100 acceptance measures whether it is worth replacing with a fused chunk kernel.
            if patch_chunk and hasattr(mod,'torch_chunk_gated_delta_rule'):
                _ORIGINAL_CHUNK=mod.torch_chunk_gated_delta_rule
                def chunk(*args,**kwargs):
                    _STATS['chunk_calls']+=1
                    q=args[0] if args else kwargs.get('query')
                    try:
                        if getattr(q,'is_cuda',False) and 1 < q.shape[1] <= 32:
                            o=torch_gdn_chunk_reference(*args,**kwargs);_STATS['chunk_triton_calls']+=0;_STATS['chunk_fallback_calls']+=1;return o
                    except Exception:pass
                    _STATS['chunk_fallback_calls']+=1;return _ORIGINAL_CHUNK(*args,**kwargs)
                mod.torch_chunk_gated_delta_rule=chunk
            _STATS['patched']=True;_STATS['patch_error']=None
    except Exception as exc:_STATS['patch_error']=f'{type(exc).__name__}: {exc}'
    return diagnostics()

def diagnostics():return dict(_STATS,triton_installed=_HAS_TRITON and find_spec('triton') is not None,decode_kernel='triton',chunk_kernel='upstream/reference')

# v1.6 grouped-head decode path: avoid materializing repeat_interleave(query/key)
# when Qwen has more value heads than key heads. Q/K L2 normalization is fused into
# the recurrence kernel so the hot path does not allocate normalized copies.
if _HAS_TRITON:
    @triton.jit
    def _gdn_decode_grouped(q,k,v,g,beta,state,out,Hk:tl.constexpr,Hv:tl.constexpr,K:tl.constexpr,V:tl.constexpr,GROUP:tl.constexpr,QSCALE:tl.constexpr,BK:tl.constexpr,BV:tl.constexpr):
        bhv=tl.program_id(0);vb=tl.program_id(1)
        bidx=bhv//Hv;hv=bhv-bidx*Hv;hk=hv//GROUP;bhk=bidx*Hk+hk
        vo=vb*BV+tl.arange(0,BV);vm=vo<V
        # Compute q/k L2 norms. This repeats per V tile but avoids separate tensors and
        # launches; A100 profiling decides whether a future two-stage variant is better.
        qss=tl.zeros((1,),tl.float32);kss=tl.zeros((1,),tl.float32)
        for k0 in range(0,K,BK):
            ko=k0+tl.arange(0,BK);km=ko<K
            qraw=tl.load(q+bhk*K+ko,mask=km,other=0.).to(tl.float32)
            kraw=tl.load(k+bhk*K+ko,mask=km,other=0.).to(tl.float32)
            qss+=tl.sum(qraw*qraw,axis=0);kss+=tl.sum(kraw*kraw,axis=0)
        iq=tl.rsqrt(qss+1e-6)*QSCALE;ik=tl.rsqrt(kss+1e-6)
        decay=tl.exp(tl.load(g+bhv));bt=tl.load(beta+bhv).to(tl.float32);mem=tl.zeros((BV,),tl.float32)
        for k0 in range(0,K,BK):
            ko=k0+tl.arange(0,BK);km=ko<K
            kv=tl.load(k+bhk*K+ko,mask=km,other=0.).to(tl.float32)*ik
            so=bhv*K*V+ko[:,None]*V+vo[None,:];m=km[:,None]&vm[None,:]
            sv=tl.load(state+so,mask=m,other=0.).to(tl.float32)*decay;tl.store(state+so,sv,mask=m);mem+=tl.sum(sv*kv[:,None],axis=0)
        vv=tl.load(v+bhv*V+vo,mask=vm,other=0.).to(tl.float32);delta=(vv-mem)*bt;acc=tl.zeros((BV,),tl.float32)
        for k0 in range(0,K,BK):
            ko=k0+tl.arange(0,BK);km=ko<K
            kv=tl.load(k+bhk*K+ko,mask=km,other=0.).to(tl.float32)*ik
            qv=tl.load(q+bhk*K+ko,mask=km,other=0.).to(tl.float32)*iq
            so=bhv*K*V+ko[:,None]*V+vo[None,:];m=km[:,None]&vm[None,:]
            sv=tl.load(state+so,mask=m,other=0.)+kv[:,None]*delta[None,:];tl.store(state+so,sv,mask=m);acc+=tl.sum(sv*qv[:,None],axis=0)
        tl.store(out+bhv*V+vo,acc,mask=vm)


def inferstack_grouped_gdn_decode(query,key,value,g,beta,initial_state=None,output_final_state=False,use_qk_l2norm_in_kernel=True,**kwargs):
    import torch
    if not _HAS_TRITON or not query.is_cuda or query.shape[1]!=1:
        raise RuntimeError('grouped GDN requires CUDA T=1')
    if not use_qk_l2norm_in_kernel:
        raise RuntimeError('v1.6 grouped kernel expects fused Q/K l2 normalization')
    dtype=query.dtype
    # [B,T,H,D] -> contiguous [B,H,D] without repeat_interleave.
    q=query[:,0].contiguous();k=key[:,0].contiguous();v=value[:,0].contiguous()
    gg=g[:,0].contiguous().float();bb=beta[:,0].contiguous()
    B,Hk,K=q.shape;Hv=value.shape[2];V=value.shape[-1]
    if Hv % Hk:
        raise RuntimeError(f'value heads {Hv} not divisible by key heads {Hk}')
    group=Hv//Hk
    state=torch.zeros((B,Hv,K,V),device=q.device,dtype=torch.float32) if initial_state is None else initial_state.float().contiguous()
    out=torch.empty((B,Hv,V),device=q.device,dtype=torch.float32)
    _gdn_decode_grouped[(B*Hv,triton.cdiv(V,32))](q,k,v,gg,bb,state,out,Hk=Hk,Hv=Hv,K=K,V=V,GROUP=group,QSCALE=1.0/(K**0.5),BK=32,BV=32)
    result=out.reshape(B,1,Hv,V).to(dtype)
    return result,(state if output_final_state else None)

def grouped_gdn_decode_reference(query,key,value,g,beta,initial_state=None,output_final_state=True,use_qk_l2norm_in_kernel=True,**kwargs):
    """Correctness oracle for the v1.6 grouped-head kernel."""
    if value.shape[2] % key.shape[2]:
        raise ValueError('value heads must be divisible by key heads')
    rep=value.shape[2]//key.shape[2]
    q=query.repeat_interleave(rep,dim=2) if rep>1 else query
    k=key.repeat_interleave(rep,dim=2) if rep>1 else key
    return torch_gdn_decode_reference(q,k,value,g,beta,initial_state,output_final_state,use_qk_l2norm_in_kernel,**kwargs)
