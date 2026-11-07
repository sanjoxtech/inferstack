from __future__ import annotations
from dataclasses import dataclass
import math

@dataclass
class SamplerStats:
    calls:int=0;triton_greedy_calls:int=0;greedy_fallback_calls:int=0;vectorized_stochastic_calls:int=0;rowwise_stochastic_calls:int=0

class FastSampler:
    def __init__(self,device='cuda:0',use_triton=True):self.device=device;self.use_triton=use_triton;self.stats=SamplerStats()
    def _history_penalty(self,logits,requests):
        import torch
        out=logits.float().clone()
        for i,r in enumerate(requests):
            p=float(r.sampling.repetition_penalty or 1.0)
            if p==1.0 or not r.generated_token_ids:continue
            ids=torch.tensor(sorted(set(r.generated_token_ids)),device=out.device,dtype=torch.long)
            vals=out[i,ids];out[i,ids]=torch.where(vals<0,vals*p,vals/p)
        return out
    def _homogeneous(self,requests):
        if not requests:return True
        a=requests[0].sampling
        key=lambda s:(float(s.temperature),float(s.top_p),s.top_k,float(s.min_p),float(s.repetition_penalty))
        return all(key(r.sampling)==key(a) for r in requests)
    def sample(self,logits,requests,step=0):
        import torch
        self.stats.calls+=1
        if all(float(r.sampling.temperature)<=0 for r in requests):
            if self.use_triton:
                from inferstack.kernels.sampler_triton import greedy_argmax, diagnostics
                before=diagnostics()['triton_calls'];out=greedy_argmax(logits);after=diagnostics()['triton_calls']
                if after>before:self.stats.triton_greedy_calls+=1
                else:self.stats.greedy_fallback_calls+=1
                return out
            self.stats.greedy_fallback_calls+=1;return torch.argmax(logits,dim=-1)
        if self._homogeneous(requests):
            self.stats.vectorized_stochastic_calls+=1;return self._sample_homogeneous(self._history_penalty(logits,requests),requests,step)
        self.stats.rowwise_stochastic_calls+=1
        rows=[self._sample_homogeneous(self._history_penalty(logits[i:i+1],[r]),[r],step) for i,r in enumerate(requests)]
        return torch.cat(rows)
    def _sample_homogeneous(self,logits,requests,step):
        import torch
        s=requests[0].sampling;temp=max(float(s.temperature),1e-6);x=logits.float()/temp
        min_p=float(s.min_p or 0.0)
        if min_p>0:
            probs=torch.softmax(x,dim=-1);mx=probs.max(dim=-1,keepdim=True).values;x=x.masked_fill(probs<mx*min_p,-float('inf'))
        top_k=s.top_k
        if top_k is not None and int(top_k)>0 and int(top_k)<x.shape[-1]:
            k=int(top_k);cut=torch.topk(x,k,dim=-1).values[:,-1:].clone();x=x.masked_fill(x<cut,-float('inf'))
        top_p=float(s.top_p or 1.0)
        if top_p<1.0:
            vals,idx=torch.sort(x,dim=-1,descending=True);p=torch.softmax(vals,dim=-1);cum=p.cumsum(dim=-1);remove=cum>top_p;remove[:,1:]=remove[:,:-1].clone();remove[:,0]=False;vals=vals.masked_fill(remove,-float('inf'));x=torch.full_like(x,-float('inf')).scatter(1,idx,vals)
        probs=torch.softmax(x,dim=-1)
        outs=[]
        for i,r in enumerate(requests):
            gen=None
            if r.sampling.seed is not None:
                gen=torch.Generator(device=probs.device);gen.manual_seed(int(r.sampling.seed)+int(step))
            outs.append(torch.multinomial(probs[i],1,generator=gen))
        return torch.cat(outs)
    def diagnostics(self):
        d=self.stats.__dict__.copy()
        try:
            from inferstack.kernels.sampler_triton import diagnostics
            d['kernel']=diagnostics()
        except Exception:pass
        return d
