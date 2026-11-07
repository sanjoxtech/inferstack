#!/usr/bin/env python
from __future__ import annotations
import argparse, asyncio, json
from pathlib import Path


def metric(evt,name,legacy):
    value=getattr(evt,name,None)
    if value is None:value=getattr(evt,legacy,0.0)
    try:return float(value or 0.0)
    except Exception:return 0.0


def main():
    p=argparse.ArgumentParser(description='Direct PyTorch CPU/CUDA operator profile of InferStack Qwen decode')
    p.add_argument('--foundry-root',default='examples/model_foundry');p.add_argument('--model-ref',default='foundry://factorylm-2b@candidate');p.add_argument('--profile',default='configs/profiles/production-v1.6.0.toml');p.add_argument('--device',default='cuda:0');p.add_argument('--steps',type=int,default=8);p.add_argument('--warmup-steps',type=int,default=3);p.add_argument('--trace',default='results/v160-matrix/native-decode-trace.json');p.add_argument('--output',default='results/v160-matrix/native-operator-profile.json')
    a=p.parse_args()
    import torch
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    from torch.profiler import profile, ProfilerActivity
    from inferstack.config import load_profile
    from inferstack.foundry import FileSystemFoundryRepository, FoundryModelLoader
    from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine
    from inferstack.core.request import Request, SamplingParams

    cfg=load_profile(a.profile);repo=FileSystemFoundryRepository(a.foundry_root);loader=FoundryModelLoader(repo,a.device);manifest,model,tok,plan=loader.load(a.model_ref,cfg)
    eng=QwenHybridContinuousEngine(model,tok,cfg,a.device);eng._maybe_patch()
    prompt='Explain continuous batching, KV cache, and kernel fusion in one paragraph.'
    ids=tuple(tok.encode(prompt,add_special_tokens=False));req=Request('profile',ids,SamplingParams(max_new_tokens=max(64,a.steps+a.warmup_steps+4),temperature=0.0));q=asyncio.Queue();job=eng._prepare_prefill([(req,q)])
    state=None
    while state is None:
        state=eng._advance_prefill(job,1)
        if job.done and state is None:raise RuntimeError('request finished during prefill; choose a different profiling prompt')
    # Operator profiling is about the decode hot path, not EOS behavior. Keep the row
    # alive for the requested number of steps even if the model happens to emit EOS.
    state.stop_sets=[set() for _ in state.stop_sets]
    for _ in range(a.warmup_steps):eng._decode(state)
    torch.cuda.synchronize()
    acts=[ProfilerActivity.CPU,ProfilerActivity.CUDA]
    with profile(activities=acts,record_shapes=True,profile_memory=True,with_stack=False) as prof:
        for _ in range(a.steps):
            eng._decode(state)
        torch.cuda.synchronize()
    rows=[]
    for e in prof.key_averages():
        self_dev=metric(e,'self_device_time_total','self_cuda_time_total');dev=metric(e,'device_time_total','cuda_time_total')
        if self_dev<=0 and dev<=0:continue
        rows.append({'name':e.key,'count':int(e.count),'self_device_us':self_dev,'device_total_us':dev,'cpu_total_us':float(getattr(e,'cpu_time_total',0.0) or 0.0),'self_cpu_us':float(getattr(e,'self_cpu_time_total',0.0) or 0.0)})
    rows.sort(key=lambda x:x['self_device_us'],reverse=True)
    total_self=sum(x['self_device_us'] for x in rows);top=rows[:40]
    for x in top:x['self_device_pct']=100.0*x['self_device_us']/total_self if total_self else 0.0
    out={'version':'1.6.0','device':torch.cuda.get_device_name(0),'steps':a.steps,'prompt_tokens':len(ids),'total_profiled_self_device_us':total_self,'top_cuda_ops':top,'gdn_kernel':eng.diagnostics().get('gdn_kernel'),'sampler':eng.diagnostics().get('sampler'),'forward_fusion':eng.diagnostics().get('forward_fusion')}
    op=Path(a.output);op.parent.mkdir(parents=True,exist_ok=True);op.write_text(json.dumps(out,indent=2,default=str)+'\n')
    trace=Path(a.trace);trace.parent.mkdir(parents=True,exist_ok=True);prof.export_chrome_trace(str(trace))
    eng._gpu_executor.shutdown(wait=False,cancel_futures=True)
    print(json.dumps(out,indent=2,default=str));print(f'\nSaved {op}\nTrace {trace}')

if __name__=='__main__':main()
