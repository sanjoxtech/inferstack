#!/usr/bin/env python
from __future__ import annotations
import argparse, concurrent.futures, json, time, urllib.request


def get(url,timeout=20):
    with urllib.request.urlopen(url,timeout=timeout) as r:return json.loads(r.read().decode())


def post(url,payload,timeout=240):
    data=json.dumps(payload).encode();req=urllib.request.Request(url,data=data,headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())


def status(base,model):return get(base.rstrip('/')+'/v1/native/status')['models'][model]

def sub(a,b,key):return float(a.get(key,0) or 0)-float(b.get(key,0) or 0)


def main():
    p=argparse.ArgumentParser(description='Low-overhead structural decode profiler for InferStack v1.6.0 profile lane')
    p.add_argument('--base-url',default='http://127.0.0.1:8000');p.add_argument('--model',default='factorylm-2b');p.add_argument('--concurrency',type=int,default=8);p.add_argument('--requests',type=int,default=16);p.add_argument('--max-tokens',type=int,default=32)
    a=p.parse_args();base=a.base_url.rstrip('/')
    # Warm one request so lazy kernels are initialized before taking the delta.
    post(base+'/v1/completions',{'model':a.model,'prompt':'Warm up the decode path.','max_tokens':8,'temperature':0.0,'stream':False})
    before=status(base,a.model);p0=before.get('decode_profile',{}) or {};tokens0=int(before.get('tokens_emitted',0) or 0)
    payload={'model':a.model,'prompt':'Explain why kernel fusion improves LLM decode in one short paragraph.','max_tokens':a.max_tokens,'temperature':0.0,'stream':False}
    t0=time.perf_counter();errors=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.concurrency) as ex:
        futs=[ex.submit(post,base+'/v1/completions',payload) for _ in range(a.requests)]
        for f in futs:
            try:f.result()
            except Exception as exc:errors.append(f'{type(exc).__name__}: {exc}')
    elapsed=time.perf_counter()-t0
    after=status(base,a.model);p1=after.get('decode_profile',{}) or {};calls=int(sub(p1,p0,'decode_calls'));tokens=int(after.get('tokens_emitted',0) or 0)-tokens0
    totals={k:sub(p1,p0,k) for k in ('decode_wall_ms','model_gpu_ms','sampler_gpu_ms','accept_wall_ms','compact_wall_ms','syncs')}
    denom=max(calls,1)
    per={k:totals[k]/denom for k in ('decode_wall_ms','model_gpu_ms','sampler_gpu_ms','accept_wall_ms','compact_wall_ms')}
    per['residual_scheduler_ms']=max(0.0,per['decode_wall_ms']-per['model_gpu_ms']-per['sampler_gpu_ms']-per['accept_wall_ms']-per['compact_wall_ms'])
    per['model_share_pct']=100.0*per['model_gpu_ms']/per['decode_wall_ms'] if per['decode_wall_ms'] else 0.0
    out={'version':'1.6.0','healthy':after.get('healthy'),'errors':errors,'concurrency':a.concurrency,'requests':a.requests,'max_tokens':a.max_tokens,'elapsed_s':elapsed,'tokens_emitted_delta':tokens,'observed_output_tps':tokens/elapsed if elapsed else 0.0,'decode_calls_delta':calls,'totals_ms':totals,'per_decode':per,'interpretation':{'model_gpu_ms':'Qwen forward including GDN/attention/MLP/cache update','sampler_gpu_ms':'GPU sampling only','accept_wall_ms':'token GPU->CPU sync + Python request/stream bookkeeping','compact_wall_ms':'batch/cache compaction bookkeeping','residual_scheduler_ms':'remaining measured per-decode wall time'}}
    print(json.dumps(out,indent=2));raise SystemExit(0 if (not errors and after.get('healthy') and calls>0) else 2)

if __name__=='__main__':main()
