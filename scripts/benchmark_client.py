#!/usr/bin/env python
from __future__ import annotations
import argparse,asyncio,json,math,statistics,time,subprocess,threading
from pathlib import Path
import httpx

def pct(xs,p):
    if not xs:return None
    ys=sorted(xs);x=(len(ys)-1)*p;lo=int(x);hi=min(lo+1,len(ys)-1);f=x-lo;return ys[lo]*(1-f)+ys[hi]*f

def stats(xs):return {'mean':statistics.mean(xs) if xs else None,'p50':pct(xs,.5),'p95':pct(xs,.95),'p99':pct(xs,.99),'min':min(xs) if xs else None,'max':max(xs) if xs else None}
class GPUMonitor:
    def __init__(self):self.stop=False;self.samples=[];self.t=None
    def start(self):self.t=threading.Thread(target=self.run,daemon=True);self.t.start()
    def run(self):
        while not self.stop:
            try:
                s=subprocess.check_output(['nvidia-smi','--query-gpu=memory.used,utilization.gpu,power.draw','--format=csv,noheader,nounits'],text=True,stderr=subprocess.DEVNULL).splitlines()[0];m,u,p=[float(x.strip()) for x in s.split(',')];self.samples.append((m,u,p))
            except Exception:pass
            time.sleep(.2)
    def close(self):self.stop=True;self.t.join(timeout=1) if self.t else None
    def summary(self):
        if not self.samples:return {'samples':0}
        return {'samples':len(self.samples),'peak_memory_mb':max(x[0] for x in self.samples),'mean_memory_mb':statistics.mean(x[0] for x in self.samples),'peak_utilization_pct':max(x[1] for x in self.samples),'mean_utilization_pct':statistics.mean(x[1] for x in self.samples),'peak_power_w':max(x[2] for x in self.samples),'mean_power_w':statistics.mean(x[2] for x in self.samples)}
async def one(client,base,model,prompt,max_tokens,temperature,top_p,timeout):
    t0=time.perf_counter();first=None;chunks=[];text='';err=None
    try:
        async with client.stream('POST',base+'/v1/completions',json={'model':model,'prompt':prompt,'max_tokens':max_tokens,'temperature':temperature,'top_p':top_p,'stream':True},timeout=timeout) as r:
            r.raise_for_status()
            async for line in r.aiter_lines():
                if not line.startswith('data: '):continue
                payload=line[6:]
                if payload=='[DONE]':break
                now=time.perf_counter()
                if first is None:first=now
                try:
                    d=json.loads(payload)
                    if 'error' in d:raise RuntimeError(d['error'])
                    piece=(d.get('choices') or [{}])[0].get('text','');text+=piece;chunks.append(now)
                except json.JSONDecodeError:pass
    except Exception as exc:err=f'{type(exc).__name__}: {exc}'
    end=time.perf_counter();n=len(chunks);ttft=(first-t0)*1000 if first else None;e2e=(end-t0)*1000;tpot=((end-first)*1000/(n-1)) if first and n>1 else None;gaps=[(chunks[i]-chunks[i-1])*1000 for i in range(1,n)]
    return {'success':err is None,'e2e_ms':e2e,'ttft_ms':ttft,'tpot_ms':tpot,'output_tokens':n,'chunk_gaps_ms':gaps,'error':err,'output_text':text}
async def run(a):
    mon=GPUMonitor();mon.start()
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=max(100,a.concurrency*2))) as c:
        for _ in range(a.warmup):await one(c,a.base_url,a.model,a.prompt,a.max_tokens,a.temperature,a.top_p,a.timeout)
        sem=asyncio.Semaphore(a.concurrency)
        async def task(i):
            async with sem:return await one(c,a.base_url,a.model,a.prompt+f'\nRequest id {i}.',a.max_tokens,a.temperature,a.top_p,a.timeout)
        t=time.perf_counter();rs=await asyncio.gather(*(task(i) for i in range(a.requests)));dur=time.perf_counter()-t
    mon.close();ok=[x for x in rs if x['success']];e=[x['e2e_ms'] for x in ok];tt=[x['ttft_ms'] for x in ok if x['ttft_ms'] is not None];tp=[x['tpot_ms'] for x in ok if x['tpot_ms'] is not None];g=[z for x in ok for z in x['chunk_gaps_ms']];tokens=sum(x['output_tokens'] for x in ok)
    return {'config':vars(a),'summary':{'sent_requests':len(rs),'completed_requests':len(ok),'failed_requests':len(rs)-len(ok),'error_rate':(len(rs)-len(ok))/len(rs) if rs else 0,'duration_s':dur,'request_throughput_rps':len(ok)/dur if dur else 0,'output_token_throughput_tps':tokens/dur if dur else 0,'total_output_tokens':tokens,'e2e_ms':stats(e),'ttft_ms':stats(tt),'tpot_ms':stats(tp),'stream_chunk_gap_ms':stats(g)},'gpu':mon.summary(),'requests':rs}
def main():
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://127.0.0.1:8000');p.add_argument('--model',default='factorylm-2b');p.add_argument('--prompt',default='Explain paged KV cache in one paragraph.');p.add_argument('--requests',type=int,default=5);p.add_argument('--concurrency',type=int,default=1);p.add_argument('--max-tokens',type=int,default=64);p.add_argument('--warmup',type=int,default=1);p.add_argument('--temperature',type=float,default=0.0);p.add_argument('--top-p',type=float,default=1.0);p.add_argument('--timeout',type=float,default=120);p.add_argument('--output')
    a=p.parse_args();d=asyncio.run(run(a));txt=json.dumps(d,indent=2);print(txt);Path(a.output).write_text(txt+'\n') if a.output else None
if __name__=='__main__':main()
