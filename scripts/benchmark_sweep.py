#!/usr/bin/env python
from __future__ import annotations
import argparse,asyncio,json
from pathlib import Path
from benchmark_client import run

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://127.0.0.1:8000');p.add_argument('--model',default='factorylm-2b');p.add_argument('--prompt',default='Explain paged KV cache in one paragraph.');p.add_argument('--concurrency',default='1,4,8,16');p.add_argument('--requests-per-concurrency',type=int,default=5);p.add_argument('--max-tokens',type=int,default=64);p.add_argument('--warmup',type=int,default=2);p.add_argument('--timeout',type=float,default=120);p.add_argument('--output-jsonl',default='results/sweep.jsonl');p.add_argument('--engine',default='inferstack_native');p.add_argument('--tokenizer',default='Qwen/Qwen3.5-2B-Base');a=p.parse_args();out=Path(a.output_jsonl);out.parent.mkdir(parents=True,exist_ok=True);out.write_text('')
    for c in [int(x) for x in a.concurrency.split(',') if x.strip()]:
        ns=argparse.Namespace(base_url=a.base_url,model=a.model,prompt=a.prompt,requests=a.requests_per_concurrency*c,concurrency=c,max_tokens=a.max_tokens,warmup=a.warmup,temperature=0.0,top_p=1.0,timeout=a.timeout,output=None)
        d=asyncio.run(run(ns));row={'engine':a.engine,'concurrency':c,'summary':d['summary'],'gpu':d['gpu']};print(json.dumps(row,indent=2));
        with out.open('a') as f:f.write(json.dumps(row)+'\n')
if __name__=='__main__':main()
