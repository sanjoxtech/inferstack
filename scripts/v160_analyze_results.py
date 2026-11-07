#!/usr/bin/env python
from __future__ import annotations
import argparse,json
from pathlib import Path


def rows(path):
    out={}
    if not path.exists():return out
    for line in path.read_text().splitlines():
        try:r=json.loads(line);out[int(r.get('concurrency'))]=r.get('summary',{})
        except Exception:pass
    return out

def metric(s,name):
    if name=='tps':return float(s.get('output_token_throughput_tps',0) or 0)
    if name=='ttft':return float((s.get('ttft_ms') or {}).get('p99',0) or 0)
    if name=='tpot':return float((s.get('tpot_ms') or {}).get('p99',0) or 0)
    return 0.0

def opmap(path):
    try:d=json.loads(path.read_text());return {x['name']:x for x in d.get('top_cuda_ops',[])}
    except Exception:return {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/v160-matrix');a=p.parse_args();root=Path(__file__).resolve().parents[1];res=root/a.results
    lanes={n:rows(res/f'{n}-sweep.jsonl') for n in ('baseline','norm','mlp','gdn','attention','full')}
    base8=lanes['baseline'].get(8,{})
    ablations=[]
    for n in ('norm','mlp','gdn','attention','full'):
        s=lanes[n].get(8,{})
        bt=metric(base8,'tps');t=metric(s,'tps');bpot=metric(base8,'tpot');pot=metric(s,'tpot')
        ablations.append({'lane':n,'c8_tps':t,'tps_speedup_vs_baseline':t/bt if bt else None,'c8_tpot_p99_ms':pot,'tpot_reduction_pct':100*(bpot-pot)/bpot if bpot else None,'c8_ttft_p99_ms':metric(s,'ttft')})
    bop=opmap(res/'baseline-operator-profile.json');fop=opmap(res/'full-operator-profile.json')
    names=['aten::mm','aten::copy_','aten::mul','aten::add','aten::rsqrt']
    opdiff=[]
    for name in names:
        b=bop.get(name,{});f=fop.get(name,{})
        opdiff.append({'op':name,'baseline_calls':b.get('count',0),'full_calls':f.get('count',0),'call_reduction_pct':100*(float(b.get('count',0))-float(f.get('count',0)))/float(b.get('count',1) or 1),'baseline_self_device_us':b.get('self_device_us',0),'full_self_device_us':f.get('self_device_us',0)})
    runtime=None
    try:runtime=json.loads((res/'runtime-profile.json').read_text())
    except Exception:pass
    out={'version':'1.6.0','baseline_c8':base8,'ablations_c8':ablations,'full_scaling':lanes['full'],'operator_call_changes':opdiff,'runtime_profile':runtime,'decision_rule':'Keep only fusion groups that improve C8 TPOT/TPS without unacceptable deterministic output divergence. Full fusion must beat the v1.6 baseline before it becomes default.'}
    (res/'analysis.json').write_text(json.dumps(out,indent=2,default=str)+'\n');print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
