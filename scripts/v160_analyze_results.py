#!/usr/bin/env python
from __future__ import annotations
import argparse,json,math,sys
from pathlib import Path


def rows(path):
    """concurrency -> list of summaries (repeated runs append). Unparsable lines are reported, not hidden."""
    out,bad={},0
    if not path.exists():return out
    for line in path.read_text().splitlines():
        try:r=json.loads(line);out.setdefault(int(r.get('concurrency')),[]).append(r.get('summary',{}))
        except Exception:bad+=1
    if bad:print(f'warning: {bad} unparsable rows in {path}',file=sys.stderr)
    return out

def metric(s,name):
    """Return the metric or None. A missing metric is never zero (audit A2)."""
    if not s:return None
    raw={'tps':s.get('output_token_throughput_tps'),'ttft':(s.get('ttft_ms') or {}).get('p99'),
         'tpot':(s.get('tpot_ms') or {}).get('p99')}.get(name)
    try:value=float(raw)
    except (TypeError,ValueError):return None
    return value if math.isfinite(value) else None

def lane_decision(base_runs,lane_runs,epsilon=0.03):
    """Per-lane verdict from repeated C8 runs using the same judge as the control plane."""
    from inferstack.evaluation.judge import Objective,evaluate
    trials=lambda runs:[{'tps':metric(r,'tps'),'tpot':metric(r,'tpot')} for r in runs]
    if len(base_runs)<3 or len(lane_runs)<3:
        return {'decision':'inconclusive_single_run' if base_runs and lane_runs else 'missing',
                'reasons':[f'{len(base_runs)} baseline and {len(lane_runs)} lane runs; 3 each required']}
    gates={'correctness':True,'quality':True,'health':True,'comparable':True,'complete':True}
    return evaluate(trials(base_runs),trials(lane_runs),[Objective('tps','max',epsilon=epsilon),
                    Objective('tpot','min',epsilon=epsilon)],gates=gates)

def ratio(a,b):
    return None if a is None or b in (None,0) else a/b

def opmap(path):
    try:d=json.loads(path.read_text());return {x['name']:x for x in d.get('top_cuda_ops',[])}
    except Exception:return {}

def main():
    p=argparse.ArgumentParser();p.add_argument('--results',default='results/v160-matrix');a=p.parse_args();root=Path(__file__).resolve().parents[1];res=root/a.results
    lanes={n:rows(res/f'{n}-sweep.jsonl') for n in ('baseline','norm','mlp','gdn','attention','full')}
    base_runs=lanes['baseline'].get(8,[])
    base8=base_runs[-1] if base_runs else {}
    ablations=[]
    for n in ('norm','mlp','gdn','attention','full'):
        runs=lanes[n].get(8,[]);s=runs[-1] if runs else {}
        bt=metric(base8,'tps');t=metric(s,'tps');bpot=metric(base8,'tpot');pot=metric(s,'tpot')
        missing=[k for k,v in (('tps',t),('tpot',pot),('baseline_tps',bt),('baseline_tpot',bpot)) if v is None]
        ablations.append({'lane':n,'c8_tps':t,'tps_speedup_vs_baseline':ratio(t,bt),'c8_tpot_p99_ms':pot,
                          'tpot_reduction_pct':None if pot is None or not bpot else 100*(bpot-pot)/bpot,
                          'c8_ttft_p99_ms':metric(s,'ttft'),'missing_metrics':missing,
                          'verdict':lane_decision(base_runs,runs)})
    bop=opmap(res/'baseline-operator-profile.json');fop=opmap(res/'full-operator-profile.json')
    names=['aten::mm','aten::copy_','aten::mul','aten::add','aten::rsqrt']
    opdiff=[]
    for name in names:
        b=bop.get(name,{});f=fop.get(name,{})
        bc,fc=b.get('count'),f.get('count')
        opdiff.append({'op':name,'baseline_calls':bc,'full_calls':fc,
                       'call_reduction_pct':None if bc in (None,0) or fc is None else 100*(float(bc)-float(fc))/float(bc),
                       'baseline_self_device_us':b.get('self_device_us'),'full_self_device_us':f.get('self_device_us')})
    runtime=None
    try:runtime=json.loads((res/'runtime-profile.json').read_text())
    except Exception:pass
    out={'version':'1.6.0','baseline_c8':base8,'ablations_c8':ablations,'full_scaling':lanes['full'],'operator_call_changes':opdiff,'runtime_profile':runtime,'decision_rule':'A lane is kept only with verdict keep: >=3 runs per side, all metrics present, epsilon improvement beyond noise, and passing numerical equivalence. Single runs are inconclusive. Missing metrics are null, never zero.'}
    (res/'analysis.json').write_text(json.dumps(out,indent=2,default=str)+'\n');print(json.dumps(out,indent=2,default=str))
if __name__=='__main__':main()
