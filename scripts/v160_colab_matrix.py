#!/usr/bin/env python
from __future__ import annotations
import argparse,json,os,subprocess,sys,time,urllib.request
from pathlib import Path
from difflib import SequenceMatcher


def get(url,timeout=10):
    with urllib.request.urlopen(url,timeout=timeout) as r:return json.loads(r.read().decode())
def post(url,payload,timeout=240):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())
def wait(base,proc,log,timeout=360):
    end=time.time()+timeout
    while time.time()<end:
        if proc.poll() is not None:return False,Path(log).read_text(errors='replace')[-16000:]
        try:return True,get(base+'/ready',2)
        except Exception:time.sleep(1)
    return False,'readiness timeout'
def stop(proc):
    if proc.poll() is not None:return
    proc.terminate()
    try:proc.wait(30)
    except subprocess.TimeoutExpired:proc.kill();proc.wait(10)
def run_capture(cmd,cwd,path):
    r=subprocess.run(cmd,cwd=cwd,text=True,capture_output=True);Path(path).write_text(r.stdout+('\nSTDERR:\n'+r.stderr if r.stderr else ''));return r
def parse_jsonl(path):
    out=[]
    if not path.exists():return out
    for line in path.read_text().splitlines():
        try:out.append(json.loads(line))
        except Exception:pass
    return out
def accept(rows):
    if not rows:return {'pass':False,'reason':'no benchmark rows'}
    bad=[]
    for r in rows:
        s=r.get('summary',{})
        if int(s.get('failed_requests',0) or 0)>0 or float(s.get('error_rate',0) or 0)>0 or int(s.get('completed_requests',0) or 0)<=0:bad.append(r.get('concurrency'))
    return {'pass':not bad,'bad_concurrency':bad,'rows':len(rows)}
def sample(base,model):
    prompts=['Explain KV cache in one sentence.','What is continuous batching?','Why does kernel fusion help decode?','Contrast prefill and decode.','What does RMSNorm do?','Why merge MLP gate and up projections?']
    out=[]
    for p in prompts:
        try:out.append(post(base+'/v1/completions',{'model':model,'prompt':p,'max_tokens':32,'temperature':0.0,'stream':False})['choices'][0]['text'])
        except Exception as exc:out.append('ERROR:'+str(exc))
    return out

def main():
    p=argparse.ArgumentParser(description='InferStack v1.6.0 forward-fusion A100 ablation matrix')
    p.add_argument('--foundry-root',default='examples/model_foundry');p.add_argument('--model-ref',default='foundry://factorylm-2b@candidate');p.add_argument('--api-model',default='factorylm-2b');p.add_argument('--tokenizer',default='Qwen/Qwen3.5-2B-Base');p.add_argument('--port',type=int,default=8000);p.add_argument('--results',default='results/v160-matrix');p.add_argument('--skip-operator-profile',action='store_true')
    a=p.parse_args();root=Path(__file__).resolve().parents[1];out=root/a.results;out.mkdir(parents=True,exist_ok=True);base=f'http://127.0.0.1:{a.port}'
    summary={'version':'1.6.0','kernel_equivalence':None,'profiles':{},'output_agreement':{},'operator_profiles':{}}
    kr=run_capture([sys.executable,'scripts/v160_kernel_equivalence.py'],root,out/'kernel-equivalence.log');summary['kernel_equivalence']={'exit':kr.returncode,'pass':kr.returncode==0,'stdout':kr.stdout[-12000:]}
    lanes=[
      ('baseline','configs/profiles/production-v1.6.0-baseline.toml','1,8'),
      ('norm','configs/profiles/production-v1.6.0-norm.toml','8'),
      ('mlp','configs/profiles/production-v1.6.0-mlp.toml','8'),
      ('gdn','configs/profiles/production-v1.6.0-gdn.toml','8'),
      ('attention','configs/profiles/production-v1.6.0-attention.toml','8'),
      ('full','configs/profiles/production-v1.6.0.toml','1,4,8,16'),
    ]
    samples={}
    for label,profile,conc in lanes:
        print(f'\n=== {label} ===',flush=True);log=out/f'{label}-server.log';lf=open(log,'w')
        proc=subprocess.Popen(['inferstack','serve','--profile',profile,'--foundry-root',a.foundry_root,'--model',a.model_ref,'--device','cuda:0','--host','127.0.0.1','--port',str(a.port)],cwd=root,stdout=lf,stderr=subprocess.STDOUT,env=os.environ.copy());ok,detail=wait(base,proc,log);row={'profile':profile,'ready':ok,'ready_detail':detail}
        if not ok:summary['profiles'][label]=row;stop(proc);lf.close();continue
        try:
            samples[label]=sample(base,a.api_model);(out/f'{label}-sample.json').write_text(json.dumps(samples[label],indent=2))
            if label=='full':
                fr=run_capture([sys.executable,'scripts/v160_feature_probe.py','--base-url',base,'--model',a.api_model,'--assert-features'],root,out/'full-feature-probe.json');row['feature_probe_exit']=fr.returncode;row['feature_probe_pass']=fr.returncode==0
            bench=out/f'{label}-sweep.jsonl';br=run_capture([sys.executable,'scripts/benchmark_sweep.py','--base-url',base,'--engine','inferstack_native','--model',a.api_model,'--tokenizer',a.tokenizer,'--concurrency',conc,'--requests-per-concurrency','5','--max-tokens','64','--warmup','2','--output-jsonl',str(bench)],root,out/f'{label}-benchmark.log');rows=parse_jsonl(bench);row['benchmark_exit']=br.returncode;row['benchmark_acceptance']=accept(rows);row['result_jsonl']=str(bench)
            try:row['status']=get(base+'/v1/native/status',10)['models'][a.api_model]
            except Exception as exc:row['status_error']=str(exc)
        finally:stop(proc);lf.close()
        summary['profiles'][label]=row;(out/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')

    # Deterministic regression signal against unfused baseline.
    base_out=samples.get('baseline') or []
    try:
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(a.tokenizer,use_fast=True)
    except Exception:tok=None
    for label,vals in samples.items():
        if not base_out or not vals:continue
        exact=sum(x==y for x,y in zip(base_out,vals))/len(base_out);sims=[]
        if tok:
            for x,y in zip(base_out,vals):sims.append(SequenceMatcher(a=tok.encode(x,add_special_tokens=False),b=tok.encode(y,add_special_tokens=False),autojunk=False).ratio())
        summary['output_agreement'][label]={'exact_string_fraction':exact,'mean_token_sequence_similarity':sum(sims)/len(sims) if sims else None,'note':'Regression signal only, not semantic quality.'}

    if not a.skip_operator_profile:
        for label,profile in [('baseline','configs/profiles/production-v1.6.0-baseline.toml'),('full','configs/profiles/production-v1.6.0.toml')]:
            op=run_capture([sys.executable,'scripts/v160_operator_profiler.py','--foundry-root',a.foundry_root,'--model-ref',a.model_ref,'--profile',profile,'--device','cuda:0','--steps','8','--warmup-steps','3','--trace',str(out/f'{label}-decode-trace.json'),'--output',str(out/f'{label}-operator-profile.json')],root,out/f'{label}-operator-profile.log');summary['operator_profiles'][label]={'exit':op.returncode,'pass':op.returncode==0,'json':str(out/f'{label}-operator-profile.json')}

    # Profile full fused lane structurally (server mode) to preserve the model-vs-runtime decomposition.
    profile='configs/profiles/production-v1.6.0-profile.toml';log=out/'profile-server.log';lf=open(log,'w');proc=subprocess.Popen(['inferstack','serve','--profile',profile,'--foundry-root',a.foundry_root,'--model',a.model_ref,'--device','cuda:0','--host','127.0.0.1','--port',str(a.port)],cwd=root,stdout=lf,stderr=subprocess.STDOUT,env=os.environ.copy());ok,detail=wait(base,proc,log);prow={'ready':ok,'ready_detail':detail}
    if ok:
        try:
            rr=run_capture([sys.executable,'scripts/v160_runtime_profiler.py','--base-url',base,'--model',a.api_model,'--concurrency','8','--requests','16','--max-tokens','32'],root,out/'runtime-profile.json');prow['runtime_profile_exit']=rr.returncode;prow['runtime_profile_pass']=rr.returncode==0
        finally:stop(proc);lf.close()
    else:stop(proc);lf.close()
    summary['profiles']['profile']=prow
    required=['baseline','norm','mlp','gdn','attention','full']
    # Liveness only: requests completed without errors and features/profiles ran.
    # This is not a performance claim; see analysis.json lane verdicts (audit A3).
    summary['liveness_pass']=bool(summary['kernel_equivalence']['pass'] and all(summary['profiles'].get(x,{}).get('ready') and summary['profiles'].get(x,{}).get('benchmark_acceptance',{}).get('pass') for x in required) and summary['profiles'].get('full',{}).get('feature_probe_pass') and summary['profiles'].get('profile',{}).get('runtime_profile_pass'))
    summary['performance_claim']='none: read analysis.json verdicts (keep requires repeated runs)'
    (out/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    ar=run_capture([sys.executable,'scripts/v160_analyze_results.py','--results',a.results],root,out/'analysis.log');summary['analysis_exit']=ar.returncode;(out/'summary.json').write_text(json.dumps(summary,indent=2,default=str)+'\n')
    print(json.dumps(summary,indent=2,default=str));print('\nSaved',out/'summary.json','and',out/'analysis.json');raise SystemExit(0 if summary['liveness_pass'] else 2)
if __name__=='__main__':main()
