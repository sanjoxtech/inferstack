#!/usr/bin/env python
from __future__ import annotations
import argparse,json,urllib.request

def get(url,timeout=20):
    with urllib.request.urlopen(url,timeout=timeout) as r:return json.loads(r.read().decode())
def post(url,payload,timeout=240):
    req=urllib.request.Request(url,data=json.dumps(payload).encode(),headers={'Content-Type':'application/json'},method='POST')
    with urllib.request.urlopen(req,timeout=timeout) as r:return json.loads(r.read().decode())

def main():
    p=argparse.ArgumentParser();p.add_argument('--base-url',default='http://127.0.0.1:8000');p.add_argument('--model',default='factorylm-2b');p.add_argument('--assert-features',action='store_true');a=p.parse_args();base=a.base_url.rstrip('/')
    # Warm + force enough decode steps that every layer/fusion path executes repeatedly.
    for prompt in ('Explain why operator fusion matters for LLM decode in one paragraph.','Compare prefill and decode in one paragraph.'):
        post(base+'/v1/completions',{'model':a.model,'prompt':prompt,'max_tokens':48,'temperature':0.0,'stream':False})
    d=get(base+'/v1/native/status')['models'][a.model];f=d.get('forward_fusion',{}) or {}
    checks={
      'installed':bool(f.get('installed')),
      'mlp_modules':int(f.get('mlp_modules',0) or 0)>0,
      'gdn_modules':int(f.get('gdn_modules',0) or 0)>0,
      'attention_modules':int(f.get('attention_modules',0) or 0)>0,
      'rmsnorm_calls':int(f.get('rmsnorm_calls',0) or 0)>0,
      'gated_rmsnorm_calls':int(f.get('gated_rmsnorm_calls',0) or 0)>0,
      'add_rmsnorm_calls':int(f.get('add_rmsnorm_calls',0) or 0)>0,
      'mlp_fused_projection_calls':int(f.get('mlp_fused_projection_calls',0) or 0)>0,
      'gdn_fused_projection_calls':int(f.get('gdn_fused_projection_calls',0) or 0)>0,
      'grouped_gdn_decode_calls':int(f.get('grouped_gdn_decode_calls',0) or 0)>0,
      'attention_fused_projection_calls':int(f.get('attention_fused_projection_calls',0) or 0)>0,
      'healthy':bool(d.get('healthy')) and not d.get('fatal_error'),
    }
    out={'version':'1.6.0','pass':all(checks.values()),'checks':checks,'forward_fusion':f,'gdn_kernel':d.get('gdn_kernel'),'decode_profile':d.get('decode_profile')}
    print(json.dumps(out,indent=2));raise SystemExit(0 if (out['pass'] or not a.assert_features) else 2)
if __name__=='__main__':main()
