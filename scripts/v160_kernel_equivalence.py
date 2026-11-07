#!/usr/bin/env python
from __future__ import annotations
import json, math


def maxdiff(a,b):
    import torch
    return float((a.float()-b.float()).abs().max().item())


def main():
    import torch
    if not torch.cuda.is_available():raise SystemExit('CUDA required')
    from transformers import AutoConfig
    from inferstack.kernels.qwen_forward_fusion import (
        fused_rmsnorm,qwen_rmsnorm_reference,fused_gated_rmsnorm,qwen_gated_rmsnorm_reference,
        fused_add_rmsnorm,add_rmsnorm_reference,fused_silu_mul,silu_mul_reference,
        fused_beta_g,beta_g_reference,
    )
    from inferstack.kernels.gdn_triton import inferstack_grouped_gdn_decode,grouped_gdn_decode_reference
    cfg=AutoConfig.from_pretrained('Qwen/Qwen3.5-2B-Base',trust_remote_code=False)
    text=getattr(cfg,'text_config',cfg)
    device='cuda';dtype=torch.bfloat16;torch.manual_seed(123)
    H=int(text.hidden_size);I=int(text.intermediate_size);Hk=int(text.linear_num_key_heads);Hv=int(text.linear_num_value_heads);K=int(text.linear_key_head_dim);V=int(text.linear_value_head_dim)
    B=8
    checks={}

    x=torch.randn(B,1,H,device=device,dtype=dtype);w=torch.randn(H,device=device,dtype=dtype)*.1
    ref=qwen_rmsnorm_reference(x,w,1e-6);got=fused_rmsnorm(x.contiguous(),w.contiguous(),1e-6);checks['rmsnorm']={'max_abs':maxdiff(got,ref),'pass':maxdiff(got,ref)<=2e-2}

    r=torch.randn_like(x);rn,rs=add_rmsnorm_reference(x,r,w,1e-6);gn,gs=fused_add_rmsnorm(x.contiguous(),r.contiguous(),w.contiguous(),1e-6)
    checks['add_rmsnorm']={'norm_max_abs':maxdiff(gn,rn),'sum_max_abs':maxdiff(gs,rs),'pass':maxdiff(gn,rn)<=3e-2 and maxdiff(gs,rs)<=2e-2}

    D=V;gx=torch.randn(B*Hv,D,device=device,dtype=dtype);gate=torch.randn_like(gx);gw=torch.randn(D,device=device,dtype=dtype)
    ref=qwen_gated_rmsnorm_reference(gx,gate,gw,1e-6);got=fused_gated_rmsnorm(gx.contiguous(),gate.contiguous(),gw.contiguous(),1e-6);checks['gated_rmsnorm']={'max_abs':maxdiff(got,ref),'pass':maxdiff(got,ref)<=3e-2}

    gu=torch.randn(B,1,2*I,device=device,dtype=dtype);ref=silu_mul_reference(gu);got=fused_silu_mul(gu.contiguous());checks['silu_mul']={'max_abs':maxdiff(got,ref),'pass':maxdiff(got,ref)<=3e-2}

    bb=torch.randn(B,1,Hv,device=device,dtype=dtype);aa=torch.randn_like(bb);alog=torch.randn(Hv,device=device,dtype=torch.float32);dt=torch.randn(Hv,device=device,dtype=dtype)
    rb,rg=beta_g_reference(bb,aa,alog,dt);gb,gg=fused_beta_g(bb.contiguous(),aa.contiguous(),alog.contiguous(),dt.contiguous())
    checks['beta_g']={'beta_max_abs':maxdiff(gb,rb),'g_max_abs':maxdiff(gg,rg),'pass':maxdiff(gb,rb)<=2e-2 and maxdiff(gg,rg)<=3e-2}

    q=torch.randn(B,1,Hk,K,device=device,dtype=dtype);k=torch.randn_like(q);v=torch.randn(B,1,Hv,V,device=device,dtype=dtype);decay=-torch.rand(B,1,Hv,device=device,dtype=torch.float32);beta=torch.sigmoid(torch.randn(B,1,Hv,device=device,dtype=dtype));state=torch.randn(B,Hv,K,V,device=device,dtype=torch.float32)*.01
    ro,rs=grouped_gdn_decode_reference(q,k,v,decay,beta,state,True,True);go,gs=inferstack_grouped_gdn_decode(q,k,v,decay,beta,state,True,True)
    checks['grouped_gdn']={'output_max_abs':maxdiff(go,ro),'state_max_abs':maxdiff(gs,rs),'pass':maxdiff(go,ro)<=5e-2 and maxdiff(gs,rs)<=5e-2,'shape':{'B':B,'Hk':Hk,'Hv':Hv,'K':K,'V':V}}

    out={'version':'1.6.0','device':torch.cuda.get_device_name(0),'dtype':'bf16','model_dims':{'hidden':H,'intermediate':I,'linear_key_heads':Hk,'linear_value_heads':Hv,'key_dim':K,'value_dim':V},'checks':checks,'pass':all(x['pass'] for x in checks.values())}
    print(json.dumps(out,indent=2));raise SystemExit(0 if out['pass'] else 2)
if __name__=='__main__':main()
