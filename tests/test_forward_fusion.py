from types import SimpleNamespace
import torch
import torch.nn as nn
import torch.nn.functional as F

from inferstack.kernels.qwen_forward_fusion import (
    qwen_rmsnorm_reference,
    qwen_gated_rmsnorm_reference,
    add_rmsnorm_reference,
    silu_mul_reference,
    beta_g_reference,
    _install_mlp,
)


def test_qwen_rmsnorm_reference_matches_formula():
    torch.manual_seed(0)
    x=torch.randn(2,3,8,dtype=torch.float32)
    w=torch.randn(8)
    eps=1e-6
    got=qwen_rmsnorm_reference(x,w,eps)
    exp=x*torch.rsqrt(x.pow(2).mean(-1,keepdim=True)+eps)*(1+w)
    assert torch.allclose(got,exp,atol=1e-6,rtol=1e-6)


def test_gated_rmsnorm_reference_matches_formula():
    torch.manual_seed(1)
    x=torch.randn(4,8);g=torch.randn(4,8);w=torch.randn(8)
    got=qwen_gated_rmsnorm_reference(x,g,w,1e-6)
    xf=x.float();exp=w.float()*(xf*torch.rsqrt(xf.pow(2).mean(-1,keepdim=True)+1e-6))*F.silu(g.float())
    assert torch.allclose(got,exp,atol=1e-6,rtol=1e-6)


def test_add_rmsnorm_reference_returns_sum_and_norm():
    x=torch.randn(2,8);r=torch.randn(2,8);w=torch.randn(8)
    n,s=add_rmsnorm_reference(x,r,w,1e-6)
    assert torch.allclose(s,x+r)
    assert torch.allclose(n,qwen_rmsnorm_reference(x+r,w,1e-6))


def test_silu_mul_reference():
    x=torch.randn(3,12);a,b=x.chunk(2,-1)
    assert torch.allclose(silu_mul_reference(x),F.silu(a)*b)


def test_beta_g_reference():
    b=torch.randn(2,4);a=torch.randn(2,4);alog=torch.randn(4);dt=torch.randn(4)
    beta,g=beta_g_reference(b,a,alog,dt)
    assert torch.allclose(beta,b.sigmoid())
    assert torch.allclose(g,-alog.float().exp()*F.softplus(a.float()+dt))


class DummyMLP(nn.Module):
    def __init__(self,h=8,i=12):
        super().__init__();self.gate_proj=nn.Linear(h,i,bias=False);self.up_proj=nn.Linear(h,i,bias=False);self.down_proj=nn.Linear(i,h,bias=False);self.act_fn=F.silu
    def forward(self,x):return self.down_proj(self.act_fn(self.gate_proj(x))*self.up_proj(x))


def test_mlp_gate_up_weight_fusion_is_numerically_equivalent_cpu():
    torch.manual_seed(2);m=DummyMLP();x=torch.randn(4,8);expected=m(x)
    cfg=SimpleNamespace(native=SimpleNamespace(fuse_mlp_gate_up=True,fuse_mlp_activation=True))
    assert _install_mlp(m,cfg)
    got=m(x)
    assert torch.allclose(got,expected,atol=1e-6,rtol=1e-5)
    assert m.gate_proj is None and m.up_proj is None
    assert m._inferstack_gate_up_weight.shape[0]==24


def test_merged_linear_projections_equal_separate_gemms():
    torch.manual_seed(3);x=torch.randn(5,8)
    w1=torch.randn(10,8);w2=torch.randn(6,8);w3=torch.randn(4,8)
    merged=F.linear(x,torch.cat([w1,w2,w3],0))
    a,b,c=merged.split([10,6,4],-1)
    assert torch.allclose(a,F.linear(x,w1),atol=1e-6)
    assert torch.allclose(b,F.linear(x,w2),atol=1e-6)
    assert torch.allclose(c,F.linear(x,w3),atol=1e-6)

from inferstack.kernels.gdn_triton import grouped_gdn_decode_reference, torch_gdn_decode_reference

def test_grouped_gdn_reference_equals_materialized_repeat():
    torch.manual_seed(4)
    B,Hk,Hv,K,V=2,2,4,3,5
    q=torch.randn(B,1,Hk,K);k=torch.randn_like(q);v=torch.randn(B,1,Hv,V)
    g=-torch.rand(B,1,Hv);beta=torch.sigmoid(torch.randn(B,1,Hv));state=torch.randn(B,Hv,K,V)
    a,s1=grouped_gdn_decode_reference(q,k,v,g,beta,state,True,True)
    rep=Hv//Hk
    b,s2=torch_gdn_decode_reference(q.repeat_interleave(rep,2),k.repeat_interleave(rep,2),v,g,beta,state,True,True)
    assert torch.allclose(a,b,atol=1e-6,rtol=1e-6)
    assert torch.allclose(s1,s2,atol=1e-6,rtol=1e-6)
