import torch
from inferstack.kernels.gdn_triton import torch_gdn_decode_reference,torch_gdn_chunk_reference
from inferstack.kernels.qwen_conv_triton import torch_conv_update_reference

def test_gdn_chunk_matches_repeated_decode():
    torch.manual_seed(0);q=torch.randn(1,4,2,3);k=torch.randn_like(q);v=torch.randn(1,4,2,5);g=-torch.rand(1,4,2);b=torch.sigmoid(torch.randn(1,4,2));a,s=torch_gdn_chunk_reference(q,k,v,g,b,None,True,True);state=None;outs=[]
    for t in range(4):o,state=torch_gdn_decode_reference(q[:,t:t+1],k[:,t:t+1],v[:,t:t+1],g[:,t:t+1],b[:,t:t+1],state,True,True);outs.append(o)
    assert torch.allclose(a,torch.cat(outs,1),atol=1e-6);assert torch.allclose(s,state,atol=1e-6)
def test_conv_reference_shift():
    h=torch.tensor([[[4.],[8.]]]);s=torch.tensor([[[1.,2.,3.],[10.,20.,30.]]]);w=torch.tensor([[1.,2.,3.],[.5,.25,.125]]);out=torch_conv_update_reference(h,s,w,None,None);assert torch.equal(s[0,0],torch.tensor([2.,3.,4.]));assert out.shape==(1,2,1)
