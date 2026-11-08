from __future__ import annotations
import importlib.util
from dataclasses import dataclass, asdict
from typing import Any

@dataclass
class RuntimePlan:
    device:str
    dtype:str
    attention_backend:str
    quantization:str
    use_cuda_graphs:bool
    use_torch_compile:bool
    reason:list[str]


def _dtype_name(name:str, device:str)->str:
    import torch
    if name!='auto': return name
    if device.startswith('cuda') and torch.cuda.is_available() and torch.cuda.is_bf16_supported(): return 'bf16'
    return 'fp16' if device.startswith('cuda') else 'fp32'

def _attention(name:str)->str:
    if name!='auto': return name
    # FlashAttention is deliberately explicit. SDPA is the reliable default and the
    # matrix can compare an upstream-kernel lane separately.
    return 'sdpa'

class FoundryModelLoader:
    def __init__(self, repo, device='cuda:0'): self.repo=repo; self.device=device
    def inspect(self, ref:str, profile=None):
        manifest=self.repo.resolve(ref)
        dtype=_dtype_name(getattr(getattr(profile,'hardware',None),'dtype','auto'), self.device)
        attention=_attention(getattr(getattr(profile,'kernels',None),'attention','auto'))
        quant=str(getattr(getattr(profile,'native',None),'quantization','none'))
        plan=RuntimePlan(self.device,dtype,attention,quant,
            bool(getattr(getattr(profile,'native',None),'explicit_cuda_graph',False)),
            bool(getattr(getattr(profile,'native',None),'compile_model',False)),
            [f'dtype={dtype}',f'attention={attention}',f'quantization={quant}'])
        return manifest, manifest.get('source',{}), plan
    def load(self, ref:str, profile):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        manifest, source, plan=self.inspect(ref, profile)
        uri=source.get('uri') or manifest['tokenizer']['uri']
        revision=source.get('revision')
        if source.get('kind','huggingface')=='huggingface':
            import re
            if not re.fullmatch(r'[0-9a-f]{40}',revision or ''):
                raise ValueError(f'Foundry manifest {ref} must pin source.revision to a 40-hex commit (got {revision!r})')
        dtype={'bf16':torch.bfloat16,'fp16':torch.float16,'fp32':torch.float32}[plan.dtype]
        kwargs=dict(source.get('load_kwargs') or {})
        kwargs.update(dict(revision=revision, trust_remote_code=bool(source.get('trust_remote_code',False)), dtype=dtype))
        if plan.attention_backend: kwargs['attn_implementation']=plan.attention_backend
        if bool(getattr(profile.native,'use_hf_kernels',False)): kwargs['use_kernels']=True
        model=AutoModelForCausalLM.from_pretrained(uri, **kwargs).to(self.device).eval()
        setattr(model,'_inferstack_text_only_load',True)
        setattr(model,'_inferstack_quantization',plan.quantization)
        if plan.quantization not in {'none','null','None',''}:
            self._quantize(model, plan.quantization, profile.native.quantization_group_size)
        tok=AutoTokenizer.from_pretrained(manifest.get('tokenizer',{}).get('uri',uri), revision=manifest.get('tokenizer',{}).get('revision',revision), use_fast=True)
        if tok.pad_token_id is None: tok.pad_token=tok.eos_token
        return manifest, model, tok, plan
    def _quantize(self, model, mode:str, group_size:int):
        import torch
        if mode=='fp8':
            cc=torch.cuda.get_device_capability(0) if torch.cuda.is_available() else (0,0)
            if cc < (8,9): raise RuntimeError(f'FP8 Native path requires compute capability >=8.9; got {cc[0]}.{cc[1]}')
        try:
            from torchao.quantization import (
                quantize_, Int8WeightOnlyConfig, Int4WeightOnlyConfig,
                Float8DynamicActivationFloat8WeightConfig,
            )
        except Exception as exc:
            raise RuntimeError('TorchAO is required for Native quantization; install .[native-quant]') from exc
        if mode=='int8':
            # LLM decode has very small M (often Bx1). The dynamic W8A8 path can select
            # small-M kernels that reject M<=16 on A100. Keep activations in BF16 and
            # quantize weights only: this is the stable TorchAO int8+bf16 inference path.
            cfg=Int8WeightOnlyConfig()
        elif mode=='int4':
            # Avoid the PLAIN Int4Tensor/MSLK dependency. TorchAO documents the
            # tile-packed HQQ path as a supported INT4 weight-only CUDA workflow.
            cfg=Int4WeightOnlyConfig(
                group_size=int(group_size),
                int4_packing_format='tile_packed_to_4d',
                int4_choose_qparams_algorithm='hqq',
            )
        elif mode=='fp8':
            cfg=Float8DynamicActivationFloat8WeightConfig()
        else:
            raise ValueError(f'unsupported quantization {mode}')
        quantize_(model,cfg,device=next(model.parameters()).device)
        setattr(model,'_inferstack_quantization_config',type(cfg).__name__)
