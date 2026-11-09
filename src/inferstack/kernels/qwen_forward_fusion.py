from __future__ import annotations

import math
import types
from dataclasses import dataclass, asdict
from typing import Any

try:
    import triton
    import triton.language as tl
    try:  # precise (correctly rounded) transcendental functions; tl.exp is a fast approximation
        from triton.language.extra import libdevice as _libdevice
    except Exception:  # noqa: BLE001
        _libdevice = None
    _HAS_TRITON = True

    @triton.jit
    def _rmsnorm_kernel(x, w, y, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        xv = tl.load(x + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(xv * xv, axis=0) / n_cols
        inv = tl.rsqrt(var + eps)
        ww = tl.load(w + offs, mask=mask, other=0.0).to(tl.float32)
        out = xv * inv * (1.0 + ww)
        tl.store(y + row * n_cols + offs, out, mask=mask)

    @triton.jit
    def _gated_rmsnorm_kernel(x, gate, w, y, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        xv = tl.load(x + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        gv = tl.load(gate + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        var = tl.sum(xv * xv, axis=0) / n_cols
        inv = tl.rsqrt(var + eps)
        ww = tl.load(w + offs, mask=mask, other=0.0).to(tl.float32)
        silu = gv * tl.sigmoid(gv)
        out = xv * inv * ww * silu
        tl.store(y + row * n_cols + offs, out, mask=mask)

    @triton.jit
    def _add_rmsnorm_kernel(x, residual, w, normed, summed, n_cols: tl.constexpr, eps: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < n_cols
        xv = tl.load(x + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        rv = tl.load(residual + row * n_cols + offs, mask=mask, other=0.0).to(tl.float32)
        sv = xv + rv
        var = tl.sum(sv * sv, axis=0) / n_cols
        inv = tl.rsqrt(var + eps)
        ww = tl.load(w + offs, mask=mask, other=0.0).to(tl.float32)
        tl.store(summed + row * n_cols + offs, sv, mask=mask)
        tl.store(normed + row * n_cols + offs, sv * inv * (1.0 + ww), mask=mask)

    @triton.jit
    def _silu_mul_kernel(x, out, cols: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        block = tl.program_id(1)
        offs = block * BLOCK + tl.arange(0, BLOCK)
        mask = offs < cols
        a = tl.load(x + row * (2 * cols) + offs, mask=mask, other=0.0).to(tl.float32)
        b = tl.load(x + row * (2 * cols) + cols + offs, mask=mask, other=0.0).to(tl.float32)
        outv = (a * tl.sigmoid(a)) * b
        tl.store(out + row * cols + offs, outv, mask=mask)

    @triton.jit
    def _beta_g_kernel(b, a, alog, dt_bias, beta, g, heads: tl.constexpr, BLOCK: tl.constexpr):
        row = tl.program_id(0)
        offs = tl.arange(0, BLOCK)
        mask = offs < heads
        bv = tl.load(b + row * heads + offs, mask=mask, other=0.0).to(tl.float32)
        av = tl.load(a + row * heads + offs, mask=mask, other=0.0).to(tl.float32)
        al = tl.load(alog + offs, mask=mask, other=0.0).to(tl.float32)
        dt = tl.load(dt_bias + offs, mask=mask, other=0.0).to(tl.float32)
        # sigmoid(b), g = -exp(A_log) * softplus(a + dt_bias)
        be = tl.sigmoid(bv)
        z = av + dt
        # g is a float32 output checked against the FP32 oracle within 1 ULP, so it uses the
        # same formula as torch.nn.functional.softplus (threshold 20, log1p(exp(z))) with
        # precise libdevice exp/log1p. The earlier fast tl.exp + log(1 + exp(-|z|)) form
        # measured 5-19 ULP on an A100 (Colab run 2026-11-09).
        sp = tl.where(z > 20.0, z, _libdevice.log1p(_libdevice.exp(z)))
        gg = -_libdevice.exp(al) * sp
        tl.store(beta + row * heads + offs, be, mask=mask)
        tl.store(g + row * heads + offs, gg, mask=mask)
except Exception:
    triton = None
    tl = None
    _libdevice = None
    _HAS_TRITON = False


@dataclass
class ForwardFusionStats:
    installed: bool = False
    rmsnorm_modules: int = 0
    gated_rmsnorm_modules: int = 0
    decoder_layers: int = 0
    mlp_modules: int = 0
    gdn_modules: int = 0
    attention_modules: int = 0
    fused_weight_bytes: int = 0
    rmsnorm_calls: int = 0
    gated_rmsnorm_calls: int = 0
    add_rmsnorm_calls: int = 0
    silu_mul_calls: int = 0
    gdn_fused_projection_calls: int = 0
    attention_fused_projection_calls: int = 0
    mlp_fused_projection_calls: int = 0
    grouped_gdn_decode_calls: int = 0
    beta_g_fused_calls: int = 0
    fallbacks: int = 0
    install_error: str | None = None


STATS = ForwardFusionStats()
_ORIGINALS: dict[str, Any] = {}


def _next_pow2(x: int) -> int:
    return 1 if x <= 1 else 1 << (x - 1).bit_length()


def qwen_rmsnorm_reference(x, weight, eps: float):
    out = x.float() * __import__('torch').rsqrt(x.float().pow(2).mean(-1, keepdim=True) + eps)
    out = out * (1.0 + weight.float())
    return out.to(x.dtype)


def qwen_gated_rmsnorm_reference(x, gate, weight, eps: float):
    import torch
    xf = x.float()
    xf = xf * torch.rsqrt(xf.pow(2).mean(-1, keepdim=True) + eps)
    out = weight.float() * xf * torch.nn.functional.silu(gate.float())
    return out.to(x.dtype)


def add_rmsnorm_reference(x, residual, weight, eps: float):
    summed = x + residual
    return qwen_rmsnorm_reference(summed, weight, eps), summed


def silu_mul_reference(x):
    import torch
    a, b = x.chunk(2, dim=-1)
    return torch.nn.functional.silu(a) * b


def beta_g_reference(b, a, alog, dt_bias):
    import torch
    return b.sigmoid(), -alog.float().exp() * torch.nn.functional.softplus(a.float() + dt_bias)


def fused_rmsnorm(x, weight, eps: float):
    import torch
    if not (_HAS_TRITON and x.is_cuda and x.is_contiguous() and weight.is_contiguous()):
        STATS.fallbacks += 1
        return qwen_rmsnorm_reference(x, weight, eps)
    d = x.shape[-1]
    block = _next_pow2(d)
    if block > 65536:
        STATS.fallbacks += 1
        return qwen_rmsnorm_reference(x, weight, eps)
    y = torch.empty_like(x)
    rows = x.numel() // d
    _rmsnorm_kernel[(rows,)](x, weight, y, n_cols=d, eps=float(eps), BLOCK=block, num_warps=4 if block <= 4096 else 8)
    STATS.rmsnorm_calls += 1
    return y


def fused_gated_rmsnorm(x, gate, weight, eps: float):
    import torch
    if not (_HAS_TRITON and x.is_cuda and x.is_contiguous() and gate.is_contiguous() and weight.is_contiguous()):
        STATS.fallbacks += 1
        return qwen_gated_rmsnorm_reference(x, gate, weight, eps)
    d = x.shape[-1]
    block = _next_pow2(d)
    if block > 65536:
        STATS.fallbacks += 1
        return qwen_gated_rmsnorm_reference(x, gate, weight, eps)
    y = torch.empty_like(x)
    rows = x.numel() // d
    _gated_rmsnorm_kernel[(rows,)](x, gate, weight, y, n_cols=d, eps=float(eps), BLOCK=block, num_warps=4)
    STATS.gated_rmsnorm_calls += 1
    return y


def fused_add_rmsnorm(x, residual, weight, eps: float):
    import torch
    if not (_HAS_TRITON and x.is_cuda and residual.is_cuda and x.is_contiguous() and residual.is_contiguous() and weight.is_contiguous()):
        STATS.fallbacks += 1
        return add_rmsnorm_reference(x, residual, weight, eps)
    d = x.shape[-1]
    block = _next_pow2(d)
    if block > 65536:
        STATS.fallbacks += 1
        return add_rmsnorm_reference(x, residual, weight, eps)
    normed = torch.empty_like(x)
    summed = torch.empty_like(x)
    rows = x.numel() // d
    _add_rmsnorm_kernel[(rows,)](x, residual, weight, normed, summed, n_cols=d, eps=float(eps), BLOCK=block, num_warps=4 if block <= 4096 else 8)
    STATS.add_rmsnorm_calls += 1
    return normed, summed


def fused_silu_mul(x):
    import torch
    if x.shape[-1] % 2:
        raise ValueError('silu_mul expects an even final dimension')
    cols = x.shape[-1] // 2
    if not (_HAS_TRITON and x.is_cuda and x.is_contiguous()):
        STATS.fallbacks += 1
        return silu_mul_reference(x)
    out = torch.empty((*x.shape[:-1], cols), dtype=x.dtype, device=x.device)
    rows = x.numel() // (2 * cols)
    block = 256
    _silu_mul_kernel[(rows, triton.cdiv(cols, block))](x, out, cols=cols, BLOCK=block)
    STATS.silu_mul_calls += 1
    return out


def fused_beta_g(b, a, alog, dt_bias):
    import torch
    heads = b.shape[-1]
    if not (_HAS_TRITON and _libdevice is not None and b.is_cuda and a.is_cuda and b.is_contiguous()
            and a.is_contiguous()):
        STATS.fallbacks += 1  # never trade float32 precision for speed: use the reference path
        return beta_g_reference(b, a, alog, dt_bias)
    beta = torch.empty_like(b)
    g = torch.empty(b.shape, dtype=torch.float32, device=b.device)
    rows = b.numel() // heads
    block = _next_pow2(heads)
    _beta_g_kernel[(rows,)](b, a, alog, dt_bias, beta, g, heads=heads, BLOCK=block)
    STATS.beta_g_fused_calls += 1
    return beta, g


def _register_fused_weight(module, name: str, tensors):
    import torch
    fused = torch.cat([t.detach() for t in tensors], dim=0).contiguous()
    module.register_buffer(name, fused, persistent=False)
    STATS.fused_weight_bytes += int(fused.numel() * fused.element_size())
    return fused


def _install_rmsnorm(mod, cfg):
    if not cfg.native.fuse_rmsnorm:
        return
    cls = getattr(mod, 'Qwen3_5RMSNorm', None)
    if cls is None:
        return
    key = 'rmsnorm_forward'
    if key not in _ORIGINALS:
        _ORIGINALS[key] = cls.forward
        def forward(self, x):
            return fused_rmsnorm(x, self.weight, self.eps)
        cls.forward = forward


def _install_gated_rmsnorm(mod, cfg):
    if not cfg.native.fuse_gated_rmsnorm:
        return
    cls = getattr(mod, 'Qwen3_5RMSNormGated', None)
    if cls is None:
        return
    key = 'gated_rmsnorm_forward'
    if key not in _ORIGINALS:
        _ORIGINALS[key] = cls.forward
        def forward(self, hidden_states, gate=None):
            if gate is None:
                return _ORIGINALS[key](self, hidden_states, gate)
            return fused_gated_rmsnorm(hidden_states, gate, self.weight, self.variance_epsilon)
        cls.forward = forward


def _install_mlp(module, cfg):
    import torch.nn.functional as F
    if not cfg.native.fuse_mlp_gate_up or getattr(module, '_inferstack_mlp_fused', False):
        return False
    gate = getattr(module, 'gate_proj', None); up = getattr(module, 'up_proj', None)
    if gate is None or up is None or gate.bias is not None or up.bias is not None:
        return False
    _register_fused_weight(module, '_inferstack_gate_up_weight', [gate.weight, up.weight])
    module.gate_proj = None
    module.up_proj = None
    use_act = bool(cfg.native.fuse_mlp_activation)
    def forward(self, x):
        gu = F.linear(x, self._inferstack_gate_up_weight)
        STATS.mlp_fused_projection_calls += 1
        act = fused_silu_mul(gu) if use_act else self.act_fn(gu[..., :gu.shape[-1]//2]) * gu[..., gu.shape[-1]//2:]
        return self.down_proj(act)
    module.forward = types.MethodType(forward, module)
    module._inferstack_mlp_fused = True
    return True


def _install_gdn(module, mod, cfg):
    import torch
    import torch.nn.functional as F
    if not cfg.native.fuse_gdn_projections or getattr(module, '_inferstack_gdn_fused', False):
        return False
    qkv = getattr(module, 'in_proj_qkv', None); zproj = getattr(module, 'in_proj_z', None)
    bproj = getattr(module, 'in_proj_b', None); aproj = getattr(module, 'in_proj_a', None)
    if any(x is None for x in (qkv,zproj,bproj,aproj)):
        return False
    if any(x.bias is not None for x in (qkv,zproj,bproj,aproj)):
        return False
    _register_fused_weight(module, '_inferstack_qkvz_weight', [qkv.weight, zproj.weight])
    _register_fused_weight(module, '_inferstack_ba_weight', [bproj.weight, aproj.weight])
    module.in_proj_qkv = module.in_proj_z = module.in_proj_b = module.in_proj_a = None
    use_beta_g = bool(cfg.native.fuse_gdn_beta_g)
    grouped_decode = bool(cfg.native.grouped_gdn_decode)

    def forward(self, hidden_states, cache_params=None, attention_mask=None, **kwargs):
        hidden_states = mod.apply_mask_to_padding_states(hidden_states, attention_mask)
        batch_size, seq_len, _ = hidden_states.shape
        use_precomputed_states = cache_params is not None and cache_params.has_previous_state(self.layer_idx)
        qkvz = F.linear(hidden_states, self._inferstack_qkvz_weight)
        qkv_width = self.conv_dim
        qkv, z = qkvz.split([qkv_width, self.value_dim], dim=-1)
        mixed_qkv = qkv.transpose(1, 2)
        ba = F.linear(hidden_states, self._inferstack_ba_weight)
        b, a = ba.split([self.num_v_heads, self.num_v_heads], dim=-1)
        STATS.gdn_fused_projection_calls += 1

        if use_precomputed_states and seq_len == 1 and not cache_params.layers[self.layer_idx].record_past:
            conv_state = cache_params.layers[self.layer_idx].conv_states[0]
            mixed_qkv = mod.causal_conv1d_update(
                mixed_qkv, conv_state, self.conv1d.weight.squeeze(1), self.conv1d.bias, self.activation,
            )
        else:
            if cache_params is not None:
                mixed_qkv = cache_params.update_conv_state(mixed_qkv, self.layer_idx, conv_kernel_size=self.conv_kernel_size)
            mixed_qkv = mod.causal_conv1d_fn(
                mixed_qkv, self.conv1d.weight.squeeze(1), self.conv1d.bias, activation=self.activation, **kwargs,
            )
            if cache_params is not None:
                mixed_qkv = mixed_qkv[:, :, -seq_len:]
        mixed_qkv = mixed_qkv.transpose(1, 2)
        query, key, value = torch.split(mixed_qkv, [self.key_dim, self.key_dim, self.value_dim], dim=-1)
        query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        beta, g = fused_beta_g(b, a, self.A_log, self.dt_bias) if use_beta_g else beta_g_reference(b, a, self.A_log, self.dt_bias)
        recurrent_state = cache_params.layers[self.layer_idx].recurrent_states[0] if use_precomputed_states else None

        # Decode can avoid materializing repeat_interleave(q/k): our grouped kernel maps
        # each value head to its key/query head directly. Prefill keeps the proven upstream path.
        if use_precomputed_states and seq_len == 1 and grouped_decode:
            try:
                from inferstack.kernels.gdn_triton import inferstack_grouped_gdn_decode
                core_attn_out, last_recurrent_state = inferstack_grouped_gdn_decode(
                    query, key, value, g, beta, recurrent_state,
                    output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                )
                STATS.grouped_gdn_decode_calls += 1
            except Exception:
                STATS.fallbacks += 1
                rep = self.num_v_heads // self.num_k_heads
                q2 = query.repeat_interleave(rep, dim=2) if rep > 1 else query
                k2 = key.repeat_interleave(rep, dim=2) if rep > 1 else key
                core_attn_out, last_recurrent_state = mod.torch_recurrent_gated_delta_rule(
                    q2, k2, value, g=g, beta=beta, initial_state=recurrent_state,
                    output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                    cu_seqlens=kwargs.pop('cu_seq_lens_q', None), **kwargs,
                )
        else:
            rep = self.num_v_heads // self.num_k_heads
            if rep > 1:
                query = query.repeat_interleave(rep, dim=2); key = key.repeat_interleave(rep, dim=2)
            core_attn_out, last_recurrent_state = mod.torch_chunk_gated_delta_rule(
                query, key, value, g=g, beta=beta, initial_state=recurrent_state,
                output_final_state=cache_params is not None, use_qk_l2norm_in_kernel=True,
                cu_seqlens=kwargs.pop('cu_seq_lens_q', None), **kwargs,
            )
        if cache_params is not None:
            cache_params.update_recurrent_state(last_recurrent_state, self.layer_idx)
        core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        z = z.reshape(-1, self.head_v_dim)
        core_attn_out = self.norm(core_attn_out, z)
        core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        return self.out_proj(core_attn_out)

    module.forward = types.MethodType(forward, module)
    module._inferstack_gdn_fused = True
    return True


def _install_attention(module, mod, cfg):
    import torch.nn.functional as F
    if not cfg.native.fuse_attention_qkv or getattr(module, '_inferstack_attention_fused', False):
        return False
    q = getattr(module,'q_proj',None); k = getattr(module,'k_proj',None); v = getattr(module,'v_proj',None)
    if any(x is None for x in (q,k,v)):
        return False
    _register_fused_weight(module, '_inferstack_qkv_weight', [q.weight,k.weight,v.weight])
    biases = [x.bias for x in (q,k,v)]
    if any(x is not None for x in biases):
        import torch
        dtype=q.weight.dtype;device=q.weight.device
        fused_bias=torch.cat([x.detach() if x is not None else torch.zeros((m.weight.shape[0],),dtype=dtype,device=device) for x,m in zip(biases,(q,k,v))])
        module.register_buffer('_inferstack_qkv_bias',fused_bias,persistent=False)
    else:
        module._inferstack_qkv_bias=None
    qout,kout,vout=q.weight.shape[0],k.weight.shape[0],v.weight.shape[0]
    module._inferstack_qkv_splits=(qout,kout,vout)
    module.q_proj=module.k_proj=module.v_proj=None

    def forward(self, hidden_states, position_embeddings, attention_mask, past_key_values=None, **kwargs):
        input_shape=hidden_states.shape[:-1]
        qkv=F.linear(hidden_states,self._inferstack_qkv_weight,self._inferstack_qkv_bias)
        qraw,kraw,vraw=qkv.split(self._inferstack_qkv_splits,dim=-1)
        STATS.attention_fused_projection_calls += 1
        qraw=qraw.view(*input_shape,-1,self.head_dim*2)
        query_states,gate=qraw.chunk(2,dim=-1)
        gate=gate.reshape(*input_shape,-1)
        query_states=self.q_norm(query_states).transpose(1,2)
        # Infer head counts from projection widths rather than assuming MHA/GQA equality.
        key_states=self.k_norm(kraw.view(*input_shape,-1,self.head_dim)).transpose(1,2)
        value_states=vraw.view(*input_shape,-1,self.head_dim).transpose(1,2)
        cos,sin=position_embeddings
        query_states,key_states=mod.apply_rotary_pos_emb(query_states,key_states,cos,sin)
        if past_key_values is not None:
            key_states,value_states=past_key_values.update(key_states,value_states,self.layer_idx)
        attention_interface=mod.ALL_ATTENTION_FUNCTIONS.get_interface(self.config._attn_implementation,mod.eager_attention_forward)
        attn_output,attn_weights=attention_interface(
            self,query_states,key_states,value_states,attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,scaling=self.scaling,**kwargs,
        )
        attn_output=attn_output.reshape(*input_shape,-1).contiguous()
        attn_output=attn_output*__import__('torch').sigmoid(gate)
        return self.o_proj(attn_output),attn_weights
    module.forward=types.MethodType(forward,module)
    module._inferstack_attention_fused=True
    return True


def _install_decoder_layer(mod, cfg):
    if not cfg.native.fuse_add_rmsnorm:
        return
    cls=getattr(mod,'Qwen3_5DecoderLayer',None)
    if cls is None:return
    key='decoder_forward'
    if key in _ORIGINALS:return
    _ORIGINALS[key]=cls.forward
    def forward(self,hidden_states,position_embeddings,attention_mask=None,position_ids=None,past_key_values=None,**kwargs):
        residual=hidden_states
        hidden_states=self.input_layernorm(hidden_states)
        if self.block_type=='linear_attention':
            hidden_states=self.linear_attn(hidden_states=hidden_states,cache_params=past_key_values,attention_mask=attention_mask,**kwargs)
        else:
            hidden_states,_=self.self_attn(hidden_states=hidden_states,attention_mask=attention_mask,position_ids=position_ids,past_key_values=past_key_values,position_embeddings=position_embeddings,**kwargs)
        hidden_states,residual=fused_add_rmsnorm(hidden_states,residual,self.post_attention_layernorm.weight,self.post_attention_layernorm.eps)
        hidden_states=self.mlp(hidden_states)
        return residual+hidden_states
    cls.forward=forward


def install_qwen35_forward_fusion(model, cfg):
    global STATS
    if not bool(getattr(cfg.native,'forward_fusion',False)):
        return diagnostics()
    try:
        from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
        _install_rmsnorm(mod,cfg);_install_gated_rmsnorm(mod,cfg);_install_decoder_layer(mod,cfg)
        for module in model.modules():
            name=module.__class__.__name__
            if name=='Qwen3_5RMSNorm':STATS.rmsnorm_modules+=1
            elif name=='Qwen3_5RMSNormGated':STATS.gated_rmsnorm_modules+=1
            elif name=='Qwen3_5DecoderLayer':STATS.decoder_layers+=1
            elif name=='Qwen3_5MLP' and _install_mlp(module,cfg):STATS.mlp_modules+=1
            elif name=='Qwen3_5GatedDeltaNet' and _install_gdn(module,mod,cfg):STATS.gdn_modules+=1
            elif name=='Qwen3_5Attention' and _install_attention(module,mod,cfg):STATS.attention_modules+=1
        STATS.installed=True;STATS.install_error=None
    except Exception as exc:
        STATS.install_error=f'{type(exc).__name__}: {exc}'
        if bool(getattr(cfg.native,'forward_fusion_strict',False)):
            raise
    return diagnostics()


def diagnostics():
    d=asdict(STATS)
    d['triton_available']=_HAS_TRITON
    return d
