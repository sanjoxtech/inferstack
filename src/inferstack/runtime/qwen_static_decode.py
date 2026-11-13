"""Static-buffer, CUDA-graph-capturable decode step for Qwen3.5 hybrid models.

Why: the native engine decodes through Hugging Face modules with growing caches, so
every step launches ~2,600 kernels from Python and cannot be graph-captured. vLLM and
SGLang replay one recorded CUDA graph per step and decode ~2x faster on the same GPU
(docs/NATIVE_GAP_ANALYSIS_2026-11-13.md).

Design
- All decode state lives in fixed buffers sized (batch bucket, max_len bucket):
  full-attention layers get a static K/V buffer, Gated DeltaNet layers keep their
  fixed-size conv and recurrent states. Every step updates them in place, so the step
  has static shapes and addresses and can be captured once and replayed.
- The step reuses the model's own modules and weights (norms, MLP, projections, rotary)
  and the modeling file's helpers, and mirrors the Transformers decode math, so the
  Transformers path is the numerical oracle. When forward fusion has replaced the
  projections with concatenated weights, the step uses those weights instead.
- Layout matches the engine's left-padded batches: every row writes its new token to
  the same physical column; a per-row mask marks which columns are real tokens.

Inputs per step (copied into static tensors before replay): next token ids [B],
positions [B], write column [1]. Output: last-token logits [B, vocab].
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from typing import Any


def _text_parts(model):
    """(text decoder module, lm_head, text config) for Qwen3_5ForCausalLM or ...ForConditionalGeneration."""
    inner = getattr(model, 'model', model)
    text = getattr(inner, 'language_model', inner)
    lm_head = getattr(model, 'lm_head', None)
    if lm_head is None or not hasattr(text, 'layers') or not hasattr(text, 'embed_tokens'):
        raise TypeError('static decode needs a Qwen3.5 causal LM with model.layers and lm_head')
    return text, lm_head, text.config


def _modeling(layer):
    mixer = getattr(layer, 'linear_attn', None) or getattr(layer, 'self_attn', None)
    return sys.modules[type(mixer).__module__]


def supports_static_decode(model) -> tuple[bool, str]:
    try:
        text, _, cfg = _text_parts(model)
    except TypeError as exc:
        return False, str(exc)
    kinds = set(getattr(cfg, 'layer_types', None) or [])
    if not kinds or not kinds <= {'full_attention', 'linear_attention'}:
        return False, f'unsupported layer types {sorted(kinds)}'
    return True, 'ok'


@dataclass
class StaticDecodeState:
    """Fixed buffers for one (batch, max_len) bucket."""
    batch: int
    max_len: int
    ids: Any
    pos: Any
    col: Any
    mask: Any
    logits: Any
    kv: dict = field(default_factory=dict)        # layer -> (k, v)  [B, Hkv, L, D]
    conv: dict = field(default_factory=dict)      # layer -> [B, conv_dim, K]
    rec: dict = field(default_factory=dict)       # layer -> [B, Hv, dk, dv]
    rows: int = 0                                 # live rows loaded


class QwenStaticDecoder:
    def __init__(self, model, device, dtype=None, gdn_backend='auto'):
        """gdn_backend: 'torch' (reference math), 'triton' (the repo's grouped GDN decode
        kernel, updates the recurrent state in place), or 'auto' (triton on CUDA when available)."""
        import torch
        ok, why = supports_static_decode(model)
        if not ok:
            raise TypeError(why)
        if gdn_backend not in ('auto', 'torch', 'triton'):
            raise ValueError(f'unknown gdn_backend {gdn_backend!r}')
        self.model = model
        self.text, self.lm_head, self.cfg = _text_parts(model)
        self.device = torch.device(device)
        self.dtype = dtype or self.text.embed_tokens.weight.dtype
        self.layer_types = list(self.cfg.layer_types)
        self.mod = _modeling(self.text.layers[0])
        self.graphs: dict = {}
        self.stats = {'captures': 0, 'replays': 0, 'eager_steps': 0, 'loads': 0}
        self.gdn_backend = self._resolve_gdn_backend(gdn_backend)
        self.stats['gdn_backend'] = self.gdn_backend

    def _resolve_gdn_backend(self, wanted):
        if wanted == 'torch':
            return 'torch'
        try:
            from inferstack.kernels import gdn_triton
            ok = gdn_triton._HAS_TRITON and self.device.type == 'cuda'
        except Exception:
            ok = False
        if wanted == 'triton' and not ok:
            raise RuntimeError('gdn_backend=triton needs CUDA and Triton')
        return 'triton' if ok else 'torch'

    # ---- buffers -----------------------------------------------------------------
    def new_state(self, batch: int, max_len: int) -> StaticDecodeState:
        import torch
        cfg, dev, dt = self.cfg, self.device, self.dtype
        st = StaticDecodeState(
            batch, max_len,
            ids=torch.zeros(batch, dtype=torch.long, device=dev),
            pos=torch.zeros(batch, dtype=torch.long, device=dev),
            col=torch.zeros(1, dtype=torch.long, device=dev),
            mask=torch.zeros(batch, max_len, dtype=torch.bool, device=dev),
            logits=torch.zeros(batch, cfg.vocab_size, dtype=dt, device=dev))
        for i, kind in enumerate(self.layer_types):
            layer = self.text.layers[i]
            if kind == 'full_attention':
                a = layer.self_attn
                shape = (batch, cfg.num_key_value_heads, max_len, a.head_dim)
                st.kv[i] = (torch.zeros(shape, dtype=dt, device=dev), torch.zeros(shape, dtype=dt, device=dev))
            else:
                g = layer.linear_attn
                st.conv[i] = torch.zeros(batch, g.conv_dim, g.conv_kernel_size, dtype=dt, device=dev)
                st.rec[i] = torch.zeros(batch, g.num_v_heads, g.head_k_dim, g.head_v_dim,
                                        dtype=torch.float32, device=dev)
        return st

    def load(self, st: StaticDecodeState, cache, attention_mask, width: int):
        """Copy a Transformers cache produced by prefill into the static buffers.

        `cache` rows map to state rows 0..b-1; its full-attention keys occupy physical
        columns 0..width-1 (left padded) and `attention_mask[:, :width]` marks real tokens.
        Rows b..batch-1 become inert (one dummy visible column, outputs ignored).
        """
        import torch
        b = int(attention_mask.shape[0])
        if b > st.batch or width > st.max_len:
            raise ValueError(f'cache ({b} rows, width {width}) exceeds state ({st.batch}, {st.max_len})')
        st.mask.zero_()
        st.mask[:b, :width].copy_(attention_mask[:, :width].bool())
        st.mask[b:, 0] = True
        for i, kind in enumerate(self.layer_types):
            src = cache.layers[i]
            if kind == 'full_attention':
                k, v = st.kv[i]
                k.zero_(); v.zero_()
                k[:b, :, :width].copy_(src.keys[:, :, :width])
                v[:b, :, :width].copy_(src.values[:, :, :width])
            else:
                st.conv[i].zero_(); st.rec[i].zero_()
                st.conv[i][:b].copy_(src.conv_states[0])
                st.rec[i][:b].copy_(src.recurrent_states[0])
        st.rows = b
        self.stats['loads'] += 1

    # ---- one decode step (static shapes; graph-capturable) --------------------------
    def _gdn(self, i, g, x, st):
        import torch
        import torch.nn.functional as F
        mod = self.mod
        B = x.shape[0]
        if getattr(g, 'in_proj_qkv', None) is None:
            # Forward fusion (kernels/qwen_forward_fusion.py) replaced the four projections
            # with two concatenated weights and removed the original modules.
            qkv, z = F.linear(x, g._inferstack_qkvz_weight).split([g.conv_dim, g.value_dim], dim=-1)
            b, a = F.linear(x, g._inferstack_ba_weight).split([g.num_v_heads, g.num_v_heads], dim=-1)
        else:
            qkv, z = g.in_proj_qkv(x), g.in_proj_z(x)
            b, a = g.in_proj_b(x), g.in_proj_a(x)
        mixed = qkv.transpose(1, 2)                                     # [B, C, 1]
        z = z.reshape(B, 1, -1, g.head_v_dim)
        conv = st.conv[i]
        w = g.conv1d.weight
        full = torch.cat([conv, mixed], dim=-1).to(w.dtype)          # [B, C, K+1]
        conv.copy_(full[:, :, -conv.shape[-1]:])
        out = F.conv1d(full, w, g.conv1d.bias, padding=0, groups=full.shape[1])[:, :, -1:]
        out = mod.ACT2FN[g.activation](out).to(x.dtype).transpose(1, 2)
        q, k, v = torch.split(out, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
        q = q.reshape(B, 1, -1, g.head_k_dim)
        k = k.reshape(B, 1, -1, g.head_k_dim)
        v = v.reshape(B, 1, -1, g.head_v_dim)
        beta = b.sigmoid()
        gate = -g.A_log.float().exp() * F.softplus(a.float() + g.dt_bias)
        if self.gdn_backend == 'triton':
            # Grouped kernel: Q/K L2 norm and head grouping fused; st.rec[i] (fp32,
            # contiguous) is updated in place, so nothing is allocated for the state.
            from inferstack.kernels.gdn_triton import inferstack_grouped_gdn_decode
            o, _ = inferstack_grouped_gdn_decode(q, k, v, gate, beta, initial_state=st.rec[i],
                                                 output_final_state=False, use_qk_l2norm_in_kernel=True)
            o = o.reshape(-1, g.head_v_dim)
            o = g.norm(o, z.reshape(-1, g.head_v_dim)).reshape(B, 1, -1)
            return g.out_proj(o)
        rep = g.num_v_heads // g.num_k_heads
        if rep > 1:
            q = q.repeat_interleave(rep, dim=2)
            k = k.repeat_interleave(rep, dim=2)
        # Recurrent gated delta rule, one token, fp32 (as torch_recurrent_gated_delta_rule).
        q, k, v = (t[:, 0].to(torch.float32) for t in (q, k, v))        # [B, H, d]
        beta, gate = beta[:, 0].float(), gate[:, 0]                    # [B, H]
        q = mod.l2norm(q, dim=-1, eps=1e-6) / (q.shape[-1] ** 0.5)
        k = mod.l2norm(k, dim=-1, eps=1e-6)
        S = st.rec[i] * gate.exp()[..., None, None]
        kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
        st.rec[i].copy_(S)
        o = (S * q.unsqueeze(-1)).sum(dim=-2)                          # [B, H, dv]
        o = o.to(x.dtype).reshape(-1, g.head_v_dim)
        o = g.norm(o, z.reshape(-1, g.head_v_dim)).reshape(B, 1, -1)
        return g.out_proj(o)

    def _attn(self, i, a, x, cos, sin, st):
        import torch
        import torch.nn.functional as F
        mod = self.mod
        B = x.shape[0]
        if getattr(a, 'q_proj', None) is None:
            # Forward fusion: one concatenated q/k/v weight, original modules removed.
            qraw, kraw, vraw = F.linear(x, a._inferstack_qkv_weight, a._inferstack_qkv_bias).split(
                a._inferstack_qkv_splits, dim=-1)
        else:
            qraw, kraw, vraw = a.q_proj(x), a.k_proj(x), a.v_proj(x)
        q, gate = torch.chunk(qraw.view(B, 1, -1, a.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(B, 1, -1)
        q = a.q_norm(q.reshape(B, 1, -1, a.head_dim)).transpose(1, 2)
        k = a.k_norm(kraw.view(B, 1, -1, a.head_dim)).transpose(1, 2)
        v = vraw.view(B, 1, -1, a.head_dim).transpose(1, 2)
        q, k = mod.apply_rotary_pos_emb(q, k, cos, sin)
        kc, vc = st.kv[i]
        kc.index_copy_(2, st.col, k)
        vc.index_copy_(2, st.col, v)
        keys = mod.repeat_kv(kc, a.num_key_value_groups)
        vals = mod.repeat_kv(vc, a.num_key_value_groups)
        o = F.scaled_dot_product_attention(q, keys, vals, attn_mask=st.mask[:, None, None, :], scale=a.scaling)
        o = o.transpose(1, 2).reshape(B, 1, -1) * torch.sigmoid(gate)
        return a.o_proj(o)

    def _step(self, st: StaticDecodeState):
        import torch
        st.mask.index_fill_(1, st.col, True)
        h = self.text.embed_tokens(st.ids[:, None])
        B = h.shape[0]
        cos, sin = self.text.rotary_emb(h, st.pos.view(1, B, 1).expand(3, B, 1))
        for i, kind in enumerate(self.layer_types):
            layer = self.text.layers[i]
            r = h
            h = layer.input_layernorm(h)
            if kind == 'linear_attention':
                h = self._gdn(i, layer.linear_attn, h, st)
            else:
                h = self._attn(i, layer.self_attn, h, cos, sin, st)
            h = r + h
            r = h
            h = r + layer.mlp(layer.post_attention_layernorm(h))
        h = self.text.norm(h)
        st.logits.copy_(self.lm_head(h[:, -1]))

    # ---- public step: eager on CPU, captured CUDA graph on GPU ---------------------
    def step(self, st: StaticDecodeState, ids, pos, col: int, *, use_graph: bool = True):
        import torch
        st.ids.copy_(ids)
        st.pos.copy_(pos)
        st.col.fill_(int(col))
        if not (use_graph and self.device.type == 'cuda'):
            with torch.inference_mode():
                self._step(st)
            self.stats['eager_steps'] += 1
            return st.logits
        key = id(st)
        graph = self.graphs.get(key)
        if graph is None:
            graph = self._capture(st)
            self.graphs[key] = graph
        graph.replay()
        self.stats['replays'] += 1
        return st.logits

    def _capture(self, st: StaticDecodeState):
        """Record one step for this state's buffers. Warm-up runs mutate state, so the
        caller's state is snapshotted and restored around capture."""
        import torch
        snap = self._snapshot(st)
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(2):
                self._step(st)
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph):
            self._step(st)
        self._restore(st, snap)
        self.stats['captures'] += 1
        return graph

    def _snapshot(self, st):
        return {'mask': st.mask.clone(), 'kv': {i: (k.clone(), v.clone()) for i, (k, v) in st.kv.items()},
                'conv': {i: t.clone() for i, t in st.conv.items()}, 'rec': {i: t.clone() for i, t in st.rec.items()}}

    def _restore(self, st, snap):
        st.mask.copy_(snap['mask'])
        for i, (k, v) in snap['kv'].items():
            st.kv[i][0].copy_(k); st.kv[i][1].copy_(v)
        for i, t in snap['conv'].items():
            st.conv[i].copy_(t)
        for i, t in snap['rec'].items():
            st.rec[i].copy_(t)

    def release(self, st: StaticDecodeState):
        self.graphs.pop(id(st), None)
