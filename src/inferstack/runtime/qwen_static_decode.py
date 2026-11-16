"""Static-buffer, CUDA-graph-capturable decode step for Qwen3.5 hybrid models.

Why: the native engine decodes through Hugging Face modules with growing caches, so
every step launches ~2,600 kernels from Python and cannot be graph-captured. vLLM and
SGLang replay one recorded CUDA graph per step and decode ~2x faster on the same GPU
(docs/NATIVE_GAP_ANALYSIS_2026-11-13.md).

Design
- Decode state lives in one pool of fixed slots (rows). Full-attention layers get a
  static K/V buffer [slots, Hkv, max_len, D]; Gated DeltaNet layers keep their
  fixed-size conv and recurrent states. Every step updates them in place.
- Row layout: token t of a request sits at column t (its position). Each row writes its
  new token at its own position, so requests of different lengths share one batch
  without left padding; a per-row mask marks which columns are real.
- Live rows are kept packed in slots 0..n-1 (see move_row). A step runs on the first
  b rows (b = smallest batch bucket >= n) and attends over the first L columns
  (L = smallest length bucket > longest position). One CUDA graph is captured per
  (b, L) pair up front, while the pool is empty, and replayed afterwards, so attention
  cost follows the live context length rather than the pool capacity.
- The step reuses the model's own modules and weights (norms, MLP, projections, rotary)
  and the modeling file's helpers, and mirrors the Transformers decode math, so the
  Transformers path is the numerical oracle. When forward fusion has replaced the
  projections with concatenated weights, the step uses those weights instead.
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


def default_buckets(limit: int, start: int) -> list[int]:
    """Powers of two from `start` below `limit`, then `limit` itself."""
    out, x = [], start
    while x < limit:
        out.append(x)
        x *= 2
    out.append(limit)
    return out


def pick_bucket(buckets, need: int) -> int:
    for b in buckets:
        if need <= b:
            return b
    raise ValueError(f'{need} exceeds the largest bucket {buckets[-1]}')


@dataclass
class StaticDecodeState:
    """Fixed buffers for a pool of `batch` slots, each holding up to `max_len` tokens."""
    batch: int
    max_len: int
    batch_buckets: list
    len_buckets: list
    ids: Any                                      # [slots] next input token
    pos: Any                                      # [slots] its position (= write column)
    mask: Any                                     # [slots, max_len] visible columns
    logits: Any                                   # [slots, vocab]
    kv: dict = field(default_factory=dict)        # layer -> (k, v)  [slots, Hkv, max_len, D]
    conv: dict = field(default_factory=dict)      # layer -> [slots, conv_dim, K]
    rec: dict = field(default_factory=dict)       # layer -> [slots, Hv, dk, dv] fp32


class QwenStaticDecoder:
    def __init__(self, model, device, dtype=None, gdn_backend='auto', attn_backend='sdpa'):
        """gdn_backend: 'torch' (reference math), 'triton' (the repo's grouped GDN decode
        kernel, updates the recurrent state in place), or 'auto' (triton on CUDA when available)."""
        import torch
        ok, why = supports_static_decode(model)
        if not ok:
            raise TypeError(why)
        if gdn_backend not in ('auto', 'torch', 'triton'):
            raise ValueError(f'unknown gdn_backend {gdn_backend!r}')
        if attn_backend not in ('sdpa', 'triton'):
            raise ValueError(f'unknown attn_backend {attn_backend!r}')
        if attn_backend == 'triton' and torch.device(device).type != 'cuda':
            raise RuntimeError('attn_backend=triton needs CUDA')
        self.attn_backend = attn_backend
        self._attn_scratch = {}
        self.model = model
        self.text, self.lm_head, self.cfg = _text_parts(model)
        self.device = torch.device(device)
        self.dtype = dtype or self.text.embed_tokens.weight.dtype
        self.layer_types = list(self.cfg.layer_types)
        self.mod = _modeling(self.text.layers[0])
        self.graphs: dict = {}
        self._pool = None
        self.stats = {'captures': 0, 'replays': 0, 'eager_steps': 0, 'loads': 0, 'rows_loaded': 0, 'row_moves': 0}
        self.gdn_backend = self._resolve_gdn_backend(gdn_backend)
        self.stats['gdn_backend'] = self.gdn_backend
        self.stats['attn_backend'] = attn_backend
        try:
            import inspect
            self._sdpa_gqa = 'enable_gqa' in inspect.signature(torch.nn.functional.scaled_dot_product_attention).parameters
        except (TypeError, ValueError):
            self._sdpa_gqa = False

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
    def new_state(self, batch: int, max_len: int, batch_buckets=None, len_buckets=None) -> StaticDecodeState:
        import torch
        cfg, dev, dt = self.cfg, self.device, self.dtype
        bb = sorted({int(x) for x in (batch_buckets or default_buckets(batch, 1)) if 0 < int(x) <= batch} | {batch})
        lb = sorted({int(x) for x in (len_buckets or default_buckets(max_len, 256)) if 0 < int(x) <= max_len} | {max_len})
        st = StaticDecodeState(
            batch, max_len, bb, lb,
            ids=torch.zeros(batch, dtype=torch.long, device=dev),
            pos=torch.zeros(batch, dtype=torch.long, device=dev),
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

    def reset(self, st: StaticDecodeState):
        """Zero every buffer (used after graph capture, whose warm-up steps write garbage)."""
        st.ids.zero_(); st.pos.zero_(); st.mask.zero_(); st.logits.zero_()
        for k, v in st.kv.values():
            k.zero_(); v.zero_()
        for t in list(st.conv.values()) + list(st.rec.values()):
            t.zero_()

    def load_rows(self, st: StaticDecodeState, cache, attention_mask, width: int, src_rows, dst_slots) -> list[int]:
        """Copy rows of a Transformers cache produced by prefill into pool slots.

        `cache` keys span physical columns 0..width-1 and `attention_mask[:, :width]`
        marks real tokens (left padding and prefix-cache layouts both work). Row r's
        real tokens are packed into columns 0..n-1 of its slot. Returns each row's n,
        which is also the position of its next token.
        """
        import torch
        if width > st.max_len:
            raise ValueError(f'cache width {width} exceeds pool max_len {st.max_len}')
        lengths = []
        m = attention_mask[:, :width].bool()
        for src, dst in zip(src_rows, dst_slots):
            if not 0 <= dst < st.batch:
                raise ValueError(f'slot {dst} outside pool of {st.batch}')
            cols = m[src].nonzero().squeeze(-1)
            n = int(cols.numel())
            if n >= st.max_len:
                raise ValueError(f'row of {n} tokens leaves no room in pool max_len {st.max_len}')
            st.mask[dst].zero_()
            st.mask[dst, :n] = True
            for i, kind in enumerate(self.layer_types):
                layer = cache.layers[i]
                if kind == 'full_attention':
                    k, v = st.kv[i]
                    k[dst, :, :n].copy_(layer.keys[src].index_select(1, cols))
                    v[dst, :, :n].copy_(layer.values[src].index_select(1, cols))
                else:
                    st.conv[i][dst].copy_(layer.conv_states[0][src])
                    st.rec[i][dst].copy_(layer.recurrent_states[0][src])
            lengths.append(n)
        self.stats['loads'] += 1
        self.stats['rows_loaded'] += len(lengths)
        return lengths

    def load(self, st: StaticDecodeState, cache, attention_mask, width: int) -> list[int]:
        """Load every cache row b into slot b."""
        b = int(attention_mask.shape[0])
        if b > st.batch:
            raise ValueError(f'cache has {b} rows, pool has {st.batch} slots')
        return self.load_rows(st, cache, attention_mask, width, range(b), range(b))

    def move_row(self, st: StaticDecodeState, src: int, dst: int, length: int):
        """Copy slot src into slot dst (to keep live rows packed). `length` = columns in use."""
        n = min(int(length), st.max_len)
        st.mask[dst].copy_(st.mask[src])
        st.ids[dst] = st.ids[src]
        st.pos[dst] = st.pos[src]
        for k, v in st.kv.values():
            k[dst, :, :n].copy_(k[src, :, :n])
            v[dst, :, :n].copy_(v[src, :, :n])
        for t in list(st.conv.values()) + list(st.rec.values()):
            t[dst].copy_(t[src])
        self.stats['row_moves'] += 1

    # ---- P1: packed prefill straight into the pool -----------------------------------
    def prefill_packed(self, st: StaticDecodeState, slots, prompts, *, backend='auto'):
        """Prefill several prompts as one flat token batch and write the results straight
        into pool slots (no padding, no Transformers cache, no load_rows copy).

        The dense parts (embeddings, norms, projections, MLP) run once over all T tokens.
        GDN layers run the causal conv and the chunked delta rule per request segment
        (FLA varlen kernels with cu_seqlens on CUDA, a per-segment reference otherwise);
        the final conv inputs and recurrent states go to st.conv/st.rec[slot]. Full-attention
        layers write K/V at columns 0..n-1 of each slot and attend causally per segment.

        backend: 'auto' (FLA on CUDA when importable), 'fla' or 'reference'.
        Returns (logits [k, vocab] at each prompt's last token, post-norm hidden [T, H],
        cu_seqlens list of k+1 offsets)."""
        import torch
        dev = self.device
        slots, prompts = [int(s) for s in slots], [list(map(int, p)) for p in prompts]
        if len(slots) != len(prompts) or not prompts:
            raise ValueError('need one slot per prompt')
        lens = [len(p) for p in prompts]
        if min(lens) < 1 or max(lens) >= st.max_len:
            raise ValueError(f'prompt lengths {lens} must be in 1..{st.max_len - 1}')
        if any(not 0 <= s < st.batch for s in slots):
            raise ValueError(f'slots {slots} outside pool of {st.batch}')
        if backend == 'auto':
            from inferstack.kernels.gdn import fla_available
            backend = 'fla' if dev.type == 'cuda' and fla_available() else 'reference'
        cu = [0]
        for n in lens:
            cu.append(cu[-1] + n)
        T = cu[-1]
        ids = torch.tensor([t for p in prompts for t in p], dtype=torch.long, device=dev)[None]
        pos = torch.cat([torch.arange(n, device=dev) for n in lens])
        cu_t = torch.tensor(cu, dtype=torch.int32 if backend == 'fla' else torch.long, device=dev)
        slot_t = torch.tensor(slots, dtype=torch.long, device=dev)
        n_t = torch.tensor(lens, dtype=torch.long, device=dev)
        st.mask[slot_t] = torch.arange(st.max_len, device=dev)[None] < n_t[:, None]
        h = self.text.embed_tokens(ids)
        cos, sin = self.text.rotary_emb(h, pos.view(1, 1, T).expand(3, 1, T))
        for i, kind in enumerate(self.layer_types):
            layer = self.text.layers[i]
            r = h
            h = layer.input_layernorm(h)
            if kind == 'linear_attention':
                h = self._gdn_packed(i, layer.linear_attn, h, st, slots, cu, cu_t, backend)
            else:
                h = self._attn_packed(i, layer.self_attn, h, cos, sin, st, slots, cu)
            h = r + h
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.text.norm(h)[0]                                         # [T, H]
        last = torch.tensor(cu[1:], device=dev) - 1
        logits = self.lm_head(h.index_select(0, last))
        self.stats['packed_prefills'] = self.stats.get('packed_prefills', 0) + 1
        self.stats['packed_rows'] = self.stats.get('packed_rows', 0) + len(slots)
        self.stats['packed_tokens'] = self.stats.get('packed_tokens', 0) + T
        return logits, h, cu

    def _gdn_packed(self, i, g, x, st, slots, cu, cu_t, backend):
        import torch
        import torch.nn.functional as F
        T = x.shape[1]
        if getattr(g, 'in_proj_qkv', None) is None:
            qkv, z = F.linear(x, g._inferstack_qkvz_weight).split([g.conv_dim, g.value_dim], dim=-1)
            b, a = F.linear(x, g._inferstack_ba_weight).split([g.num_v_heads, g.num_v_heads], dim=-1)
        else:
            qkv, z = g.in_proj_qkv(x), g.in_proj_z(x)
            b, a = g.in_proj_b(x), g.in_proj_a(x)
        w = g.conv1d.weight.squeeze(1)                                   # [C, K]
        K = w.shape[-1]
        beta = b.sigmoid()
        gate = -g.A_log.float().exp() * F.softplus(a.float() + g.dt_bias)
        if backend == 'fla':
            from fla.modules.causal_conv1d import causal_conv1d
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            mixed, conv_final = causal_conv1d(qkv, weight=w, bias=g.conv1d.bias, initial_state=None,
                                              output_final_state=True, activation=g.activation,
                                              backend='triton', cu_seqlens=cu_t)
            q, k, v = torch.split(mixed, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
            q = q.reshape(1, T, -1, g.head_k_dim)
            k = k.reshape(1, T, -1, g.head_k_dim)
            v = v.reshape(1, T, -1, g.head_v_dim)
            o, rec_final = chunk_gated_delta_rule(q, k, v, g=gate, beta=beta, scale=g.head_k_dim ** -0.5,
                                                  initial_state=None, output_final_state=True,
                                                  use_qk_l2norm_in_kernel=True, cu_seqlens=cu_t)
            slot_t = torch.tensor(slots, dtype=torch.long, device=x.device)
            st.conv[i].index_copy_(0, slot_t, conv_final.to(st.conv[i].dtype))
            st.rec[i].index_copy_(0, slot_t, rec_final.float())
            o = o.to(x.dtype)
        else:
            from inferstack.kernels.gdn import gdn
            from inferstack.kernels.gdn_conv import causal_conv_prefill
            outs = []
            for j, s in enumerate(slots):
                a0, a1 = cu[j], cu[j + 1]
                mixed, cf = causal_conv_prefill(qkv[:, a0:a1], w, g.conv1d.bias, None,
                                                activation=g.activation, backend='reference')
                q, k, v = torch.split(mixed, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
                n = a1 - a0
                q = q.reshape(1, n, -1, g.head_k_dim)
                k = k.reshape(1, n, -1, g.head_k_dim)
                v = v.reshape(1, n, -1, g.head_v_dim)
                o, rf = gdn(q, k, v, gate[:, a0:a1], beta[:, a0:a1], None, output_state=True,
                            qk_l2norm=True, backend='recurrent_reference')
                st.conv[i][s].copy_(cf[0])
                st.rec[i][s].copy_(rf[0])
                outs.append(o)
            o = torch.cat(outs, dim=1)
        o = g.norm(o.reshape(-1, g.head_v_dim), z.reshape(-1, g.head_v_dim)).reshape(1, T, -1)
        return g.out_proj(o)

    def _attn_packed(self, i, a, x, cos, sin, st, slots, cu):
        import torch
        import torch.nn.functional as F
        mod = self.mod
        T = x.shape[1]
        if getattr(a, 'q_proj', None) is None:
            qraw, kraw, vraw = F.linear(x, a._inferstack_qkv_weight, a._inferstack_qkv_bias).split(
                a._inferstack_qkv_splits, dim=-1)
        else:
            qraw, kraw, vraw = a.q_proj(x), a.k_proj(x), a.v_proj(x)
        q, gate = torch.chunk(qraw.view(1, T, -1, a.head_dim * 2), 2, dim=-1)
        gate = gate.reshape(1, T, -1)
        q = a.q_norm(q.reshape(1, T, -1, a.head_dim)).transpose(1, 2)    # [1, Hq, T, D]
        k = a.k_norm(kraw.view(1, T, -1, a.head_dim)).transpose(1, 2)    # [1, Hkv, T, D]
        v = vraw.view(1, T, -1, a.head_dim).transpose(1, 2)
        q, k = mod.apply_rotary_pos_emb(q, k, cos, sin)
        kc, vc = st.kv[i]
        outs = []
        for j, s in enumerate(slots):
            a0, a1 = cu[j], cu[j + 1]
            n = a1 - a0
            kc[s, :, :n].copy_(k[0, :, a0:a1])
            vc[s, :, :n].copy_(v[0, :, a0:a1])
            qs, ks, vs = q[:, :, a0:a1], k[:, :, a0:a1], v[:, :, a0:a1]
            if self._sdpa_gqa:
                o = F.scaled_dot_product_attention(qs, ks, vs, is_causal=True, scale=a.scaling,
                                                   enable_gqa=a.num_key_value_groups > 1)
            else:
                o = F.scaled_dot_product_attention(qs, mod.repeat_kv(ks, a.num_key_value_groups),
                                                   mod.repeat_kv(vs, a.num_key_value_groups),
                                                   is_causal=True, scale=a.scaling)
            outs.append(o)
        o = torch.cat(outs, dim=2).transpose(1, 2).reshape(1, T, -1) * torch.sigmoid(gate)
        return a.o_proj(o)

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
        conv = st.conv[i][:B]
        rec = st.rec[i][:B]
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
            # Grouped kernel: Q/K L2 norm and head grouping fused; the recurrent state
            # (fp32, contiguous leading-rows view) is updated in place.
            from inferstack.kernels.gdn_triton import inferstack_grouped_gdn_decode
            o, _ = inferstack_grouped_gdn_decode(q, k, v, gate, beta, initial_state=rec,
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
        S = rec * gate.exp()[..., None, None]
        kv_mem = (S * k.unsqueeze(-1)).sum(dim=-2)
        delta = (v - kv_mem) * beta.unsqueeze(-1)
        S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
        rec.copy_(S)
        o = (S * q.unsqueeze(-1)).sum(dim=-2)                          # [B, H, dv]
        o = o.to(x.dtype).reshape(-1, g.head_v_dim)
        o = g.norm(o, z.reshape(-1, g.head_v_dim)).reshape(B, 1, -1)
        return g.out_proj(o)

    def _attn(self, i, a, x, cos, sin, st, L, write_idx, mask):
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
        kc[:B].scatter_(2, write_idx, k)                              # each row at its own position
        vc[:B].scatter_(2, write_idx, v)
        if self.attn_backend == 'triton':
            # N5.1: split-KV decode kernel, one KV read per group, stops at each row's position.
            from inferstack.kernels.decode_attention import decode_attention, make_scratch
            key = (id(st), i)
            if key not in self._attn_scratch:
                self._attn_scratch[key] = make_scratch(st.batch, kc.shape[1], q.shape[1] // kc.shape[1], a.head_dim,
                                                       st.max_len, x.device)
            o = decode_attention(q[:, :, 0], kc[:B], vc[:B], st.mask[:B], st.pos[:B], L, a.scaling,
                                 backend='triton', scratch=self._attn_scratch[key])[:, :, None]
            o = o.transpose(1, 2).reshape(B, 1, -1) * torch.sigmoid(gate)
            return a.o_proj(o)
        keys, vals = kc[:B, :, :L], vc[:B, :, :L]
        if self._sdpa_gqa:
            o = F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask, scale=a.scaling,
                                               enable_gqa=a.num_key_value_groups > 1)
        else:
            keys = mod.repeat_kv(keys, a.num_key_value_groups)
            vals = mod.repeat_kv(vals, a.num_key_value_groups)
            o = F.scaled_dot_product_attention(q, keys, vals, attn_mask=mask, scale=a.scaling)
        o = o.transpose(1, 2).reshape(B, 1, -1) * torch.sigmoid(gate)
        return a.o_proj(o)

    def _step(self, st: StaticDecodeState, B: int, L: int):
        ids, pos = st.ids[:B], st.pos[:B]
        st.mask[:B].scatter_(1, pos.view(B, 1), True)
        mask = st.mask[:B, None, None, :L]
        h = self.text.embed_tokens(ids[:, None])
        cos, sin = self.text.rotary_emb(h, pos.view(1, B, 1).expand(3, B, 1))
        write_idx = None
        for i, kind in enumerate(self.layer_types):
            layer = self.text.layers[i]
            r = h
            h = layer.input_layernorm(h)
            if kind == 'linear_attention':
                h = self._gdn(i, layer.linear_attn, h, st)
            else:
                if write_idx is None:
                    a = layer.self_attn
                    write_idx = pos.view(B, 1, 1, 1).expand(B, self.cfg.num_key_value_heads, 1, a.head_dim)
                h = self._attn(i, layer.self_attn, h, cos, sin, st, L, write_idx, mask)
            h = r + h
            r = h
            h = r + layer.mlp(layer.post_attention_layernorm(h))
        h = self.text.norm(h)
        st.logits[:B].copy_(self.lm_head(h[:, -1]))

    # ---- public API ----------------------------------------------------------------
    def capture(self, st: StaticDecodeState, batches=None, lengths=None) -> int:
        """Record one CUDA graph per (batch bucket, length bucket). Warm-up steps write
        into the buffers, so call this while the pool holds no live rows; buffers are
        zeroed afterwards. Returns the number of graphs captured (0 off CUDA)."""
        import torch
        if self.device.type != 'cuda':
            return 0
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        n = 0
        for B in (batches or st.batch_buckets):
            for L in (lengths or st.len_buckets):
                key = (id(st), int(B), int(L))
                if key in self.graphs:
                    continue
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.inference_mode(), torch.cuda.stream(stream):
                    self._step(st, B, L)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(graph, pool=self._pool):
                    self._step(st, B, L)
                self.graphs[key] = graph
                n += 1
        torch.cuda.synchronize()
        self.reset(st)
        self.stats['captures'] += n
        return n

    def step(self, st: StaticDecodeState, ids, pos, rows: int | None = None, *, length: int | None = None,
             use_graph: bool = True):
        """One decode step for live slots 0..rows-1. ids/pos: [rows] next token and its
        position. length: 1 + the largest position (pass it when known to avoid a sync).
        Returns logits [rows, vocab] (a view of the pool's buffer; consume before the next step)."""
        import torch
        rows = int(ids.shape[0]) if rows is None else int(rows)
        if length is None:
            length = int(pos[:rows].max().item()) + 1
        if length > st.max_len:
            raise ValueError(f'position {length - 1} beyond pool max_len {st.max_len}')
        B = pick_bucket(st.batch_buckets, rows)
        L = pick_bucket(st.len_buckets, int(length))
        st.ids[:rows].copy_(ids[:rows])
        st.pos[:rows].copy_(pos[:rows])
        if B > rows:
            st.ids[rows:B].zero_()
            st.pos[rows:B].zero_()
        graph = self.graphs.get((id(st), B, L)) if use_graph else None
        if graph is None:
            with torch.inference_mode():
                self._step(st, B, L)
            self.stats['eager_steps'] += 1
        else:
            graph.replay()
            self.stats['replays'] += 1
        return st.logits[:rows]

    def release(self, st: StaticDecodeState):
        # Scratch is keyed by id(st) too: a later state can reuse the id with other shapes
        # (seen in the N6.1b bench: a stale smaller scratch broke the plain lane at batch 16).
        for key in [k for k in self.graphs if k[0] == id(st)]:
            self.graphs.pop(key)
        for key in [k for k in self._attn_scratch if k[0] == id(st)]:
            self._attn_scratch.pop(key)
