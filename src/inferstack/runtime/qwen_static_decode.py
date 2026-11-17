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


def _fla_cached_tensors() -> list:
    """Every result currently held by FLA's tensor_cache-decorated helpers (identity-keyed
    memo deques reached through each wrapper's closure)."""
    import sys
    from collections import deque
    keep, seen = [], set()
    for name, mod in list(sys.modules.items()):
        if not name.startswith('fla') or mod is None:
            continue
        for obj in list(vars(mod).values()):
            if not callable(obj) or getattr(obj, '__wrapped__', None) is None or id(obj) in seen:
                continue
            seen.add(id(obj))
            for cell in getattr(obj, '__closure__', None) or ():
                try:
                    val = cell.cell_contents
                except ValueError:
                    continue
                if isinstance(val, deque):
                    keep.extend(entry[-1] for entry in list(val) if isinstance(entry, tuple))
    return keep


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
    def snap_align(self, backend='auto'):
        """P5: block boundaries prefill_packed(snap=...) can record must be multiples of this
        (FLA keeps chunk-start states every 64 tokens; the reference path records any boundary)."""
        if backend == 'auto':
            from inferstack.kernels.gdn import fla_available
            backend = 'fla' if self.device.type == 'cuda' and fla_available() else 'reference'
        if backend != 'fla':
            return 1
        from inferstack.kernels import fla_gdr_states
        return fla_gdr_states.CHUNK if fla_gdr_states.available() else 0

    def prefill_packed(self, st: StaticDecodeState, slots, prompts, *, starts=None, backend='auto',
                       snap=None, snap_out=None):
        """Prefill several prompts as one flat token batch and write the results straight
        into pool slots (no padding, no Transformers cache, no load_rows copy).

        The dense parts (embeddings, norms, projections, MLP) run once over all T tokens.
        GDN layers run the causal conv and the chunked delta rule per request segment
        (FLA varlen kernels with cu_seqlens on CUDA, a per-segment reference otherwise);
        the final conv inputs and recurrent states go to st.conv/st.rec[slot]. Full-attention
        layers write K/V at columns 0..n-1 of each slot and attend causally per segment.

        starts (P2): per row, how many leading prompt tokens the slot already holds (K/V
        columns 0..s-1 plus the GDN conv/recurrent state after them, e.g. a restored prefix
        snapshot). Only prompt[s:] is computed, from that state. Default all 0.

        backend: 'auto' (FLA on CUDA when importable), 'fla' or 'reference'.

        snap (P5): per row, a block boundary b (starts[j] < b < len(prompt), b - starts[j] a multiple
        of snap_align()) at which to record the GDN conv/recurrent state in this same pass, or 0.
        snap_out (a dict) receives {'rows': [j...], 'conv': {layer: [R, D, W]}, 'rec': {layer:
        [R, HV, K, V] fp32}} for those rows; K/V for columns < b are in the slot after the call.
        Lets a cold prompt leave a prefix-cache snapshot without a second prefill of its prefix.
        Returns (logits [k, vocab] at each prompt's last token, post-norm hidden [T, H] of the
        computed tokens, cu_seqlens list of k+1 offsets into them)."""
        import torch
        dev = self.device
        slots, prompts = [int(s) for s in slots], [list(map(int, p)) for p in prompts]
        if len(slots) != len(prompts) or not prompts:
            raise ValueError('need one slot per prompt')
        full = [len(p) for p in prompts]
        starts = [0] * len(prompts) if starts is None else [int(x) for x in starts]
        if len(starts) != len(prompts) or any(not 0 <= s0 < n for s0, n in zip(starts, full)):
            raise ValueError(f'starts {starts} must leave at least one token of {full}')
        lens = [n - s0 for n, s0 in zip(full, starts)]
        if min(lens) < 1 or max(full) >= st.max_len:
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
        ids = torch.tensor([t for p, s0 in zip(prompts, starts) for t in p[s0:]], dtype=torch.long, device=dev)[None]
        pos = torch.cat([torch.arange(s0, n, device=dev) for s0, n in zip(starts, full)])
        cu_t = torch.tensor(cu, dtype=torch.int32 if backend == 'fla' else torch.long, device=dev)
        slot_t = torch.tensor(slots, dtype=torch.long, device=dev)
        n_t = torch.tensor(full, dtype=torch.long, device=dev)
        st.mask[slot_t] = torch.arange(st.max_len, device=dev)[None] < n_t[:, None]
        # P5: per-prefill device tensors are built once here, not per layer. A host list copied to
        # the GPU inside the layer loop (or a boolean-mask assignment) synchronises the stream, so
        # the GPU drained and then idled while FLA's ~1.6 ms of Python per call launched the next
        # layer (about 30 ms per long prefill on L4).
        sn = None
        if snap is not None and any(snap):
            if snap_out is None or len(snap) != len(prompts):
                raise ValueError('snap needs one boundary per prompt and a snap_out dict')
            al = self.snap_align(backend)
            rows = [j for j, b in enumerate(snap) if b]
            for j in rows:
                b, s0 = int(snap[j]), starts[j]
                if not al or not s0 < b < full[j] or (b - s0) % al:
                    raise ValueError(f'snap boundary {b} not recordable for row {j} (start {s0}, len {full[j]}, align {al})')
            K0 = None
            for g0 in (self.text.layers[i].linear_attn for i, kd in enumerate(self.layer_types) if kd == 'linear_attention'):
                K0 = g0.conv1d.weight.shape[-1]
                break
            # token rows (in the packed batch) of the W conv inputs before each boundary, and each
            # boundary's chunk index in FLA's h; one host->device copy per prefill
            tok = [cu[j] + int(snap[j]) - starts[j] - K0 + w for j in rows for w in range(K0)]
            if backend == 'fla':
                from inferstack.kernels.fla_gdr_states import CHUNK, chunk_offsets
                off = chunk_offsets(lens)
                hix = [off[j] + (int(snap[j]) - starts[j]) // CHUNK for j in rows]
            else:
                hix = []
            idx = torch.tensor(tok + hix, dtype=torch.long, device=dev)
            sn = {'rows': rows, 'b': [int(snap[j]) - starts[j] for j in rows], 'W': K0,
                  'tok': idx[:len(tok)], 'h': idx[len(tok):]}
            snap_out.clear()
            snap_out.update(rows=rows, conv={}, rec={})
        keep = None
        if any(starts):   # 1.0 for rows continuing from the slot's state, 0.0 for fresh rows
            keep = torch.tensor([float(s0 > 0) for s0 in starts], device=dev)
        h = self.text.embed_tokens(ids)
        cos, sin = self.text.rotary_emb(h, pos.view(1, 1, T).expand(3, 1, T))
        for i, kind in enumerate(self.layer_types):
            layer = self.text.layers[i]
            r = h
            h = layer.input_layernorm(h)
            if kind == 'linear_attention':
                h = self._gdn_packed(i, layer.linear_attn, h, st, slots, cu, cu_t, backend, starts,
                                     slot_t=slot_t, keep=keep, sn=sn, snap_out=snap_out)
            else:
                h = self._attn_packed(i, layer.self_attn, h, cos, sin, st, slots, cu, starts)
            h = r + h
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.text.norm(h)[0]                                         # [T, H]
        last = torch.tensor(cu[1:], device=dev) - 1
        logits = self.lm_head(h.index_select(0, last))
        self.stats['packed_prefills'] = self.stats.get('packed_prefills', 0) + 1
        self.stats['packed_rows'] = self.stats.get('packed_rows', 0) + len(slots)
        self.stats['packed_tokens'] = self.stats.get('packed_tokens', 0) + T
        return logits, h, cu

    def _gdn_packed(self, i, g, x, st, slots, cu, cu_t, backend, starts, *, slot_t=None, keep=None, sn=None,
                    snap_out=None):
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
        cont = any(starts)
        if backend == 'fla':
            from fla.modules.causal_conv1d import causal_conv1d
            from fla.ops.gated_delta_rule import chunk_gated_delta_rule
            if slot_t is None:
                slot_t = torch.tensor(slots, dtype=torch.long, device=x.device)
            init_conv = init_rec = None
            if cont:
                # Continuing rows start from the slot's conv/recurrent state, fresh rows from zeros.
                # FLA's varlen conv takes one initial state per sequence (P4: checked equal to
                # per-segment calls on L4, max diff 0.0), so this stays a single call.
                # Zeroed with torch.where, not a boolean-mask assignment (that one syncs the stream);
                # where() also keeps a stale non-finite value in a fresh slot from leaking in.
                if keep is None:
                    keep = torch.tensor([float(s0 > 0) for s0 in starts], device=x.device)
                live = keep > 0
                init_conv = st.conv[i].index_select(0, slot_t).to(qkv.dtype)
                init_conv = torch.where(live[:, None, None], init_conv, torch.zeros((), dtype=init_conv.dtype,
                                                                                     device=x.device))
                init_rec = st.rec[i].index_select(0, slot_t).float()
                init_rec = torch.where(live[:, None, None, None], init_rec, torch.zeros((), device=x.device))
            mixed, conv_final = causal_conv1d(qkv, weight=w, bias=g.conv1d.bias, initial_state=init_conv,
                                              output_final_state=True, activation=g.activation,
                                              backend='triton', cu_seqlens=cu_t)
            q, k, v = torch.split(mixed, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
            q = q.reshape(1, T, -1, g.head_k_dim)
            k = k.reshape(1, T, -1, g.head_k_dim)
            v = v.reshape(1, T, -1, g.head_v_dim)
            if sn is None:
                o, rec_final = chunk_gated_delta_rule(q, k, v, g=gate, beta=beta, scale=g.head_k_dim ** -0.5,
                                                      initial_state=init_rec, output_final_state=True,
                                                      use_qk_l2norm_in_kernel=True, cu_seqlens=cu_t)
            else:   # P5: same math, plus FLA's chunk-start states h; the boundary state is h at b's chunk
                from inferstack.kernels.fla_gdr_states import chunk_gdr_with_states
                o, rec_final, hs = chunk_gdr_with_states(q, k, v, gate, beta, g.head_k_dim ** -0.5, init_rec, cu_t)
                R, W = len(sn['rows']), sn['W']
                snap_out['conv'][i] = qkv[0].index_select(0, sn['tok']).view(R, W, -1).transpose(1, 2) \
                    .to(st.conv[i].dtype).contiguous()
                snap_out['rec'][i] = hs[0].index_select(0, sn['h']).float()
            st.conv[i].index_copy_(0, slot_t, conv_final.to(st.conv[i].dtype))
            st.rec[i].index_copy_(0, slot_t, rec_final.float())
            o = o.to(x.dtype)
        else:
            from inferstack.kernels.gdn import gdn
            from inferstack.kernels.gdn_conv import causal_conv_prefill
            outs = []
            snaps = {} if sn is None else dict(zip(sn['rows'], sn['b']))
            sc, sr = [], []
            for j, s in enumerate(slots):
                a0, a1 = cu[j], cu[j + 1]
                cf = st.conv[i][s:s + 1] if starts[j] else None
                rf = st.rec[i][s:s + 1].clone() if starts[j] else None
                # P5 snap: run the segment in two pieces at the boundary (exact; reference path only)
                cuts = [a0] + ([a0 + snaps[j]] if j in snaps else []) + [a1]
                for c0, c1 in zip(cuts, cuts[1:]):
                    mixed, cf = causal_conv_prefill(qkv[:, c0:c1], w, g.conv1d.bias, cf,
                                                    activation=g.activation, backend='reference')
                    q, k, v = torch.split(mixed, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
                    n = c1 - c0
                    q = q.reshape(1, n, -1, g.head_k_dim)
                    k = k.reshape(1, n, -1, g.head_k_dim)
                    v = v.reshape(1, n, -1, g.head_v_dim)
                    o, rf = gdn(q, k, v, gate[:, c0:c1], beta[:, c0:c1], rf, output_state=True,
                                qk_l2norm=True, backend='recurrent_reference')
                    if j in snaps and c1 != a1:
                        sc.append(cf[0].to(st.conv[i].dtype).clone())
                        sr.append(rf[0].float().clone())
                    outs.append(o)
                st.conv[i][s].copy_(cf[0])
                st.rec[i][s].copy_(rf[0])
            o = torch.cat(outs, dim=1)
            if sn is not None:
                snap_out['conv'][i] = torch.stack(sc)
                snap_out['rec'][i] = torch.stack(sr)
        o = g.norm(o.reshape(-1, g.head_v_dim), z.reshape(-1, g.head_v_dim)).reshape(1, T, -1)
        return g.out_proj(o)

    def _attn_packed(self, i, a, x, cos, sin, st, slots, cu, starts):
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
            s0 = starts[j]
            n = s0 + a1 - a0
            kc[s, :, s0:n].copy_(k[0, :, a0:a1])
            vc[s, :, s0:n].copy_(v[0, :, a0:a1])
            qs = q[:, :, a0:a1]
            ks, vs = kc[s:s + 1, :, :n], vc[s:s + 1, :, :n]
            if s0:   # queries at positions s0..n-1 see columns <= their own position
                causal, msk = False, (torch.arange(n, device=x.device)[None] <=
                                      torch.arange(s0, n, device=x.device)[:, None])
            else:
                causal, msk = True, None
            if self._sdpa_gqa:
                o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=msk, is_causal=causal, scale=a.scaling,
                                                   enable_gqa=a.num_key_value_groups > 1)
            else:
                o = F.scaled_dot_product_attention(qs, mod.repeat_kv(ks, a.num_key_value_groups),
                                                   mod.repeat_kv(vs, a.num_key_value_groups),
                                                   attn_mask=msk, is_causal=causal, scale=a.scaling)
            outs.append(o)
        o = torch.cat(outs, dim=2).transpose(1, 2).reshape(1, T, -1) * torch.sigmoid(gate)
        return a.o_proj(o)

    # ---- P4: small packed prefills as CUDA graphs ------------------------------------
    # A small packed prefill is host-bound (L4: ~21 ms of GPU work inside ~65 ms wall, ~1200
    # launches; FLA's chunk call alone ~1.2 ms of Python per layer). FLA's varlen kernels can be
    # captured, but read cu_seqlens on the host at capture, so a graph is only valid for the
    # split it was captured with. Each graph therefore has a fixed shape: k rows, each padded
    # to L tokens (split 0, L, 2L, ...). Padding tokens are inert in the delta rule (beta = 0,
    # g = 0 leaves the state unchanged; L4 probe: outputs equal, state diff <= 3e-8), the conv
    # final state is gathered from the real tokens, and padded K/V columns land past each
    # row's length where the pool mask hides them until decode overwrites them.
    PREFILL_GRAPH_LENS = (16, 32, 64, 128)
    PREFILL_GRAPH_MAX_ROWS = 8

    def prefill_graph_key(self, st, prompts, starts=None):
        """(k, L) bucket for these prompts, or None when they do not fit a graph."""
        k = len(prompts)
        if not 1 <= k <= self.PREFILL_GRAPH_MAX_ROWS or self.device.type != 'cuda':
            return None
        starts = [0] * k if starts is None else starts
        new = max(len(p) - s0 for p, s0 in zip(prompts, starts))
        L = next((x for x in self.PREFILL_GRAPH_LENS if x >= new), None)
        if L is None or any(s0 + L > st.max_len for s0 in starts):
            return None
        return k, L

    def _pg_buffers(self, st, k, L):
        import torch
        dev, T = self.device, k * L
        return {'ids': torch.zeros((1, T), dtype=torch.long, device=dev),
                'pos': torch.zeros((T,), dtype=torch.long, device=dev),
                'valid': torch.zeros((T,), dtype=torch.bool, device=dev),
                'slot': torch.zeros((k,), dtype=torch.long, device=dev),
                'cont': torch.zeros((k,), dtype=torch.bool, device=dev),
                'nnew': torch.ones((k,), dtype=torch.long, device=dev),
                'full': torch.ones((k,), dtype=torch.long, device=dev),
                'cu': torch.arange(0, T + 1, L, dtype=torch.int32, device=dev)}

    def _pg_fill(self, buf, k, L, slots, prompts, starts):
        import torch
        ids, pos, valid = [], [], []
        for p, s0 in zip(prompts, starts):
            new = p[s0:]
            n = len(new)
            ids += new + [0] * (L - n)
            pos += list(range(s0, s0 + L))
            valid += [True] * n + [False] * (L - n)
        host = {'ids': torch.tensor([ids]), 'pos': torch.tensor(pos), 'valid': torch.tensor(valid),
                'slot': torch.tensor(slots), 'cont': torch.tensor([s0 > 0 for s0 in starts]),
                'nnew': torch.tensor([len(p) - s0 for p, s0 in zip(prompts, starts)]),
                'full': torch.tensor([len(p) for p in prompts])}
        for name, t in host.items():
            buf[name].copy_(t, non_blocking=True)

    def _pg_forward(self, st, buf, k, L):
        """The graph body: fixed shapes, every per-call value read from `buf`."""
        import torch
        dev, T = self.device, k * L
        slot_t = buf['slot']
        st.mask[slot_t] = torch.arange(st.max_len, device=dev)[None] < buf['full'][:, None]
        h = self.text.embed_tokens(buf['ids'])
        cos, sin = self.text.rotary_emb(h, buf['pos'].view(1, 1, T).expand(3, 1, T))
        for i, kind in enumerate(self.layer_types):
            layer = self.text.layers[i]
            r = h
            h = layer.input_layernorm(h)
            if kind == 'linear_attention':
                h = self._gdn_graphed(i, layer.linear_attn, h, st, buf, k, L)
            else:
                h = self._attn_graphed(i, layer.self_attn, h, cos, sin, st, buf, k, L)
            h = r + h
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.text.norm(h)[0]                                         # [T, H]
        last = torch.arange(k, device=dev) * L + buf['nnew'] - 1
        buf['hidden'] = h
        buf['logits'] = self.lm_head(h.index_select(0, last))

    def _gdn_graphed(self, i, g, x, st, buf, k, L):
        import torch
        import torch.nn.functional as F
        from fla.modules.causal_conv1d import causal_conv1d
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule
        T = x.shape[1]
        if getattr(g, 'in_proj_qkv', None) is None:
            qkv, z = F.linear(x, g._inferstack_qkvz_weight).split([g.conv_dim, g.value_dim], dim=-1)
            b, a = F.linear(x, g._inferstack_ba_weight).split([g.num_v_heads, g.num_v_heads], dim=-1)
        else:
            qkv, z = g.in_proj_qkv(x), g.in_proj_z(x)
            b, a = g.in_proj_b(x), g.in_proj_a(x)
        w = g.conv1d.weight.squeeze(1)                                   # [C, W]
        W = w.shape[-1]
        valid = buf['valid'].view(1, T, 1)
        beta = b.sigmoid() * valid                                       # padding: beta = 0
        gate = (-g.A_log.float().exp() * F.softplus(a.float() + g.dt_bias)) * valid   # and g = 0
        slot_t, cont = buf['slot'], buf['cont'].view(k, 1, 1)
        init_conv = st.conv[i].index_select(0, slot_t).to(qkv.dtype) * cont           # [k, C, W]
        init_rec = st.rec[i].index_select(0, slot_t).float() * cont.unsqueeze(-1)
        mixed, _ = causal_conv1d(qkv, weight=w, bias=g.conv1d.bias, initial_state=init_conv,
                                 output_final_state=False, activation=g.activation,
                                 backend='triton', cu_seqlens=buf['cu'])
        # conv final state = the last W inputs up to each row's last real token
        seq = torch.cat([init_conv, qkv[0].view(k, L, -1).transpose(1, 2)], dim=-1)   # [k, C, W+L]
        cols = (buf['nnew'].view(k, 1) + torch.arange(W, device=x.device)).view(k, 1, W)
        conv_final = seq.gather(2, cols.expand(k, seq.shape[1], W))
        q, kk, v = torch.split(mixed, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
        o, rec_final = chunk_gated_delta_rule(q.reshape(1, T, -1, g.head_k_dim), kk.reshape(1, T, -1, g.head_k_dim),
                                              v.reshape(1, T, -1, g.head_v_dim), g=gate, beta=beta,
                                              scale=g.head_k_dim ** -0.5, initial_state=init_rec,
                                              output_final_state=True, use_qk_l2norm_in_kernel=True,
                                              cu_seqlens=buf['cu'])
        st.conv[i].index_copy_(0, slot_t, conv_final.to(st.conv[i].dtype))
        st.rec[i].index_copy_(0, slot_t, rec_final.float())
        o = g.norm(o.to(x.dtype).reshape(-1, g.head_v_dim), z.reshape(-1, g.head_v_dim)).reshape(1, T, -1)
        return g.out_proj(o)

    def _attn_graphed(self, i, a, x, cos, sin, st, buf, k, L):
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
        kx = a.k_norm(kraw.view(1, T, -1, a.head_dim)).transpose(1, 2)   # [1, Hkv, T, D]
        v = vraw.view(1, T, -1, a.head_dim).transpose(1, 2)
        q, kx = mod.apply_rotary_pos_emb(q, kx, cos, sin)
        kc, vc = st.kv[i]
        rows = buf['slot'].repeat_interleave(L)                          # [T]
        cols = buf['pos']                                                # [T]
        kc[rows, :, cols] = kx[0].transpose(0, 1).to(kc.dtype)           # [T, Hkv, D]
        vc[rows, :, cols] = v[0].transpose(0, 1).to(vc.dtype)
        ks, vs = kc.index_select(0, buf['slot']), vc.index_select(0, buf['slot'])     # [k, Hkv, S, D]
        qs = q[0].view(q.shape[1], k, L, -1).transpose(0, 1)             # [k, Hq, L, D]
        S = st.max_len
        msk = (torch.arange(S, device=x.device).view(1, 1, 1, S) <= buf['pos'].view(k, 1, L, 1))
        if self._sdpa_gqa:
            o = F.scaled_dot_product_attention(qs, ks, vs, attn_mask=msk, scale=a.scaling,
                                               enable_gqa=a.num_key_value_groups > 1)
        else:
            o = F.scaled_dot_product_attention(qs, mod.repeat_kv(ks, a.num_key_value_groups),
                                               mod.repeat_kv(vs, a.num_key_value_groups),
                                               attn_mask=msk, scale=a.scaling)
        o = o.transpose(1, 2).reshape(1, T, -1) * torch.sigmoid(gate)   # [k, L, Hq*D] -> [1, T, .]
        return a.o_proj(o)

    def capture_prefill_graph(self, st, k, L):
        """Capture the (k, L) prefill graph. The warm-up runs write pool slots 0..k-1, so call
        this only while no request is live (engine warm-up), then reset the pool."""
        import torch
        key = (id(st), 'prefill', k, L)
        if key in self.graphs:
            return self.graphs[key]
        buf = self._pg_buffers(st, k, L)
        # warm-up (Triton autotune / compile) on scratch rows, then capture; the pool state the
        # warm-up touches is overwritten by every real call, which writes all it reads
        buf['slot'].copy_(torch.arange(k, device=self.device))
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.inference_mode(), torch.cuda.stream(stream):
            for _ in range(2):
                self._pg_forward(st, buf, k, L)
        torch.cuda.current_stream().wait_stream(stream)
        if getattr(self, '_pg_pool', None) is None:   # one pool for all prefill graphs
            self._pg_pool = torch.cuda.graph_pool_handle()
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode(), torch.cuda.graph(graph, pool=self._pg_pool):
            self._pg_forward(st, buf, k, L)
        # FLA memoises cu_seqlens-derived index tensors (chunk indices etc.) in small identity
        # caches (fla.utils tensor_cache, 4 entries). The capture hit those caches, so the graph
        # reads tensors it does not own; once evicted they were freed and reused, and replay
        # hit an illegal address (L4, v1.16 smoke). Keep every cached result alive with the graph.
        buf['_fla_keep'] = _fla_cached_tensors()
        self.graphs[key] = (graph, buf)
        self.stats['prefill_captures'] = self.stats.get('prefill_captures', 0) + 1
        return self.graphs[key]

    def prefill_graphed(self, st, slots, prompts, *, starts=None, capture=False):
        """prefill_packed for small cohorts through a captured graph. Same return contract
        (logits [k, V], hidden [T_new, H] of the real tokens, cu list). Returns None when the
        prompts do not fit a bucket, or the bucket has no graph and capture=False."""
        import torch
        slots, prompts = [int(s) for s in slots], [list(map(int, p)) for p in prompts]
        starts = [0] * len(prompts) if starts is None else [int(x) for x in starts]
        key = self.prefill_graph_key(st, prompts, starts)
        if key is None:
            return None
        k, L = key
        entry = self.graphs.get((id(st), 'prefill', k, L))
        if entry is None:
            if not capture:
                return None
            entry = self.capture_prefill_graph(st, k, L)
        graph, buf = entry
        self._pg_fill(buf, k, L, slots, prompts, starts)
        graph.replay()
        nnew = [len(p) - s0 for p, s0 in zip(prompts, starts)]
        cu = [0]
        for n in nnew:
            cu.append(cu[-1] + n)
        real = torch.cat([torch.arange(j * L, j * L + n, device=self.device) for j, n in enumerate(nnew)])
        self.stats['packed_prefills'] = self.stats.get('packed_prefills', 0) + 1
        self.stats['packed_rows'] = self.stats.get('packed_rows', 0) + k
        self.stats['packed_tokens'] = self.stats.get('packed_tokens', 0) + cu[-1]
        self.stats['prefill_replays'] = self.stats.get('prefill_replays', 0) + 1
        return buf['logits'].clone(), buf['hidden'].index_select(0, real), cu

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
