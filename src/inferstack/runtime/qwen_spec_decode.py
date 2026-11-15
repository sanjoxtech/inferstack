"""N6.1: lossless one-draft speculative decoding with Qwen3.5's own MTP head.

N6.0 (docs/verification/evidence/n60-mtp-acceptance-*) measured the checkpoint's MTP head on
scorecard prompts. Its first draft matches the greedy next token 86-96% of the time. This module
turns that head into a decode lane. Each step is one captured CUDA graph:

1. verify: the main model runs two tokens per row, [last accepted token x at p, draft d at p+1];
2. greedy: y0 = argmax at p (the true next token), y1 = argmax at p+1 (the token after d);
3. accept d iff y0 == d. The row emits [y0, y1] when accepted, else [y0]. Output is exactly
   greedy decoding (up to bf16 numerics of a 2-token forward vs a 1-token one);
4. commit the linear-attention state. GDN conv/recurrent state cannot be rolled back like a KV
   cache, so the verify pass saves the state after token 1 and restores it on rows that rejected
   (full-attention KV at p+1 is simply masked and overwritten next step);
5. draft: the MTP layer runs on both positions with its own KV cache. Entry q consumes
   (embedding of token q+1, final hidden state at q). It writes entries p and p+1 and takes the
   output of the last accepted position, then the shared lm_head gives the next draft;
6. the graph writes the next step's inputs itself (new last token, draft, position).

The host only reads y and accept to emit tokens. It never feeds tokens back.

MTP convention (from N6.0): h is the final hidden state after the final norm, and the MTP entry
for position q uses rotary position q. Head structure:
fc(concat(pre_fc_norm_embedding(emb), pre_fc_norm_hidden(h))) -> one full-attention decoder layer
-> norm -> the model's lm_head.

The checkpoint ships these 15 `mtp.*` tensors; transformers drops them at load
(`_keys_to_ignore_on_load_unexpected`). `load_mtp_tensors` reads them from the pinned snapshot.
"""
from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from inferstack.runtime.qwen_static_decode import QwenStaticDecoder, StaticDecodeState, pick_bucket

DRAFTS = 1   # tokens drafted per step (the verify pass runs DRAFTS + 1 tokens)


def load_mtp_tensors(repo_id: str, revision: str) -> dict:
    """The checkpoint's `mtp.*` tensors (dropped by transformers at load)."""
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    idx = json.loads(Path(hf_hub_download(repo_id, 'model.safetensors.index.json', revision=revision)).read_text())
    shards: dict = {}
    for k, f in idx['weight_map'].items():
        if k.startswith('mtp.'):
            shards.setdefault(f, []).append(k)
    out = {}
    for f, keys in shards.items():
        with safe_open(hf_hub_download(repo_id, f, revision=revision), framework='pt') as h:
            for k in keys:
                out[k] = h.get_tensor(k)
    if not out:
        raise RuntimeError(f'{repo_id}@{revision} has no mtp.* tensors')
    return out


class MTPHead:
    """The MTP layer as plain modules. embed_tokens, rotary and lm_head stay shared with the model."""

    def __init__(self, decoder: QwenStaticDecoder, tensors: dict | None = None, *, seed: int | None = None):
        import torch
        text, cfg = decoder.text, decoder.cfg
        full = next(i for i, t in enumerate(cfg.layer_types) if t == 'full_attention')
        # Fresh layer of the model's own class (forward fusion may have rewritten the live layers).
        self.layer = type(text.layers[full])(cfg, full)
        self.norm_e, self.norm_h, self.norm = (copy.deepcopy(text.norm) for _ in range(3))
        H = cfg.hidden_size
        self.fc = torch.nn.Linear(2 * H, H, bias=False)
        if tensors is not None:
            sd = {k[len('mtp.layers.0.'):]: v for k, v in tensors.items() if k.startswith('mtp.layers.0.')}
            missing, unexpected = self.layer.load_state_dict(sd, strict=False)
            missing = [m for m in missing if not m.endswith('inv_freq')]
            if missing or unexpected:
                raise RuntimeError(f'MTP layer mismatch: missing={missing} unexpected={unexpected}')
            for mod, key in ((self.norm_e, 'mtp.pre_fc_norm_embedding.weight'),
                             (self.norm_h, 'mtp.pre_fc_norm_hidden.weight'), (self.norm, 'mtp.norm.weight')):
                mod.weight.data.copy_(tensors[key])
            self.fc.weight.data.copy_(tensors['mtp.fc.weight'])
        else:   # tests: a random head (drafts are mostly wrong, which exercises the reject path)
            g = torch.Generator().manual_seed(0 if seed is None else seed)
            with torch.no_grad():
                for m in (self.layer, self.norm_e, self.norm_h, self.norm, self.fc):
                    for p in m.parameters():
                        p.add_(torch.randn(p.shape, generator=g) * 0.05)
        for m in (self.layer, self.norm_e, self.norm_h, self.norm, self.fc):
            m.to(device=decoder.device, dtype=decoder.dtype).eval()


@dataclass
class SpecState:
    """Per-slot speculative buffers, alongside a StaticDecodeState of the same pool."""
    tok: Any                                       # [slots, 2] (last accepted token, draft)
    pos: Any                                       # [slots] position of the last accepted token
    y: Any                                         # [slots, 2] greedy tokens from the verify pass
    accept: Any                                    # [slots] bool
    offs: Any                                      # [2] = (0, 1)
    mtp_kv: tuple = ()                             # (k, v) [slots, Hkv, max_len, D]
    conv_alt: dict = field(default_factory=dict)   # GDN state after verify token 1
    rec_alt: dict = field(default_factory=dict)


class QwenSpecDecoder(QwenStaticDecoder):
    """Static decode lane plus the N6.1 MTP verify/draft step. Plain `step` still works."""

    def __init__(self, model, device, dtype=None, gdn_backend='auto', attn_backend='sdpa', *,
                 mtp_tensors: dict | None = None, mtp_seed: int | None = None):
        super().__init__(model, device, dtype=dtype, gdn_backend=gdn_backend, attn_backend=attn_backend)
        self.mtp = MTPHead(self, mtp_tensors, seed=mtp_seed)
        self.spec_graphs: dict = {}
        self.stats.update(spec_captures=0, spec_replays=0, spec_eager_steps=0, spec_rows_loaded=0)

    # ---- buffers -----------------------------------------------------------------
    def new_spec_state(self, st: StaticDecodeState) -> SpecState:
        import torch
        dev, n = self.device, st.batch
        a = self.mtp.layer.self_attn
        shape = (n, self.cfg.num_key_value_heads, st.max_len, a.head_dim)
        sp = SpecState(tok=torch.zeros(n, 2, dtype=torch.long, device=dev),
                       pos=torch.zeros(n, dtype=torch.long, device=dev),
                       y=torch.zeros(n, 2, dtype=torch.long, device=dev),
                       accept=torch.zeros(n, dtype=torch.bool, device=dev),
                       offs=torch.arange(2, device=dev),
                       mtp_kv=(torch.zeros(shape, dtype=self.dtype, device=dev),
                               torch.zeros(shape, dtype=self.dtype, device=dev)))
        for i, t in st.conv.items():
            sp.conv_alt[i] = torch.zeros_like(t)
            sp.rec_alt[i] = torch.zeros_like(st.rec[i])
        return sp

    def reset_spec(self, st: StaticDecodeState, sp: SpecState):
        self.reset(st)
        for t in (sp.tok, sp.pos, sp.y, sp.accept, *sp.mtp_kv, *sp.conv_alt.values(), *sp.rec_alt.values()):
            t.zero_()

    # ---- multi-token pieces (T tokens per row) -----------------------------------
    def _attn_t(self, a, x, cos, sin, kc, vc, mask, pos, L, backend, skey):
        """Full attention for T consecutive tokens per row at positions pos..pos+T-1.
        kc/vc/mask are views of the rows in use. Writes K/V, then attends causally."""
        import torch
        import torch.nn.functional as F
        B, T, _ = x.shape
        D = a.head_dim
        if getattr(a, 'q_proj', None) is None:
            qraw, kraw, vraw = F.linear(x, a._inferstack_qkv_weight, a._inferstack_qkv_bias).split(
                a._inferstack_qkv_splits, dim=-1)
        else:
            qraw, kraw, vraw = a.q_proj(x), a.k_proj(x), a.v_proj(x)
        q, gate = torch.chunk(qraw.view(B, T, -1, D * 2), 2, dim=-1)
        gate = gate.reshape(B, T, -1)
        q = a.q_norm(q.reshape(B, T, -1, D)).transpose(1, 2)
        k = a.k_norm(kraw.view(B, T, -1, D)).transpose(1, 2)
        v = vraw.view(B, T, -1, D).transpose(1, 2)
        q, k = self.mod.apply_rotary_pos_emb(q, k, cos, sin)
        cols = pos[:, None] + torch.arange(T, device=x.device)                  # [B, T]
        idx = cols.view(B, 1, T, 1).expand(B, k.shape[1], T, D)
        kc.scatter_(2, idx, k.to(kc.dtype))
        vc.scatter_(2, idx, v.to(vc.dtype))
        if backend == 'triton':
            from inferstack.kernels.decode_attention import decode_attention, make_scratch
            if skey not in self._attn_scratch:
                # One scratch for every layer of the step: the launches run in order on one stream.
                self._attn_scratch[skey] = make_scratch(skey[2], kc.shape[1], q.shape[1] // kc.shape[1], D,
                                                        kc.shape[2], x.device)
            o = torch.stack([decode_attention(q[:, :, t].contiguous(), kc, vc, mask, pos + t, L, a.scaling,
                                              backend='triton', scratch=self._attn_scratch[skey])
                             for t in range(T)], dim=2)
        else:
            c = torch.arange(L, device=x.device)
            m = mask[:, None, None, :L] & (c[None, None, None, :] <= cols[:, None, :, None])
            keys, vals = kc[:, :, :L], vc[:, :, :L]
            if self._sdpa_gqa:
                o = F.scaled_dot_product_attention(q, keys, vals, attn_mask=m, scale=a.scaling,
                                                   enable_gqa=a.num_key_value_groups > 1)
            else:
                o = F.scaled_dot_product_attention(q, self.mod.repeat_kv(keys, a.num_key_value_groups),
                                                   self.mod.repeat_kv(vals, a.num_key_value_groups),
                                                   attn_mask=m, scale=a.scaling)
        o = o.transpose(1, 2).reshape(B, T, -1) * torch.sigmoid(gate)
        return a.o_proj(o)

    def _gdn_t(self, i, g, x, st, sp):
        """Gated DeltaNet over T tokens, state updated in place. Saves the state after the
        first token into sp.*_alt so rejected rows can be restored."""
        import torch
        import torch.nn.functional as F
        mod = self.mod
        B, T, _ = x.shape
        if getattr(g, 'in_proj_qkv', None) is None:
            qkv, z = F.linear(x, g._inferstack_qkvz_weight).split([g.conv_dim, g.value_dim], dim=-1)
            b, a = F.linear(x, g._inferstack_ba_weight).split([g.num_v_heads, g.num_v_heads], dim=-1)
        else:
            qkv, z = g.in_proj_qkv(x), g.in_proj_z(x)
            b, a = g.in_proj_b(x), g.in_proj_a(x)
        conv, rec = st.conv[i][:B], st.rec[i][:B]
        K = conv.shape[-1]
        w = g.conv1d.weight
        full = torch.cat([conv, qkv.transpose(1, 2).to(w.dtype)], dim=-1)            # [B, C, K+T]
        sp.conv_alt[i][:B].copy_(full[:, :, 1:K + 1])
        conv.copy_(full[:, :, T:])
        out = F.conv1d(full, w, g.conv1d.bias, padding=0, groups=full.shape[1])[:, :, -T:]
        out = mod.ACT2FN[g.activation](out).to(x.dtype).transpose(1, 2)            # [B, T, C]
        q, k, v = torch.split(out, [g.key_dim, g.key_dim, g.value_dim], dim=-1)
        q = q.reshape(B, T, -1, g.head_k_dim)
        k = k.reshape(B, T, -1, g.head_k_dim)
        v = v.reshape(B, T, -1, g.head_v_dim)
        beta = b.sigmoid()
        gate = -g.A_log.float().exp() * F.softplus(a.float() + g.dt_bias)
        outs = []
        if self.gdn_backend == 'triton':
            from inferstack.kernels.gdn_triton import inferstack_grouped_gdn_decode
            for t in range(T):
                o, _ = inferstack_grouped_gdn_decode(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], gate[:, t:t + 1],
                                                     beta[:, t:t + 1], initial_state=rec, output_final_state=False,
                                                     use_qk_l2norm_in_kernel=True)
                if t == 0:
                    sp.rec_alt[i][:B].copy_(rec)
                outs.append(o)
        else:
            rep = g.num_v_heads // g.num_k_heads
            if rep > 1:
                q = q.repeat_interleave(rep, dim=2)
                k = k.repeat_interleave(rep, dim=2)
            S = rec
            for t in range(T):
                qt, kt, vt = (u[:, t].to(torch.float32) for u in (q, k, v))
                bt, gt = beta[:, t].float(), gate[:, t]
                qt = mod.l2norm(qt, dim=-1, eps=1e-6) / (qt.shape[-1] ** 0.5)
                kt = mod.l2norm(kt, dim=-1, eps=1e-6)
                S = S * gt.exp()[..., None, None]
                delta = (vt - (S * kt.unsqueeze(-1)).sum(dim=-2)) * bt.unsqueeze(-1)
                S = S + kt.unsqueeze(-1) * delta.unsqueeze(-2)
                if t == 0:
                    sp.rec_alt[i][:B].copy_(S)
                outs.append((S * qt.unsqueeze(-1)).sum(dim=-2).to(x.dtype)[:, None])
            rec.copy_(S)
        o = torch.cat(outs, dim=1).reshape(-1, g.head_v_dim)
        o = g.norm(o, z.reshape(-1, g.head_v_dim)).reshape(B, T, -1)
        return g.out_proj(o)

    def _mtp_layer(self, x, cos, sin, sp, rows, mask, pos, L, backend, skey):
        lay = self.mtp.layer
        r = x
        h = r + self._attn_t(lay.self_attn, lay.input_layernorm(x), cos, sin, sp.mtp_kv[0][rows], sp.mtp_kv[1][rows],
                             mask, pos, L, backend, skey)
        return h + lay.mlp(lay.post_attention_layernorm(h))

    def _mtp_in(self, nxt, hidden):
        import torch
        m = self.mtp
        return m.fc(torch.cat([m.norm_e(self.text.embed_tokens(nxt)), m.norm_h(hidden)], dim=-1))

    # ---- the speculative step (static shapes; graph-capturable) ----------------------
    def _spec_step(self, st: StaticDecodeState, sp: SpecState, B: int, L: int):
        import torch
        tok, p = sp.tok[:B], sp.pos[:B]
        cols = p[:, None] + sp.offs                                       # [B, 2]
        mask = st.mask[:B]
        mask.scatter_(1, cols, True)
        h = self.text.embed_tokens(tok)                                   # [B, 2, H]
        cos, sin = self.text.rotary_emb(h, cols[None].expand(3, B, 2))
        skey = (id(st), 'spec', st.batch)
        for i, kind in enumerate(self.layer_types):
            layer = self.text.layers[i]
            r = h
            h = layer.input_layernorm(h)
            if kind == 'linear_attention':
                h = self._gdn_t(i, layer.linear_attn, h, st, sp)
            else:
                kc, vc = st.kv[i]
                h = self._attn_t(layer.self_attn, h, cos, sin, kc[:B], vc[:B], mask, p, L, self.attn_backend, skey)
            h = r + h
            h = h + layer.mlp(layer.post_attention_layernorm(h))
        h = self.text.norm(h)                                             # post-norm hidden [B, 2, H]
        y = self.lm_head(h).argmax(-1)                                    # [B, 2]
        acc = y[:, 0] == tok[:, 1]
        # Rows that rejected the draft keep the GDN state after token 1.
        keep = acc.view(B, 1, 1)
        for i in st.conv:
            st.conv[i][:B].copy_(torch.where(keep, st.conv[i][:B], sp.conv_alt[i][:B]))
            st.rec[i][:B].copy_(torch.where(keep.view(B, 1, 1, 1), st.rec[i][:B], sp.rec_alt[i][:B]))
        mask.scatter_(1, cols[:, 1:], acc[:, None])                       # rejected draft column is not real
        # MTP: entries p and p+1 = (token after, hidden at). Draft from the last accepted one.
        first = torch.where(acc, tok[:, 1], y[:, 0])
        nxt = torch.stack([first, y[:, 1]], dim=1)
        hm = self._mtp_layer(self._mtp_in(nxt, h), cos, sin, sp, slice(0, B), mask, p, L, self.attn_backend, skey)
        sel = torch.where(acc[:, None], hm[:, 1], hm[:, 0])
        draft = self.lm_head(self.mtp.norm(sel)).argmax(-1)
        sp.y[:B].copy_(y)
        sp.accept[:B].copy_(acc)
        sp.tok[:B].copy_(torch.stack([torch.where(acc, y[:, 1], y[:, 0]), draft], dim=1))
        sp.pos[:B].add_(1 + acc.long())

    # ---- public API ----------------------------------------------------------------
    def load_spec_row(self, st: StaticDecodeState, sp: SpecState, slot: int, prompt_ids, hidden, first_token: int):
        """After `load_rows` put a prompt of n tokens into `slot`: fill the MTP cache for
        positions 0..n-1 and the first draft. prompt_ids [n]; hidden [n, H] (post-final-norm
        hidden states of the prompt, from `prefill_with_hidden`); first_token = greedy token at n."""
        import torch
        n = int(prompt_ids.shape[-1])
        if not bool(st.mask[slot, :n].all()) or bool(st.mask[slot, n:].any()):
            raise ValueError('load_spec_row expects the packed layout written by load_rows')
        dev = self.device
        with torch.inference_mode():
            nxt = torch.cat([prompt_ids.view(-1)[1:].to(dev), torch.tensor([int(first_token)], device=dev)])[None]
            x = self._mtp_in(nxt, hidden.view(1, n, -1).to(device=dev, dtype=self.dtype))
            posn = torch.arange(n, device=dev)
            cos, sin = self.text.rotary_emb(x, posn.view(1, 1, n).expand(3, 1, n))
            hm = self._mtp_layer(x, cos, sin, sp, slice(slot, slot + 1), st.mask[slot:slot + 1],
                                 torch.zeros(1, dtype=torch.long, device=dev), n, 'sdpa', None)
            draft = self.lm_head(self.mtp.norm(hm[:, -1])).argmax(-1)
            sp.tok[slot, 0] = int(first_token)
            sp.tok[slot, 1] = draft[0]
            sp.pos[slot] = n
        self.stats['spec_rows_loaded'] += 1
        return int(draft[0])

    def capture_spec(self, st: StaticDecodeState, sp: SpecState, batches=None, lengths=None) -> int:
        """One graph per (batch bucket, length bucket); call while the pool is empty."""
        import torch
        if self.device.type != 'cuda':
            return 0
        if self._pool is None:
            self._pool = torch.cuda.graph_pool_handle()
        n = 0
        for B in (batches or st.batch_buckets):
            for L in (lengths or st.len_buckets):
                key = (id(st), int(B), int(L))
                if key in self.spec_graphs:
                    continue
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.inference_mode(), torch.cuda.stream(stream):
                    self._spec_step(st, sp, B, L)
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.inference_mode(), torch.cuda.graph(graph, pool=self._pool):
                    self._spec_step(st, sp, B, L)
                self.spec_graphs[key] = graph
                n += 1
        torch.cuda.synchronize()
        self.reset_spec(st, sp)
        self.stats['spec_captures'] += n
        return n

    def spec_step(self, st: StaticDecodeState, sp: SpecState, rows: int, *, length: int, use_graph: bool = True):
        """One verify+draft step for live slots 0..rows-1. length: 2 + the largest position
        (the host tracks it from `accept`). Returns views (y [rows, 2], accept [rows]): each
        row emits y[r, 0], plus y[r, 1] when accept[r]. Consume before the next step."""
        import torch
        if length > st.max_len:
            raise ValueError(f'verify column {length - 1} beyond pool max_len {st.max_len}')
        B = pick_bucket(st.batch_buckets, rows)
        L = pick_bucket(st.len_buckets, int(length))
        if B > rows:                      # padding rows: the graph advances them, keep them at column 0
            sp.tok[rows:B].zero_()
            sp.pos[rows:B].zero_()
        graph = self.spec_graphs.get((id(st), B, L)) if use_graph else None
        if graph is None:
            with torch.inference_mode():
                self._spec_step(st, sp, B, L)
            self.stats['spec_eager_steps'] += 1
        else:
            graph.replay()
            self.stats['spec_replays'] += 1
        return sp.y[:rows], sp.accept[:rows]

    def release(self, st: StaticDecodeState):
        super().release(st)
        for key in [k for k in self.spec_graphs if k[0] == id(st)]:
            self.spec_graphs.pop(key)


def prefill_with_hidden(model, input_ids):
    """Transformers prefill of one unpadded row: (cache, post-final-norm hidden [n, H], greedy next token)."""
    import torch
    inner = getattr(model, 'model', model)
    text = getattr(inner, 'language_model', inner)
    cap = {}
    hook = text.norm.register_forward_hook(lambda m, i, o: cap.update(h=o))
    try:
        with torch.inference_mode():
            out = model(input_ids=input_ids, use_cache=True, logits_to_keep=1)
    finally:
        hook.remove()
    return out.past_key_values, cap['h'][0], int(out.logits[0, -1].argmax(-1))
