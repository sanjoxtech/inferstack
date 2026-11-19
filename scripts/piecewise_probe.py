#!/usr/bin/env python
"""P13 S0: would a piecewise-graphed mixed prefill+decode step beat today's separate steps on L4?

No engine change. On the served lane (profile v1.36: 48 slots, spec decode, lazy banks), measures:

  decode_ms[B]       today's whole-step spec graph, B live rows (position 200)
  prefill_ms[k, L]   today's separate packed prefill of k prompts x L tokens (graphed when a bucket
                     fits, as served), wall and GPU
  dense_ms[N]        every dense op of one step (norms, projections, MLP; all layers) on N tokens:
                     one graph, and as 25 piece graphs replayed in turn (the piecewise launch cost)
  pmix[P]            the prefill-side mixers on P tokens split in rows of 64: per GDN layer FLA varlen
                     conv + chunk delta rule, per attention layer the per-row SDPA of _attn_packed;
                     GPU time and host time (time to return, no sync)

and projects, for B decode rows plus a P-token prefill cohort:

  mixed_gpu  = dense_ms[2B + P] (pieces) + (decode_ms[B] - dense_ms[2B]) + pmix_gpu[P]
  host       = 25 piece replays + decode mixer launches (measured: decode graph host time is ~0) + pmix_host[P]
  separate   = decode_ms[B] + prefill_ms[cohort]
  gate (plan): mixed_gpu <= decode_ms[B] + 0.6 * prefill_ms[cohort]  and  host < mixed_gpu

  python scripts/piecewise_probe.py --profile configs/profiles/production-v1.36.0.toml --out OUT/probe
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', default='configs/profiles/production-v1.36.0.toml')
    ap.add_argument('--foundry-root', default='examples/model_foundry')
    ap.add_argument('--model-ref', default='foundry://factorylm-2b@candidate')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--decode-rows', type=int, nargs='*', default=[16, 32, 48])
    ap.add_argument('--prefill-tokens', type=int, nargs='*', default=[64, 128, 256, 512])
    ap.add_argument('--row-tokens', type=int, default=64)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    import torch.nn.functional as F
    from inferstack.config import load_profile
    from inferstack.foundry import FileSystemFoundryRepository, FoundryModelLoader
    from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine
    from inferstack.runtime.qwen_static_decode import pick_bucket
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    dev = torch.device(a.device)
    cfg = load_profile(a.profile)
    _, model, tok, _ = FoundryModelLoader(FileSystemFoundryRepository(a.foundry_root), a.device).load(a.model_ref, cfg)
    eng = QwenHybridContinuousEngine(model, tok, cfg, a.device)
    eng._maybe_patch()
    eng._maybe_static_lane()
    lane = eng._lane
    dec, st, sp = lane.dec, lane.st, lane.sp
    text, H = dec.text, dec.text.norm.weight.shape[0]
    g = torch.Generator(device=dev).manual_seed(0)
    res = {'meta': {'profile': a.profile, 'gpu': torch.cuda.get_device_name(0), 'slots': st.batch,
                    'layer_types': list(dec.layer_types), 'static_diag': eng._static_diag}}

    def gpu_ms(fn, iters=a.iters):
        """Median GPU time of fn() with events, after 3 warm-ups."""
        ts = []
        for it in range(iters + 3):
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); fn(); e1.record()
            torch.cuda.synchronize()
            if it >= 3:
                ts.append(e0.elapsed_time(e1))
        return statistics.median(ts)

    def host_ms(fn, iters=a.iters):
        """Median time for fn() to return (no sync inside), GPU drained before each call."""
        ts = []
        for it in range(iters + 3):
            torch.cuda.synchronize()
            t0 = time.perf_counter(); fn(); t1 = time.perf_counter()
            torch.cuda.synchronize()
            if it >= 3:
                ts.append((t1 - t0) * 1000)
        return statistics.median(ts)

    # ---- separate prefill (as served: graphed when a bucket fits) ------------------------
    def prompts(k, L):
        return [torch.randint(100, 20000, (L,), generator=g, device=dev).tolist() for _ in range(k)]
    prefill = {}
    for P in a.prefill_tokens:
        k = max(1, P // a.row_tokens)
        L = P // k
        ps = prompts(k, L)
        key = dec.prefill_graph_key(st, ps)
        with torch.inference_mode():
            if key:
                dec.capture_prefill_graph(st, *key)

            def run():
                if not (key and dec.prefill_graphed(st, list(range(k)), ps) is not None):
                    dec.prefill_packed(st, list(range(k)), ps)
            prefill[P] = {'rows': k, 'tokens_per_row': L, 'graphed': bool(key), 'gpu_ms': gpu_ms(run),
                          'wall_ms': host_ms(lambda: (run(), torch.cuda.synchronize()))}
        print(json.dumps({'prefill': P, **prefill[P]}), flush=True)
    dec.reset_spec(st, sp)
    res['prefill'] = prefill

    # ---- today's decode step --------------------------------------------------------------
    decode = {}
    for B in a.decode_rows:
        def fill():
            dec.reset_spec(st, sp)
            st.mask[:B, :200].fill_(True)
            sp.mtp_mask[:B, :200].fill_(True)
            sp.pos[:B].fill_(200)
            sp.tok[:B].copy_(torch.randint(100, 20000, sp.tok[:B].shape, device=dev, generator=g))
        ts, hs = [], []
        for it in range(a.iters + 3):
            fill(); torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter(); e0.record()
            dec.spec_step(st, sp, B, length=202)
            e1.record(); t1 = time.perf_counter(); torch.cuda.synchronize()
            if it >= 3:
                ts.append(e0.elapsed_time(e1)); hs.append((t1 - t0) * 1000)
        decode[B] = {'gpu_ms': statistics.median(ts), 'host_ms': statistics.median(hs)}
        # eager step: its host time bounds the decode-side launches a mixed step would make outside graphs
        Bb = pick_bucket(st.batch_buckets, B)
        Lb = pick_bucket(st.len_buckets, 202)

        def eager():
            with torch.inference_mode():
                dec._spec_step(st, sp, Bb, Lb)
        fill()
        decode[B]['eager_host_ms'] = host_ms(eager, iters=5)
        decode[B]['eager_gpu_ms'] = gpu_ms(eager, iters=5)
        print(json.dumps({'decode_rows': B, **decode[B]}), flush=True)
    dec.reset_spec(st, sp)
    res['decode'] = decode

    # ---- dense ops on N tokens: one graph vs 25 piece graphs ------------------------------
    layers = [(i, kind, text.layers[i]) for i, kind in enumerate(dec.layer_types)]

    def dense_in(layer, kind, h):
        x = layer.input_layernorm(h)
        if kind == 'linear_attention':
            gm = layer.linear_attn
            if getattr(gm, 'in_proj_qkv', None) is None:
                return F.linear(x, gm._inferstack_qkvz_weight), F.linear(x, gm._inferstack_ba_weight)
            return gm.in_proj_qkv(x), gm.in_proj_z(x), gm.in_proj_b(x), gm.in_proj_a(x)
        at = layer.self_attn
        if getattr(at, 'q_proj', None) is None:
            return (F.linear(x, at._inferstack_qkv_weight, at._inferstack_qkv_bias),)
        return at.q_proj(x), at.k_proj(x), at.v_proj(x)

    def dense_out(layer, kind, h, o):
        mix = layer.linear_attn.out_proj(o) if kind == 'linear_attention' else layer.self_attn.o_proj(o)
        h = h + mix
        return h + layer.mlp(layer.post_attention_layernorm(h))

    def o_width(layer, kind):
        return layer.linear_attn.out_proj.in_features if kind == 'linear_attention' else layer.self_attn.o_proj.in_features

    dense = {}
    Ns = sorted({2 * B for B in a.decode_rows} | {2 * B + P for B in a.decode_rows for P in a.prefill_tokens})
    for N in Ns:
        h0 = torch.randn(1, N, H, device=dev, dtype=dec.dtype, generator=g) * 0.1
        os_ = {i: torch.randn(1, N, o_width(l, k), device=dev, dtype=dec.dtype, generator=g) * 0.1 for i, k, l in layers}
        buf = {'h': h0.clone()}

        def whole():
            h = buf['h']
            for i, kind, layer in layers:
                dense_in(layer, kind, h)
                h = dense_out(layer, kind, h, os_[i])
            buf['out'] = h

        def piece(j):   # piece j: dense_out of layer j-1 (if any) + dense_in of layer j (if any)
            def f():
                h = buf['h']
                if j > 0:
                    i, kind, layer = layers[j - 1]
                    h = dense_out(layer, kind, h, os_[i])
                if j < len(layers):
                    i, kind, layer = layers[j]
                    dense_in(layer, kind, h)
                buf['p'] = h
            return f
        with torch.inference_mode():
            graphs = {}
            for name, fn in [('whole', whole)] + [(f'p{j}', piece(j)) for j in range(len(layers) + 1)]:
                s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s):
                    fn(); fn()
                torch.cuda.current_stream().wait_stream(s)
                gr = torch.cuda.CUDAGraph()
                with torch.cuda.graph(gr):
                    fn()
                graphs[name] = gr
            pieces = [graphs[f'p{j}'] for j in range(len(layers) + 1)]
            dense[N] = {'whole_gpu_ms': gpu_ms(graphs['whole'].replay),
                        'pieces_gpu_ms': gpu_ms(lambda: [p.replay() for p in pieces]),
                        'pieces_host_ms': host_ms(lambda: [p.replay() for p in pieces]),
                        'pieces': len(pieces)}
            del graphs, pieces
        torch.cuda.empty_cache()
        print(json.dumps({'dense_tokens': N, **dense[N]}), flush=True)
    res['dense'] = dense

    # ---- prefill-side mixers on P tokens (eager, FLA varlen) -------------------------------
    from fla.modules.causal_conv1d import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    pmix = {}
    for P in a.prefill_tokens:
        k = max(1, P // a.row_tokens)
        L = P // k
        cu = torch.tensor([j * L for j in range(k + 1)], dtype=torch.int32, device=dev)
        ins = {}
        for i, kind, layer in layers:
            if kind == 'linear_attention':
                gm = layer.linear_attn
                ins[i] = dict(qkv=torch.randn(1, P, gm.conv_dim, device=dev, dtype=dec.dtype, generator=g),
                              gate=-torch.rand(1, P, gm.num_v_heads, device=dev, generator=g),
                              beta=torch.rand(1, P, gm.num_v_heads, device=dev, dtype=dec.dtype, generator=g))
            else:
                at = layer.self_attn
                Hq = dec.cfg.num_attention_heads
                Hk = dec.cfg.num_key_value_heads
                ins[i] = dict(q=torch.randn(1, Hq, P, at.head_dim, device=dev, dtype=dec.dtype, generator=g),
                              k=torch.randn(1, Hk, P, at.head_dim, device=dev, dtype=dec.dtype, generator=g),
                              v=torch.randn(1, Hk, P, at.head_dim, device=dev, dtype=dec.dtype, generator=g))

        def mixers():
            for i, kind, layer in layers:
                x = ins[i]
                if kind == 'linear_attention':
                    gm = layer.linear_attn
                    w = gm.conv1d.weight.squeeze(1)
                    mixed, _ = causal_conv1d(x['qkv'], weight=w, bias=gm.conv1d.bias, initial_state=None,
                                             output_final_state=True, activation=gm.activation, backend='triton',
                                             cu_seqlens=cu)
                    q, kk, v = torch.split(mixed, [gm.key_dim, gm.key_dim, gm.value_dim], dim=-1)
                    chunk_gated_delta_rule(q.reshape(1, P, -1, gm.head_k_dim), kk.reshape(1, P, -1, gm.head_k_dim),
                                           v.reshape(1, P, -1, gm.head_v_dim), g=x['gate'], beta=x['beta'],
                                           scale=gm.head_k_dim ** -0.5, initial_state=None, output_final_state=True,
                                           use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
                else:
                    at = layer.self_attn
                    for j in range(k):
                        sl = slice(j * L, (j + 1) * L)
                        F.scaled_dot_product_attention(x['q'][:, :, sl], x['k'][:, :, sl], x['v'][:, :, sl],
                                                       is_causal=True, scale=at.scaling,
                                                       enable_gqa=at.num_key_value_groups > 1)
        with torch.inference_mode():
            pmix[P] = {'rows': k, 'gpu_ms': gpu_ms(mixers), 'host_ms': host_ms(mixers)}
        print(json.dumps({'prefill_mixers_tokens': P, **pmix[P]}), flush=True)
    res['prefill_mixers'] = pmix

    # ---- projection ---------------------------------------------------------------------------
    proj = []
    for B in a.decode_rows:
        Bb = pick_bucket(st.batch_buckets, B)
        dec_mix = decode[B]['gpu_ms'] - dense[2 * B]['pieces_gpu_ms']
        for P in a.prefill_tokens:
            mixed = dense[2 * B + P]['pieces_gpu_ms'] + dec_mix + pmix[P]['gpu_ms']
            # conservative: the whole eager step's host time stands in for the decode mixers' launches
            host = dense[2 * B + P]['pieces_host_ms'] + pmix[P]['host_ms'] + decode[B]['eager_host_ms']
            separate = decode[B]['gpu_ms'] + prefill[P]['gpu_ms']
            r = {'decode_rows': B, 'bucket': Bb, 'prefill_tokens': P, 'mixed_gpu_ms': mixed, 'host_ms_est': host,
                 'separate_gpu_ms': separate, 'saving_ms': separate - mixed,
                 'gate_ms': decode[B]['gpu_ms'] + 0.6 * prefill[P]['gpu_ms'],
                 'pass': bool(mixed <= decode[B]['gpu_ms'] + 0.6 * prefill[P]['gpu_ms'] and host < mixed)}
            proj.append(r)
            print(json.dumps({'projection': True, **{k2: (round(v, 2) if isinstance(v, float) else v) for k2, v in r.items()}}),
                  flush=True)
    res['projection'] = proj
    (out / 'piecewise_probe.json').write_text(json.dumps(res, indent=2, default=str))


if __name__ == '__main__':
    main()
