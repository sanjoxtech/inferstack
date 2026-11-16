#!/usr/bin/env python
"""P4 probe: can FLA's varlen GDN kernels live inside a CUDA graph?

P4 profiling showed a small packed prefill is host-bound (GPU busy ~21 ms of ~68 ms wall; FLA's
chunk_gated_delta_rule costs ~1.2 ms of Python per layer). A CUDA graph would remove that, but
only if the kernels neither sync to the host nor read cu_seqlens on the host while capturing.

Checks, on one GDN-sized problem:
  1. host reads: grep FLA's chunk / conv / index helpers for .tolist() / .item() / .cpu().
  2. capture: torch.cuda.graph around causal_conv1d + chunk_gated_delta_rule with static
     q/k/v/g/beta/cu_seqlens/initial_state buffers.
  3. replay with NEW contents in the same buffers (different segment split, same total T):
     max |diff| against eager on the new contents. A graph that baked cu_seqlens in at capture
     shows a large diff here.
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    import fla
    out = {'fla_version': getattr(fla, '__version__', '?')}
    root = Path(fla.__file__).parent
    hits = []
    for rel in ('ops/gated_delta_rule/chunk.py', 'ops/common/chunk_delta_h.py', 'ops/common/chunk_o.py',
                'ops/common/chunk_scaled_dot_kkt.py', 'ops/utils/index.py', 'ops/utils/solve_tril.py',
                'ops/utils/cumsum.py', 'modules/causal_conv1d.py', 'ops/gated_delta_rule/wy_fast.py',
                'modules/l2norm.py', 'ops/utils/op.py'):
        p = root / rel
        if not p.exists():
            hits.append({'file': rel, 'missing': True})
            continue
        for n, line in enumerate(p.read_text().splitlines(), 1):
            if re.search(r'\.tolist\(\)|\.item\(\)|\.cpu\(\)|cache', line):
                hits.append({'file': rel, 'line': n, 'text': line.strip()[:160]})
    out['host_read_lines'] = hits

    from fla.modules.causal_conv1d import causal_conv1d
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    dev, dt = 'cuda', torch.bfloat16
    T, H, Hv, dk, dv, W = 192, 16, 16, 128, 128, 4
    C = 2 * H * dk + Hv * dv
    torch.manual_seed(0)
    w = torch.randn(C, W, device=dev, dtype=dt) * 0.3
    s_qkv = torch.randn(1, T, C, device=dev, dtype=dt)
    s_g = -torch.rand(1, T, Hv, device=dev) * 0.5
    s_beta = torch.rand(1, T, Hv, device=dev, dtype=dt)
    s_conv0 = torch.randn(4, C, W, device=dev, dtype=dt)
    s_rec0 = torch.randn(4, Hv, dk, dv, device=dev) * 0.1
    s_cu = torch.tensor([0, 48, 96, 144, 192], dtype=torch.int32, device=dev)

    def run(qkv, g, beta, conv0, rec0, cu):
        mixed, cf = causal_conv1d(qkv, weight=w, bias=None, initial_state=conv0, output_final_state=True,
                                  activation='silu', backend='triton', cu_seqlens=cu)
        q, k, v = torch.split(mixed, [H * dk, H * dk, Hv * dv], dim=-1)
        o, rf = chunk_gated_delta_rule(q.reshape(1, T, H, dk), k.reshape(1, T, H, dk), v.reshape(1, T, Hv, dv),
                                       g=g, beta=beta, scale=dk ** -0.5, initial_state=rec0,
                                       output_final_state=True, use_qk_l2norm_in_kernel=True, cu_seqlens=cu)
        return o, cf, rf

    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.inference_mode():
            for _ in range(3):
                run(s_qkv, s_g, s_beta, s_conv0, s_rec0, s_cu)
        torch.cuda.current_stream().wait_stream(s)
        graph = torch.cuda.CUDAGraph()
        with torch.inference_mode():
            with torch.cuda.graph(graph):
                g_o, g_cf, g_rf = run(s_qkv, s_g, s_beta, s_conv0, s_rec0, s_cu)
        out['capture'] = 'ok'
        results = []
        for split in ([0, 48, 96, 144, 192], [0, 10, 100, 101, 192], [0, 1, 2, 3, 192]):
            with torch.inference_mode():
                s_qkv.copy_(torch.randn_like(s_qkv))
                s_cu.copy_(torch.tensor(split, dtype=torch.int32, device=dev))
                graph.replay()
                e_o, e_cf, e_rf = run(s_qkv, s_g, s_beta, s_conv0, s_rec0, s_cu)
            torch.cuda.synchronize()
            results.append({'split': split, 'o': float((g_o - e_o).abs().max()),
                            'conv_final': float((g_cf - e_cf).abs().max()),
                            'rec_final': float((g_rf - e_rf).abs().max())})
        out['replay_vs_eager'] = results
        # timing: eager vs replay
        import time
        for name, fn in (('eager', lambda: run(s_qkv, s_g, s_beta, s_conv0, s_rec0, s_cu)), ('replay', graph.replay)):
            with torch.inference_mode():
                for _ in range(3):
                    fn()
                torch.cuda.synchronize()
                t0 = time.perf_counter()
                for _ in range(20):
                    fn()
                torch.cuda.synchronize()
            out[f'{name}_ms'] = (time.perf_counter() - t0) * 1e3 / 20
    except Exception as e:  # noqa: BLE001
        out['capture'] = f'failed: {type(e).__name__}: {e}'[:600]
    print(json.dumps(out, indent=2))
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')


if __name__ == '__main__':
    main()
