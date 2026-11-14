#!/usr/bin/env python3
"""E-N3-2/E-N3-3 on a GPU: GDN prefill backends against the per-token oracle, and timing.

For the model's real GDN geometry (read from its config) and a grid of batch x tokens,
runs every available prefill backend of inferstack.kernels.gdn and reports:

  * max abs / relative error of output and final state vs `recurrent_reference` (FP32
    per-token oracle on the same BF16 inputs; checked on the smaller shapes),
  * median CUDA ms per call (after warmup) for chunk_reference and fla_chunk.

No crossover is assumed: the timing table is the evidence for the qualification of
`gdn_prefill_backend` per GPU and shape.

  python scripts/gdn_backend_check.py --model Qwen/Qwen3.5-2B-Base --out gdn-check.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))


def geometry(model, revision):
    from transformers import AutoConfig
    c = AutoConfig.from_pretrained(model, revision=revision)
    c = getattr(c, 'text_config', c)
    return {'Hk': c.linear_num_key_heads, 'Hv': c.linear_num_value_heads,
            'K': c.linear_key_head_dim, 'V': c.linear_value_head_dim,
            'gdn_layers': sum(1 for t in getattr(c, 'layer_types', []) if t == 'linear_attention')}


def make(B, T, geo, dtype, seed):
    import torch
    gen = torch.Generator(device='cuda').manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen, device='cuda')  # noqa: E731
    q = r(B, T, geo['Hk'], geo['K']).to(dtype)
    k = r(B, T, geo['Hk'], geo['K']).to(dtype)
    v = r(B, T, geo['Hv'], geo['V']).to(dtype)
    g = -torch.rand(B, T, geo['Hv'], generator=gen, device='cuda') * 0.3
    beta = torch.rand(B, T, geo['Hv'], generator=gen, device='cuda').to(dtype)
    s0 = r(B, geo['Hv'], geo['K'], geo['V']) * 0.05
    return q, k, v, g, beta, s0


def err(a, b):
    a, b = a.float(), b.float()
    d = (a - b).abs()
    return {'max_abs': d.max().item(), 'max_rel': (d.max() / b.abs().max().clamp_min(1e-6)).item()}


def time_ms(fn, iters):
    import torch
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    out = []
    for _ in range(iters):
        a, b = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        a.record(); fn(); b.record(); torch.cuda.synchronize()
        out.append(a.elapsed_time(b))
    return statistics.median(out)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--model', default='Qwen/Qwen3.5-2B-Base')
    ap.add_argument('--revision', default='b1485b2fa6dfa1287294f269f5fb618e03d52d7c')
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16])
    ap.add_argument('--tokens', type=int, nargs='+', default=[64, 128, 512, 2048])
    ap.add_argument('--oracle-max-tokens', type=int, default=512)
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--out')
    args = ap.parse_args()
    import torch
    from inferstack.kernels import gdn as G
    geo = geometry(args.model, args.revision)
    backends = ['chunk_reference'] + (['fla_chunk'] if G.fla_available() else [])
    report = {'gpu': torch.cuda.get_device_name(0), 'torch': torch.__version__, 'geometry': geo,
              'chunk_reference_impl': G.diagnostics()['chunk_reference_impl'],
              'fla_available': G.fla_available(), 'fla_pin': G.FLA_PIN, 'rows': []}
    print(json.dumps({k: report[k] for k in ('gpu', 'geometry', 'fla_available', 'chunk_reference_impl')}), flush=True)
    for B in args.batches:
        for T in args.tokens:
            q, k, v, g, beta, s0 = make(B, T, geo, torch.bfloat16, seed=B * 10007 + T)
            row = {'B': B, 'T': T}
            ref = None
            if T <= args.oracle_max_tokens and B * T <= 4 * args.oracle_max_tokens:
                ref = G.gdn(q, k, v, g, beta, s0, backend='recurrent_reference')
            for name in backends:
                try:
                    o, s = G.gdn(q, k, v, g, beta, s0, backend=name)
                    if ref is not None:
                        row[f'{name}_out'] = err(o, ref[0])
                        row[f'{name}_state'] = err(s, ref[1])
                    row[f'{name}_ms'] = time_ms(lambda: G.gdn(q, k, v, g, beta, s0, backend=name), args.iters)
                except Exception as exc:  # recorded, not hidden
                    row[f'{name}_error'] = f'{type(exc).__name__}: {exc}'[:300]
            if 'chunk_reference_ms' in row and 'fla_chunk_ms' in row:
                row['speedup_fla'] = row['chunk_reference_ms'] / row['fla_chunk_ms']
            report['rows'].append(row)
            print(json.dumps(row), flush=True)
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
