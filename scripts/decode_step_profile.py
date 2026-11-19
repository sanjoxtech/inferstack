#!/usr/bin/env python
"""P10 diagnosis: how does one spec decode step scale with live rows, and which kernels take it?

Builds the engine's static spec lane from a serving profile (same decoder, pool and CUDA graphs as
`inferstack serve`), fills rows 0..B-1 with a realistic position, then measures:
  - graph-replay GPU time of one spec step for each batch bucket at each length bucket
  - one torch.profiler table of the eager step (CUDA kernels grouped by name) at --prof-rows
State values are random, so outputs are meaningless; only timing is measured.

  python scripts/decode_step_profile.py --profile configs/profiles/production-v1.35.0-s48.toml --out OUT/decode
"""
from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', default='configs/profiles/production-v1.35.0-s48.toml')
    ap.add_argument('--foundry-root', default='examples/model_foundry')
    ap.add_argument('--model-ref', default='foundry://factorylm-2b@candidate')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--iters', type=int, default=20)
    ap.add_argument('--positions', type=int, nargs='*', default=[200, 450, 1600])
    ap.add_argument('--prof-rows', type=int, nargs='*', default=[8, 32, 48])
    ap.add_argument('--prof-position', type=int, default=200)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    from inferstack.config import load_profile
    from inferstack.foundry import FileSystemFoundryRepository, FoundryModelLoader
    from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    cfg = load_profile(a.profile)
    _, model, tok, _ = FoundryModelLoader(FileSystemFoundryRepository(a.foundry_root), a.device).load(a.model_ref, cfg)
    eng = QwenHybridContinuousEngine(model, tok, cfg, a.device)
    eng._maybe_patch()
    eng._maybe_static_lane()
    lane = eng._lane
    assert lane is not None and lane.sp is not None, eng._static_diag
    dec, st, sp = lane.dec, lane.st, lane.sp
    g = torch.Generator(device=a.device).manual_seed(0)

    def fill(B, pos):
        dec.reset_spec(st, sp)
        st.mask[:B, :pos].fill_(True)
        sp.mtp_mask[:B, :pos].fill_(True)
        sp.pos[:B].fill_(pos)
        sp.tok[:B].copy_(torch.randint(100, 20000, (B, 2), device=a.device, generator=g))

    def time_step(B, pos, use_graph=True):
        ts = []
        for it in range(a.iters + 3):
            fill(B, pos)
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record()
            dec.spec_step(st, sp, B, length=pos + 2, use_graph=use_graph)
            e1.record()
            torch.cuda.synchronize()
            if it >= 3:
                ts.append(e0.elapsed_time(e1))
        return statistics.median(ts)

    rows = []
    buckets = sorted(set(int(b) for b in st.batch_buckets))
    for pos in a.positions:
        for B in buckets:
            if pos + 4 > st.max_len:
                continue
            r = {'rows': B, 'position': pos, 'graph_ms': time_step(B, pos)}
            r['ms_per_row'] = r['graph_ms'] / B
            print(json.dumps(r), flush=True)
            rows.append(r)

    tables = {}
    from torch.profiler import ProfilerActivity, profile
    for B in a.prof_rows:
        B = min(B, st.batch)
        fill(B, a.prof_position)
        dec.spec_step(st, sp, B, length=a.prof_position + 2, use_graph=False)   # warm the eager path
        fill(B, a.prof_position)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for _ in range(3):
                dec.spec_step(st, sp, B, length=a.prof_position + 2, use_graph=False)
            torch.cuda.synchronize()
        kern = {}
        for ev in prof.key_averages():
            t = getattr(ev, 'device_time_total', None)
            if t is None:
                t = getattr(ev, 'cuda_time_total', 0)
            if t and ev.device_type is not None and 'CUDA' in str(ev.device_type):
                kern[ev.key] = kern.get(ev.key, 0.0) + t / 3 / 1000.0
        top = sorted(kern.items(), key=lambda kv: -kv[1])[:30]
        tables[B] = {'eager_gpu_ms_sum': sum(kern.values()), 'top_kernels_ms': top}
        print(json.dumps({'prof_rows': B, 'eager_gpu_ms_sum': round(sum(kern.values()), 3)}), flush=True)
        for k, v in top[:15]:
            print(f'  {v:8.3f} ms  {k[:110]}', flush=True)
        (out / f'prof_rows{B}.txt').write_text(prof.key_averages().table(sort_by='cuda_time_total', row_limit=60))

    meta = {'profile': a.profile, 'slots': st.batch, 'max_len': st.max_len, 'batch_buckets': buckets,
            'len_buckets': list(st.len_buckets), 'gpu': torch.cuda.get_device_name(0),
            'layer_types': list(dec.layer_types), 'static_diag': eng._static_diag}
    (out / 'decode_step.json').write_text(json.dumps({'meta': meta, 'scaling': rows, 'kernels': tables}, indent=2, default=str))


if __name__ == '__main__':
    main()
