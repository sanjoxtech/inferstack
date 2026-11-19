#!/usr/bin/env python
"""P12 diagnosis: how close do the decode step's GEMMs get to the L4 memory-bandwidth floor?

Records every F.linear call (weight shape, rows M) of one eager spec step of the served lane at
--rows live rows, then times each distinct (weight shape, M) with CUDA-graph replay (no launch
overhead) under several GEMM back ends:
  cublas      : torch default
  cublaslt    : torch.backends.cuda.preferred_blas_library('cublaslt')
  autotune    : torch.compile(mode='max-autotune-no-cudagraphs') of F.linear (Triton templates)
Reports per back end the summed GEMM time of one step and the achieved weight-read bandwidth.
A back end that wins here still has to win end to end (scorecard A/B).

  python scripts/gemm_bw_bench.py --profile configs/profiles/production-v1.36.0.toml --out OUT/gemm
"""
from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', default='configs/profiles/production-v1.36.0.toml')
    ap.add_argument('--foundry-root', default='examples/model_foundry')
    ap.add_argument('--model-ref', default='foundry://factorylm-2b@candidate')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--rows', type=int, nargs='*', default=[1, 16, 32, 48])
    ap.add_argument('--position', type=int, default=200)
    ap.add_argument('--iters', type=int, default=50)
    ap.add_argument('--backends', nargs='*', default=['cublas', 'cublaslt', 'autotune'])
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
    cfg = load_profile(a.profile)
    _, model, tok, _ = FoundryModelLoader(FileSystemFoundryRepository(a.foundry_root), a.device).load(a.model_ref, cfg)
    eng = QwenHybridContinuousEngine(model, tok, cfg, a.device)
    eng._maybe_patch()
    eng._maybe_static_lane()
    lane = eng._lane
    dec, st, sp = lane.dec, lane.st, lane.sp

    def record(B):
        calls = Counter()
        orig = F.linear

        def spy(x, w, b=None):
            calls[(tuple(w.shape), int(x.numel() // x.shape[-1]), str(w.dtype))] += 1
            return orig(x, w, b)
        dec.reset_spec(st, sp)
        st.mask[:B, :a.position].fill_(True)
        sp.mtp_mask[:B, :a.position].fill_(True)
        sp.pos[:B].fill_(a.position)
        F.linear = spy
        torch.nn.functional.linear = spy
        try:
            with torch.inference_mode():
                dec._spec_step(st, sp, pick_bucket(st.batch_buckets, B), pick_bucket(st.len_buckets, a.position + 2))
        finally:
            F.linear = orig
            torch.nn.functional.linear = orig
        torch.cuda.synchronize()
        return calls

    def graph_time(fn, x):
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                fn(x)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            for _ in range(10):
                fn(x)
        ts = []
        for _ in range(a.iters):
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            e0.record(); g.replay(); e1.record(); torch.cuda.synchronize()
            ts.append(e0.elapsed_time(e1) / 10)
        return statistics.median(ts)

    results = {}
    compiled = {}
    for B in a.rows:
        calls = record(B)
        rows = []
        for (wshape, M, dt), n in sorted(calls.items(), key=lambda kv: -kv[0][0][0] * kv[0][0][1] * kv[1]):
            w = torch.randn(wshape, device=a.device, dtype=torch.bfloat16) * 0.02
            x = torch.randn(M, wshape[1], device=a.device, dtype=torch.bfloat16)
            r = {'w': list(wshape), 'M': M, 'calls': n, 'mb': w.numel() * 2 / 1e6}
            for be in a.backends:
                try:
                    if be == 'cublas':
                        torch.backends.cuda.preferred_blas_library('cublas')
                        fn = lambda t, w=w: F.linear(t, w)   # noqa: E731
                    elif be == 'cublaslt':
                        torch.backends.cuda.preferred_blas_library('cublaslt')
                        fn = lambda t, w=w: F.linear(t, w)   # noqa: E731
                    else:
                        torch.backends.cuda.preferred_blas_library('cublas')
                        key = (tuple(wshape), M)
                        if key not in compiled:
                            compiled[key] = torch.compile(lambda t, w=w: F.linear(t, w), mode='max-autotune-no-cudagraphs',
                                                          dynamic=False)
                            compiled[key](x)
                        fn = compiled[key]
                    ms = graph_time(fn, x)
                    r[be] = {'ms': ms, 'gbs': r['mb'] / ms}
                except Exception as exc:   # noqa: BLE001
                    r[be] = {'error': f'{type(exc).__name__}: {str(exc)[:200]}'}
            torch.backends.cuda.preferred_blas_library('cublas')
            rows.append(r)
            print(json.dumps({'rows': B, **r}), flush=True)
        tot = {be: sum(r[be]['ms'] * r['calls'] for r in rows if 'ms' in r.get(be, {})) for be in a.backends}
        best = sum(min(r[be]['ms'] for be in a.backends if 'ms' in r.get(be, {})) * r['calls'] for r in rows)
        mb = sum(r['mb'] * r['calls'] for r in rows)
        results[B] = {'gemms': rows, 'step_gemm_ms': tot, 'step_gemm_ms_best_per_shape': best, 'weight_mb_per_step': mb}
        print(json.dumps({'rows': B, 'step_gemm_ms': {k: round(v, 3) for k, v in tot.items()},
                          'best_per_shape': round(best, 3), 'weight_mb': round(mb, 1),
                          'floor_ms_at_300gbs': round(mb / 300, 3)}), flush=True)
    (out / 'gemm_bw.json').write_text(json.dumps({'gpu': torch.cuda.get_device_name(0), 'profile': a.profile,
                                                 'results': results}, indent=2, default=str))


if __name__ == '__main__':
    main()
