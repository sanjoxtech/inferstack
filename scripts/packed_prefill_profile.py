#!/usr/bin/env python
"""P4 diagnosis: where does a small packed prefill spend its time?

For a set of packed-prefill shapes (rows x new tokens, optionally continuing from a restored
prefix), measure on the real model:
  wall_ms  : host wall time of one prefill_packed call (synchronised at both ends)
  gpu_ms   : CUDA-event time of the same call
  cpu_bound: wall_ms - gpu_ms is large when Python/launch overhead dominates
and one torch.profiler table (top CPU ops and top CUDA kernels) for the c1 cache-hit shape.
Also checks whether FLA's varlen causal_conv1d accepts per-sequence initial_state, which
would let the continuation path drop its per-segment conv loop.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', default='configs/profiles/production-v1.15.0.toml')
    ap.add_argument('--foundry-root', default='examples/model_foundry')
    ap.add_argument('--model-ref', default='foundry://factorylm-2b@candidate')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--iters', type=int, default=10)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    from inferstack.config import load_profile
    from inferstack.foundry import FileSystemFoundryRepository, FoundryModelLoader
    from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine
    from inferstack.runtime.qwen_static_decode import QwenStaticDecoder
    cfg = load_profile(a.profile)
    _, model, tok, _ = FoundryModelLoader(FileSystemFoundryRepository(a.foundry_root), a.device).load(a.model_ref, cfg)
    eng = QwenHybridContinuousEngine(model, tok, cfg, a.device)
    eng._maybe_patch()
    n = cfg.native
    dec = QwenStaticDecoder(model, a.device, gdn_backend=str(n.static_decode_gdn_backend),
                            attn_backend=str(getattr(n, 'static_decode_attention', 'sdpa') or 'sdpa'))
    st = dec.new_state(16, 4096)
    g = torch.Generator().manual_seed(0)

    def toks(m):
        return torch.randint(100, 20000, (m,), generator=g).tolist()

    prefix = toks(1152)
    # (name, prompts, starts)
    shapes = [
        ('c1_short_40', [toks(40)], None),
        ('c1_med_300', [toks(300)], None),
        ('c1_long_1200', [toks(1200)], None),
        ('c1_hit_1152+48', [prefix + toks(48)], [1152]),
        ('c4_short_40', [toks(40) for _ in range(4)], None),
        ('c4_hit_1152+48', [prefix + toks(48) for _ in range(4)], [1152] * 4),
        ('c8_hit_1152+48', [prefix + toks(48) for _ in range(8)], [1152] * 8),
        ('c8_med_300', [toks(300) for _ in range(8)], None),
    ]
    rows = []
    for name, prompts, starts in shapes:
        slots = list(range(len(prompts)))
        if starts:   # put the prefix state into the slots once
            with torch.inference_mode():
                dec.prefill_packed(st, slots, [p[:starts[0]] for p in prompts])
        walls, gpus = [], []
        for it in range(a.iters + 2):
            if starts:   # restore prefix columns' state is already there; GDN state must be reset
                with torch.inference_mode():
                    dec.prefill_packed(st, slots, [p[:starts[0]] for p in prompts])
            torch.cuda.synchronize()
            e0, e1 = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            e0.record()
            with torch.inference_mode():
                dec.prefill_packed(st, slots, prompts, starts=starts)
            e1.record()
            torch.cuda.synchronize()
            if it >= 2:
                walls.append((time.perf_counter() - t0) * 1e3)
                gpus.append(e0.elapsed_time(e1))
        row = {'shape': name, 'rows': len(prompts), 'new_tokens': sum(len(p) - (s or 0) for p, s in
                                                                        zip(prompts, starts or [0] * len(prompts))),
               'wall_ms': statistics.median(walls), 'gpu_ms': statistics.median(gpus)}
        row['host_gap_ms'] = row['wall_ms'] - row['gpu_ms']
        print(json.dumps(row), flush=True)
        rows.append(row)

    # decode step for scale (graph replay, B=1 and 8)
    steps = {}
    for B in (1, 8):
        ids = torch.ones(B, dtype=torch.long, device=a.device)
        pos = torch.full((B,), 1200, device=a.device)
        with torch.inference_mode():
            for _ in range(3):
                dec.step(st, ids, pos, rows=B, length=1300)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        with torch.inference_mode():
            for _ in range(20):
                dec.step(st, ids, pos, rows=B, length=1300)
        torch.cuda.synchronize()
        steps[f'decode_step_B{B}_ms'] = (time.perf_counter() - t0) * 1e3 / 20
    print(json.dumps(steps), flush=True)

    # torch.profiler on the c1 cache-hit shape
    from torch.profiler import ProfilerActivity, profile
    prompts, starts = [prefix + toks(48)], [1152]
    with torch.inference_mode():
        dec.prefill_packed(st, [0], [prefix])
    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        with torch.inference_mode():
            dec.prefill_packed(st, [0], prompts, starts=starts)
        torch.cuda.synchronize()
    ka = prof.key_averages()
    cpu_top = ka.table(sort_by='self_cpu_time_total', row_limit=25)
    cuda_top = ka.table(sort_by='self_cuda_time_total', row_limit=25)
    n_kernels = sum(e.count for e in ka if e.device_type.name == 'CUDA')
    print(cpu_top)
    print(cuda_top)

    # FLA varlen conv with per-sequence initial_state: equal to per-segment calls?
    conv_check = {}
    try:
        from fla.modules.causal_conv1d import causal_conv1d
        D, W = 2048, 4
        lens = [7, 50, 3]
        x = torch.randn(1, sum(lens), D, device=a.device, dtype=torch.bfloat16)
        w = torch.randn(D, W, device=a.device, dtype=torch.bfloat16)
        init = torch.randn(len(lens), D, W, device=a.device, dtype=torch.bfloat16)
        cu = torch.tensor([0, 7, 57, 60], dtype=torch.int32, device=a.device)
        y, f = causal_conv1d(x, weight=w, bias=None, initial_state=init, output_final_state=True,
                             activation='silu', backend='triton', cu_seqlens=cu)
        ys, fs = [], []
        for j in range(len(lens)):
            yj, fj = causal_conv1d(x[:, cu[j]:cu[j + 1]], weight=w, bias=None, initial_state=init[j:j + 1],
                                   output_final_state=True, activation='silu', backend='triton')
            ys.append(yj)
            fs.append(fj)
        conv_check = {'ok': True, 'y_maxdiff': float((y - torch.cat(ys, 1)).abs().max()),
                      'final_maxdiff': float((f - torch.cat(fs, 0)).abs().max())}
    except Exception as e:  # noqa: BLE001
        conv_check = {'ok': False, 'error': f'{type(e).__name__}: {e}'[:400]}
    print(json.dumps({'fla_varlen_conv_initial_state': conv_check}), flush=True)

    out = {'version': 1, 'gpu': torch.cuda.get_device_name(0), 'iters': a.iters, 'shapes': rows,
           'decode': steps, 'profile_cuda_kernel_launches': n_kernels, 'conv_varlen_init': conv_check,
           'profile_cpu_top': cpu_top, 'profile_cuda_top': cuda_top}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    eng._gpu_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
