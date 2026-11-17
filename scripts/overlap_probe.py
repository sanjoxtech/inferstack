#!/usr/bin/env python
"""P7 step 1: can decode keep running while a long prefill runs on the same L4?

Decode: 16 live rows (slots 0..15), graph-replayed `dec.step` on the main stream.
Prefill: packed prefill of K long prompts into top slots (16..), on a second CUDA stream,
launched from a second Python thread so its host work does not block the decode loop.

Reports (median ms):
  decode_alone      : one decode step, nothing else running
  prefill_alone     : the packed prefill, nothing else running
  overlap           : decode steps completed while the prefill ran, their step times, and the
                      prefill's wall time when sharing the GPU
  sequential_stall  : what decode rows see today (one prefill blocks every row for its duration)

If decode steps stay well under the 40 ms SLO v2 per-token limit during the prefill and the
prefill slows by < ~1.5x, intra-device overlap (docs/P7_PREFILL_DECODE_OVERLAP_PLAN.md) is worth
building.
"""
from __future__ import annotations

import argparse
import json
import statistics
import threading
import time
from pathlib import Path


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', default='configs/profiles/production-v1.18.0.toml')
    ap.add_argument('--foundry-root', default='examples/model_foundry')
    ap.add_argument('--model-ref', default='foundry://factorylm-2b@candidate')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--rows', type=int, default=16)
    ap.add_argument('--prompt-tokens', type=int, default=2000)
    ap.add_argument('--prefill-rows', type=int, nargs='*', default=[1, 2])
    ap.add_argument('--reps', type=int, default=5)
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
    R, K = a.rows, max(a.prefill_rows)
    st = dec.new_state(R + K, 4096)
    g = torch.Generator().manual_seed(0)

    def toks(m):
        return torch.randint(100, 20000, (m,), generator=g).tolist()

    with torch.inference_mode():   # live decode rows: 16 x 1200-token contexts
        for j in range(R):
            dec.prefill_packed(st, [j], [toks(1200)])
    ids = torch.ones(R, dtype=torch.long, device=a.device)
    pos0 = torch.full((R,), 1200, device=a.device)
    L = 1200 + 400

    def step(i):
        dec.step(st, ids, pos0 + i, rows=R, length=L)

    with torch.inference_mode():   # capture/warm the decode graph and the prefill shapes
        for i in range(5):
            step(i)
        for k in a.prefill_rows:
            dec.prefill_packed(st, list(range(R, R + k)), [toks(a.prompt_tokens) for _ in range(k)])
    torch.cuda.synchronize()

    def time_decode(nsteps=30):
        ts = []
        with torch.inference_mode():
            for i in range(nsteps):
                t = time.perf_counter()
                step(i)
                torch.cuda.current_stream().synchronize()
                ts.append((time.perf_counter() - t) * 1e3)
        return ts

    res = {'gpu': torch.cuda.get_device_name(0), 'rows': R, 'prompt_tokens': a.prompt_tokens,
           'decode_alone_ms': statistics.median(time_decode()), 'cases': []}
    side = torch.cuda.Stream()
    for k in a.prefill_rows:
        slots = list(range(R, R + k))
        alone, walls, steps_during, step_ms = [], [], [], []
        for _ in range(a.reps):
            prompts = [toks(a.prompt_tokens) for _ in range(k)]
            torch.cuda.synchronize()
            t = time.perf_counter()
            with torch.inference_mode():
                dec.prefill_packed(st, slots, prompts)
            torch.cuda.synchronize()
            alone.append((time.perf_counter() - t) * 1e3)
        for _ in range(a.reps):
            prompts = [toks(a.prompt_tokens) for _ in range(k)]
            done = threading.Event()
            box = {}

            def run_prefill():
                with torch.inference_mode(), torch.cuda.stream(side):
                    t0 = time.perf_counter()
                    dec.prefill_packed(st, slots, prompts)
                    side.synchronize()
                    box['ms'] = (time.perf_counter() - t0) * 1e3
                done.set()
            torch.cuda.synchronize()
            th = threading.Thread(target=run_prefill)
            th.start()
            ts = []
            i = 0
            with torch.inference_mode():
                while not done.is_set():
                    t = time.perf_counter()
                    step(i)
                    torch.cuda.current_stream().synchronize()
                    ts.append((time.perf_counter() - t) * 1e3)
                    i += 1
            th.join()
            walls.append(box['ms'])
            steps_during.append(len(ts))
            step_ms += ts
        row = {'prefill_rows': k, 'prefill_alone_ms': statistics.median(alone),
               'overlap_prefill_ms': statistics.median(walls), 'overlap_decode_steps': statistics.median(steps_during),
               'overlap_decode_step_ms_p50': statistics.median(step_ms) if step_ms else None,
               'overlap_decode_step_ms_p95': sorted(step_ms)[int(0.95 * (len(step_ms) - 1))] if step_ms else None,
               'sequential_stall_ms': statistics.median(alone)}
        row['prefill_slowdown'] = row['overlap_prefill_ms'] / row['prefill_alone_ms']
        print(json.dumps(row), flush=True)
        res['cases'].append(row)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(res, indent=2) + '\n')
    print(json.dumps({'decode_alone_ms': res['decode_alone_ms']}), flush=True)
    eng._gpu_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
