#!/usr/bin/env python
"""N2: where does native prefill time go on the prefill_heavy workload?

Diagnosis only. It changes no source and produces no scored numbers: it loads the
native engine in-process (like scripts/v160_operator_profiler.py), feeds it the exact
prefill_heavy prompts the scorecard uses, and drives the prefill path directly, so
scheduler queueing is excluded by construction.

Modes
  phases  For each (profile, cohort size, chunk size, prefix cache on/off) runs a full
          prefill and splits its wall time into model forward, prefix-cache snapshot
          and the remainder (sampling, accept, Python). Chunk size and prefix cache are
          the engine's own config knobs, varied here only to measure their cost.
  ops     torch.profiler operator table + Chrome trace for one prefill.

The L4 scorecard (docs/verification/COLAB_L4_SCORECARD_2026-11-13.md) gives the served
TTFT; served TTFT minus the pure prefill time measured here is time spent waiting in
the scheduler.
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scorecard_prompts(case_name='prefill_heavy', n=1):
    """The scorecard's own deterministic prompts, so this matches what was measured."""
    spec = importlib.util.spec_from_file_location('n2_engine_scorecard', ROOT / 'scripts' / 'engine_scorecard.py')
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    wl = sc.DEFAULT_WORKLOAD
    case = next(c for c in wl['cases'] if c['name'] == case_name)
    return sc.prompts(case, n, wl['seed']), case


def _sync_fn(device):
    if str(device).startswith('cuda'):
        import torch
        return torch.cuda.synchronize
    return lambda: None


def set_prefix_cache(eng, enabled):
    """Fresh (empty) prefix cache, or a disabled one, using the engine's own cache class."""
    from inferstack.runtime.hybrid_prefix_cache import HybridPrefixStateCache
    n = eng.cfg.native
    eng._prefix_cache = HybridPrefixStateCache(
        n.prefix_state_cache_max_entries if enabled else 0, n.prefix_cache_min_tokens,
        n.prefix_cache_block_tokens, n.prefix_state_cache_max_mb * 1024 * 1024)


def measure_prefill(eng, token_lists, *, chunk, prefix_cache, max_new_tokens=32):
    """Run one cohort's prefill to completion and split its wall time.

    Returns ms totals: wall, model (forward calls), snapshot (prefix-cache puts),
    other (= wall - model - snapshot: prepare, sampling, accept, Python).
    """
    from inferstack.core.request import Request, SamplingParams

    sync = _sync_fn(eng.device)
    eng.cfg.native.prefill_chunk_tokens = int(chunk)
    set_prefix_cache(eng, prefix_cache)
    timers = {'model': 0.0, 'snapshot': 0.0, 'model_calls': 0, 'snapshots': 0}
    orig_call, orig_store = eng._model_call, eng._maybe_store_prefix

    def timed_call(**kw):
        sync()
        t = time.perf_counter()
        out = orig_call(**kw)
        sync()
        timers['model'] += (time.perf_counter() - t) * 1000
        timers['model_calls'] += 1
        return out

    def timed_store(job):
        before = eng._stats['prefix_block_snapshots']
        sync()
        t = time.perf_counter()
        orig_store(job)
        sync()
        timers['snapshot'] += (time.perf_counter() - t) * 1000
        timers['snapshots'] += eng._stats['prefix_block_snapshots'] - before

    eng._model_call, eng._maybe_store_prefix = timed_call, timed_store
    try:
        stamp = time.time_ns()
        cohort = [(Request(f'n2-{stamp}-{i}', tuple(ids), SamplingParams(max_new_tokens=max_new_tokens, temperature=0.0)),
                   asyncio.Queue()) for i, ids in enumerate(token_lists)]
        sync()
        t0 = time.perf_counter()
        job = eng._prepare_prefill(cohort)
        while not job.done:
            eng._advance_prefill(job, 1)
        sync()
        wall = (time.perf_counter() - t0) * 1000
    finally:
        eng._model_call, eng._maybe_store_prefix = orig_call, orig_store
    return {'wall_ms': wall, 'model_ms': timers['model'], 'snapshot_ms': timers['snapshot'],
            'other_ms': max(0.0, wall - timers['model'] - timers['snapshot']),
            'model_calls': timers['model_calls'], 'snapshots': timers['snapshots'],
            'prompt_tokens': [len(x) for x in token_lists]}


def median_rows(rows):
    keys = ('wall_ms', 'model_ms', 'snapshot_ms', 'other_ms')
    out = {k: statistics.median(r[k] for r in rows) for k in keys}
    out.update(min_wall_ms=min(r['wall_ms'] for r in rows), max_wall_ms=max(r['wall_ms'] for r in rows),
               model_calls=rows[0]['model_calls'], snapshots=rows[0]['snapshots'],
               prompt_tokens_mean=statistics.mean(rows[0]['prompt_tokens']), repeats=len(rows))
    return out


def run_phases(eng, tok, *, batches, chunks, repeats):
    texts, case = scorecard_prompts(n=max(batches))
    ids = [tok.encode(t, add_special_tokens=False) for t in texts]
    results = []
    for b in batches:
        for chunk in chunks:
            for pc in (True, False):
                row = {'batch': b, 'chunk': chunk, 'prefix_cache': pc}
                try:
                    measure_prefill(eng, ids[:b], chunk=chunk, prefix_cache=pc, max_new_tokens=case['max_tokens'])  # warmup
                    row.update(median_rows([measure_prefill(eng, ids[:b], chunk=chunk, prefix_cache=pc,
                                                            max_new_tokens=case['max_tokens']) for _ in range(repeats)]))
                except Exception as exc:  # e.g. CUDA OOM at a large cohort/chunk: record, keep going
                    row['error'] = f'{type(exc).__name__}: {str(exc)[:300]}'
                    _free_cuda()
                print(json.dumps(row), flush=True)
                results.append(row)
    return results


def _free_cuda():
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception:
        pass


def run_ops(eng, tok, *, batch, chunk, prefix_cache, trace_path, shapes=False):
    import torch
    from torch.profiler import ProfilerActivity, profile
    texts, case = scorecard_prompts(n=batch)
    ids = [tok.encode(t, add_special_tokens=False) for t in texts]
    measure_prefill(eng, ids, chunk=chunk, prefix_cache=prefix_cache, max_new_tokens=case['max_tokens'])  # warmup
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=shapes) as prof:
        timing = measure_prefill(eng, ids, chunk=chunk, prefix_cache=prefix_cache, max_new_tokens=case['max_tokens'])
        torch.cuda.synchronize()
    rows = []
    for e in prof.key_averages(group_by_input_shape=shapes):
        dev = float(getattr(e, 'self_device_time_total', None) or getattr(e, 'self_cuda_time_total', 0.0) or 0.0)
        cpu = float(getattr(e, 'self_cpu_time_total', 0.0) or 0.0)
        if dev <= 0 and cpu <= 0:
            continue
        rows.append({'name': e.key, 'count': int(e.count), 'self_device_us': dev, 'self_cpu_us': cpu,
                     **({'input_shapes': str(e.input_shapes)[:300]} if shapes else {})})
    total_dev = sum(r['self_device_us'] for r in rows) or 1.0
    rows.sort(key=lambda r: r['self_device_us'], reverse=True)
    for r in rows:
        r['self_device_pct'] = 100.0 * r['self_device_us'] / total_dev
    Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
    prof.export_chrome_trace(str(trace_path))
    return {'batch': batch, 'chunk': chunk, 'prefix_cache': prefix_cache, 'timing': timing,
            'total_self_device_ms': total_dev / 1000, 'top_device_ops': rows[:40],
            'top_cpu_ops': sorted(rows, key=lambda r: r['self_cpu_us'], reverse=True)[:20], 'trace': str(trace_path)}


def load_engine(profile, foundry_root, model_ref, device):
    from inferstack.config import load_profile
    from inferstack.foundry import FileSystemFoundryRepository, FoundryModelLoader
    from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine
    cfg = load_profile(profile)
    _, model, tok, _ = FoundryModelLoader(FileSystemFoundryRepository(foundry_root), device).load(model_ref, cfg)
    eng = QwenHybridContinuousEngine(model, tok, cfg, device)
    eng._maybe_patch()
    return eng, tok


def main():
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument('mode', choices=['phases', 'ops'])
    p.add_argument('--profile', default='configs/profiles/production-v1.6.0-baseline.toml')
    p.add_argument('--foundry-root', default='examples/model_foundry')
    p.add_argument('--model-ref', default='foundry://factorylm-2b@candidate')
    p.add_argument('--device', default='cuda:0')
    p.add_argument('--batches', type=int, nargs='+', default=[1, 4])
    p.add_argument('--chunks', type=int, nargs='+', default=[128, 512, 2048])
    p.add_argument('--repeats', type=int, default=3)
    p.add_argument('--batch', type=int, default=1, help='ops: cohort size')
    p.add_argument('--chunk', type=int, default=128, help='ops: chunk size')
    p.add_argument('--no-prefix-cache', action='store_true', help='ops: disable prefix cache')
    p.add_argument('--shapes', action='store_true', help='ops: group the operator table by input shapes')
    p.add_argument('--trace', default='results/n2/prefill-trace.json')
    p.add_argument('--out', required=True)
    a = p.parse_args()
    import torch
    if a.device.startswith('cuda') and not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    eng, tok = load_engine(a.profile, a.foundry_root, a.model_ref, a.device)
    ident = {'profile': a.profile, 'model_ref': a.model_ref, 'device': a.device,
             'gpu': torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None,
             'forward_fusion': eng.diagnostics().get('forward_fusion'), 'gdn_kernel': eng.diagnostics().get('gdn_kernel')}
    if a.mode == 'phases':
        body = {'rows': run_phases(eng, tok, batches=a.batches, chunks=a.chunks, repeats=a.repeats)}
    else:
        body = run_ops(eng, tok, batch=a.batch, chunk=a.chunk, prefix_cache=not a.no_prefix_cache, trace_path=a.trace, shapes=a.shapes)
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({'version': 1, 'mode': a.mode, 'identity': ident, **body}, indent=2, default=str) + '\n')
    eng._gpu_executor.shutdown(wait=False, cancel_futures=True)
    print(f'Saved {out}')


if __name__ == '__main__':
    main()
