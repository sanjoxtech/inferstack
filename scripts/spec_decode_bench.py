#!/usr/bin/env python
"""N6.1 bench: MTP speculative decode lane vs the plain static decode lane, on the real model.

Both lanes run CUDA-graph replay in one QwenSpecDecoder after the same per-row prefill, and
generate N greedy tokens per row (free-running, not teacher-forced). Reports per case:

- step time (median ms), tokens emitted per row per step (1 + acceptance), output tok/s;
- acceptance (draft accepted / steps);
- agreement with the plain lane: rows whose first N tokens are identical, and the mean index
  of the first difference where they differ (bf16 2-token vs 1-token forward numerics).

Model-step evidence, not a scored serving number.

  python scripts/spec_decode_bench.py --profile configs/profiles/production-v1.13.0.toml --out spec.json
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = 'Qwen/Qwen3.5-2B-Base'
REVISION = 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c'


def prompts(n, words, seed=20261109):
    import importlib.util
    spec = importlib.util.spec_from_file_location('spec_engine_scorecard', ROOT / 'scripts' / 'engine_scorecard.py')
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc.prompts({'name': f'bench{words}', 'prompt_words': words}, n, seed)


def prefill_rows(model, dec, st, sp, toks):
    import torch
    from inferstack.runtime.qwen_spec_decode import prefill_with_hidden
    firsts = []
    for slot, t in enumerate(toks):
        ids = torch.tensor([t], device=dec.device)
        cache, hidden, first = prefill_with_hidden(model, ids)
        dec.load_rows(st, cache, torch.ones_like(ids), ids.shape[1], [0], [slot])
        if sp is not None:
            dec.load_spec_row(st, sp, slot, ids[0], hidden, first)
        firsts.append(first)
        del cache
    return firsts


def plain_lane(model, dec, toks, n, max_len):
    import torch
    B = len(toks)
    st = dec.new_state(B, max_len, batch_buckets=[B])
    dec.capture(st)
    firsts = prefill_rows(model, dec, st, None, toks)
    out = [[f] for f in firsts]
    ids = torch.tensor(firsts, device=dec.device)
    pos = torch.tensor([len(t) for t in toks], device=dec.device)
    longest = int(pos.max())
    ms = []
    for s in range(n - 1):
        torch.cuda.synchronize(); t0 = time.perf_counter()
        lg = dec.step(st, ids, pos, B, length=longest + s + 1)
        ids = lg.argmax(-1)
        host = ids.tolist()
        ms.append((time.perf_counter() - t0) * 1000)
        for r in range(B):
            out[r].append(host[r])
        pos = pos + 1
    dec.release(st)
    del st
    return out, ms


def spec_lane(model, dec, toks, n, max_len):
    import torch
    B = len(toks)
    st = dec.new_state(B, max_len, batch_buckets=[B])
    sp = dec.new_spec_state(st)
    t0 = time.perf_counter()
    graphs = dec.capture_spec(st, sp)
    capture_s = time.perf_counter() - t0
    firsts = prefill_rows(model, dec, st, sp, toks)
    out = [[f] for f in firsts]
    pos = [len(t) for t in toks]
    ms, per_step, accepts = [], [], 0
    steps = 0
    while min(map(len, out)) < n:
        torch.cuda.synchronize(); t0 = time.perf_counter()
        y, acc = dec.spec_step(st, sp, B, length=max(pos) + 2)
        yh, ah = y.tolist(), acc.tolist()
        ms.append((time.perf_counter() - t0) * 1000)
        emitted = 0
        for r in range(B):
            out[r].append(yh[r][0])
            emitted += 1
            if ah[r]:
                out[r].append(yh[r][1])
                emitted += 1
                accepts += 1
            pos[r] += 1 + int(ah[r])
        per_step.append(emitted)
        steps += 1
    dec.release(st)
    del st, sp
    return out, ms, per_step, accepts / (steps * B), graphs, capture_s


def bench_case(model, tok, dec, batch, words, n):
    import torch
    toks = [tok.encode(t, add_special_tokens=False) for t in prompts(batch, words)]
    max_len = 256 * (-(-(max(map(len, toks)) + n + 8) // 256))
    plain, pms = plain_lane(model, dec, toks, n, max_len)
    torch.cuda.empty_cache()
    spec, sms, per_step, acceptance, graphs, capture_s = spec_lane(model, dec, toks, n, max_len)
    torch.cuda.empty_cache()
    same, first_diff = 0, []
    for a, b in zip(plain, spec):
        a, b = a[:n], b[:n]
        if a == b:
            same += 1
        else:
            first_diff.append(next(i for i, (x, y) in enumerate(zip(a, b)) if x != y))
    warm_p = pms[3:] if len(pms) > 6 else pms
    warm_s = sms[3:] if len(sms) > 6 else sms
    plain_tps = batch * len(pms) / (sum(pms) / 1000)
    spec_tps = sum(per_step) / (sum(sms) / 1000)
    return {'batch': batch, 'prompt_words': words, 'prompt_tokens_max': max(map(len, toks)), 'new_tokens': n,
            'max_len': max_len,
            'plain': {'ms_per_step_median': statistics.median(warm_p), 'tok_s': plain_tps},
            'spec': {'ms_per_step_median': statistics.median(warm_s), 'tok_s': spec_tps, 'steps': len(sms),
                     'tokens_per_row_step': sum(per_step) / (len(sms) * batch), 'acceptance': acceptance,
                     'graphs': graphs, 'capture_s': capture_s},
            'speedup_tok_s': spec_tps / plain_tps,
            'agreement': {'rows_identical': same, 'rows': batch,
                          'mean_first_diff': (sum(first_diff) / len(first_diff)) if first_diff else None}}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', default='configs/profiles/production-v1.13.0.toml')
    ap.add_argument('--foundry-root', default='examples/model_foundry')
    ap.add_argument('--model-ref', default='foundry://factorylm-2b@candidate')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16])
    ap.add_argument('--words', type=int, nargs='+', default=[48, 400])
    ap.add_argument('--new-tokens', type=int, default=128)
    ap.add_argument('--attn', default=None, help='sdpa|triton (default: the profile static_decode_attention)')
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    from inferstack.config import load_profile
    from inferstack.foundry import FileSystemFoundryRepository, FoundryModelLoader
    from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine
    from inferstack.runtime.qwen_spec_decode import QwenSpecDecoder, load_mtp_tensors
    cfg = load_profile(a.profile)
    _, model, tok, _ = FoundryModelLoader(FileSystemFoundryRepository(a.foundry_root), a.device).load(a.model_ref, cfg)
    eng = QwenHybridContinuousEngine(model, tok, cfg, a.device)
    eng._maybe_patch()   # the kernels the served profile installs
    attn = a.attn or str(getattr(cfg.native, 'static_decode_attention', 'sdpa') or 'sdpa')
    dec = QwenSpecDecoder(model, a.device, gdn_backend='auto', attn_backend=attn,
                          mtp_tensors=load_mtp_tensors(REPO_ID, REVISION))
    rows = []
    for words in a.words:
        for b in a.batches:
            try:
                row = bench_case(model, tok, dec, b, words, a.new_tokens)
            except Exception as exc:
                row = {'batch': b, 'prompt_words': words, 'error': f'{type(exc).__name__}: {str(exc)[:400]}',
                       'traceback': traceback.format_exc()[-2500:]}
                torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            rows.append(row)
    out = {'version': 1, 'profile': a.profile, 'attn_backend': attn, 'gdn_backend': dec.gdn_backend,
           'gpu': torch.cuda.get_device_name(0), 'model': f'{REPO_ID}@{REVISION}',
           'decoder_stats': dec.stats, 'rows': rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str) + '\n')
    eng._gpu_executor.shutdown(wait=False, cancel_futures=True)
    print('Saved', a.out)


if __name__ == '__main__':
    main()
