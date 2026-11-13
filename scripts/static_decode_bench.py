#!/usr/bin/env python
"""Decode-step benchmark: current native decode (Transformers modules, growing cache)
versus the static-buffer decode (eager and CUDA-graph replay), on the real model.

Measures model-step time only (no HTTP/scheduler), at several batch sizes, after the
same prefill. Also checks agreement: greedy tokens from the static path are compared
with the Transformers path step by step (teacher-forced on the reference tokens), and
the max logit difference is reported. Structural evidence, not a scored serving number.
"""
from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def prompts(n, words, seed=20261109):
    import importlib.util
    spec = importlib.util.spec_from_file_location('sd_engine_scorecard', ROOT / 'scripts' / 'engine_scorecard.py')
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc.prompts({'name': f'bench{words}', 'prompt_words': words}, n, seed)


def left_pad(tok_lists, pad, device):
    import torch
    w = max(map(len, tok_lists))
    ids = torch.full((len(tok_lists), w), pad, dtype=torch.long, device=device)
    mask = torch.zeros((len(tok_lists), w), dtype=torch.long, device=device)
    for i, t in enumerate(tok_lists):
        ids[i, w - len(t):] = torch.tensor(t, device=device)
        mask[i, w - len(t):] = 1
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    return ids, mask, pos, w


def bench_case(model, tok, decoders, batch, words, steps, bucket_len):
    import torch
    dev = decoders['static_graph'].device
    toks = [tok.encode(t, add_special_tokens=False) for t in prompts(batch, words)]
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    ids, mask, pos, w = left_pad(toks, pad, dev)
    lengths = torch.tensor([len(t) for t in toks], device=dev)

    def prefill():
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
        return out.past_key_values, out.logits[:, -1].argmax(-1)

    # Reference: current native decode path (Transformers modules + installed kernels).
    cache, nxt = prefill()
    m, p, ref_tokens, ref_logits, ref_ms = mask, lengths.clone(), [], [], []
    for s in range(steps):
        m = torch.cat([m, torch.ones((batch, 1), dtype=m.dtype, device=dev)], dim=1)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        with torch.inference_mode():
            out = model(input_ids=nxt[:, None], attention_mask=m, position_ids=p[:, None],
                        past_key_values=cache, use_cache=True, logits_to_keep=1)
        torch.cuda.synchronize(); ref_ms.append((time.perf_counter() - t0) * 1000)
        cache = out.past_key_values
        lg = out.logits[:, -1].float()
        ref_logits.append(lg); ref_tokens.append(nxt.clone())
        nxt = lg.argmax(-1); p = p + 1
    ref_tokens.append(nxt.clone())

    results = {}
    for mode in ('static_eager', 'static_graph_torch_gdn', 'static_graph'):
        dec = decoders[mode]
        cache, first = prefill()
        st = dec.new_state(batch, bucket_len)
        dec.load(st, cache, mask, w)
        p = lengths.clone(); ms = []; agree = 0; max_diff = 0.0
        for s in range(steps):
            feed = ref_tokens[s]  # teacher-forced on the reference so step s compares like with like
            torch.cuda.synchronize(); t0 = time.perf_counter()
            lg = dec.step(st, feed, p, w + s, use_graph=(mode != 'static_eager'))
            torch.cuda.synchronize(); ms.append((time.perf_counter() - t0) * 1000)
            lg = lg.float()
            agree += int((lg.argmax(-1) == ref_logits[s].argmax(-1)).sum())
            max_diff = max(max_diff, float((lg - ref_logits[s]).abs().max()))
            p = p + 1
        dec.release(st)
        warm = ms[3:] if len(ms) > 6 else ms  # first graph step includes capture
        results[mode] = {'gdn_backend': dec.gdn_backend, 'ms_per_step_median': statistics.median(warm), 'ms_first_step': ms[0],
                         'greedy_agreement': agree / (steps * batch), 'max_abs_logit_diff': max_diff}
    warm_ref = ref_ms[3:] if len(ref_ms) > 6 else ref_ms
    results['native_current'] = {'ms_per_step_median': statistics.median(warm_ref)}
    return {'batch': batch, 'prompt_words': words, 'prompt_tokens_max': w, 'steps': steps, **results,
            'speedup_graph_vs_current': results['native_current']['ms_per_step_median'] / results['static_graph']['ms_per_step_median']}


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', default='configs/profiles/production-v1.6.0-baseline.toml')
    ap.add_argument('--foundry-root', default='examples/model_foundry')
    ap.add_argument('--model-ref', default='foundry://factorylm-2b@candidate')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--batches', type=int, nargs='+', default=[1, 4, 16])
    ap.add_argument('--words', type=int, nargs='+', default=[48, 400])
    ap.add_argument('--steps', type=int, default=64)
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
    eng._maybe_patch()  # same kernels the served native profile installs
    decoders = {'static_eager': QwenStaticDecoder(model, a.device, gdn_backend='torch'),
                'static_graph_torch_gdn': QwenStaticDecoder(model, a.device, gdn_backend='torch'),
                'static_graph': QwenStaticDecoder(model, a.device, gdn_backend='auto')}
    import traceback
    rows = []
    for words in a.words:
        for b in a.batches:
            try:
                bucket = 256 * (-(-(int(words * 1.4) + a.steps + 16) // 256))
                row = bench_case(model, tok, decoders, b, words, a.steps, bucket)
            except Exception as exc:
                row = {'batch': b, 'prompt_words': words, 'error': f'{type(exc).__name__}: {str(exc)[:400]}',
                       'traceback': traceback.format_exc()[-2000:]}
                torch.cuda.empty_cache()
            print(json.dumps(row), flush=True)
            rows.append(row)
    out = {'version': 1, 'profile': a.profile, 'gpu': torch.cuda.get_device_name(0),
           'gdn_kernel': eng.diagnostics().get('gdn_kernel'), 'decoder_stats': {k: d.stats for k, d in decoders.items()}, 'rows': rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2, default=str) + '\n')
    eng._gpu_executor.shutdown(wait=False, cancel_futures=True)
    print('Saved', a.out)


if __name__ == '__main__':
    main()
