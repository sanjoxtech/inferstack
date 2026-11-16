#!/usr/bin/env python
"""P1 check on the real model: is packed prefill as close to Transformers as the old path?

For each prompt set, three paths produce a greedy continuation:
  ref    : Transformers, one prompt at a time (no padding), its own cache. The oracle.
  old    : today's path. Left-padded Transformers cohort, then load_rows into the static
           pool, then static decode.
  packed : QwenStaticDecoder.prefill_packed into the pool, then static decode.
Both pool paths decode eagerly with the same decoder settings as the served profile.
Reported per prompt: first-token logit max |diff| vs ref, and how many greedy tokens match
ref before the first divergence. In bf16 neither pool path is bit-identical to ref; the
question is whether packed diverges earlier or more often than old.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def scorecard_prompts(case, n, seed=20261109):
    import importlib.util
    spec = importlib.util.spec_from_file_location('pp_engine_scorecard', ROOT / 'scripts' / 'engine_scorecard.py')
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    return sc.prompts(case, n, seed)


def ref_greedy(model, toks, steps, dev):
    import torch
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([toks], device=dev), use_cache=True, logits_to_keep=1)
        first = out.logits[0, -1].float()
        cache, tok, seq = out.past_key_values, out.logits[:, -1].argmax(-1), []
        for s in range(steps):
            seq.append(int(tok))
            out = model(input_ids=tok[:, None], position_ids=torch.tensor([[len(toks) + s]], device=dev),
                        past_key_values=cache, use_cache=True)
            cache, tok = out.past_key_values, out.logits[:, -1].argmax(-1)
    return first, seq


def pool_decode(dec, st, first_logits, lengths, steps):
    import torch
    nxt = first_logits.argmax(-1)
    pos = torch.tensor(lengths, device=nxt.device)
    seqs = [[int(t)] for t in nxt.tolist()]
    for _ in range(steps - 1):
        with torch.inference_mode():
            nxt = dec.step(st, nxt, pos, length=int(max(lengths)) + _ + 1, use_graph=False).argmax(-1)
        pos = pos + 1
        for j, t in enumerate(nxt.tolist()):
            seqs[j].append(int(t))
    return seqs


def agree(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', default='configs/profiles/production-v1.15.0.toml')
    ap.add_argument('--foundry-root', default='examples/model_foundry')
    ap.add_argument('--model-ref', default='foundry://factorylm-2b@candidate')
    ap.add_argument('--device', default='cuda:0')
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
    eng._maybe_patch()
    n = cfg.native
    dec = QwenStaticDecoder(model, a.device, gdn_backend=str(n.static_decode_gdn_backend),
                            attn_backend=str(getattr(n, 'static_decode_attention', 'sdpa') or 'sdpa'))
    pad = tok.pad_token_id if tok.pad_token_id is not None else 0
    sets = [({'name': 'shared_prefix', 'prompt_words': 1200, 'shared_prefix': True}, 8),
            ({'name': 'prefill_heavy', 'prompt_words': 1500}, 4),
            ({'name': 'interactive', 'prompt_words': 48}, 8)]
    rows = []
    for case, k in sets:
        toks = [tok.encode(t, add_special_tokens=False) for t in scorecard_prompts(case, k)]
        lengths = [len(t) for t in toks]
        refs = [ref_greedy(model, t, a.steps, a.device) for t in toks]
        # old path: left-padded cohort, load_rows
        st_old = dec.new_state(16, 4096)
        w = max(lengths)
        ids = torch.full((k, w), pad, dtype=torch.long, device=a.device)
        mask = torch.zeros((k, w), dtype=torch.long, device=a.device)
        for i, t in enumerate(toks):
            ids[i, w - len(t):] = torch.tensor(t, device=a.device)
            mask[i, w - len(t):] = 1
        pos = (mask.cumsum(-1) - 1).clamp_min(0)
        with torch.inference_mode():
            out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True, logits_to_keep=1)
        old_first = out.logits[:, -1].float()
        dec.load_rows(st_old, out.past_key_values, mask, w, range(k), range(k))
        del out
        old = pool_decode(dec, st_old, old_first, lengths, a.steps)
        del st_old
        # packed path
        st_new = dec.new_state(16, 4096)
        with torch.inference_mode():
            new_first, _, _ = dec.prefill_packed(st_new, range(k), toks)
        new_first = new_first.float()
        new = pool_decode(dec, st_new, new_first, lengths, a.steps)
        del st_new
        torch.cuda.empty_cache()
        for j in range(k):
            rf, rs = refs[j]
            row = {'case': case['name'], 'i': j, 'prompt_tokens': lengths[j],
                   'old_first_maxdiff': float((old_first[j] - rf).abs().max()),
                   'packed_first_maxdiff': float((new_first[j] - rf).abs().max()),
                   'old_agree': agree(old[j], rs), 'packed_agree': agree(new[j], rs),
                   'old_vs_packed_agree': agree(old[j], new[j])}
            print(json.dumps(row), flush=True)
            rows.append(row)
    summ = {}
    for name in ('shared_prefix', 'prefill_heavy', 'interactive'):
        rs = [r for r in rows if r['case'] == name]
        summ[name] = {p: {'full_match': sum(r[f'{p}_agree'] == a.steps for r in rs), 'rows': len(rs),
                          'mean_agree': sum(r[f'{p}_agree'] for r in rs) / len(rs),
                          'max_first_maxdiff': max(r[f'{p}_first_maxdiff'] for r in rs)} for p in ('old', 'packed')}
    out = {'version': 1, 'profile': a.profile, 'steps': a.steps, 'gpu': torch.cuda.get_device_name(0),
           'summary': summ, 'rows': rows}
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + '\n')
    print(json.dumps(summ, indent=2))
    eng._gpu_executor.shutdown(wait=False, cancel_futures=True)


if __name__ == '__main__':
    main()
