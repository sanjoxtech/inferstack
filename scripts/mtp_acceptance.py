#!/usr/bin/env python
"""N6.0: how often would Qwen3.5's own MTP head guess the next tokens right?

Offline and teacher-forced, so it changes no serving code. For scorecard prompts it:

1. generates the greedy continuation with the plain Transformers model (the target);
2. runs one forward over prompt + continuation, capturing the final hidden states before and
   after the final norm;
3. builds the MTP head from the checkpoint's `mtp.*` tensors (dropped by transformers):
   pre_fc_norm_embedding / pre_fc_norm_hidden -> fc(concat) -> one full-attention decoder layer
   -> norm -> the shared lm_head. Its layer and norms are copies of the model's own classes;
4. scores draft 1: at position t, MTP(emb(x[t+1]), h[t]) should predict x[t+2]. Drafts 2 and 3
   are chained the way speculative decoding uses them: the MTP output hidden state at t plus the
   next token, scored only where every earlier draft was right.

Two hidden-state conventions are scored (final hidden before vs after the norm); the better one
is the convention to build with. Reports per case:

- acceptance a1, a2|1, a3|2;
- expected tokens per verify step for k = 1..3 drafts (1 + a1 + a1*a2 + ...).

Measured on the continuation region only. Evidence for whether N6 speculative decoding pays off
on this workload, not a speed number.

  python scripts/mtp_acceptance.py --out mtp.json
"""
from __future__ import annotations

import argparse
import copy
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
REPO_ID = 'Qwen/Qwen3.5-2B-Base'
REVISION = 'b1485b2fa6dfa1287294f269f5fb618e03d52d7c'


def scorecard_prompts(case_name, n):
    spec = importlib.util.spec_from_file_location('mtp_engine_scorecard', ROOT / 'scripts' / 'engine_scorecard.py')
    sc = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(sc)
    wl = sc.DEFAULT_WORKLOAD
    case = next(c for c in wl['cases'] if c['name'] == case_name)
    return sc.prompts(case, n, wl['seed']), case


def load_mtp_tensors(repo_id, revision):
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    idx = json.loads(Path(hf_hub_download(repo_id, 'model.safetensors.index.json', revision=revision)).read_text())
    shards = {}
    for k, f in idx['weight_map'].items():
        if k.startswith('mtp.'):
            shards.setdefault(f, []).append(k)
    out = {}
    for f, keys in shards.items():
        with safe_open(hf_hub_download(repo_id, f, revision=revision), framework='pt') as h:
            for k in keys:
                out[k] = h.get_tensor(k)
    return out


class MTPHead:
    def __init__(self, model, tensors, device, dtype):
        import torch
        text = model.model if hasattr(model.model, 'layers') else model.model.language_model
        cfg = getattr(model.config, 'text_config', model.config)
        full = next(i for i, t in enumerate(cfg.layer_types) if t == 'full_attention')
        self.text = text
        self.layer = copy.deepcopy(text.layers[full])
        self.layer.self_attn.layer_idx = 0
        sd = {k[len('mtp.layers.0.'):]: v for k, v in tensors.items() if k.startswith('mtp.layers.0.')}
        missing, unexpected = self.layer.load_state_dict(sd, strict=False)
        if unexpected or [m for m in missing if not m.endswith('rotary_emb.inv_freq')]:
            raise RuntimeError(f'MTP layer mismatch: missing={missing} unexpected={unexpected}')
        self.norm_e, self.norm_h, self.norm = (copy.deepcopy(text.norm) for _ in range(3))
        for mod, key in ((self.norm_e, 'mtp.pre_fc_norm_embedding.weight'), (self.norm_h, 'mtp.pre_fc_norm_hidden.weight'),
                         (self.norm, 'mtp.norm.weight')):
            mod.weight.data.copy_(tensors[key])
        w = tensors['mtp.fc.weight']
        self.fc = torch.nn.Linear(w.shape[1], w.shape[0], bias=False)
        self.fc.weight.data.copy_(w)
        for m in (self.layer, self.norm_e, self.norm_h, self.norm, self.fc):
            m.to(device=device, dtype=dtype).eval()
        self.lm_head = model.lm_head
        self.embed = text.embed_tokens

    def __call__(self, next_ids, hidden, positions):
        """next_ids [1,T] (token t+1), hidden [1,T,H] (state at t), positions [1,T] -> (logits, hidden_out)."""
        import torch
        x = self.fc(torch.cat([self.norm_e(self.embed(next_ids)), self.norm_h(hidden)], dim=-1))
        cos_sin = self.text.rotary_emb(x, positions[None].expand(3, *positions.shape))
        out = self.layer(x, position_embeddings=cos_sin, attention_mask=None, position_ids=positions)
        h = out[0] if isinstance(out, tuple) else out
        h = self.norm(h)
        return self.lm_head(h), h


def score_sequence(model, mtp, ids, prompt_len):
    """ids [1, N] = prompt + greedy continuation. Returns per-convention hit lists for drafts 1..3."""
    import torch
    cap = {}
    text = mtp.text
    hook = text.norm.register_forward_hook(lambda m, i, o: cap.update(pre=i[0], post=o))
    with torch.inference_mode():
        model(input_ids=ids, use_cache=False, logits_to_keep=1)
    hook.remove()
    N = ids.shape[1]
    res = {}
    with torch.inference_mode():
        for conv in ('post', 'pre'):
            h = cap[conv][:, :N - 2]                         # states at t = 0..N-3
            pos = torch.arange(N - 2, device=ids.device)[None]
            hits = []
            tok = ids[:, 1:N - 1]                            # x[t+1]
            ok = None
            for d in range(3):                               # draft d+1 predicts x[t+2+d]
                T = N - 2 - d
                if T <= prompt_len:
                    break
                logits, h_out = mtp(tok[:, :T], h[:, :T], pos[:, :T])
                pred = logits.argmax(-1)[0]
                target = ids[0, 2 + d:2 + d + T]
                hit = (pred == target)
                region = torch.arange(T, device=ids.device) >= max(0, prompt_len - 2 - d)
                cond = region if ok is None else (region & ok[:T])
                hits.append((int((hit & cond).sum()), int(cond.sum())))
                ok = hit if ok is None else (ok[:T] & hit)
                h = h_out                                    # chain: next draft uses the MTP hidden
                tok = ids[:, 2 + d:2 + d + T]                # and the next (accepted) token
                pos = pos[:, :T]
            res[conv] = hits
    return res


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--profile', default='configs/profiles/production-v1.6.0-baseline.toml')
    ap.add_argument('--foundry-root', default='examples/model_foundry')
    ap.add_argument('--model-ref', default='foundry://factorylm-2b@candidate')
    ap.add_argument('--device', default='cuda:0')
    ap.add_argument('--cases', nargs='+', default=['interactive', 'moderate', 'prefill_heavy', 'shared_prefix'])
    ap.add_argument('--prompts', type=int, default=6)
    ap.add_argument('--new-tokens', type=int, default=128)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()
    import torch
    from inferstack.config import load_profile
    from inferstack.foundry import FileSystemFoundryRepository, FoundryModelLoader
    cfg = load_profile(a.profile)
    _, model, tok, _ = FoundryModelLoader(FileSystemFoundryRepository(a.foundry_root), a.device).load(a.model_ref, cfg)
    model.eval()
    dtype = next(model.parameters()).dtype
    tensors = load_mtp_tensors(REPO_ID, REVISION)
    mtp = MTPHead(model, tensors, a.device, dtype)
    report = {'version': 1, 'model': f'{REPO_ID}@{REVISION}', 'mtp_tensors': sorted(tensors),
              'gpu': torch.cuda.get_device_name(0) if a.device.startswith('cuda') else None, 'cases': {}}
    for case_name in a.cases:
        texts, case = scorecard_prompts(case_name, a.prompts)
        agg = {c: [[0, 0], [0, 0], [0, 0]] for c in ('post', 'pre')}
        for text in texts:
            ids = torch.tensor([tok.encode(text, add_special_tokens=False)], device=a.device)
            with torch.inference_mode():
                gen = model.generate(input_ids=ids, attention_mask=torch.ones_like(ids), max_new_tokens=a.new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id or tok.eos_token_id)
            r = score_sequence(model, mtp, gen, ids.shape[1])
            for conv, hits in r.items():
                for d, (h, n) in enumerate(hits):
                    agg[conv][d][0] += h; agg[conv][d][1] += n
        out = {}
        for conv, rows in agg.items():
            acc = [h / n if n else None for h, n in rows]
            exp, run = [1.0], 1.0
            for x in acc:
                if x is None:
                    break
                run *= x; exp.append(exp[-1] + run)
            out[conv] = {'acceptance': acc, 'counts': rows, 'expected_tokens_per_step': exp}
        report['cases'][case_name] = out
        print(case_name, json.dumps(out), flush=True)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(report, indent=2) + '\n')
    print('Saved', a.out)


if __name__ == '__main__':
    main()
