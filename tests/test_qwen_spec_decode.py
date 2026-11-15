"""N6.1 MTP speculative decode lane: output must equal plain greedy decoding.

CPU, tiny random Qwen3.5 model in fp32, Transformers greedy decode as the oracle. Drafts are
forced right, wrong, or mixed, so the accept path, the reject path (GDN state restore) and
their alternation are all exercised. CUDA graph replay and acceptance with the real MTP head
are measured by scripts/spec_decode_bench.py.
"""
import pytest

torch = pytest.importorskip('torch')
qwen = pytest.importorskip('transformers.models.qwen3_5.modeling_qwen3_5')
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

from inferstack.runtime.qwen_spec_decode import QwenSpecDecoder, prefill_with_hidden  # noqa: E402

PROMPTS = [[5, 7, 11, 13, 2, 9, 4], [3, 1, 8], [60, 61, 62, 63, 64]]


def tiny_model(seed=0):
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=16,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=['linear_attention', 'linear_attention', 'full_attention', 'linear_attention'],
        max_position_embeddings=256, rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                                                      'partial_rotary_factor': 0.5, 'mrope_section': [2, 1, 1]},
        attn_implementation='sdpa')
    model = qwen.Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def hf_greedy(model, prompt, n):
    with torch.inference_mode():
        out = model(input_ids=torch.tensor([prompt]), use_cache=True)
        cache, tok, toks = out.past_key_values, out.logits[:, -1].argmax(-1), []
        for s in range(n):
            toks.append(int(tok))
            out = model(input_ids=tok[:, None], position_ids=torch.tensor([[len(prompt) + s]]),
                        past_key_values=cache, use_cache=True)
            cache, tok = out.past_key_values, out.logits[:, -1].argmax(-1)
    return toks


def setup(model, prompts, pool=None, max_len=64, mtp_seed=1):
    dec = QwenSpecDecoder(model, 'cpu', dtype=torch.float32, mtp_seed=mtp_seed)
    st = dec.new_state(pool or len(prompts), max_len)
    sp = dec.new_spec_state(st)
    emitted = []
    for slot, p in enumerate(prompts):
        ids = torch.tensor([p])
        cache, hidden, first = prefill_with_hidden(model, ids)
        assert dec.load_rows(st, cache, torch.ones(1, len(p), dtype=torch.long), len(p), [0], [slot]) == [len(p)]
        dec.load_spec_row(st, sp, slot, ids[0], hidden, first)
        emitted.append([first])
    return dec, st, sp, emitted


def run(model, prompts, n, draft_rule=None, pool=None):
    """Spec-decode until every row has n tokens. draft_rule(step, row, oracle_next) -> forced draft or None."""
    oracle = [hf_greedy(model, p, n + 2) for p in prompts]
    dec, st, sp, emitted = setup(model, prompts, pool)
    pos = [len(p) for p in prompts]
    accepts, step = [], 0
    while min(map(len, emitted)) < n:
        if draft_rule is not None:
            for r in range(len(prompts)):
                forced = draft_rule(step, r, oracle[r][min(len(emitted[r]), n + 1)])
                if forced is not None:
                    sp.tok[r, 1] = forced
        assert all(int(sp.tok[r, 0]) == emitted[r][-1] for r in range(len(prompts)))
        y, acc = dec.spec_step(st, sp, len(prompts), length=max(pos) + 2, use_graph=False)
        for r in range(len(prompts)):
            emitted[r].append(int(y[r, 0]))
            if bool(acc[r]):
                emitted[r].append(int(y[r, 1]))
            pos[r] += 1 + int(acc[r])
            accepts.append(bool(acc[r]))
            assert int(sp.pos[r]) == pos[r]
        step += 1
    for r in range(len(prompts)):
        assert emitted[r][:n] == oracle[r][:n], f'row {r}: {emitted[r][:n]} vs {oracle[r][:n]}'
    return accepts, dec


def test_right_drafts_are_all_accepted_and_output_is_greedy():
    accepts, _ = run(tiny_model(), PROMPTS, 16, lambda s, r, nxt: nxt)
    assert all(accepts)


def test_wrong_drafts_are_all_rejected_and_output_is_greedy():
    accepts, _ = run(tiny_model(), PROMPTS, 12, lambda s, r, nxt: (nxt + 1) % 97)
    assert not any(accepts)


def test_alternating_accept_and_reject_restores_linear_attention_state():
    rule = lambda s, r, nxt: nxt if (s + r) % 3 else (nxt + 5) % 97   # noqa: E731
    accepts, _ = run(tiny_model(seed=2), PROMPTS, 20, rule)
    assert any(accepts) and not all(accepts)


def test_mtp_head_drafts_keep_output_greedy_with_padding_rows():
    accepts, dec = run(tiny_model(seed=3), PROMPTS, 14, None, pool=4)   # bucket 4 for 3 rows
    assert dec.stats['spec_eager_steps'] > 0


def test_lane_draft_equals_offline_mtp_over_the_full_sequence():
    """With right drafts, the draft the graph writes for position P+1 must equal the MTP head
    run offline (causal, one pass) over the greedy sequence: entry P-1 = (token P, hidden P-1)."""
    model = tiny_model(seed=5)
    prompt, n = PROMPTS[0], 10
    greedy = hf_greedy(model, prompt, n + 4)
    dec, st, sp, emitted = setup(model, [prompt], pool=1)
    lane = {}                                     # position of the drafted token -> draft
    pos = len(prompt)
    lane[pos + 1] = int(sp.tok[0, 1])
    while len(emitted[0]) < n:
        sp.tok[0, 1] = greedy[len(emitted[0])]
        y, acc = dec.spec_step(st, sp, 1, length=pos + 2, use_graph=False)
        assert bool(acc[0])
        emitted[0] += [int(y[0, 0]), int(y[0, 1])]
        pos += 2
        lane[pos + 1] = int(sp.tok[0, 1])
    seq = prompt + greedy[:len(emitted[0])]
    _, hidden, _ = prefill_with_hidden(model, torch.tensor([seq]))
    st2 = dec.new_state(1, 64)
    sp2 = dec.new_spec_state(st2)
    N = len(seq)
    st2.mask[0, :N - 1] = True
    with torch.inference_mode():
        x = dec._mtp_in(torch.tensor([seq[1:]]), hidden[None, :N - 1])
        posn = torch.arange(N - 1)
        cos, sin = dec.text.rotary_emb(x, posn.view(1, 1, -1).expand(3, 1, N - 1))
        hm = dec._mtp_layer(x, cos, sin, sp2, slice(0, 1), st2.mask[0:1], torch.zeros(1, dtype=torch.long),
                            N - 1, 'sdpa', None)
        offline = dec.lm_head(dec.mtp.norm(hm[0])).argmax(-1)    # offline[q] predicts seq[q + 2]
    for P1, d in lane.items():
        if P1 - 2 < N - 1:
            assert d == int(offline[P1 - 2]), (P1, d, int(offline[P1 - 2]))
