"""Static decode step vs the Transformers Qwen3.5 decode path (numerical oracle).

Needs transformers with Qwen3.5 (>=5.22). Skipped otherwise. Runs on CPU with a tiny
random model; GPU graph replay is checked by scripts/static_decode_bench.py.
"""
import pytest

torch = pytest.importorskip('torch')
qwen = pytest.importorskip('transformers.models.qwen3_5.modeling_qwen3_5')
from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5TextConfig  # noqa: E402

from inferstack.runtime.qwen_static_decode import QwenStaticDecoder, supports_static_decode  # noqa: E402


def tiny_model(seed=0, kv_heads=2, v_heads=4):
    torch.manual_seed(seed)
    cfg = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=kv_heads, head_dim=16,
        linear_conv_kernel_dim=4, linear_key_head_dim=8, linear_value_head_dim=8,
        linear_num_key_heads=2, linear_num_value_heads=v_heads,
        layer_types=['linear_attention', 'linear_attention', 'full_attention', 'linear_attention'],
        max_position_embeddings=256, rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                                                      'partial_rotary_factor': 0.5, 'mrope_section': [2, 1, 1]},
        attn_implementation='sdpa')
    model = qwen.Qwen3_5ForCausalLM(cfg).eval()
    # Random init leaves A_log/dt_bias/norm weights at trivial values; perturb so the test bites.
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    return model


def left_padded(prompts, pad=0):
    w = max(map(len, prompts))
    ids = torch.full((len(prompts), w), pad, dtype=torch.long)
    mask = torch.zeros((len(prompts), w), dtype=torch.long)
    for i, p in enumerate(prompts):
        ids[i, w - len(p):] = torch.tensor(p)
        mask[i, w - len(p):] = 1
    pos = (mask.cumsum(-1) - 1).clamp_min(0)
    return ids, mask, pos, w


def run_both(model, prompts, steps, batch_bucket=None, max_len=64, len_buckets=None):
    ids, mask, pos, w = left_padded(prompts)
    B = len(prompts)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    cache = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1)
    dec = QwenStaticDecoder(model, 'cpu', dtype=torch.float32)
    st = dec.new_state(batch_bucket or B, max_len, len_buckets=len_buckets)
    lengths = dec.load(st, cache, mask, w)
    assert lengths == [len(p) for p in prompts]
    positions = torch.tensor([len(p) for p in prompts])
    hf_mask = mask
    ref_logits, got_logits = [], []
    for s in range(steps):
        hf_mask = torch.cat([hf_mask, torch.ones((B, 1), dtype=torch.long)], dim=1)
        with torch.inference_mode():
            ref = model(input_ids=nxt[:, None], attention_mask=hf_mask, position_ids=positions[:, None],
                        past_key_values=cache, use_cache=True)
        cache = ref.past_key_values
        got = dec.step(st, nxt, positions).clone()
        ref_logits.append(ref.logits[:, -1].float())
        got_logits.append(got.float())
        nxt = ref.logits[:, -1].argmax(-1)
        positions = positions + 1
    return torch.stack(ref_logits), torch.stack(got_logits), dec


def hf_greedy(model, prompt, steps):
    """Reference continuation for one prompt (unbatched Transformers decode)."""
    ids = torch.tensor([prompt])
    with torch.inference_mode():
        out = model(input_ids=ids, use_cache=True)
    cache, tok, toks, logits = out.past_key_values, out.logits[:, -1].argmax(-1), [], []
    for s in range(steps):
        toks.append(int(tok))
        with torch.inference_mode():
            out = model(input_ids=tok[:, None], position_ids=torch.tensor([[len(prompt) + s]]),
                        past_key_values=cache, use_cache=True)
        cache = out.past_key_values
        logits.append(out.logits[0, -1].float())
        tok = out.logits[:, -1].argmax(-1)
    return toks, logits


def test_supports_qwen35_text_model():
    assert supports_static_decode(tiny_model()) == (True, 'ok')


def test_single_row_matches_transformers():
    ref, got, dec = run_both(tiny_model(), [[5, 7, 11, 13, 2, 9]], steps=6)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    assert dec.stats['eager_steps'] == 6 and dec.stats['loads'] == 1


def test_left_padded_batch_matches_transformers():
    prompts = [[5, 7, 11, 13, 2, 9, 4, 4], [3, 1, 8], [6, 6, 6, 1, 2]]
    ref, got, _ = run_both(tiny_model(seed=1), prompts, steps=8)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_bucket_padding_rows_do_not_change_live_rows():
    prompts = [[5, 7, 11, 13], [3, 1, 8, 2, 2]]
    ref, got, _ = run_both(tiny_model(seed=2), prompts, steps=5, batch_bucket=8)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)
    assert torch.isfinite(got).all()


def test_gqa_and_head_repeat_shapes():
    ref, got, _ = run_both(tiny_model(seed=3, kv_heads=1, v_heads=2), [[1, 2, 3, 4, 5], [9, 8]], steps=4)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_forward_fused_model_matches_transformers():
    """native_full installs forward fusion, which removes the original projection modules
    (seen on L4 2026-11-13: 'NoneType' object is not callable). The static step must use
    the fused weights."""
    from types import SimpleNamespace
    from inferstack.kernels.qwen_forward_fusion import install_qwen35_forward_fusion
    model = tiny_model(seed=4)
    flags = ('forward_fusion', 'fuse_rmsnorm', 'fuse_add_rmsnorm', 'fuse_gated_rmsnorm', 'fuse_mlp_gate_up',
             'fuse_mlp_activation', 'fuse_gdn_projections', 'fuse_gdn_beta_g', 'grouped_gdn_decode',
             'fuse_attention_qkv', 'forward_fusion_strict')
    diag = install_qwen35_forward_fusion(model, SimpleNamespace(native=SimpleNamespace(**dict.fromkeys(flags, True))))
    assert diag['gdn_modules'] >= 3 and diag['attention_modules'] >= 1, diag
    assert model.model.layers[0].linear_attn.in_proj_qkv is None
    assert model.model.layers[2].self_attn.q_proj is None
    ref, got, _ = run_both(model, [[5, 7, 11, 13, 2], [3, 1, 8]], steps=5)
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_gdn_backend_resolution_on_cpu():
    model = tiny_model()
    assert QwenStaticDecoder(model, 'cpu').gdn_backend == 'torch'
    assert QwenStaticDecoder(model, 'cpu', gdn_backend='torch').stats['gdn_backend'] == 'torch'
    with pytest.raises(RuntimeError):
        QwenStaticDecoder(model, 'cpu', gdn_backend='triton')
    with pytest.raises(ValueError):
        QwenStaticDecoder(model, 'cpu', gdn_backend='fla')


def test_load_rejects_oversized_cache():
    model = tiny_model()
    ids, mask, pos, w = left_padded([[1, 2, 3, 4]] * 3)
    with torch.inference_mode():
        cache = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True).past_key_values
    dec = QwenStaticDecoder(model, 'cpu', dtype=torch.float32)
    with pytest.raises(ValueError):
        dec.load(dec.new_state(2, 64), cache, mask, w)
    with pytest.raises(ValueError):
        dec.load(dec.new_state(4, 3), cache, mask, w)


def test_length_buckets_slice_attention():
    """Attention runs over the smallest length bucket covering the live rows."""
    prompts = [[5, 7, 11, 13, 2, 9], [3, 1, 8]]
    ref, got, dec = run_both(tiny_model(seed=5), prompts, steps=8, max_len=64, len_buckets=[8, 16, 64])
    torch.testing.assert_close(got, ref, rtol=1e-4, atol=1e-4)


def test_rows_of_different_lengths_and_move_row():
    """Two requests decode together; row 0 finishes, row 1 moves to slot 0 and continues.
    Every logit must match each request decoded alone by Transformers."""
    model = tiny_model(seed=6)
    pa, pb = [5, 7, 11, 13, 2, 9, 4], [3, 1]
    ta, la = hf_greedy(model, pa, 3)
    tb, lb = hf_greedy(model, pb, 7)
    ids, mask, pos, w = left_padded([pa, pb])
    with torch.inference_mode():
        cache = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True).past_key_values
    dec = QwenStaticDecoder(model, 'cpu', dtype=torch.float32)
    st = dec.new_state(4, 32, len_buckets=[8, 32])
    assert dec.load(st, cache, mask, w) == [len(pa), len(pb)]
    for s in range(3):
        got = dec.step(st, torch.tensor([ta[s], tb[s]]), torch.tensor([len(pa) + s, len(pb) + s]))
        torch.testing.assert_close(got[0].float(), la[s], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(got[1].float(), lb[s], rtol=1e-4, atol=1e-4)
    dec.move_row(st, 1, 0, len(pb) + 3)
    for s in range(3, 7):
        got = dec.step(st, torch.tensor([tb[s]]), torch.tensor([len(pb) + s]))
        torch.testing.assert_close(got[0].float(), lb[s], rtol=1e-4, atol=1e-4)
    assert dec.stats['row_moves'] == 1


def test_load_rows_into_chosen_slots_with_holes_in_mask():
    """Prefix-cache layouts leave holes in the attention mask; load_rows packs the real
    columns. Oracle: Transformers decoding from the same cache with the same mask."""
    model = tiny_model(seed=7)
    ids = torch.tensor([[4, 9, 0, 0, 2, 6, 1]])
    mask = torch.tensor([[1, 1, 0, 0, 1, 1, 1]])
    pos = torch.tensor([[0, 1, 0, 0, 2, 3, 4]])
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    cache, tok = out.past_key_values, out.logits[:, -1].argmax(-1)
    dec = QwenStaticDecoder(model, 'cpu', dtype=torch.float32)
    st = dec.new_state(4, 16)
    assert dec.load_rows(st, cache, mask, 7, [0], [2]) == [5]
    dec.move_row(st, 2, 0, 5)          # live rows are packed from slot 0
    m = mask
    for s in range(3):
        m = torch.cat([m, torch.ones((1, 1), dtype=torch.long)], dim=1)
        with torch.inference_mode():
            ref = model(input_ids=tok[:, None], attention_mask=m, position_ids=torch.tensor([[5 + s]]),
                        past_key_values=cache, use_cache=True)
        cache = ref.past_key_values
        got = dec.step(st, tok, torch.tensor([5 + s]))
        torch.testing.assert_close(got[0].float(), ref.logits[0, -1].float(), rtol=1e-4, atol=1e-4)
        tok = ref.logits[:, -1].argmax(-1)


# ---- P1: packed prefill straight into the pool ------------------------------------------
@pytest.mark.parametrize('fused', [False, True])
def test_packed_prefill_matches_transformers(fused):
    """prefill_packed (flat tokens, per-segment GDN/attention, written into chosen slots)
    then static decode must equal unbatched Transformers greedy decode, logits included."""
    model = tiny_model(seed=5)
    if fused:
        from types import SimpleNamespace
        from inferstack.kernels.qwen_forward_fusion import install_qwen35_forward_fusion
        flags = ('forward_fusion', 'fuse_rmsnorm', 'fuse_add_rmsnorm', 'fuse_gated_rmsnorm', 'fuse_mlp_gate_up',
                 'fuse_mlp_activation', 'fuse_gdn_projections', 'fuse_gdn_beta_g', 'grouped_gdn_decode',
                 'fuse_attention_qkv', 'forward_fusion_strict')
        install_qwen35_forward_fusion(model, SimpleNamespace(native=SimpleNamespace(**dict.fromkeys(flags, True))))
    prompts = [[5, 7, 11, 13, 2, 9, 4, 4, 1, 3], [3], [6, 6, 6, 1, 2], [8, 2]]
    slots = [3, 0, 5, 1]                     # out of order, with holes
    steps = 5
    refs = [hf_greedy(model, p, steps + 1) for p in prompts]
    dec = QwenStaticDecoder(model, 'cpu', dtype=torch.float32)
    st = dec.new_state(8, 32)
    st.mask.fill_(True)                      # stale columns from a previous occupant must be cleared
    with torch.inference_mode():
        logits, hid, cu = dec.prefill_packed(st, slots, prompts)
    assert cu == [0, 10, 11, 16, 18] and hid.shape[0] == 18
    for j, p in enumerate(prompts):
        with torch.inference_mode():
            ref = model(input_ids=torch.tensor([p])).logits[0, -1].float()
        torch.testing.assert_close(logits[j].float(), ref, rtol=1e-4, atol=1e-4)
        assert int(st.mask[slots[j]].sum()) == len(p)
    assert [int(t) for t in logits.argmax(-1)] == [r[0][0] for r in refs]
    # Rows in slots 0 and 1 (prompts 1 and 3) then decode together and stay greedy-equal.
    nxt = torch.tensor([refs[1][0][0], refs[3][0][0]])
    pos = torch.tensor([len(prompts[1]), len(prompts[3])])
    for t in range(steps):
        got = dec.step(st, nxt, pos).float()
        torch.testing.assert_close(got[0], refs[1][1][t], rtol=1e-4, atol=1e-4)
        torch.testing.assert_close(got[1], refs[3][1][t], rtol=1e-4, atol=1e-4)
        nxt, pos = got.argmax(-1), pos + 1


def test_packed_prefill_then_batched_decode_is_greedy():
    model = tiny_model(seed=6)
    prompts = [[1, 2, 3, 4, 5, 6, 7], [9, 8], [4, 4, 4, 4]]
    steps = 6
    refs = [hf_greedy(model, p, steps + 1)[0] for p in prompts]
    dec = QwenStaticDecoder(model, 'cpu', dtype=torch.float32)
    st = dec.new_state(4, 32)
    with torch.inference_mode():
        logits, _, _ = dec.prefill_packed(st, [0, 1, 2], prompts)
    nxt = logits.argmax(-1)
    pos = torch.tensor([len(p) for p in prompts])
    out = [[int(t)] for t in nxt]
    for s in range(steps):
        nxt = dec.step(st, nxt, pos).argmax(-1)
        pos = pos + 1
        for j in range(3):
            out[j].append(int(nxt[j]))
    assert out == refs


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FLA varlen kernels need CUDA')
def test_packed_prefill_fla_matches_reference_on_cuda():
    """FLA varlen conv + chunked delta rule (cu_seqlens) vs the per-segment reference:
    same logits within bf16-kernel tolerance, same pool state, same greedy first token."""
    from inferstack.kernels.gdn import fla_available
    if not fla_available():
        pytest.skip('flash-linear-attention not installed')
    torch.manual_seed(7)
    cfg = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=128, intermediate_size=192, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=['linear_attention', 'linear_attention', 'full_attention', 'linear_attention'],
        max_position_embeddings=512, rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                                                      'partial_rotary_factor': 0.5, 'mrope_section': [4, 2, 2]},
        attn_implementation='sdpa')
    model = qwen.Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    model = model.cuda()
    g = torch.Generator().manual_seed(0)
    prompts = [torch.randint(0, 97, (n,), generator=g).tolist() for n in (130, 1, 67, 200)]
    slots = [2, 0, 3, 1]
    out = {}
    for be in ('reference', 'fla'):
        dec = QwenStaticDecoder(model, 'cuda', dtype=torch.float32)
        st = dec.new_state(4, 256)
        with torch.inference_mode():
            logits, _, _ = dec.prefill_packed(st, slots, prompts, backend=be)
        out[be] = (logits.float(), st)
    (lr, sr), (lf, sf) = out['reference'], out['fla']
    torch.testing.assert_close(lf, lr, rtol=2e-2, atol=2e-2)
    for i in sr.conv:
        torch.testing.assert_close(sf.conv[i], sr.conv[i], rtol=2e-2, atol=2e-2)
        torch.testing.assert_close(sf.rec[i], sr.rec[i], rtol=2e-2, atol=2e-2)
    for i in sr.kv:
        torch.testing.assert_close(sf.kv[i][0], sr.kv[i][0], rtol=2e-2, atol=2e-2)
    assert torch.equal(sf.mask, sr.mask)


# ---- P2: continuation from a slot's state (prefix snapshots) ------------------------------
def test_packed_prefill_continues_from_restored_prefix():
    """Prefill a 16-token prefix, snapshot it (PoolPrefixCache), then prefill three prompts in
    one call: one continuing from the snapshot restored into another slot, one continuing from
    a slot copy, one fresh (start 0). Logits must equal unbatched Transformers, and decode
    after it must stay greedy-equal."""
    from inferstack.runtime.pool_prefix_cache import PoolPrefixCache
    model = tiny_model(seed=8)
    pre = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    prompts = [pre + [1, 2, 3], pre + [4], [6, 1, 2, 5, 5]]
    steps = 4
    refs = [hf_greedy(model, p, steps + 1) for p in prompts]
    dec = QwenStaticDecoder(model, 'cpu', dtype=torch.float32)
    st = dec.new_state(6, 40)
    pc = PoolPrefixCache(8, min_tokens=8, block_tokens=8, max_bytes=1 << 26)
    with torch.inference_mode():
        dec.prefill_packed(st, [5], [pre])
    assert pc.put(pre, st, 5) and pc.boundary(len(prompts[0])) == 16
    st.mask.fill_(True)
    pc.copy_slot(st, 5, 1, 16)
    e = pc.longest(prompts[0])
    assert e is not None and len(e.tokens) == 16 and pc.longest(pre) is None   # never the whole prompt
    pc.restore(e, st, 0)
    with torch.inference_mode():
        logits, hid, cu = dec.prefill_packed(st, [0, 1, 2], prompts, starts=[16, 16, 0])
    assert cu == [0, 3, 4, 9] and hid.shape[0] == 9
    for j, p in enumerate(prompts):
        with torch.inference_mode():
            ref = model(input_ids=torch.tensor([p])).logits[0, -1].float()
        torch.testing.assert_close(logits[j].float(), ref, rtol=1e-4, atol=1e-4)
        assert int(st.mask[j].sum()) == len(p)
    nxt = logits.argmax(-1)
    pos = torch.tensor([len(p) for p in prompts])
    for t in range(steps):
        got = dec.step(st, nxt, pos).float()
        for j in range(3):
            torch.testing.assert_close(got[j], refs[j][1][t], rtol=1e-4, atol=1e-4)
        nxt, pos = got.argmax(-1), pos + 1


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FLA varlen kernels need CUDA')
def test_packed_continuation_fla_matches_reference_on_cuda():
    from inferstack.kernels.gdn import fla_available
    if not fla_available():
        pytest.skip('flash-linear-attention not installed')
    torch.manual_seed(9)
    cfg = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=128, intermediate_size=192, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=['linear_attention', 'linear_attention', 'full_attention', 'linear_attention'],
        max_position_embeddings=512, rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                                                      'partial_rotary_factor': 0.5, 'mrope_section': [4, 2, 2]},
        attn_implementation='sdpa')
    model = qwen.Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    model = model.cuda()
    g = torch.Generator().manual_seed(1)
    pre = torch.randint(0, 97, (128,), generator=g).tolist()
    prompts = [pre + torch.randint(0, 97, (n,), generator=g).tolist() for n in (40, 1)] + \
              [torch.randint(0, 97, (90,), generator=g).tolist()]
    out = {}
    for be in ('reference', 'fla'):
        dec = QwenStaticDecoder(model, 'cuda', dtype=torch.float32)
        st = dec.new_state(4, 256)
        with torch.inference_mode():
            dec.prefill_packed(st, [0], [pre], backend=be)
            for k, v in st.kv.values():
                k[1, :, :128].copy_(k[0, :, :128]); v[1, :, :128].copy_(v[0, :, :128])
            for t in list(st.conv.values()) + list(st.rec.values()):
                t[1].copy_(t[0])
            logits, _, _ = dec.prefill_packed(st, [0, 1, 2], prompts, starts=[128, 128, 0], backend=be)
            full, _, _ = dec.prefill_packed(st, [3], [prompts[0]], backend=be)
        out[be] = (logits.float(), full.float())
    (lr, fr), (lf, ff) = out['reference'], out['fla']
    torch.testing.assert_close(lf, lr, rtol=2e-2, atol=2e-2)
    torch.testing.assert_close(lf[0], ff[0], rtol=2e-2, atol=2e-2)   # continuation == from scratch


def test_packed_prefill_snap_records_boundary_state():
    """P5 single-pass snapshot: prefill_packed(snap=...) records, for the snapped rows, the GDN
    conv/recurrent state at the boundary inside the full prefill. It must equal the state a
    separate prefill of the prefix leaves, the logits must equal the call without snap, and the
    snapshot (with the slot's K/V columns) must serve a continuation."""
    from inferstack.runtime.pool_prefix_cache import PoolPrefixCache
    model = tiny_model(seed=8)
    pre = [9, 8, 7, 6, 5, 4, 3, 2, 1, 2, 3, 4, 5, 6, 7, 8]
    prompts = [pre + [1, 2, 3], [6, 1, 2, 5, 5], pre + [4, 4]]
    dec = QwenStaticDecoder(model, 'cpu', dtype=torch.float32)
    assert dec.snap_align() == 1
    st = dec.new_state(6, 40)
    with torch.inference_mode():
        plain, _, _ = dec.prefill_packed(st, [0, 1, 2], prompts)
        so = {}
        snapped, _, _ = dec.prefill_packed(st, [0, 1, 2], prompts, snap=[16, 0, 8], snap_out=so)
        torch.testing.assert_close(snapped, plain, rtol=1e-5, atol=1e-5)
        assert so['rows'] == [0, 2] and set(so['conv']) == set(st.conv) and set(so['rec']) == set(st.rec)
        dec.prefill_packed(st, [4, 5], [pre, pre[:8]])
        for i in st.conv:
            for r, s in enumerate((4, 5)):
                torch.testing.assert_close(so['conv'][i][r], st.conv[i][s], rtol=1e-5, atol=1e-5)
                torch.testing.assert_close(so['rec'][i][r], st.rec[i][s].float(), rtol=1e-5, atol=1e-5)
        dec.prefill_packed(st, [0], [prompts[0]], snap=[16], snap_out=so)
        pc = PoolPrefixCache(8, min_tokens=8, block_tokens=8, max_bytes=1 << 26)
        assert pc.put(pre, st, 0, conv={i: c[0] for i, c in so['conv'].items()},
                      rec={i: x[0] for i, x in so['rec'].items()})
        pc.restore(pc.longest(prompts[2]), st, 3)
        cont, _, _ = dec.prefill_packed(st, [3], [prompts[2]], starts=[16])
    torch.testing.assert_close(cont[0], plain[2], rtol=1e-4, atol=1e-4)
    with pytest.raises(ValueError):
        dec.prefill_packed(st, [0], [pre], snap=[16], snap_out={})   # boundary must be inside the prompt


@pytest.mark.skipif(not torch.cuda.is_available(), reason='FLA varlen kernels need CUDA')
@pytest.mark.parametrize('dtype', ['float32', 'bfloat16'])
def test_packed_prefill_snap_fla_matches_prefix_prefill_on_cuda(dtype):
    """P5: on FLA the boundary state comes from the delta-rule kernel's chunk-start states h
    (stored in the model dtype). It must match the state a prefill of the prefix alone leaves
    (bf16: within rounding), with logits unchanged versus the plain call."""
    from inferstack.kernels.gdn import fla_available
    from inferstack.kernels import fla_gdr_states
    if not fla_available() or not fla_gdr_states.available():
        pytest.skip('flash-linear-attention 0.6.0 not installed')
    dt = getattr(torch, dtype)
    torch.manual_seed(9)
    cfg = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=128, intermediate_size=192, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=['linear_attention', 'linear_attention', 'full_attention', 'linear_attention'],
        max_position_embeddings=512, rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                                                      'partial_rotary_factor': 0.5, 'mrope_section': [4, 2, 2]},
        attn_implementation='sdpa')
    model = qwen.Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    model = model.cuda().to(dt)
    g = torch.Generator().manual_seed(1)
    rnd = lambda n: torch.randint(0, 97, (n,), generator=g).tolist()   # noqa: E731
    prompts = [rnd(200), rnd(90), rnd(129)]
    dec = QwenStaticDecoder(model, 'cuda', dtype=dt)
    assert dec.snap_align() == 64
    st = dec.new_state(6, 256)
    tol = dict(rtol=1e-4, atol=1e-4) if dt == torch.float32 else dict(rtol=3e-2, atol=3e-2)
    with torch.inference_mode():
        plain, _, _ = dec.prefill_packed(st, [0, 1, 2], prompts)
        so = {}
        snapped, _, _ = dec.prefill_packed(st, [0, 1, 2], prompts, snap=[128, 0, 128], snap_out=so)
        assert so['rows'] == [0, 2]
        torch.testing.assert_close(snapped.float(), plain.float(), **tol)
        dec.prefill_packed(st, [3, 4], [prompts[0][:128], prompts[2][:128]])
        for i in st.conv:
            for r, s in enumerate((3, 4)):
                torch.testing.assert_close(so['conv'][i][r].float(), st.conv[i][s].float(), rtol=0, atol=0)
                torch.testing.assert_close(so['rec'][i][r], st.rec[i][s].float(), **tol)
        with pytest.raises(ValueError):
            dec.prefill_packed(st, [0], [prompts[0]], snap=[96], snap_out={})   # not a chunk start



# ---- P4: small packed prefills as CUDA graphs -------------------------------------------
@pytest.mark.skipif(not torch.cuda.is_available(), reason='prefill graphs need CUDA')
def test_prefill_graphed_fla_matches_eager_on_cuda():
    """prefill_graphed (rows padded to a bucket, inert padding, fixed split, graph replay) gives
    the same logits, hidden states, pool K/V and GDN states as eager prefill_packed, for fresh
    rows and rows continuing from a restored prefix, across two replays of one graph."""
    from inferstack.kernels.gdn import fla_available
    if not fla_available():
        pytest.skip('flash-linear-attention not installed')
    torch.manual_seed(11)
    cfg = Qwen3_5TextConfig(
        vocab_size=97, hidden_size=128, intermediate_size=192, num_hidden_layers=4,
        num_attention_heads=4, num_key_value_heads=2, head_dim=32,
        linear_conv_kernel_dim=4, linear_key_head_dim=32, linear_value_head_dim=32,
        linear_num_key_heads=2, linear_num_value_heads=4,
        layer_types=['linear_attention', 'linear_attention', 'full_attention', 'linear_attention'],
        max_position_embeddings=512, rope_parameters={'rope_type': 'default', 'rope_theta': 10000.0,
                                                      'partial_rotary_factor': 0.5, 'mrope_section': [4, 2, 2]},
        attn_implementation='sdpa')
    model = qwen.Qwen3_5ForCausalLM(cfg).eval()
    with torch.no_grad():
        for p in model.parameters():
            p.add_(torch.randn_like(p) * 0.05)
    model = model.cuda()
    g = torch.Generator().manual_seed(2)
    pre = torch.randint(0, 97, (128,), generator=g).tolist()

    def rnd(n):
        return torch.randint(0, 97, (n,), generator=g).tolist()

    def seed_prefix(dec, st, rows):
        dec.prefill_packed(st, [0], [pre])
        for r in rows:
            if r == 0:
                continue
            for k, v in st.kv.values():
                k[r, :, :128].copy_(k[0, :, :128]); v[r, :, :128].copy_(v[0, :, :128])
            for t in list(st.conv.values()) + list(st.rec.values()):
                t[r].copy_(t[0])

    cases = [([rnd(40), rnd(3), rnd(17)], [0, 0, 0]),                       # fresh, mixed lengths
             ([pre + rnd(40), pre + rnd(1), rnd(60)], [128, 128, 0]),       # continuation + fresh
             ([rnd(5), rnd(33), rnd(64)], [0, 0, 0])]                       # second replay, other lengths
    dg = QwenStaticDecoder(model, 'cuda', dtype=torch.float32)
    sg = dg.new_state(4, 512)
    with torch.inference_mode():
        dg.capture_prefill_graph(sg, 3, 64)
        dg.capture_prefill_graph(sg, 1, 16)
        # churn FLA's identity caches and the allocator with eager varlen calls on fresh
        # cu_seqlens tensors: graphs must not depend on tensors the caches drop (the v1.16 smoke
        # crashed with an illegal address this way while a single-graph test passed)
        for n in range(6):
            dg.prefill_packed(sg, [0, 1], [rnd(9 + n), rnd(30 + 2 * n)])
            torch.empty(1 << 20, device='cuda').fill_(float('nan'))
    dg.reset(sg)
    for prompts, starts in cases:
        de = QwenStaticDecoder(model, 'cuda', dtype=torch.float32)
        se = de.new_state(4, 512)
        with torch.inference_mode():
            seed_prefix(de, se, [j for j, s in enumerate(starts) if s])
            seed_prefix(dg, sg, [j for j, s in enumerate(starts) if s])
            le, he, cue = de.prefill_packed(se, [0, 1, 2], prompts, starts=starts)
            res = dg.prefill_graphed(sg, [0, 1, 2], prompts, starts=starts)
        assert res is not None
        lg, hg, cug = res
        assert cug == cue
        torch.testing.assert_close(lg, le, rtol=1e-3, atol=1e-3)
        torch.testing.assert_close(hg, he, rtol=1e-3, atol=1e-3)
        for j, p in enumerate(prompts):
            n = len(p)
            assert bool(sg.mask[j, :n].all()) and not bool(sg.mask[j, n:].any())
            for (kg, vg), (ke, ve) in zip(sg.kv.values(), se.kv.values()):
                torch.testing.assert_close(kg[j, :, :n], ke[j, :, :n], rtol=1e-3, atol=1e-3)
                torch.testing.assert_close(vg[j, :, :n], ve[j, :, :n], rtol=1e-3, atol=1e-3)
        for i in sg.conv:
            torch.testing.assert_close(sg.conv[i][:3], se.conv[i][:3], rtol=1e-3, atol=1e-3)
            torch.testing.assert_close(sg.rec[i][:3], se.rec[i][:3], rtol=1e-3, atol=1e-3)
    assert dg.stats['prefill_replays'] == 3
    # prompts that do not fit a bucket fall back (None), as do uncaptured buckets
    with torch.inference_mode():
        assert dg.prefill_graphed(sg, [0], [rnd(200)]) is None
        assert dg.prefill_graphed(sg, [0, 1], [rnd(5), rnd(5)]) is None
