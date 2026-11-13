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


def run_both(model, prompts, steps, batch_bucket=None, max_len=64):
    ids, mask, pos, w = left_padded(prompts)
    B = len(prompts)
    with torch.inference_mode():
        out = model(input_ids=ids, attention_mask=mask, position_ids=pos, use_cache=True)
    cache = out.past_key_values
    nxt = out.logits[:, -1].argmax(-1)
    dec = QwenStaticDecoder(model, 'cpu', dtype=torch.float32)
    st = dec.new_state(batch_bucket or B, max_len)
    dec.load(st, cache, mask, w)
    positions = torch.tensor([len(p) for p in prompts])
    hf_mask = mask
    ref_logits, got_logits = [], []
    for s in range(steps):
        col = w + s
        hf_mask = torch.cat([hf_mask, torch.ones((B, 1), dtype=torch.long)], dim=1)
        with torch.inference_mode():
            ref = model(input_ids=nxt[:, None], attention_mask=hf_mask, position_ids=positions[:, None],
                        past_key_values=cache, use_cache=True)
        cache = ref.past_key_values
        ids_in = torch.zeros(st.batch, dtype=torch.long)
        ids_in[:B] = nxt
        pos_in = torch.zeros(st.batch, dtype=torch.long)
        pos_in[:B] = positions
        got = dec.step(st, ids_in, pos_in, col)[:B].clone()
        ref_logits.append(ref.logits[:, -1].float())
        got_logits.append(got.float())
        nxt = ref.logits[:, -1].argmax(-1)
        positions = positions + 1
    return torch.stack(ref_logits), torch.stack(got_logits), dec


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
