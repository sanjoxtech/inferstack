"""N3.2 GDN primitive: every backend computes the contract; dispatch is the primitive's job."""
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('transformers.models.qwen3_5.modeling_qwen3_5')

from inferstack.kernels import gdn as G  # noqa: E402


def inputs(B=2, T=5, Hk=2, Hv=4, K=16, V=8, seed=0, state=True, dtype=torch.float32):
    gen = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=gen)  # noqa: E731
    q, k, v = r(B, T, Hk, K).to(dtype), r(B, T, Hk, K).to(dtype), r(B, T, Hv, V).to(dtype)
    g = -torch.rand(B, T, Hv, generator=gen) * 0.5          # log decay <= 0
    beta = torch.rand(B, T, Hv, generator=gen)
    s0 = r(B, Hv, K, V) * 0.1 if state else None
    return q, k, v, g, beta, s0


@pytest.mark.parametrize('T', [1, 3, 64, 130])
@pytest.mark.parametrize('with_state', [False, True])
def test_chunk_reference_matches_per_token_oracle(T, with_state):
    args = inputs(T=T, state=with_state, seed=T)
    o_ref, s_ref = G.gdn(*args, backend='recurrent_reference')
    o, s = G.gdn(*args, backend='chunk_reference')
    assert o.shape == (2, T, 4, 8) and s.shape == (2, 4, 16, 8)
    torch.testing.assert_close(o, o_ref, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(s, s_ref, atol=2e-4, rtol=2e-4)


def test_upstream_recurrent_matches_oracle_for_one_token():
    args = inputs(T=1, seed=7)
    o_ref, s_ref = G.gdn(*args, backend='recurrent_reference')
    o, s = G.gdn(*args, backend='recurrent_upstream')
    torch.testing.assert_close(o, o_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s, s_ref, atol=1e-5, rtol=1e-5)


def test_prefill_then_decode_equals_one_long_pass():
    """State hand-off: chunked prefill of 40 tokens + 3 decode steps == one 43-token pass."""
    q, k, v, g, beta, s0 = inputs(T=43, seed=3)
    o_all, s_all = G.gdn(q, k, v, g, beta, s0, backend='recurrent_reference')
    o_pre, s = G.gdn(q[:, :40], k[:, :40], v[:, :40], g[:, :40], beta[:, :40], s0, backend='chunk_reference')
    outs = [o_pre]
    for t in range(40, 43):
        o, s = G.gdn(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], g[:, t:t + 1], beta[:, t:t + 1], s)
        outs.append(o)
    torch.testing.assert_close(torch.cat(outs, 1), o_all, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(s, s_all, atol=2e-4, rtol=2e-4)


def test_auto_dispatch_and_cpu_fallback_are_counted():
    G.STATS.clear()
    G.gdn(*inputs(T=1))                       # CPU: Triton decode unavailable -> upstream recurrent
    G.gdn(*inputs(T=9))                       # prefill -> chunked algebra
    G.gdn(*inputs(T=1), prefer_triton_decode=False)
    assert G.STATS['recurrent_triton_fallback'] == 1 and G.STATS['recurrent_upstream'] == 1
    assert G.STATS['chunk_reference'] == 2
    assert G.select_backend(torch.zeros(1, 1, 1, 1)) == 'recurrent_triton'
    assert G.select_backend(torch.zeros(1, 4, 1, 1)) == 'chunk_reference'


def test_explicit_backend_does_not_silently_fall_back():
    with pytest.raises(RuntimeError):
        G.gdn(*inputs(T=1), backend='recurrent_triton')   # no CUDA here
    with pytest.raises(ValueError):
        G.gdn(*inputs(T=2), backend='nope')


def test_rejects_bad_head_grouping_and_shapes():
    q, k, v, g, beta, s0 = inputs(Hk=3, Hv=4)
    with pytest.raises(ValueError):
        G.gdn(q, k, v, g, beta, s0, backend='chunk_reference')
    q, k, v, g, beta, s0 = inputs()
    with pytest.raises(ValueError):
        G.gdn(q[:, :2], k, v, g, beta, s0)


def test_output_dtype_follows_query_and_state_is_fp32():
    o, s = G.gdn(*inputs(T=6, dtype=torch.bfloat16), backend='chunk_reference')
    assert o.dtype == torch.bfloat16 and s.dtype == torch.float32


def test_prefill_backend_selection():
    q = torch.zeros(1, 8, 1, 1)
    assert G.select_backend(q, prefill='reference') == 'chunk_reference'
    assert G.select_backend(q, prefill='fla') == 'fla_chunk'
    assert G.select_backend(q, prefill='auto') == 'chunk_reference'  # CPU tensors never pick fla
    with pytest.raises(ValueError):
        G.select_backend(q, prefill='nope')


def test_fla_backend_requires_cuda_and_package():
    with pytest.raises((RuntimeError, ImportError, ModuleNotFoundError)):
        G.gdn(*inputs(T=8), backend='fla_chunk')


@pytest.mark.skipif(not (torch.cuda.is_available() and G.fla_available()), reason='needs CUDA + flash-linear-attention')
def test_fla_chunk_matches_oracle_on_gpu():
    args = [x.cuda() if x is not None else None for x in inputs(T=130, K=64, V=64)]
    o_ref, s_ref = G.gdn(*args, backend='recurrent_reference')
    o, s = G.gdn(*args, backend='fla_chunk')
    torch.testing.assert_close(o, o_ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(s, s_ref, atol=2e-3, rtol=2e-3)
