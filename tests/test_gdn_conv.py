"""N4.2 GDN prefill conv primitive: reference == upstream math; fla == reference on CUDA."""
import asyncio

import pytest

torch = pytest.importorskip('torch')
F = torch.nn.functional

from inferstack.kernels import gdn_conv as C  # noqa: E402


def upstream(x, weight, bias, state):
    """transformers Qwen3.5 prefill conv: cache concat (or zero pad), F.conv1d, slice, SiLU."""
    B, T, D = x.shape
    W = weight.shape[1]
    xt = x.transpose(1, 2)
    full = torch.cat([state, xt], -1) if state is not None else xt
    y = F.conv1d(full, weight.unsqueeze(1), bias, padding=W - 1, groups=D)[:, :, :full.shape[-1]]
    y = F.silu(y)[:, :, -T:]
    prev = state if state is not None else torch.zeros(B, D, W)
    return y.transpose(1, 2), torch.cat([prev, xt], -1)[..., -W:]


def data(B=2, T=9, D=6, W=4, seed=0, state=True):
    g = torch.Generator().manual_seed(seed)
    x = torch.randn(B, T, D, generator=g)
    w = torch.randn(D, W, generator=g)
    b = torch.randn(D, generator=g)
    s = torch.randn(B, D, W, generator=g) if state else None
    return x, w, b, s


@pytest.mark.parametrize('T', [1, 3, 4, 9, 70])
@pytest.mark.parametrize('with_state', [False, True])
def test_reference_matches_upstream(T, with_state):
    x, w, b, s = data(T=T, state=with_state, seed=T)
    y, final = C.causal_conv_prefill(x, w, b, s)
    y_ref, f_ref = upstream(x, w, b, s)
    torch.testing.assert_close(y, y_ref, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(final, f_ref)


def test_chunked_equals_one_pass():
    x, w, b, _ = data(T=33, seed=5, state=False)
    y_all, f_all = C.causal_conv_prefill(x, w, b)
    y1, s = C.causal_conv_prefill(x[:, :20], w, b)
    y2, s = C.causal_conv_prefill(x[:, 20:22], w, b, s)      # shorter than the kernel
    y3, s = C.causal_conv_prefill(x[:, 22:], w, b, s)
    torch.testing.assert_close(torch.cat([y1, y2, y3], 1), y_all, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(s, f_all)


def test_strided_projection_view_is_accepted():
    """The layer passes a column slice of the fused qkvz projection (not contiguous)."""
    x, w, b, s = data(T=7, D=6)
    wide = torch.cat([x, torch.randn(2, 7, 5)], -1)[..., :6]
    torch.testing.assert_close(C.causal_conv_prefill(wide, w, b, s)[0], C.causal_conv_prefill(x.contiguous(), w, b, s)[0])


def test_rejects_bad_shapes_and_backend():
    x, w, b, s = data()
    with pytest.raises(ValueError):
        C.causal_conv_prefill(x, w[:3], b, s)
    with pytest.raises(ValueError):
        C.causal_conv_prefill(x, w, b, s[:, :, :2])
    with pytest.raises(ValueError):
        C.causal_conv_prefill(x, w, b, s, backend='nope')
    with pytest.raises(RuntimeError):
        C.causal_conv_prefill(x, w, b, s, backend='fla')     # CPU tensors


needs_fla = pytest.mark.skipif(not (torch.cuda.is_available() and C.fla_conv_available()),
                               reason='needs CUDA + flash-linear-attention')


@needs_fla
@pytest.mark.parametrize('dtype', [torch.float32, torch.bfloat16])
@pytest.mark.parametrize('T', [1, 3, 64, 1000])
@pytest.mark.parametrize('with_state', [False, True])
def test_fla_matches_reference_on_gpu(dtype, T, with_state):
    x, w, b, s = (t.cuda() if t is not None else None for t in data(B=3, T=T, D=96, seed=T, state=with_state))
    x, w, b = x.to(dtype), w.to(dtype), b.to(dtype)
    s = None if s is None else s.to(dtype)
    y, f = C.causal_conv_prefill(x, w, b, s, backend='fla')
    y_ref, f_ref = C.causal_conv_prefill(x, w, b, s, backend='reference')
    tol = 2e-2 if dtype == torch.bfloat16 else 1e-4
    torch.testing.assert_close(y.float(), y_ref.float(), atol=tol, rtol=tol)
    torch.testing.assert_close(f.float(), f_ref.float())


@needs_fla
def test_engine_greedy_outputs_same_with_fla_conv():
    """Tiny real Qwen3.5 on CUDA with forward fusion: fla conv == reference conv, token for token."""
    pytest.importorskip('transformers.models.qwen3_5.modeling_qwen3_5')
    from inferstack.core.request import Request, SamplingParams
    from inferstack.runtime.qwen_continuous import QwenHybridContinuousEngine
    from test_qwen_static_decode import tiny_model
    from test_qwen_static_lane import PLAN, Tok, cfg, drain

    def run(conv):
        c = cfg(False, prefix=True)
        c.native.forward_fusion = True
        c.native.fuse_gdn_projections = True
        c.native.gdn_conv_backend = conv

        async def go():
            e = QwenHybridContinuousEngine(tiny_model(seed=11).to('cuda', torch.float32), Tok(), c, 'cuda:0')
            await e.start()

            async def one(delay, rid, prompt, max_new):
                await asyncio.sleep(delay)
                q = await e.submit(Request(rid, tuple(prompt), SamplingParams(max_new_tokens=max_new, temperature=0.0)))
                return rid, await drain(q)
            res = dict(await asyncio.gather(*(one(*p) for p in PLAN)))
            await e.close()
            return res
        return asyncio.run(go())

    C.STATS.clear()
    ref = run('reference')
    assert C.STATS['fla'] == 0
    got = run('fla')
    assert C.STATS['fla'] > 0, 'fla conv never ran: fusion not installed on the GDN layers?'
    assert got == ref


def test_missing_fla_backend_fails_before_touching_the_layer(monkeypatch):
    """n51: the fla availability check ran after the projections were removed, leaving a broken
    layer behind a swallowed error. It must raise first and leave the module intact."""
    pytest.importorskip('transformers.models.qwen3_5.modeling_qwen3_5')
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    from inferstack.kernels import gdn as G
    from inferstack.kernels import qwen_forward_fusion as FF
    from test_qwen_static_decode import tiny_model
    from test_qwen_static_lane import cfg
    model = tiny_model(seed=3)
    layer = next(m for m in model.modules() if hasattr(m, 'in_proj_qkv'))
    c = cfg(False)
    c.native.forward_fusion = True
    c.native.fuse_gdn_projections = True
    c.native.gdn_prefill_backend = 'fla'
    monkeypatch.setattr(G, 'fla_available', lambda: False)
    with pytest.raises(FF.FusionBackendUnavailable):
        FF._install_gdn(layer, mod, c)
    assert layer.in_proj_qkv is not None and not getattr(layer, '_inferstack_gdn_fused', False)
