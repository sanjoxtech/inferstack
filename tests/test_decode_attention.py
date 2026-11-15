"""N5.1 decode attention: reference semantics on CPU; Triton split-KV kernel == reference on CUDA."""
import pytest

torch = pytest.importorskip('torch')

from inferstack.kernels import decode_attention as DA  # noqa: E402


def data(B=3, Hq=8, Hkv=2, D=64, Lmax=96, seed=0, device='cpu', dtype=torch.float32, gaps=True):
    g = torch.Generator().manual_seed(seed)
    q = torch.randn(B, Hq, D, generator=g)
    k = torch.randn(B, Hkv, Lmax, D, generator=g)
    v = torch.randn(B, Hkv, Lmax, D, generator=g)
    pos = torch.randint(0, Lmax, (B,), generator=g)
    mask = torch.zeros(B, Lmax, dtype=torch.bool)
    for b in range(B):
        start = int(torch.randint(0, int(pos[b]) + 1, (1,), generator=g))
        mask[b, start:int(pos[b]) + 1] = True                  # left padding before start
        if gaps and pos[b] > 8:
            mask[b, start + 1:start + 3] = False                # prefix-cache style gap
        mask[b, int(pos[b])] = True                             # the current token is always visible
        mask[b, int(pos[b]) + 1:] = bool(b % 2)                 # stale True past pos must be ignored
    to = lambda t: t.to(device=device, dtype=dtype if t.is_floating_point() else t.dtype)  # noqa: E731
    return to(q), to(k), to(v), mask.to(device), pos.to(device)


def manual(q, k, v, mask, pos, L, scale):
    B, Hq, D = q.shape
    G = Hq // k.shape[1]
    out = torch.empty_like(q, dtype=torch.float32)
    for b in range(B):
        cols = [c for c in range(L) if mask[b, c] and c <= pos[b]]
        for h in range(Hq):
            kk, vv = k[b, h // G, cols].float(), v[b, h // G, cols].float()
            w = torch.softmax((kk @ q[b, h].float()) * scale, 0)
            out[b, h] = w @ vv
    return out


def test_reference_matches_manual_and_ignores_columns_past_pos():
    q, k, v, mask, pos = data()
    L = 96
    got = DA.decode_attention(q, k, v, mask, pos, L, 0.125)
    torch.testing.assert_close(got.float(), manual(q, k, v, mask, pos, L, 0.125), atol=1e-5, rtol=1e-5)


def test_rejects_bad_backend_and_shapes():
    q, k, v, mask, pos = data()
    with pytest.raises(ValueError):
        DA.decode_attention(q, k, v, mask, pos, 96, 0.1, backend='nope')
    with pytest.raises(ValueError):
        DA.decode_attention(q[:, :7], k, v, mask, pos, 96, 0.1)
    with pytest.raises(RuntimeError):
        DA.decode_attention(q, k, v, mask, pos, 96, 0.1, backend='triton')   # CPU tensors


needs_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and DA.HAS_TRITON), reason='needs CUDA + triton')


@needs_cuda
@pytest.mark.parametrize('B,D,Lmax,L', [(1, 256, 64, 64), (3, 256, 600, 512), (16, 256, 2048, 2048), (5, 128, 300, 300)])
@pytest.mark.parametrize('dtype', [torch.bfloat16, torch.float32])
def test_triton_matches_reference(B, D, Lmax, L, dtype):
    q, k, v, mask, pos = data(B=B, D=D, Lmax=Lmax, seed=B * 7 + D, device='cuda', dtype=dtype)
    pos = pos.clamp_max(L - 1)
    mask[torch.arange(B, device='cuda'), pos] = True
    ref = DA.decode_attention(q, k, v, mask, pos, L, D ** -0.5)
    got = DA.decode_attention(q, k, v, mask, pos, L, D ** -0.5, backend='triton')
    tol = 2e-2 if dtype == torch.bfloat16 else 2e-3       # fp32 inputs go through tf32 dots
    torch.testing.assert_close(got.float(), ref.float(), atol=tol, rtol=tol)


@needs_cuda
def test_triton_replays_in_a_cuda_graph():
    q, k, v, mask, pos = data(B=4, D=256, Lmax=1024, device='cuda', dtype=torch.bfloat16)
    scratch = DA.make_scratch(4, 2, 4, 256, 1024, 'cuda')
    out = torch.empty_like(q)
    s = torch.cuda.Stream()
    s.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(s):
        out.copy_(DA.decode_attention(q, k, v, mask, pos, 1024, 0.0625, backend='triton', scratch=scratch))
    torch.cuda.current_stream().wait_stream(s)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out.copy_(DA.decode_attention(q, k, v, mask, pos, 1024, 0.0625, backend='triton', scratch=scratch))
    pos.add_(0)
    q.normal_()                                    # new inputs, same addresses
    graph.replay()
    ref = DA.decode_attention(q, k, v, mask, pos, 1024, 0.0625)
    torch.testing.assert_close(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
