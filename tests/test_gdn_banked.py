"""N6.1b banked GDN decode: torch reference vs token-by-token grouped reference; Triton vs reference."""
import pytest

torch = pytest.importorskip('torch')

from inferstack.kernels import gdn_banked as GB  # noqa: E402
from inferstack.kernels.gdn_triton import grouped_gdn_decode_reference  # noqa: E402


def data(B=3, T=2, Hk=2, Hv=4, K=16, V=8, slots=4, seed=0, device='cpu'):
    g = torch.Generator().manual_seed(seed)
    r = lambda *s: torch.randn(*s, generator=g)   # noqa: E731
    q, k, v = r(B, T, Hk, K), r(B, T, Hk, K), r(B, T, Hv, V)
    gate = -torch.rand(B, T, Hv, generator=g)
    beta = torch.rand(B, T, Hv, generator=g)
    banks = r(2, slots, Hv, K, V)
    sel = torch.tensor([0, 1, 1, 0][:slots], dtype=torch.int32)
    return [t.to(device) for t in (q, k, v, gate, beta, banks, sel)]


def stepwise(q, k, v, gate, beta, state):
    outs = []
    for t in range(q.shape[1]):
        o, state = grouped_gdn_decode_reference(q[:, t:t + 1], k[:, t:t + 1], v[:, t:t + 1], gate[:, t:t + 1],
                                                beta[:, t:t + 1], state, True, True)
        outs.append((o, state))
    return outs


@pytest.mark.parametrize('T', [1, 2])
def test_reference_matches_token_by_token_and_banks_follow_the_contract(T):
    q, k, v, gate, beta, banks, sel = data(T=T)
    B = q.shape[0]
    before = banks.clone()
    cur0 = torch.stack([before[int(sel[b]), b] for b in range(B)])
    steps = stepwise(q, k, v, gate, beta, cur0)
    out = GB.gdn_banked(q, k, v, gate, beta, banks, sel, backend='torch')
    for t in range(T):
        torch.testing.assert_close(out[:, t], steps[t][0][:, 0].float(), atol=1e-5, rtol=1e-5)
    for b in range(B):
        s = int(sel[b])
        torch.testing.assert_close(banks[s, b], steps[-1][1][b], atol=1e-5, rtol=1e-5)        # current = last
        other = steps[0][1][b] if T == 2 else before[1 - s, b]                                  # other = after token 1
        torch.testing.assert_close(banks[1 - s, b], other, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(banks[:, B:], before[:, B:])                                     # unused slots untouched


def test_rejects_bad_shapes():
    q, k, v, gate, beta, banks, sel = data(T=2)
    with pytest.raises(ValueError):
        GB.gdn_banked(q.repeat(1, 2, 1, 1), k.repeat(1, 2, 1, 1), v.repeat(1, 2, 1, 1), gate.repeat(1, 2, 1),
                      beta.repeat(1, 2, 1), banks, sel, backend='torch')


needs_cuda = pytest.mark.skipif(not (torch.cuda.is_available() and GB.HAS_TRITON), reason='needs CUDA + triton')


@needs_cuda
@pytest.mark.parametrize('T', [1, 2])
@pytest.mark.parametrize('B,Hk,Hv,K,V', [(1, 16, 16, 128, 128), (5, 16, 32, 128, 128), (3, 2, 4, 64, 48)])
def test_triton_matches_reference(T, B, Hk, Hv, K, V):
    q, k, v, gate, beta, banks, sel = data(B=B, T=T, Hk=Hk, Hv=Hv, K=K, V=V, slots=B + 2, seed=B + T, device='cuda')
    sel = (torch.arange(B + 2, device='cuda') % 2).to(torch.int32)
    q, k, v, beta = q.bfloat16(), k.bfloat16(), v.bfloat16(), beta.bfloat16()
    ref_banks = banks.clone()
    ref = GB.gdn_banked(q, k, v, gate, beta, ref_banks, sel, backend='torch')
    got = GB.gdn_banked(q, k, v, gate, beta, banks, sel)
    torch.testing.assert_close(got, ref, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(banks, ref_banks, atol=2e-3, rtol=2e-3)


def _lazy_then_fixup(q, k, v, gate, beta, banks, sel, acc, backend):
    """P10 lazy mode end to end: lazy step, fix-up of rejected rows, selector update.
    Returns outputs and each row's current state afterwards."""
    B = q.shape[0]
    out = GB.gdn_banked(q, k, v, gate, beta, banks, sel, backend=backend, lazy=True)
    GB.gdn_banked_fixup(k[:, 0], v[:, 0], gate[:, 0], beta[:, 0], banks, sel, acc, backend=backend)
    new = torch.where(acc, 1 - sel[:B], sel[:B])
    return out, torch.stack([banks[int(new[b]), b] for b in range(B)])


def test_lazy_bank_with_fixup_matches_eager_banks():
    """Accepted rows end on token 2's state, rejected rows on token 1's, as in the eager banks."""
    q, k, v, gate, beta, banks, sel = data(B=4, T=2, slots=4)
    acc = torch.tensor([True, False, True, False])
    eager = banks.clone()
    ref_out = GB.gdn_banked(q, k, v, gate, beta, eager, sel, backend='torch')
    ref_sel = torch.where(acc, sel, 1 - sel)
    ref_cur = torch.stack([eager[int(ref_sel[b]), b] for b in range(4)])
    out, cur = _lazy_then_fixup(q, k, v, gate, beta, banks, sel, acc, 'torch')
    torch.testing.assert_close(out, ref_out)
    torch.testing.assert_close(cur, ref_cur, atol=1e-5, rtol=1e-5)


@needs_cuda
@pytest.mark.parametrize('B,Hk,Hv,K,V', [(1, 16, 16, 128, 128), (7, 16, 32, 128, 128), (3, 2, 4, 64, 48)])
def test_triton_lazy_bank_and_fixup_match_reference(B, Hk, Hv, K, V):
    q, k, v, gate, beta, banks, sel = data(B=B, T=2, Hk=Hk, Hv=Hv, K=K, V=V, slots=B + 2, seed=B, device='cuda')
    sel = (torch.arange(B + 2, device='cuda') % 2).to(torch.int32)
    acc = (torch.arange(B, device='cuda') % 3) != 1
    q, k, v, beta = q.bfloat16(), k.bfloat16(), v.bfloat16(), beta.bfloat16()
    ref_banks = banks.clone()
    ref_out, ref_cur = _lazy_then_fixup(q, k, v, gate, beta, ref_banks, sel, acc, 'torch')
    out, cur = _lazy_then_fixup(q, k, v, gate, beta, banks, sel, acc, 'triton')
    torch.testing.assert_close(out, ref_out, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(cur, ref_cur, atol=2e-3, rtol=2e-3)
    torch.testing.assert_close(banks[:, B:], ref_banks[:, B:])


@pytest.mark.parametrize('backend', ['torch', pytest.param('triton', marks=needs_cuda)])
def test_lazy_three_tokens_with_partial_accept_matches_stepwise(backend):
    """P11: a lazy T = 3 step plus gdn_banked_fixup_n leaves each row on the state after the tokens
    it kept (1, 2 or all 3), equal to running those tokens one by one."""
    dev = 'cuda' if backend == 'triton' else 'cpu'
    B, Hk, Hv, K, V = 6, 2, 4, 64, 48
    q, k, v, gate, beta, banks, sel = data(B=B, T=3, Hk=Hk, Hv=Hv, K=K, V=V, slots=B + 1, seed=3, device=dev)
    sel = (torch.arange(B + 1, device=dev) % 2).to(torch.int32)
    cnt = torch.tensor([1, 2, 3, 3, 1, 2], device=dev)
    before = banks.clone()
    cur0 = torch.stack([before[int(sel[b]), b] for b in range(B)])
    steps = stepwise(q.float(), k.float(), v.float(), gate, beta.float(), cur0.clone())
    out = GB.gdn_banked(q, k, v, gate, beta, banks, sel, backend=backend, lazy=True)
    GB.gdn_banked_fixup_n(k[:, :2], v[:, :2], gate[:, :2], beta[:, :2], banks, sel, cnt, backend=backend)
    new = torch.where(cnt == 3, 1 - sel[:B], sel[:B])
    tol = dict(atol=2e-3, rtol=2e-3) if backend == 'triton' else dict(atol=1e-5, rtol=1e-5)
    for t in range(3):
        torch.testing.assert_close(out[:, t], steps[t][0][:, 0].float(), **tol)
    for b in range(B):
        torch.testing.assert_close(banks[int(new[b]), b], steps[int(cnt[b]) - 1][1][b], **tol)
    torch.testing.assert_close(banks[:, B:], before[:, B:])
