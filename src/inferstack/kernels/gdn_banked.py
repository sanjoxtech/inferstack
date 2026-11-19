"""N6.1b: banked Gated DeltaNet decode for T = 1 or 2 tokens per row.

The N6.1 verify pass ran the grouped decode kernel once per token, copied the state after
token 1 aside, then restored rejected rows with torch.where. Each pass streams the fp32
recurrent state (1 MB per row per layer on Qwen3.5-2B). The grouped kernel itself streams it
four times per token (load/store after the decay, load/store after the update). At batch 16 that
was ~2 GB of extra traffic per verify step.

Here every row has two state banks and a selector `sel[b]` that says which bank is current.
One program owns a (row, value head, V tile) slice of the state:

1. it loads the slice once (from bank sel) and keeps it in registers;
2. it runs the T tokens (decay, delta rule, output), with Q/K L2 norm fused;
3. for T = 2 it stores the state after token 1 into the other bank, and after token 2 into the
   current bank.

After the verify, a rejected row only flips its selector (`sel = where(accept, sel, 1 - sel)`),
so nothing is copied back. Traffic per row and layer: 1 read + T writes.

Layout:
    q, k  : [B, T, Hk, K] contiguous      v : [B, T, Hv, V] contiguous
    g     : [B, T, Hv] fp32 (log decay)   beta : [B, T, Hv]
    banks : [2, slots, Hv, K, V] fp32 contiguous; rows 0..B-1 are used
    sel   : [slots] int32 (0 or 1)
    returns o [B, T, Hv, V] fp32

`reference` is the torch oracle with the same semantics (also the CPU path).

P10 lazy mode (`lazy=True`): T = 2 stores only the state after token 2, into the other bank, and
leaves the current bank (state before the step) untouched: 1 read + 1 write. Drafts are accepted
~96% of the time on Qwen3.5-2B, so the state after token 1 is rarely needed. After the verify:
accepted rows flip `sel` (to the bank holding token 2's state); rejected rows keep `sel` and
`gdn_banked_fixup` rebuilds token 1's state in place from the kept state and token 1's k, v,
gate, beta (masked loads, so accepted rows cost no state traffic).
"""
from __future__ import annotations

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _banked_kernel(q, k, v, g, beta, banks, sel, out, bank_stride,
                       Hk: tl.constexpr, Hv: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                       GROUP: tl.constexpr, T: tl.constexpr, QSCALE: tl.constexpr, BV: tl.constexpr,
                       LAZY: tl.constexpr):
        bhv = tl.program_id(0); vb = tl.program_id(1)
        b = bhv // Hv; hv = bhv - b * Hv; hk = hv // GROUP
        ko = tl.arange(0, K)
        vo = vb * BV + tl.arange(0, BV); vm = vo < V
        s = tl.load(sel + b).to(tl.int64)
        offs = bhv * K * V + ko[:, None] * V + vo[None, :]
        cur = banks + s * bank_stride + offs
        oth = banks + (1 - s) * bank_stride + offs
        S = tl.load(cur, mask=vm[None, :], other=0.0)
        for t in tl.static_range(T):
            qk = ((b * T + t) * Hk + hk) * K + ko
            qv = tl.load(q + qk).to(tl.float32)
            kv = tl.load(k + qk).to(tl.float32)
            qv = qv * tl.rsqrt(tl.sum(qv * qv, 0) + 1e-6) * QSCALE
            kv = kv * tl.rsqrt(tl.sum(kv * kv, 0) + 1e-6)
            gi = (b * T + t) * Hv + hv
            S = S * tl.exp(tl.load(g + gi))
            mem = tl.sum(S * kv[:, None], 0)
            vv = tl.load(v + gi * V + vo, mask=vm, other=0.0).to(tl.float32)
            delta = (vv - mem) * tl.load(beta + gi).to(tl.float32)
            S = S + kv[:, None] * delta[None, :]
            tl.store(out + gi * V + vo, tl.sum(S * qv[:, None], 0), mask=vm)
            if T == 2 and t == 0 and not LAZY:
                tl.store(oth, S, mask=vm[None, :])
        if LAZY and T == 2:
            tl.store(oth, S, mask=vm[None, :])
        else:
            tl.store(cur, S, mask=vm[None, :])

    @triton.jit
    def _fixup_kernel(k, v, g, beta, banks, sel, acc, bank_stride,
                      Hk: tl.constexpr, Hv: tl.constexpr, K: tl.constexpr, V: tl.constexpr,
                      GROUP: tl.constexpr, BV: tl.constexpr):
        bhv = tl.program_id(0); vb = tl.program_id(1)
        b = bhv // Hv; hv = bhv - b * Hv; hk = hv // GROUP
        rej = tl.load(acc + b) == 0
        ko = tl.arange(0, K)
        vo = vb * BV + tl.arange(0, BV); vm = (vo < V) & rej
        s = tl.load(sel + b).to(tl.int64)
        ptr = banks + s * bank_stride + bhv * K * V + ko[:, None] * V + vo[None, :]
        S = tl.load(ptr, mask=vm[None, :], other=0.0)
        kv = tl.load(k + (b * Hk + hk) * K + ko, mask=(ko < K) & rej, other=0.0).to(tl.float32)
        kv = kv * tl.rsqrt(tl.sum(kv * kv, 0) + 1e-6)
        gi = b * Hv + hv
        S = S * tl.exp(tl.load(g + gi))
        mem = tl.sum(S * kv[:, None], 0)
        vv = tl.load(v + gi * V + vo, mask=vm, other=0.0).to(tl.float32)
        S = S + kv[:, None] * ((vv - mem) * tl.load(beta + gi).to(tl.float32))[None, :]
        tl.store(ptr, S, mask=vm[None, :])

    HAS_TRITON = True
except Exception:  # CPU-only environments
    HAS_TRITON = False


def _l2(x):
    import torch
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + 1e-6)


def reference(q, k, v, g, beta, banks, sel, lazy=False):
    """Same contract as `gdn_banked`, in torch (fp32)."""
    import torch
    B, T, Hk, K = q.shape
    Hv = v.shape[2]
    rep = Hv // Hk
    qf = _l2(q.float()).repeat_interleave(rep, dim=2) / (K ** 0.5)
    kf = _l2(k.float()).repeat_interleave(rep, dim=2)
    vf, gf, bf = v.float(), g.float(), beta.float()
    s = sel[:B].long().view(B, 1, 1, 1)
    S = torch.where(s == 0, banks[0, :B], banks[1, :B])
    outs, first = [], None
    for t in range(T):
        S = S * gf[:, t].exp()[..., None, None]
        mem = (S * kf[:, t].unsqueeze(-1)).sum(-2)
        S = S + kf[:, t].unsqueeze(-1) * ((vf[:, t] - mem) * bf[:, t].unsqueeze(-1)).unsqueeze(-2)
        outs.append((S * qf[:, t].unsqueeze(-1)).sum(-2))
        if t == 0:
            first = S
    if T == 2 and lazy:                  # token 2's state into the other bank, current kept
        banks[0, :B].copy_(torch.where(s == 0, banks[0, :B], S))
        banks[1, :B].copy_(torch.where(s == 0, S, banks[1, :B]))
    elif T == 2:
        banks[0, :B].copy_(torch.where(s == 0, S, first))
        banks[1, :B].copy_(torch.where(s == 0, first, S))
    else:
        banks[0, :B].copy_(torch.where(s == 0, S, banks[0, :B]))
        banks[1, :B].copy_(torch.where(s == 0, banks[1, :B], S))
    return torch.stack(outs, dim=1)


def fixup_reference(k, v, g, beta, banks, sel, acc):
    """Torch oracle for `gdn_banked_fixup`: k [B, Hk, K], v [B, Hv, V], g/beta [B, Hv]."""
    import torch
    B, Hk, K = k.shape
    Hv = v.shape[1]
    kf = _l2(k.float()).repeat_interleave(Hv // Hk, dim=1)
    s = sel[:B].long().view(B, 1, 1, 1)
    S = torch.where(s == 0, banks[0, :B], banks[1, :B])
    S = S * g.float().exp()[..., None, None]
    mem = (S * kf.unsqueeze(-1)).sum(-2)
    S = S + kf.unsqueeze(-1) * ((v.float() - mem) * beta.float().unsqueeze(-1)).unsqueeze(-2)
    rej = (~acc[:B].bool()).view(B, 1, 1, 1)
    banks[0, :B].copy_(torch.where(rej & (s == 0), S, banks[0, :B]))
    banks[1, :B].copy_(torch.where(rej & (s == 1), S, banks[1, :B]))


def gdn_banked_fixup(k, v, g, beta, banks, sel, acc, *, backend='triton'):
    """P10 lazy mode: for rows with acc == False, apply token 1 (k, v, g, beta) to bank[sel] in place."""
    import torch
    B, Hk, K = k.shape
    Hv, V = v.shape[1], v.shape[2]
    if backend == 'torch' or not (HAS_TRITON and k.is_cuda):
        return fixup_reference(k, v, g, beta, banks, sel, acc)
    k, v = k.contiguous(), v.contiguous()
    g, beta = g.float().contiguous(), beta.contiguous()
    a = acc[:B].to(torch.int32)
    BV = 32
    _fixup_kernel[(B * Hv, triton.cdiv(V, BV))](k, v, g, beta, banks, sel, a, banks.stride(0),
                                               Hk=Hk, Hv=Hv, K=K, V=V, GROUP=Hv // Hk, BV=BV, num_warps=4)


def gdn_banked(q, k, v, g, beta, banks, sel, *, backend='triton', lazy=False):
    import torch
    B, T, Hk, K = q.shape
    Hv, V = v.shape[2], v.shape[3]
    if T not in (1, 2) or Hv % Hk or banks.shape[0] != 2 or banks.dtype != torch.float32:
        raise ValueError(f'bad banked GDN inputs q={tuple(q.shape)} v={tuple(v.shape)} banks={tuple(banks.shape)}')
    if backend == 'torch':
        return reference(q, k, v, g, beta, banks, sel, lazy)
    if not HAS_TRITON or not q.is_cuda:
        raise RuntimeError('banked GDN triton kernel needs CUDA + triton')
    if K & (K - 1) or not banks.is_contiguous():
        raise ValueError('banked GDN kernel needs power-of-two K and contiguous banks')
    q, k, v = q.contiguous(), k.contiguous(), v.contiguous()
    g, beta = g.float().contiguous(), beta.contiguous()
    out = torch.empty((B, T, Hv, V), device=q.device, dtype=torch.float32)
    BV = 32
    _banked_kernel[(B * Hv, triton.cdiv(V, BV))](q, k, v, g, beta, banks, sel, out, banks.stride(0),
                                                 Hk=Hk, Hv=Hv, K=K, V=V, GROUP=Hv // Hk, T=T,
                                                 QSCALE=1.0 / (K ** 0.5), BV=BV, LAZY=bool(lazy), num_warps=4)
    return out
