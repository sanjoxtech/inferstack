"""N5.1: GQA decode attention for the static decode lane (one query token per row).

The lane used SDPA with a boolean mask over the whole length bucket. On the L4 that runs a
prefill-shaped (mem-efficient) kernel: every column of the bucket is read even past a row's
current position, and each query head reads its shared KV head again. At batch 16 the step grew
from 24.1 ms (~420-token context) to 34.3 ms (~1250 tokens; bucket 2048), in
docs/verification/evidence/n50-decode-step-l4-48c555b-20261115/.

This kernel is split-KV (flash-decoding):

- one program handles (row, kv head, column split);
- it loads each KV block once for all query heads in the group;
- it stops at the row's own end (pos + 1);
- it applies the lane's mask, because prefix-cache rows can have padding gaps;
- a second kernel merges the splits.

The grid depends only on (B, L bucket), so it can be captured in a CUDA graph.

Contract:
    q    : [B, Hq, D]           (one token per row; Hq % Hkv == 0)
    k, v : [B, Hkv, Lmax, D]    lane cache (only the first L columns are visible)
    mask : [B, Lmax] bool       True = attend
    pos  : [B] long             position of the current token (columns > pos are never read)
    returns [B, Hq, D] in q's dtype

`reference` (SDPA on the same mask, columns > pos masked) is the oracle.
"""
from __future__ import annotations

BACKENDS = ('sdpa', 'triton')
SPLIT = 256          # columns per split program
BLOCK_N = 32         # columns per inner iteration

try:
    import triton
    import triton.language as tl

    @triton.jit
    def _split_kernel(q, k, v, mask, pos, m_out, l_out, o_out,
                      sqb, sqh, skb, skh, skl, smb, ssb, ssh, sss, sob, soh, sos, sog,
                      scale, L,
                      G: tl.constexpr, GP: tl.constexpr, D: tl.constexpr,
                      SPLIT: tl.constexpr, BLOCK_N: tl.constexpr):
        b = tl.program_id(0); h = tl.program_id(1); s = tl.program_id(2)
        end = tl.minimum(tl.load(pos + b) + 1, L)
        start = s * SPLIT
        offs_g = tl.arange(0, GP); offs_d = tl.arange(0, D)
        gmask = offs_g < G
        qp = q + b * sqb + (h * G + offs_g)[:, None] * sqh + offs_d[None, :]
        qt = tl.load(qp, mask=gmask[:, None], other=0.0)
        m_i = tl.full((GP,), float('-inf'), tl.float32)
        l_i = tl.zeros((GP,), tl.float32)
        acc = tl.zeros((GP, D), tl.float32)
        for n0 in range(start, start + SPLIT, BLOCK_N):
            if n0 < end:
                offs_n = n0 + tl.arange(0, BLOCK_N)
                valid = (offs_n < end) & (tl.load(mask + b * smb + offs_n, mask=offs_n < end, other=0) != 0)
                kp = k + b * skb + h * skh + offs_n[:, None] * skl + offs_d[None, :]
                vp = v + b * skb + h * skh + offs_n[:, None] * skl + offs_d[None, :]
                kt = tl.load(kp, mask=valid[:, None], other=0.0)
                sc = tl.dot(qt, tl.trans(kt)) * scale                       # [GP, BLOCK_N]
                sc = tl.where(valid[None, :], sc, float('-inf'))
                m_new = tl.maximum(m_i, tl.max(sc, 1))
                m_safe = tl.where(m_new == float('-inf'), 0.0, m_new)
                p = tl.exp(sc - m_safe[:, None])
                alpha = tl.where(m_i == float('-inf'), 0.0, tl.exp(m_i - m_safe))
                vt = tl.load(vp, mask=valid[:, None], other=0.0)
                acc = acc * alpha[:, None] + tl.dot(p.to(vt.dtype), vt)
                l_i = l_i * alpha + tl.sum(p, 1)
                m_i = m_new
        base = b * ssb + h * ssh + s * sss
        tl.store(m_out + base + offs_g, m_i, mask=gmask)
        tl.store(l_out + base + offs_g, l_i, mask=gmask)
        op = o_out + b * sob + h * soh + s * sos + offs_g[:, None] * sog + offs_d[None, :]
        tl.store(op, acc, mask=gmask[:, None])

    @triton.jit
    def _merge_kernel(m_in, l_in, o_in, out, ssb, ssh, sss, sob, soh, sos, sog, s_outb, s_outh,
                      G: tl.constexpr, D: tl.constexpr, S: tl.constexpr):
        b = tl.program_id(0); hq = tl.program_id(1)
        h = hq // G; g = hq % G
        offs_d = tl.arange(0, D)
        m = tl.full((1,), float('-inf'), tl.float32)
        for s in range(S):
            m = tl.maximum(m, tl.load(m_in + b * ssb + h * ssh + s * sss + g + tl.zeros((1,), tl.int32)))
        m_safe = tl.where(m == float('-inf'), 0.0, m)
        l = tl.zeros((1,), tl.float32)
        acc = tl.zeros((D,), tl.float32)
        for s in range(S):
            ms = tl.load(m_in + b * ssb + h * ssh + s * sss + g + tl.zeros((1,), tl.int32))
            w = tl.where(ms == float('-inf'), 0.0, tl.exp(ms - m_safe))
            l += w * tl.load(l_in + b * ssb + h * ssh + s * sss + g + tl.zeros((1,), tl.int32))
            acc += w * tl.load(o_in + b * sob + h * soh + s * sos + g * sog + offs_d)
        res = acc / tl.where(l == 0.0, 1.0, l)
        tl.store(out + b * s_outb + hq * s_outh + offs_d, res.to(out.dtype.element_ty))

    HAS_TRITON = True
except Exception:  # CPU-only environments
    HAS_TRITON = False


def reference(q, k, v, mask, pos, L, scale):
    import torch
    import torch.nn.functional as F
    B, Hq, D = q.shape
    Hkv = k.shape[1]
    cols = torch.arange(L, device=q.device)
    m = mask[:, :L] & (cols[None, :] <= pos[:, None])
    kk = k[:, :, :L].repeat_interleave(Hq // Hkv, dim=1)
    vv = v[:, :, :L].repeat_interleave(Hq // Hkv, dim=1)
    o = F.scaled_dot_product_attention(q[:, :, None].float(), kk.float(), vv.float(),
                                       attn_mask=m[:, None, None, :], scale=scale)
    return o[:, :, 0].to(q.dtype)


def triton_decode(q, k, v, mask, pos, L, scale, scratch=None):
    """Two launches; `scratch` (from `make_scratch`) keeps addresses fixed for CUDA graphs."""
    import torch
    if not HAS_TRITON or not q.is_cuda:
        raise RuntimeError('triton decode attention needs CUDA + triton')
    B, Hq, D = q.shape
    Hkv = k.shape[1]
    G = Hq // Hkv
    S = -(-int(L) // SPLIT)
    m_s, l_s, o_s = scratch if scratch is not None else make_scratch(B, Hkv, G, D, int(L), q.device)
    m_s, l_s, o_s = m_s[:B, :, :S], l_s[:B, :, :S], o_s[:B, :, :S]
    if mask.dtype != torch.uint8 and mask.dtype != torch.bool:
        raise ValueError('mask must be bool/uint8')
    mk = mask.view(torch.uint8) if mask.dtype == torch.bool else mask
    GP = max(16, triton.next_power_of_2(G))
    _split_kernel[(B, Hkv, S)](
        q, k, v, mk, pos, m_s, l_s, o_s,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), k.stride(2), mk.stride(0),
        m_s.stride(0), m_s.stride(1), m_s.stride(2), o_s.stride(0), o_s.stride(1), o_s.stride(2), o_s.stride(3),
        float(scale), int(L), G=G, GP=GP, D=D, SPLIT=SPLIT, BLOCK_N=BLOCK_N, num_warps=4, num_stages=2)
    out = torch.empty_like(q)
    _merge_kernel[(B, Hq)](m_s, l_s, o_s, out, m_s.stride(0), m_s.stride(1), m_s.stride(2),
                           o_s.stride(0), o_s.stride(1), o_s.stride(2), o_s.stride(3), out.stride(0), out.stride(1),
                           G=G, D=D, S=S, num_warps=4)
    return out


def make_scratch(B, Hkv, G, D, Lmax, device):
    import torch
    S = -(-int(Lmax) // SPLIT)
    return (torch.empty(B, Hkv, S, G, dtype=torch.float32, device=device),
            torch.empty(B, Hkv, S, G, dtype=torch.float32, device=device),
            torch.empty(B, Hkv, S, G, D, dtype=torch.float32, device=device))


def decode_attention(q, k, v, mask, pos, L, scale, *, backend='sdpa', scratch=None):
    if backend not in BACKENDS:
        raise ValueError(f'unknown decode attention backend {backend!r}; choose from {BACKENDS}')
    if q.dim() != 3 or k.dim() != 4 or q.shape[1] % k.shape[1]:
        raise ValueError(f'bad shapes q={tuple(q.shape)} k={tuple(k.shape)}')
    if backend == 'triton':
        return triton_decode(q, k, v, mask, pos, L, scale, scratch)
    return reference(q, k, v, mask, pos, L, scale)
