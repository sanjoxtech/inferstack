"""P5: FLA's chunked gated delta rule forward that also returns the per-chunk states.

FLA's public `chunk_gated_delta_rule` keeps `h` (the recurrent state at the start of every
64-token chunk, [1, NT, HV, K, V]) internal. A prefix-cache snapshot at a block boundary b
inside a prompt needs exactly h at the chunk that starts at b, so a cold prompt can leave a
snapshot from its single prefill pass instead of a second prefill of its prefix (SGLang's
gdn_prefill_track records the same thing; ideas only, no code taken).

This mirrors the forward of fla-core 0.6.0 (MIT): chunk.ChunkGatedDeltaRuleFunction.forward
with use_qk_l2norm_in_kernel=True, gate not in kernel, no context parallelism, followed by
chunk_gated_delta_rule_fwd. It calls the same internal ops in the same order, so o and
final_state equal the public call's. Pinned to fla-core 0.6.0; other versions raise.

h is stored by FLA in k's dtype (bf16 here): a boundary state read from it is rounded to bf16,
unlike final_state (fp32).
"""
from __future__ import annotations

CHUNK = 64
FLA_VERSION = '0.6.0'


def available():
    try:
        import fla
        return getattr(fla, '__version__', '') == FLA_VERSION
    except Exception:
        return False


def chunk_gdr_with_states(q, k, v, g, beta, scale, initial_state, cu_seqlens):
    """q, k: [1, T, H, K]; v: [1, T, HV, V]; g, beta: [1, T, HV]; cu_seqlens int32 [N+1].
    Returns (o [1, T, HV, V], final_state [N, HV, K, V] fp32, h [1, NT, HV, K, V])."""
    import fla
    if getattr(fla, '__version__', '') != FLA_VERSION:
        raise RuntimeError(f'fla_gdr_states mirrors fla-core {FLA_VERSION}, found {fla.__version__}')
    from fla.modules.norm.l2norm import l2norm_fwd
    from fla.ops.common.chunk_delta_h import chunk_gated_delta_rule_fwd_h
    from fla.ops.common.chunk_o import chunk_fwd_o
    from fla.ops.gated_delta_rule.chunk_fwd import chunk_gated_delta_rule_fwd_intra
    from fla.ops.utils import chunk_local_cumsum
    from fla.ops.utils.constant import RCP_LN2
    from fla.ops.utils.index import prepare_chunk_indices
    q, k, v, g, beta = (x.contiguous() for x in (q, k, v, g, beta))
    if initial_state is not None:
        initial_state = initial_state.contiguous()
    q, _ = l2norm_fwd(q)
    k, _ = l2norm_fwd(k)
    chunk_indices = prepare_chunk_indices(cu_seqlens, CHUNK)
    g = chunk_local_cumsum(g, chunk_size=CHUNK, scale=RCP_LN2, cu_seqlens=cu_seqlens, chunk_indices=chunk_indices)
    w, u, _ = chunk_gated_delta_rule_fwd_intra(k=k, v=v, g=g, beta=beta, cu_seqlens=cu_seqlens,
                                               chunk_indices=chunk_indices, chunk_size=CHUNK)
    h, v_new, final_state = chunk_gated_delta_rule_fwd_h(k=k, w=w, u=u, g=g, initial_state=initial_state,
                                                         output_final_state=True, cu_seqlens=cu_seqlens,
                                                         chunk_indices=chunk_indices, state_v_first=False,
                                                         chunk_size=CHUNK)
    o = chunk_fwd_o(q=q, k=k, v=v_new, h=h, g=g, scale=scale, cu_seqlens=cu_seqlens,
                    chunk_indices=chunk_indices, state_v_first=False, chunk_size=CHUNK)
    return o, final_state, h


def chunk_offsets(lens):
    """Host-side index of each sequence's first chunk in h (FLA's prepare_chunk_offsets)."""
    out = [0]
    for n in lens:
        out.append(out[-1] + (int(n) + CHUNK - 1) // CHUNK)
    return out
