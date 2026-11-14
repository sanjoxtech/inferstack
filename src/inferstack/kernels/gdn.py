"""N3.2: one native Gated DeltaNet primitive for decode and prefill.

The model layer calls `gdn(...)` with un-repeated query/key heads and gets the attention
output plus the final recurrent state. Which algorithm runs (Triton recurrent decode, the
upstream chunked algebra, a per-token reference, later a fused high-token prefill kernel)
is the primitive's decision, not the layer's or the scheduler's. This is the boundary
llama.cpp draws with its single gated_delta_net op (docs/research/LLAMA_CPP_V1_CODE_STUDY_2026-11-14.md §2–4).

Contract (same as transformers' torch_*_gated_delta_rule):

    query, key : [B, T, Hk, K]      value : [B, T, Hv, V]      Hv % Hk == 0
    g          : [B, T, Hv]  log-space decay (state *= exp(g) each token)
    beta       : [B, T, Hv]  write strength
    state      : [B, Hv, K, V] or None (zeros)
    returns    : out [B, T, Hv, V] in query's dtype, state' [B, Hv, K, V] fp32 (or None)

Q/K are L2-normalized when `qk_l2norm` (eps 1e-6, as FLA) and Q is scaled by K**-0.5.
Every backend computes exactly this; `recurrent_reference` is the numerical oracle.
"""
from __future__ import annotations

from collections import Counter

BACKENDS = ('auto', 'recurrent_triton', 'recurrent_upstream', 'chunk_reference', 'fla_chunk', 'recurrent_reference')
PREFILL_CHOICES = ('reference', 'fla', 'auto')
FLA_PIN = 'flash-linear-attention==0.6.0 / fla-core==0.6.0 (MIT)'
STATS: Counter = Counter()


def _repeat_heads(query, key, value):
    hk, hv = key.shape[2], value.shape[2]
    if hv % hk:
        raise ValueError(f'value heads {hv} not divisible by key heads {hk}')
    rep = hv // hk
    if rep == 1:
        return query, key
    return query.repeat_interleave(rep, dim=2), key.repeat_interleave(rep, dim=2)


def _torch_function(fn):
    """Strip every wrapper transformers puts around a GDN rule down to the torch function:
    functools.wraps layers (__wrapped__) and the `kernels` package's Func module, whose
    forward closes over the wrapped function as `func`."""
    seen = set()
    while id(fn) not in seen:
        seen.add(id(fn))
        inner = getattr(fn, '__wrapped__', None)
        if inner is not None and inner is not fn:
            fn = inner
            continue
        forward = getattr(getattr(fn, 'forward', None), '__func__', None)
        cells = dict(zip(forward.__code__.co_freevars, (c.cell_contents for c in forward.__closure__ or ()))) \
            if forward is not None and forward.__closure__ else {}
        if callable(cells.get('func')):
            fn = cells['func']
            continue
        break
    return fn


def _upstream_chunk():
    """transformers' torch chunked algebra, with its hub/package dispatch stripped.

    On the GPU image transformers wraps torch_chunk_gated_delta_rule in a `kernels` Func
    module around a dispatcher that runs FLA whenever flash-linear-attention is importable.
    `chunk_reference` must stay the PyTorch algorithm whatever is installed, so A/B runs
    (reference vs fla_chunk) are real."""
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    from inferstack.kernels import gdn_triton
    return _torch_function(gdn_triton._ORIGINAL_CHUNK or mod.torch_chunk_gated_delta_rule)


def _run_recurrent_triton(query, key, value, g, beta, state, output_state, qk_l2norm):
    from inferstack.kernels.gdn_triton import inferstack_grouped_gdn_decode
    return inferstack_grouped_gdn_decode(query, key, value, g, beta, state,
                                         output_final_state=output_state, use_qk_l2norm_in_kernel=qk_l2norm)


def _run_recurrent_upstream(query, key, value, g, beta, state, output_state, qk_l2norm, **kwargs):
    from transformers.models.qwen3_5 import modeling_qwen3_5 as mod
    q, k = _repeat_heads(query, key, value)
    return mod.torch_recurrent_gated_delta_rule(q, k, value, g=g, beta=beta, initial_state=state,
                                                output_final_state=output_state, use_qk_l2norm_in_kernel=qk_l2norm,
                                                **kwargs)


def _run_chunk_reference(query, key, value, g, beta, state, output_state, qk_l2norm, **kwargs):
    from inferstack.kernels import gdn_triton
    q, k = _repeat_heads(query, key, value)
    if gdn_triton._STATS.get('patched') and query.is_cuda and 1 < query.shape[1] <= 32:
        # Same short-chunk rule as the installed gdn_triton chunk wrapper.
        return gdn_triton.torch_gdn_chunk_reference(q, k, value, g, beta, state, output_state, qk_l2norm)
    return _upstream_chunk()(q, k, value, g=g, beta=beta, initial_state=state, output_final_state=output_state,
                             use_qk_l2norm_in_kernel=qk_l2norm, **kwargs)


def fla_available():
    try:
        from fla.ops.gated_delta_rule import chunk_gated_delta_rule  # noqa: F401
        return True
    except Exception:
        return False


def _run_fla_chunk(query, key, value, g, beta, state, output_state, qk_l2norm):
    """N3.3 high-token path: FLA's fused chunked kernel (Triton). Native GVA (no head
    repeat), [B, T, H, ...] layout, Q/K L2 norm in the kernel, scale K**-0.5."""
    from fla.ops.gated_delta_rule import chunk_gated_delta_rule
    if not query.is_cuda:
        raise RuntimeError('fla_chunk needs CUDA')
    out, final = chunk_gated_delta_rule(
        query, key, value, g=g.float(), beta=beta, scale=key.shape[-1] ** -0.5,
        initial_state=None if state is None else state.float().contiguous(),
        output_final_state=output_state, use_qk_l2norm_in_kernel=qk_l2norm)
    return out.to(query.dtype), (None if final is None else final.float())


def _run_recurrent_reference(query, key, value, g, beta, state, output_state, qk_l2norm):
    from inferstack.kernels.gdn_triton import torch_gdn_chunk_reference
    q, k = _repeat_heads(query, key, value)
    return torch_gdn_chunk_reference(q, k, value, g, beta, state, output_state, qk_l2norm)


def select_backend(query, *, prefer_triton_decode=True, prefill='reference'):
    """Backend for this call shape when the caller asks for 'auto'.

    prefill: 'reference' (upstream chunked algebra), 'fla' (fused kernel, required), or
    'auto' (fla when it is importable and the tensors are on CUDA)."""
    # One token: try the Triton recurrent kernel; without CUDA/Triton it falls back to the
    # upstream recurrent rule inside gdn(), exactly as the pre-N3.2 layer code did.
    if query.shape[1] == 1 and prefer_triton_decode:
        return 'recurrent_triton'
    if prefill not in PREFILL_CHOICES:
        raise ValueError(f'unknown prefill backend {prefill!r}; expected one of {PREFILL_CHOICES}')
    if prefill == 'fla' or (prefill == 'auto' and query.is_cuda and fla_available()):
        return 'fla_chunk'
    return 'chunk_reference'


def gdn(query, key, value, g, beta, state=None, *, output_state=True, qk_l2norm=True, backend='auto',
        prefer_triton_decode=True, prefill='reference', **kwargs):
    """Gated DeltaNet over T tokens. See the module docstring for the contract."""
    if backend not in BACKENDS:
        raise ValueError(f'unknown GDN backend {backend!r}; expected one of {BACKENDS}')
    if key.shape[:2] != query.shape[:2] or value.shape[:2] != query.shape[:2]:
        raise ValueError('query/key/value must share [B, T]')
    chosen = select_backend(query, prefer_triton_decode=prefer_triton_decode, prefill=prefill) \
        if backend == 'auto' else backend
    if chosen == 'recurrent_triton':
        try:
            out = _run_recurrent_triton(query, key, value, g, beta, state, output_state, qk_l2norm)
            STATS['recurrent_triton'] += 1
            return out
        except Exception:
            if backend != 'auto':
                raise
            STATS['recurrent_triton_fallback'] += 1
            chosen = 'recurrent_upstream'
    if chosen == 'recurrent_upstream':
        STATS['recurrent_upstream'] += 1
        return _run_recurrent_upstream(query, key, value, g, beta, state, output_state, qk_l2norm, **kwargs)
    if chosen == 'fla_chunk':
        STATS['fla_chunk'] += 1
        return _run_fla_chunk(query, key, value, g, beta, state, output_state, qk_l2norm)
    if chosen == 'chunk_reference':
        STATS['chunk_reference'] += 1
        return _run_chunk_reference(query, key, value, g, beta, state, output_state, qk_l2norm, **kwargs)
    STATS['recurrent_reference'] += 1
    return _run_recurrent_reference(query, key, value, g, beta, state, output_state, qk_l2norm)


def diagnostics():
    ref = _upstream_chunk()
    return {'backends': list(BACKENDS[1:]), 'calls': dict(STATS), 'fla_available': fla_available(),
            'fla_pin': FLA_PIN,
            'chunk_reference_impl': f"{getattr(ref, '__module__', '?')}.{getattr(ref, '__qualname__', type(ref).__name__)}"}
