"""N4.2: the Gated DeltaNet short causal convolution for prefill, as one primitive.

Upstream (transformers Qwen3.5) runs the prefill conv as: transpose to [B, D, T], concat
the cached left context, depthwise F.conv1d, slice, separate SiLU, transpose back. On the
L4 (B=16 x ~1560 tokens) that is ~160 ms of conv_depthwise2d plus ~48 ms of SiLU and the
transposes' copies (docs/verification/N42_PREFILL_DIAGNOSIS_L4_2026-11-15.md). FLA ships a
fused Triton kernel that reads [B, T, D] directly and applies SiLU in the same pass.

Contract (layout of the model's in_proj output, no transpose):

    x             : [B, T, D]
    weight        : [D, W]            depthwise kernel (conv1d.weight.squeeze(1))
    bias          : [D] or None
    initial_state : [B, D, W] or None  the last W inputs before x (zeros if None); this
                    is what transformers' linear-attention cache keeps in conv_states
    returns       : y [B, T, D] in x's dtype, final_state [B, D, W] (last W inputs) or None

`reference` is the oracle (same math as upstream: zero left padding, depthwise conv,
optional SiLU); `fla` is the fused kernel (CUDA only).
"""
from __future__ import annotations

from collections import Counter

BACKENDS = ('reference', 'fla')
STATS: Counter = Counter()


def _reference(x, weight, bias, initial_state, activation, output_final_state):
    import torch
    import torch.nn.functional as F
    B, T, D = x.shape
    W = weight.shape[-1]
    xt = x.transpose(1, 2)
    left = (initial_state[..., 1:] if initial_state is not None
            else torch.zeros((B, D, W - 1), dtype=x.dtype, device=x.device))
    full = torch.cat([left.to(x.dtype), xt], dim=-1)                       # [B, D, W-1+T]
    y = F.conv1d(full.to(weight.dtype), weight.unsqueeze(1), bias, groups=D)  # [B, D, T]
    if activation in ('silu', 'swish'):
        y = F.silu(y)
    elif activation is not None:
        raise ValueError(f'unsupported activation {activation!r}')
    final = None
    if output_final_state:
        prev = initial_state.to(x.dtype) if initial_state is not None else torch.zeros((B, D, W), dtype=x.dtype, device=x.device)
        final = torch.cat([prev, xt], dim=-1)[..., -W:].contiguous()
    return y.transpose(1, 2).to(x.dtype), final


def _fla(x, weight, bias, initial_state, activation, output_final_state):
    if not x.is_cuda:
        raise RuntimeError('fla conv needs CUDA')
    from fla.modules.causal_conv1d import causal_conv1d
    y, final = causal_conv1d(x, weight=weight, bias=bias,
                             initial_state=None if initial_state is None else initial_state.to(x.dtype).contiguous(),
                             output_final_state=output_final_state, activation=activation, backend='triton')
    return y, final


def causal_conv_prefill(x, weight, bias=None, initial_state=None, *, activation='silu',
                        output_final_state=True, backend='reference'):
    if backend not in BACKENDS:
        raise ValueError(f'unknown conv backend {backend!r}; choose from {BACKENDS}')
    if x.dim() != 3 or weight.dim() != 2 or weight.shape[0] != x.shape[-1]:
        raise ValueError(f'bad shapes x={tuple(x.shape)} weight={tuple(weight.shape)}')
    if initial_state is not None and tuple(initial_state.shape) != (x.shape[0], x.shape[2], weight.shape[1]):
        raise ValueError(f'initial_state {tuple(initial_state.shape)} != (B, D, W)')
    STATS[backend] += 1
    run = _fla if backend == 'fla' else _reference
    return run(x, weight, bias, initial_state, activation, output_final_state)


def fla_conv_available():
    try:
        from fla.modules.causal_conv1d import causal_conv1d  # noqa: F401
        return True
    except Exception:
        return False
