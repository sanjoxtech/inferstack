#!/usr/bin/env python
"""Fused-kernel numerical equivalence against FP32 oracles, measured in ULPs.

Each fused kernel is compared with an FP32 computation of the same math on the
same BF16 inputs. BF16 outputs pass when max error <= 1 ULP and mean <= 0.5 ULP
(single correct rounding), or when they are no worse than the existing BF16
reference implementation (which may round twice). FP32 recurrent state uses a
relative bound. Bounds are declared per check and printed with the results.

Supersedes the v1.6 fixed absolute tolerances (2e-2..5e-2), which rejected
correctly rounded outputs at magnitudes >= 4 (see V2_AGENT_HANDOFF ledger,
baseline reproduction 2026-11-07).
"""
from __future__ import annotations

import json
import sys


def main():
    import torch
    if not torch.cuda.is_available():
        raise SystemExit('CUDA required')
    from transformers import AutoConfig
    from inferstack.evaluation.numerical import check, check_relative
    from inferstack.kernels.gdn_triton import grouped_gdn_decode_reference, inferstack_grouped_gdn_decode
    from inferstack.kernels.qwen_forward_fusion import (add_rmsnorm_reference, beta_g_reference, fused_add_rmsnorm,
                                                        fused_beta_g, fused_gated_rmsnorm, fused_rmsnorm,
                                                        fused_silu_mul, qwen_gated_rmsnorm_reference,
                                                        qwen_rmsnorm_reference, silu_mul_reference)
    cfg = AutoConfig.from_pretrained('Qwen/Qwen3.5-2B-Base', trust_remote_code=False)
    text = getattr(cfg, 'text_config', cfg)
    device, bf = 'cuda', torch.bfloat16
    H, I = int(text.hidden_size), int(text.intermediate_size)
    Hk, Hv = int(text.linear_num_key_heads), int(text.linear_num_value_heads)
    K, V = int(text.linear_key_head_dim), int(text.linear_value_head_dim)
    B = 8
    checks = []
    for seed in (123, 7, 2026):
        torch.manual_seed(seed)
        tag = f'/seed{seed}'
        x = torch.randn(B, 1, H, device=device, dtype=bf)
        w = torch.randn(H, device=device, dtype=bf) * .1
        checks.append(check('rmsnorm' + tag, fused_rmsnorm(x.contiguous(), w.contiguous(), 1e-6),
                            qwen_rmsnorm_reference(x.float(), w.float(), 1e-6), dtype='bfloat16',
                            reference=qwen_rmsnorm_reference(x, w, 1e-6)))

        r = torch.randn_like(x)
        gn, gs = fused_add_rmsnorm(x.contiguous(), r.contiguous(), w.contiguous(), 1e-6)
        rn, rs = add_rmsnorm_reference(x, r, w, 1e-6)
        s32 = x.float() + r.float()
        checks.append(check('add_rmsnorm.sum' + tag, gs, s32, dtype='bfloat16', reference=rs))
        checks.append(check('add_rmsnorm.norm' + tag, gn, qwen_rmsnorm_reference(s32, w.float(), 1e-6),
                            dtype='bfloat16', reference=rn))

        gx = torch.randn(B * Hv, V, device=device, dtype=bf)
        gate, gw = torch.randn_like(gx), torch.randn(V, device=device, dtype=bf)
        checks.append(check('gated_rmsnorm' + tag, fused_gated_rmsnorm(gx.contiguous(), gate.contiguous(),
                                                                         gw.contiguous(), 1e-6),
                            qwen_gated_rmsnorm_reference(gx.float(), gate.float(), gw.float(), 1e-6),
                            dtype='bfloat16', reference=qwen_gated_rmsnorm_reference(gx, gate, gw, 1e-6)))

        gu = torch.randn(B, 1, 2 * I, device=device, dtype=bf)
        a, b = gu.float().chunk(2, -1)
        checks.append(check('silu_mul' + tag, fused_silu_mul(gu.contiguous()), torch.nn.functional.silu(a) * b,
                            dtype='bfloat16', reference=silu_mul_reference(gu)))

        bb = torch.randn(B, 1, Hv, device=device, dtype=bf)
        aa = torch.randn_like(bb)
        alog = torch.randn(Hv, device=device, dtype=torch.float32)
        dt = torch.randn(Hv, device=device, dtype=bf)
        gb, gg = fused_beta_g(bb.contiguous(), aa.contiguous(), alog.contiguous(), dt.contiguous())
        rb, rg = beta_g_reference(bb, aa, alog, dt)
        ob, og = beta_g_reference(bb.float(), aa.float(), alog, dt.float())
        checks.append(check('beta_g.beta' + tag, gb, ob, dtype=str(gb.dtype).replace('torch.', ''), reference=rb))
        checks.append(check('beta_g.g' + tag, gg, og, dtype=str(gg.dtype).replace('torch.', ''), reference=rg))

        q = torch.randn(B, 1, Hk, K, device=device, dtype=bf)
        k = torch.randn_like(q)
        v = torch.randn(B, 1, Hv, V, device=device, dtype=bf)
        decay = -torch.rand(B, 1, Hv, device=device, dtype=torch.float32)
        beta = torch.sigmoid(torch.randn(B, 1, Hv, device=device, dtype=bf))
        state = torch.randn(B, Hv, K, V, device=device, dtype=torch.float32) * .01
        go, gst = inferstack_grouped_gdn_decode(q, k, v, decay, beta, state.clone(), True, True)
        ro, rst = grouped_gdn_decode_reference(q, k, v, decay, beta, state.clone(), True, True)
        oo, ost = grouped_gdn_decode_reference(q.float(), k.float(), v.float(), decay, beta.float(), state.clone(),
                                               True, True)
        # Recurrent decode reduces over K: allow 2 ULP on the BF16 output, relative bound on FP32 state.
        checks.append(check('grouped_gdn.output' + tag, go, oo, dtype=str(go.dtype).replace('torch.', ''),
                            max_ulp=2.0, max_mean_ulp=0.5, reference=ro))
        checks.append(check_relative('grouped_gdn.state' + tag, gst, ost, max_rel=1e-3, floor=1e-4))

    out = {'version': '2.0.0', 'method': 'fp32_oracle_ulp', 'device': torch.cuda.get_device_name(0),
           'torch': torch.__version__, 'cuda': torch.version.cuda,
           'model_dims': {'hidden': H, 'intermediate': I, 'linear_key_heads': Hk, 'linear_value_heads': Hv,
                          'key_dim': K, 'value_dim': V, 'batch': B},
           'checks': checks, 'failed': [c['check'] for c in checks if not c['pass']],
           'pass': all(c['pass'] for c in checks)}
    print(json.dumps(out, indent=2))
    raise SystemExit(0 if out['pass'] else 2)


if __name__ == '__main__':
    sys.exit(main())
