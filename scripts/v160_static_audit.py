#!/usr/bin/env python
from __future__ import annotations
import json
from pathlib import Path
from inferstack.config import load_profile


def main():
    root=Path(__file__).resolve().parents[1]
    cfg=load_profile(root/'configs/profiles/production-v1.6.0.toml')
    ff=(root/'src/inferstack/kernels/qwen_forward_fusion.py').read_text()
    g=(root/'src/inferstack/kernels/gdn_triton.py').read_text()
    q=(root/'src/inferstack/runtime/qwen_continuous.py').read_text()
    checks={
      'forward_fusion_master':cfg.native.forward_fusion,
      'fused_qwen_rmsnorm':'_rmsnorm_kernel' in ff and cfg.native.fuse_rmsnorm,
      'fused_add_rmsnorm':'_add_rmsnorm_kernel' in ff and cfg.native.fuse_add_rmsnorm,
      'fused_gated_rmsnorm':'_gated_rmsnorm_kernel' in ff and cfg.native.fuse_gated_rmsnorm,
      'merged_mlp_gate_up':'_inferstack_gate_up_weight' in ff and cfg.native.fuse_mlp_gate_up,
      'fused_silu_mul':'_silu_mul_kernel' in ff and cfg.native.fuse_mlp_activation,
      'merged_gdn_qkvz_ba':'_inferstack_qkvz_weight' in ff and '_inferstack_ba_weight' in ff and cfg.native.fuse_gdn_projections,
      'fused_gdn_beta_g':'_beta_g_kernel' in ff and cfg.native.fuse_gdn_beta_g,
      'grouped_gdn_no_repeat':'_gdn_decode_grouped' in g and 'inferstack_grouped_gdn_decode' in g and cfg.native.grouped_gdn_decode,
      'merged_attention_qkv':'_inferstack_qkv_weight' in ff and cfg.native.fuse_attention_qkv,
      'runtime_diagnostics':"'forward_fusion':ff" in q,
      'ablation_profiles':all((root/f'configs/profiles/{n}.toml').exists() for n in ('production-v1.6.0-baseline','production-v1.6.0-norm','production-v1.6.0-mlp','production-v1.6.0-gdn','production-v1.6.0-attention','production-v1.6.0')),
    }
    out={'version':'1.6.0','pass':all(checks.values()),'implemented':sum(checks.values()),'total':len(checks),'checks':checks,'meaning':'Static feature-wiring audit only. A100 kernel equivalence and benchmark ablations are separate acceptance layers.'}
    print(json.dumps(out,indent=2));raise SystemExit(0 if out['pass'] else 2)
if __name__=='__main__':main()
