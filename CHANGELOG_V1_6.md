# InferStack v1.6.0 Changelog

## Forward-fusion engine

- Added Qwen3.5 zero-centered RMSNorm Triton kernel.
- Added fused residual-add + RMSNorm Triton kernel.
- Added GDN RMSNorm+SiLU-gate Triton kernel.
- Added merged MLP gate/up projection and fused SiLU*mul path.
- Added merged Qwen3.5 GDN `qkvz` and `ba` projections.
- Added fused GDN beta/decay preparation.
- Added grouped-head GDN decode kernel that avoids Q/K repeat materialization.
- Added merged full-attention Q/K/V projection.
- Added runtime fusion diagnostics and per-feature counters.

## Measurement harness

- Added CUDA numerical-equivalence script for all new Triton arithmetic paths.
- Added C8 ablation profiles for Norm, MLP, GDN and attention projection fusion.
- Added baseline vs full-fusion PyTorch operator traces.
- Added analysis report for TPS/TPOT speedup and operator-call reductions.
- Added deterministic output-agreement signal against the unfused baseline.
