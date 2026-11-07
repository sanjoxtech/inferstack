# InferStack v1.6.0 Release Gate — Qwen Forward Fusion

## Why this release exists

v1.5 solved serving-side serialization and proved scaling. v1.5.5 profiling then isolated the next bottleneck: ~98.69% of measured C8 decode wall time was inside the Qwen forward pass. v1.6 is the first release dedicated to reducing that forward cost.

## Implemented optimization groups

- zero-centered Qwen RMSNorm Triton
- residual-add + RMSNorm Triton
- GDN gated-RMSNorm Triton
- merged MLP gate/up GEMM
- Triton SiLU*mul
- merged GDN qkv+z and b+a GEMMs
- fused beta/g preparation
- grouped-head recurrent GDN decode without materialized Q/K repeat
- merged full-attention Q/K/V GEMM

## Three acceptance layers

1. **CPU/static:** exact reference formulas, weight-merging equivalence, source/profile wiring.
2. **A100 kernel equivalence:** Triton norm/gating/GDN paths versus PyTorch references using actual Qwen3.5 dimensions.
3. **A100 performance:** independent ablation at C8 plus full fusion scaling and operator-profile comparison.

## Not claimed yet

- vLLM parity
- a specific v1.6 speedup
- graph replay improvement
- hybrid paged KV
- fully fused transformer layer

The A100 matrix decides which v1.6 fusion groups graduate into the production default.
