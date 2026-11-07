# InferStack v1.6.0 — Qwen Forward Fusion Candidate

Private development repository for InferStack's inference runtime and planned
optimization control plane. The imported v1.6.0 candidate is the starting point;
the control plane is not implemented yet. See [architecture and roadmap](docs/ARCHITECTURE.md).

## GitHub and Colab workflow

- Develop and review source changes here.
- Download **Code → Download ZIP** while signed in to GitHub, then follow
  [the Colab guide](COLAB_V1_6.md). No separately packaged release ZIP is needed.
- Record the tested commit SHA alongside benchmark results. Download results
  before ending a Colab runtime; generated results and model weights are not tracked.

For a local checkout after GitHub CLI authentication:

```bash
gh repo clone sanjoxtech/inferstack
cd inferstack
```

v1.6 starts where v1.5.5 profiling ended. The serving system is no longer the dominant cost: at C8, measured decode was ~44.834 ms and the Qwen model forward alone consumed ~44.249 ms (98.69%). The v1.6 objective is therefore **not another scheduler rewrite**. It is to reduce per-token work inside Qwen3.5 itself.

## Data-backed motivation

The v1.5.5 A100 profile showed:

- Qwen forward: ~44.249 ms / decode
- total decode: ~44.834 ms
- scheduler + sampler + streaming + compaction combined: ~0.586 ms
- `aten::mm`: ~21.3% CUDA self time, 1,496 calls in the profiling window
- `aten::copy_`: ~8.5%, 3,320 calls
- many elementwise launches: `mul`, `add`, `rsqrt`, normalization and gating ops

So v1.6 attacks **GEMM count, memory movement, and tiny elementwise kernels**.

## v1.6 fusion plan

### 1. Qwen zero-centered RMSNorm Triton
Replaces the eager `float -> square -> mean -> rsqrt -> scale -> weight -> cast` chain with one kernel.

### 2. Fused residual-add + RMSNorm
The mid-layer pattern:

```text
mixer output + residual -> RMSNorm
```

is executed as one Triton kernel rather than a standalone add followed by norm.

### 3. Fused gated RMSNorm for GDN
Qwen3.5 GDN uses RMSNorm followed by SiLU gating. v1.6 combines normalization, weight and gate into one kernel.

### 4. MLP gate/up projection fusion
Instead of:

```text
gate_proj(x)   -> GEMM #1
up_proj(x)     -> GEMM #2
SiLU(gate)*up
```

v1.6 concatenates the two weights and computes:

```text
gate_up(x)     -> GEMM #1
fused SiLU*mul -> Triton
```

The down projection remains unchanged.

### 5. GDN projection fusion
Qwen3.5 eager GDN performs four projections:

```text
qkv
z
b
a
```

v1.6 uses two GEMMs:

```text
qkvz
ba
```

matching the structural idea used by optimized inference engines.

### 6. Fused beta/g preparation
The GDN `sigmoid(b)` and `-exp(A_log)*softplus(a+dt_bias)` path is combined into one Triton kernel.

### 7. Grouped-head GDN decode
The eager Qwen path materializes repeated Q/K heads before recurrent GDN decode when value-head count exceeds key-head count. v1.6 adds a grouped-head Triton path that maps value heads to key/query heads directly and avoids `repeat_interleave` materialization.

### 8. Attention Q/K/V projection fusion
The full-attention layers merge Q, K and V projection weights into one GEMM, then split the output logically.

## What v1.6 intentionally does not change

The following v1.5.5 systems remain intact:

- continuous batching
- mid-generation admission
- live refill
- active compaction
- decode-first scheduling
- hybrid prefix cache / COW
- dedicated GPU worker thread
- thread-safe streaming
- custom GDN recurrence kernel
- causal-conv Triton kernel
- Triton greedy sampler

Those have already been measured and are not the current bottleneck.

## Local validation

```bash
pip install -e ".[dev]"
pytest -q
python scripts/v160_static_audit.py
inferstack native plan --profile configs/profiles/production-v1.6.0.toml
```

Local acceptance proves reference math and feature wiring. It does **not** prove CUDA performance.

## A100 validation

Run:

```bash
python scripts/v160_colab_matrix.py \
  --foundry-root examples/model_foundry \
  --model-ref foundry://factorylm-2b@candidate \
  --api-model factorylm-2b \
  --tokenizer Qwen/Qwen3.5-2B-Base
```

The matrix runs CUDA numerical equivalence first, then these lanes:

```text
baseline            C1, C8
RMSNorm fusion      C8
MLP fusion          C8
GDN fusion          C8
attention QKV       C8
full v1.6 fusion    C1, C4, C8, C16
```

It also profiles baseline and full-fusion operator counts so we can verify that `mm`, `copy_`, `mul`, `add`, and `rsqrt` calls actually decrease.

## Acceptance rule

A fusion is kept only if it:

1. passes CUDA numerical equivalence where applicable;
2. keeps the runtime healthy with zero benchmark request errors;
3. improves C8 TPOT/TPS versus the unfused v1.6 baseline;
4. does not introduce unacceptable deterministic output divergence.

**No v1.6 speedup is claimed until that A100 matrix is run.**
