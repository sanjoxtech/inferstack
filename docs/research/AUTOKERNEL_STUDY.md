# AutoKernel study for P10 (2026-11-17)

Repo: https://github.com/RightNow-AI/autokernel (MIT, commit 77bfaa4, 2026-11-08, ~1.6k stars).
Read only; nothing run and no code copied. Ideas below would be re-implemented in our own code
(MIT would also allow reuse with its notice in THIRD_PARTY_NOTICES.md).

## What it is

An autoresearch loop for GPU kernels (after karpathy/autoresearch): profile a PyTorch model,
extract the hottest ops as standalone Triton or CUDA kernels, then let a coding agent edit one
file (`kernel.py`), run a fixed benchmark (`bench.py`), and keep or revert, ~90 s per experiment.
`orchestrate.py` picks the next kernel by Amdahl's law (expected end-to-end gain). `verify.py`
checks the whole model at the end. The agent's instructions are one long `program.md`
(phases: model analysis with a human, multi-kernel loop, integration; a 6-tier optimisation
playbook starting with block-size tuning and memory access).

## What we should take (ideas)

1. **The harness is fixed and the agent never edits it.** Same rule as our scorecard; for P10 the
   agent may only touch an allowlisted kernel file.
2. **Adversarial correctness inputs.** `bench.py` designs inputs so cheats fail loudly: peaked
   attention (scores sd 4) so a kernel that drops distant keys or attends past the causal edge
   misses tolerance; paged KV with a randomly permuted block table and a poisoned null page /
   tail slots (finite values x100) so a kernel that ignores the table or reads past kv_len fails
   by orders of magnitude. Five stages from a tight-tolerance smoke test up to sizes and dtypes.
   Our kernel gates should copy this design: inputs where a wrong shortcut is visibly wrong.
3. **Per-dtype tolerances** (bf16 atol/rtol 2e-2 or 2e-3 by op) and a roofline report (% of
   memory or compute peak), so "faster" is judged against the hardware limit, not only the
   previous version.
4. **Amdahl ordering.** Rank kernels by share of end-to-end time before spending agent time.
5. **results.tsv + one branch per campaign**: every experiment logged, keep/revert explicit.

## What not to take

- Microbenchmark wins as the verdict. A kernel faster in `bench.py` can be flat end to end
  (AutoTuneBench: KernelBench L1 median 1.0001x). Our verdict stays the L4 scorecard A/B.
- Its shipped kernels (GQA prefill, flash attention, matmul, rmsnorm...) do not cover our
  hot paths (gated delta rule, packed varlen GDN conv, spec-decode verify).

## Fit with InferStack (first target candidates)

Our own Triton kernels: `kernels/decode_attention.py` (GQA decode), `gdn_triton.py`,
`gdn_banked.py`, `qwen_conv_triton.py`, `qwen_forward_fusion.py`, `sampler_triton.py`.
FLA's GDN kernels are a dependency, not ours to edit.

Proposed P10 step 1: profile one decode step and one packed prefill on L4 (we have
`scripts/packed_prefill_profile.py`), rank our kernels by time (Amdahl), pick the top one we own,
write a fixed bench with adversarial inputs for it, then run the edit/bench/keep loop with a
configurable model (OpenRouter/DeepSeek or Claude/Codex). A kept kernel must also pass the
engine tests and win the 3-round scorecard A/B.
