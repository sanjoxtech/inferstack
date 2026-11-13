# N2 native prefill diagnosis (Colab L4) — 13 November 2026

Diagnosis for NATIVE_ENGINE_OPTIMIZATION_PLAN step N2, on the largest gap measured by
the L4 scorecard (COLAB_L4_SCORECARD_2026-11-13.md): prefill_heavy TTFT.

- Evidence archive: `inferstack-colab-l4-n2-20261113T094642Z.zip` (user download),
  sha256 `3e96827c4db1bf2f692035e432ce68a2169c3eb546506cf012b12b0fc96046f5`. Chrome traces
  (80–106 MB each) stay in the archive; phase/operator JSON and the summary are committed
  under `evidence/n2-l4-20261113/`.
- Source pinned at `476bf3f`; notebook `docs/notebooks/colab_n2_prefill_diagnosis.ipynb`.
- 1 × NVIDIA L4; model `Qwen/Qwen3.5-2B-Base@b1485b2f` (BF16) through the foundry candidate.
- Method: `scripts/n2_prefill_diagnosis.py` drives the native prefill path in-process with
  the scorecard's own prefill_heavy prompts (1551–1570 tokens). Scheduler queueing is
  excluded by construction. 1 warmup + 3 repeats per configuration; ranges in brackets.
  Not a scored speed run: chunk size and prefix cache were varied only to measure cost.

## Prefill time (median ms)

| profile | cohort | chunk 128 (current) | chunk 512 | chunk 2048 | served TTFT p50 |
|---|---|---|---|---|---|
| native_unfused | 1 | 978 [974–987] | 401 | 214 | 1136 (c=1) |
| native_unfused | 4 | 1005 [997–1007] | 854 | 963 | 1010 (c=4) |
| native_full | 1 | 868 [868–874] | 372 | 201 | 1017 (c=1) |
| native_full | 4 | 889 [882–890] | 807 | 841 | 899 (c=4) |

Model forward is 97–99% of prefill wall time in every configuration.

## Findings, ranked

1. **Small prefill chunks dominate single-request prefill.** With the current
   `prefill_chunk_tokens = 128` a ~1570-token prompt takes 13 forward passes. Going to one
   2048-token pass cuts prefill from 978 to 214 ms (native_unfused, 4.6×) and 868 to
   201 ms (native_full). Each extra small forward costs ~60 ms of fixed work at cohort 1:
   it re-reads all weights, runs the vocab projection (`gemvx`, ~4 ms per call), and issues
   ~2,600 kernel launches (34,060 launches per chunk-128 prefill in the trace).
   Estimated ceiling at c=1: TTFT ~1136 → ~370 ms (214 ms prefill + the ~160 ms served
   overhead below), still ~3× SGLang's 120 ms. The cost is decode stalls: one 2048-token
   forward blocks interleaved decode for ~200 ms instead of ~75 ms. That trade-off has to
   be measured with the scorecard, not assumed.
2. **At cohort 4 prefill is compute-bound and chunk size barely matters** (854–1005 ms for
   ~6,240 tokens, about 7k tokens/s; 2048 is slower than 512). The GPU time is in GEMMs
   (`aten::mm`, `ampere_bf16_s1688gemm` tiles) and in the Gated DeltaNet prefill running
   as a chain of PyTorch ops: 2,268 `bmm`, 468 FP32 `linalg_solve_triangular`/`trsm`,
   918 FP32 `sgemm`, ~8,600 `copy_` per prefill. The engine reports
   `decode_kernel: triton` but `chunk_kernel: upstream/reference`: prefill GDN has no
   custom kernel. A fused chunked GDN prefill kernel is the likely next lever. That is
   kernel work (N3) and needs owner go-ahead.
3. **Prefix-cache snapshots are not the problem**: ≤13 ms of 978 (1.3%) at cohort 1, zero
   at cohort 4 (snapshots are only taken for single-request cohorts). Rejected.
4. **Scheduler/HTTP overhead is small**: served TTFT minus prefill is ~160 ms at c=1 and
   ~0–10 ms at c=4. It is not the main gap at these concurrencies; c=16 queueing
   (4.1 s TTFT) was not isolated here.

## Caveats

- In-process prefill only; the ops runs include profiler overhead (wall 1.4–1.8 s), so
  use the phases table for times and the ops tables only for shares.
- native_full has its forward fusions installed (24 decoder layers); in prefill its
  kernel mix is close to native_unfused and it is ~10% faster at chunk 128.
- `install-core.log` has pip's usual dependency-resolver warning about Colab's
  preinstalled packages; the install and preflight succeeded.
