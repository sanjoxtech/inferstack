# Static decode + CUDA graph, first L4 measurement — 13 November 2026

Step 1 of docs/NATIVE_GAP_ANALYSIS_2026-11-13.md. Model-step timing only (no HTTP or
scheduler), so it is structural evidence, not a scored serving result.

- Archive: `inferstack-colab-l4-static-decode-20261113T102302Z.zip` (user download),
  sha256 `9bbfd15f595b074953e2f30d83667e625ba9683be5f5b3990529bf6975a7fc8e`; JSON and
  summary under `evidence/static-decode-l4-20261113/`.
- Source `405692e`, 1 × NVIDIA L4, `Qwen/Qwen3.5-2B-Base@b1485b2f` BF16, 64 decode steps
  after the same prefill, median of steps 4–64; teacher-forced on the current path's tokens.

## native_unfused (ms per decode step)

| prompt words | batch | current native | static eager | static graph | speedup | greedy agreement | max logit diff |
|---|---|---|---|---|---|---|---|
| 48 | 1 | 43.3 | 43.4 | 18.4 | 2.35× | 1.000 | 0.19 |
| 48 | 4 | 46.1 | 44.6 | 21.5 | 2.14× | 1.000 | 0.31 |
| 48 | 16 | 46.3 | 45.2 | 29.1 | 1.59× | 1.000 | 0.53 |
| 400 | 1 | 43.2 | 43.2 | 18.8 | 2.30× | 1.000 | 0.16 |
| 400 | 4 | 46.4 | 44.7 | 22.5 | 2.06× | 0.996 | 0.19 |
| 400 | 16 | 46.1 | 45.0 | 33.5 | 1.38× | 0.996 | 0.28 |

Reading:
- Same math without the graph (static eager) is as slow as today, so the gain is the
  graph removing launch overhead, as the gap analysis predicted.
- Batch 1 at 18.4 ms is close to vLLM/SGLang's ~16 ms client decode on the same GPU.
- Batch 16 (29–34 ms vs vLLM's 19.3) is limited by GPU work inside the step. The
  recurrent update ran as PyTorch ops materializing [B, H, 128, 128] FP32 temporaries
  in 18 layers.
- Agreement: 1 differing greedy token in 256 at batch 4/16 with 400-word prompts
  (logit diff ≤ 0.53, BF16). Acceptable for a structural step; the full scorecard and
  quality checks decide acceptance.

## native_full

All 12 cells failed with `TypeError: 'NoneType' object is not callable`: forward
fusion removes the original q/k/v and GDN input projection modules and keeps
concatenated weights. Fixed in `ff92334` (uses the fused weights; a new test runs the
real fusion installer and reproduces the error on the old code).

## Next run (`ff92334`)

Re-run with native_full fixed and the Triton GDN decode kernel inside the graph
(`gdn_backend='auto'`), with the PyTorch-GDN graph kept as a comparison column.
