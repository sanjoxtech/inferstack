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

## Second run (`ff92334`): native_full fixed, Triton GDN inside the graph

- Archive: `inferstack-colab-l4-static-decode-20261113T163206Z.zip`, sha256
  `cc17f9fee72241c0ce19f56b9f21a0f4828e32da9b31379b44828c3c7e8da029`; evidence under
  `evidence/static-decode-l4-ff92334-20261113/`.
- All 24 cells ran (0 errors). Graph decoders: 6 captures, 384 replays each; the
  Triton-GDN decoder reports `gdn_backend: triton`.

ms per decode step (median of steps 4–64):

| profile | words | batch | native now | graph + torch GDN | graph + Triton GDN | speedup | agreement | max logit diff |
|---|---|---|---|---|---|---|---|---|
| native_unfused | 48 | 1 | 45.0 | 18.4 | 17.6 | 2.56× | 1.000 | 0.16 |
| native_unfused | 48 | 16 | 48.1 | 29.0 | 23.5 | 2.05× | 1.000 | 0.53 |
| native_unfused | 400 | 16 | 48.0 | 33.4 | 27.8 | 1.72× | 0.996 | 0.38 |
| native_full | 48 | 1 | 30.0 | 17.2 | 16.4 | 1.83× | 1.000 | 0.16 |
| native_full | 48 | 4 | 35.4 | 20.3 | 18.3 | 1.94× | 1.000 | 0.38 |
| native_full | 48 | 16 | 36.4 | 27.9 | 22.2 | 1.64× | 0.999 | 0.50 |
| native_full | 400 | 1 | 29.9 | 17.6 | 16.7 | 1.78× | 0.984 | 0.16 |
| native_full | 400 | 16 | 35.3 | 32.1 | 26.4 | 1.34× | 0.994 | 0.31 |

Full table: `evidence/static-decode-l4-ff92334-20261113/static-decode-summary.md`.

Reading:
- The Triton GDN kernel inside the graph cuts batch 16 by 17–20% versus the PyTorch GDN
  graph. Batch 1 (16.4 ms) now matches vLLM/SGLang's ~16 ms client decode.
- Batch 16 remains 15–37% above vLLM's 19.3 ms (22.2 short prompts, 26.4 long). The gap
  grows with prompt length, which points at full-attention cost: the 6 attention
  layers attend over the whole padded bucket (masked), not the live length. Next
  structural candidates: length-aware attention (paged/varlen) and a fused decode
  attention kernel.
- Agreement: native_full 400-word batch 1 is 0.984 (1 token in 64) in every static
  mode including eager, so it is a fused-vs-static numerical difference (logit diff
  0.16, BF16), not a graph or Triton fault. Full scorecard and quality checks decide.
- Still model-step timing only. The decision to keep it comes from wiring it into the
  served engine and running the full L4 scorecard.
