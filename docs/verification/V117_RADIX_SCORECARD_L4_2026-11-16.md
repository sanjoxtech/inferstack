# v1.17 (P5: sync removal + first-sight prefix snapshots), native scorecard, L4, 2026-11-16

Commit `125e923` · Qwen3.5-2B-Base@b1485b2f · BF16 · 1× L4 (Colab) · 3 rounds · default workload.
Profiles: `native_graphs` (v1.16) vs `native_radix` (v1.17). Equal work in every cell, 100% success.
Tests on the runtime: `test_qwen_static_lane.py` + `test_qwen_static_decode.py` pass (with and without FLA).
Evidence: `docs/verification/evidence/p5-radix-l4-125e923-20261116/`.

## How we got here (rival study first)

1. **Profile (0d5e5cc).** A 1200-token eager prefill: 67 ms of kernels in 100 ms of wall time. Each
   GDN layer copied a host list to the GPU, which syncs the stream, so the GPU idled while FLA's
   ~1.6 ms of Python per call launched the next layer.
2. **Sync removal (f89cb99).** Per-prefill device tensors are built once; fresh-row state is zeroed with
   `torch.where`. The sync counter shows 7 syncs per prefill, all before the layer loop (was +18).
   1×1200 went from 100 to 86–90 ms, 1×300 from 65 to 60 ms; 8×300 unchanged.
3. **SGLang's own prefill (742e0f6, `sglang.benchmark.one_batch`, warm run):**

   | shape | SGLang 0.5.23 | ours (after 2) |
   |---|---|---|
   | 1×300 | 42.6 ms | 62.5 ms |
   | 1×1200 | 80.0 ms | 85.7 ms |
   | 8×300 | 156.3 ms | 165.7 ms |
   | 16×1200 | 1295 ms | 1357 ms |

   Per-token prefill is within 5% at size, so prefill speed does not explain prefill_heavy c4–c16.
4. **Cause.** `engine_scorecard.prompts()` gives request *i* the same text in every concurrency cell,
   so c4/c8/c16 repeat prompts from earlier cells. SGLang's radix cache and vLLM's automatic prefix
   cache store every prompt on first sight. Ours stored a prefix only once a recent prompt shared it:
   prefill_heavy c16 hit 16 of 32 repeats (v1.16 lane stats).
5. **Fix (125e923).** `prefix_cache_store_first_sight` snapshots every cold prompt's last block
   boundary; v1.17 also raises the pool prefix cache to 128 entries / 3 GB.

## Result: v1.17 vs v1.16 (output tok/s, 3 rounds)

| cell | v1.16 | v1.17 | relation | TTFT p50 v1.16 → v1.17 (ms) |
|---|---|---|---|---|
| prefill_heavy c4 | 137.8 | 176.4 | ahead ×1.28 | 144 → 107 |
| prefill_heavy c8 | 201.8 | 240.8 | ahead ×1.19 | 156 → 142 |
| prefill_heavy c16 | 233.0 | 279.4 | ahead ×1.20 | 328 → 223 |
| prefill_heavy c1 | 64.9 | 63.0 | behind ×0.97 | 131 → 147 |
| moderate c1 | 92.6 | 93.4 | overlap | 73 → 97 |
| other 11 cells | | | overlap or ahead ×1.00–1.01 | |

Cost: a cold prompt now takes two packed prefill calls (prefix, then the rest) to leave a snapshot,
which adds ~16–24 ms TTFT at c1 for long cold prompts.

Against the v1.16 run's rivals (same workload, different session): prefill_heavy c4 176 vs SGLang 160
(now ahead), c8 241 vs 242 (tie), c16 279 vs 324 (still behind ×0.86); shared_prefix c16 482 vs 556.

## Decision: KEEP as the next candidate, with one follow-up

Net: +19% to +28% in three cells, −3% in one, TTFT +16–24 ms on cold long c1 prompts. The cost
comes from the two-pass snapshot, not the policy. SGLang avoids it by recording the state at the
block boundary inside the single prefill pass (`gdn_prefill_track`: the delta-rule kernel's
per-chunk state at the boundary). Next: single-pass snapshot. The conv state is the last W−1 inputs
before the boundary and K/V are already in the slot; only the recurrent state needs the chunk state
at the boundary (FLA computes it in `chunk_gated_delta_rule_fwd_h`, not returned by the public API).

Note for the scorecard: the cross-cell prompt reuse rewards prefix caching in every cell above c1.
It is the same for all engines and the caches are legitimate features; a cold-only workload would
measure prefill speed alone.
