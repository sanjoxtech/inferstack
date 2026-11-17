# v1.18 (P5: single-pass prefix snapshots), native scorecard, L4, 2026-11-17

Commit `0aed0e2` (code `134f3d1`) · Qwen3.5-2B-Base@b1485b2f · BF16 · 1× L4 (Colab) · 3 rounds · default workload.
Profiles: `native_radix` (v1.17, baseline) vs `native_single` (v1.18). Equal work in every cell, 100% success.
Tests: `test_qwen_static_lane.py` + `test_qwen_static_decode.py` pass with FLA 0.6.0 on L4 (25 tests, includes
the new fp32 and bf16 snapshot checks); the local suite passes (476, Qwen3.5 tests skip locally).
Evidence: `docs/verification/evidence/p5-single-l4-0aed0e2-20261117/`.

## What changed

v1.17 stored a cold prompt's prefix snapshot by prefilling in two calls (prefix, then the rest),
which cost 16–24 ms TTFT on long cold prompts. v1.18 records the snapshot inside the one prefill:

- K/V for columns < b are already in the slot after the prefill.
- Conv state at b = the W (4) raw conv inputs before b, gathered from the packed batch.
- Recurrent state at b = FLA's chunk-start state `h` for the chunk that starts at b
  (b is a multiple of the 128-token block, so of FLA's 64-token chunk). The public FLA API keeps
  `h` internal; `src/inferstack/kernels/fla_gdr_states.py` mirrors fla-core 0.6.0's forward (MIT)
  with the same internal ops and also returns `h`. Same idea as SGLang's `gdn_prefill_track`
  (ideas only).
- Rows that share a prefix inside a cohort keep the prefix pass (members copy the leader's state).

`h` is stored in the model dtype (bf16), so a snapshot's recurrent state is bf16-rounded, unlike
the fp32 final state. The bf16 test bounds the difference against a prefix-only prefill.

First run (`134f3d1`) stopped on my test asking for a bit-exact conv state: it differed by 3e-7
(fp32), because the prefix-only prefill's projection is a GEMM over a different T. Fixed the test
to compare within rounding (`0aed0e2`); log kept as `pytest-fla-134f3d1-strict-conv.log`.

## Result: v1.18 vs v1.17 (3 rounds)

| cell | TTFT p50 v1.17 → v1.18 (ms) | tok/s v1.17 → v1.18 | relation |
|---|---|---|---|
| moderate c1 | 94.9 → 75.0 | 94.3 → 93.4 | behind ×0.99 |
| moderate c4 | 87.8 → 67.7 | 334.7 → 339.0 | ahead ×1.01 |
| moderate c8 | 92.9 → 83.9 | 583.9 → 595.6 | ahead ×1.02 |
| moderate c16 | 137.3 → 111.5 | 913.9 → 948.1 | ahead ×1.04 |
| prefill_heavy c1 | 143.5 → 126.7 | 64.0 → 65.6 | ahead ×1.03 |
| prefill_heavy c4 | 104.9 → 94.8 | 178.9 → 181.2 | ahead ×1.01 |
| prefill_heavy c8 | 139.7 → 123.6 | 244.8 → 245.6 | overlap |
| prefill_heavy c16 | 261.5 → 219.1 | 283.5 → 290.3 | ahead ×1.02 |
| shared_prefix c1 | 39.9 → 39.8 | 79.6 → 80.5 | ahead ×1.01 |
| shared_prefix c16 | 98.5 → 83.5 | 490.3 → 484.6 | overlap (E2E p95 1672 → 1835) |
| other 6 cells | equal | | overlap ×0.98–1.01 |

The cold-prompt TTFT cost of v1.17 is gone and then some: moderate c1 is back below v1.16 (73 ms
in the v1.16 run, 75 here), prefill_heavy c1 TTFT 143 → 127 ms. Throughput ahead in 7 cells,
overlap in 8, behind ×0.99 in moderate c1 (decode 10.2 → 10.4 ms/tok; a 1% shift).

## Decision: KEEP v1.18 as the candidate

Watch: shared_prefix c16 E2E p95 rose 1672 → 1835 ms with throughput overlapping; decode ms/tok
27.2 → 28.3. Next round should look at whether the extra snapshot entries (every lone cold prompt
now stores one) compete with shared-prefix entries in the 128-entry LRU.

Against rivals (v1.16-run numbers, different session): prefill_heavy c16 290 vs SGLang 324 (×0.90,
was ×0.86); shared_prefix c16 485 vs 556 still behind. Remaining gap at c16 is decode under load
(48.7 ms/tok vs SGLang 21.6): the next structural item.
