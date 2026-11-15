# Engine scorecard

Runs: l4-1794709727 · baseline: `native_conv` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_conv | 2 | 63.9 [63.8–64.0] | 2209.7 [2203.1–2216.3] | 16.9 [16.8–16.9] | 58.1 [57.9–58.2] |  |  |
| interactive | 1 | native_drain | 2 | 63.7 [63.0–64.4] | 2213.1 [2213.0–2213.3] | 16.9 [16.9–16.9] | 57.9 [57.9–58.0] | insufficient_rounds ×1.00 | True |
| interactive | 16 | native_conv | 2 | 274.7 [110.6–438.8] | 3988.6 [3331.2–4646.0] | 24.4 [24.3–24.4] | 593.1 [551.7–634.4] |  |  |
| interactive | 16 | native_drain | 2 | 126.3 [113.3–139.4] | 3334.6 [3326.5–3342.8] | 24.6 [24.5–24.6] | 631.1 [629.0–633.2] | insufficient_rounds ×1.06 | True |
| moderate | 1 | native_conv | 2 | 123.2 [122.9–123.6] | 4557.9 [4554.7–4561.2] | 17.3 [17.3–17.3] | 56.4 [56.3–56.4] |  |  |
| moderate | 1 | native_drain | 2 | 122.4 [121.2–123.7] | 4555.1 [4548.5–4561.7] | 17.3 [17.3–17.3] | 56.3 [56.3–56.4] | insufficient_rounds ×1.00 | True |
| moderate | 16 | native_conv | 2 | 597.6 [565.7–629.4] | 15902.2 [8102.2–23702.2] | 29.4 [29.2–29.6] | 424.5 [339.7–509.3] |  |  |
| moderate | 16 | native_drain | 2 | 582.9 [547.9–617.9] | 8578.1 [8164.9–8991.4] | 29.3 [29.3–29.4] | 501.1 [496.0–506.1] | insufficient_rounds ×1.18 | True |
| prefill_heavy | 1 | native_conv | 2 | 175.8 [175.1–176.5] | 735.4 [734.4–736.5] | 18.0 [18.0–18.0] | 43.7 [43.7–43.8] |  |  |
| prefill_heavy | 1 | native_drain | 2 | 174.8 [174.6–174.9] | 734.3 [733.2–735.4] | 17.9 [17.9–17.9] | 43.8 [43.8–43.8] | insufficient_rounds ×1.00 | True |
| prefill_heavy | 16 | native_conv | 2 | 1393.0 [1335.4–1450.7] | 5580.7 [5235.3–5926.0] | 85.5 [85.4–85.7] | 120.8 [113.5–128.1] |  |  |
| prefill_heavy | 16 | native_drain | 2 | 821.9 [819.8–824.0] | 4294.2 [4277.1–4311.3] | 88.3 [88.2–88.4] | 141.9 [141.6–142.1] | insufficient_rounds ×1.17 | True |
| shared_prefix | 1 | native_conv | 2 | 70.2 [69.4–70.9] | 1250.6 [1250.0–1251.1] | 18.0 [17.9–18.0] | 51.6 [51.6–51.6] |  |  |
| shared_prefix | 1 | native_drain | 2 | 69.7 [69.2–70.1] | 1252.9 [1252.8–1253.0] | 18.0 [18.0–18.0] | 51.5 [51.5–51.5] | insufficient_rounds ×1.00 | True |
| shared_prefix | 16 | native_conv | 2 | 186.0 [176.9–195.0] | 3527.8 [3258.7–3796.8] | 46.6 [46.5–46.8] | 263.8 [242.7–284.9] |  |  |
| shared_prefix | 16 | native_drain | 2 | 178.4 [175.8–180.9] | 3304.3 [3298.9–3309.7] | 46.9 [46.8–47.0] | 283.0 [282.6–283.4] | insufficient_rounds ×1.07 | True |
