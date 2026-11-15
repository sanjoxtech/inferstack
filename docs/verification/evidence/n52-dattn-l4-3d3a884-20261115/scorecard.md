# Engine scorecard

Runs: l4-1794726090 · baseline: `native_dattn` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_dattn | 2 | 63.5 [63.2–63.7] | 2206.8 [2205.9–2207.7] | 16.8 [16.8–16.8] | 58.2 [58.2–58.2] |  |  |
| interactive | 1 | native_drain | 2 | 64.3 [64.1–64.5] | 2224.3 [2212.4–2236.1] | 16.9 [16.9–16.9] | 57.9 [57.7–58.1] | insufficient_rounds ×0.99 | True |
| interactive | 16 | native_dattn | 2 | 141.5 [103.1–179.8] | 3815.6 [3809.9–3821.3] | 23.2 [23.1–23.2] | 634.1 [630.1–638.1] |  |  |
| interactive | 16 | native_drain | 2 | 449.1 [112.8–785.3] | 4014.4 [3326.3–4702.5] | 24.5 [24.5–24.6] | 564.3 [498.2–630.3] | insufficient_rounds ×0.89 | True |
| moderate | 1 | native_dattn | 2 | 123.2 [122.0–124.3] | 4466.3 [4466.1–4466.5] | 16.9 [16.9–16.9] | 57.6 [57.6–57.6] |  |  |
| moderate | 1 | native_drain | 2 | 122.5 [121.6–123.4] | 4573.7 [4553.7–4593.7] | 17.4 [17.3–17.4] | 56.2 [56.0–56.4] | insufficient_rounds ×0.98 | True |
| moderate | 16 | native_dattn | 2 | 597.3 [594.7–600.0] | 7002.9 [6653.2–7352.7] | 23.5 [23.5–23.5] | 616.1 [607.1–625.0] |  |  |
| moderate | 16 | native_drain | 2 | 569.5 [536.1–602.9] | 8319.5 [8252.0–8387.0] | 29.7 [29.6–29.7] | 500.2 [499.9–500.4] | insufficient_rounds ×0.81 | True |
| prefill_heavy | 1 | native_dattn | 2 | 170.2 [169.8–170.7] | 703.6 [701.7–705.6] | 17.0 [17.0–17.0] | 45.8 [45.7–45.9] |  |  |
| prefill_heavy | 1 | native_drain | 2 | 174.3 [170.3–178.2] | 741.8 [733.5–750.1] | 18.1 [18.0–18.3] | 43.4 [43.0–43.9] | insufficient_rounds ×0.95 | True |
| prefill_heavy | 16 | native_dattn | 2 | 758.0 [753.5–762.4] | 3655.0 [3641.9–3668.1] | 71.7 [71.4–72.1] | 169.7 [168.9–170.4] |  |  |
| prefill_heavy | 16 | native_drain | 2 | 808.2 [794.7–821.7] | 4522.6 [4201.5–4843.7] | 86.4 [86.2–86.6] | 142.0 [139.1–145.0] | insufficient_rounds ×0.84 | True |
| shared_prefix | 1 | native_dattn | 2 | 70.6 [70.6–70.6] | 1185.9 [1184.3–1187.5] | 17.0 [17.0–17.1] | 54.3 [54.2–54.4] |  |  |
| shared_prefix | 1 | native_drain | 2 | 71.8 [70.5–73.1] | 1260.6 [1251.4–1269.8] | 18.1 [18.0–18.2] | 51.0 [50.7–51.4] | insufficient_rounds ×0.94 | True |
| shared_prefix | 16 | native_dattn | 2 | 163.9 [163.7–164.1] | 2626.9 [2625.8–2628.1] | 35.6 [35.5–35.6] | 369.9 [369.2–370.5] |  |  |
| shared_prefix | 16 | native_drain | 2 | 191.2 [188.4–194.0] | 3398.7 [3347.8–3449.6] | 47.4 [47.2–47.7] | 269.3 [260.0–278.6] | insufficient_rounds ×0.73 | True |
