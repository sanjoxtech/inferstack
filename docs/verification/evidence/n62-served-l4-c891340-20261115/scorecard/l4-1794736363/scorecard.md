# Engine scorecard

Runs: l4-1794736363 · baseline: `native_dattn` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_dattn | 2 | 65.8 [65.6–66.0] | 2205.1 [2195.1–2215.0] | 16.8 [16.7–16.9] | 58.2 [57.8–58.5] |  |  |
| interactive | 1 | native_spec | 2 | 65.2 [64.6–65.8] | 1385.8 [1380.4–1391.3] | 10.3 [10.2–10.3] | 93.8 [93.5–94.0] | insufficient_rounds ×1.61 | True |
| interactive | 16 | native_dattn | 2 | 110.3 [106.0–114.6] | 3493.0 [3181.2–3804.7] | 23.1 [22.9–23.3] | 637.8 [609.2–666.5] |  |  |
| interactive | 16 | native_spec | 2 | 193.3 [191.4–195.1] | 2992.3 [2406.5–3578.1] | 15.9 [15.7–16.1] | 867.1 [806.1–928.1] | insufficient_rounds ×1.36 | True |
| moderate | 1 | native_dattn | 2 | 127.1 [126.7–127.5] | 4471.8 [4450.6–4492.9] | 17.0 [16.9–17.1] | 57.4 [57.1–57.7] |  |  |
| moderate | 1 | native_spec | 2 | 124.5 [124.5–124.6] | 2809.4 [2808.4–2810.3] | 10.4 [10.4–10.4] | 92.1 [92.1–92.2] | insufficient_rounds ×1.61 | True |
| moderate | 16 | native_dattn | 2 | 624.4 [601.9–647.0] | 7810.8 [7400.0–8221.6] | 23.7 [23.6–23.7] | 591.3 [584.4–598.2] |  |  |
| moderate | 16 | native_spec | 2 | 462.7 [461.0–464.4] | 5213.7 [5206.7–5220.8] | 17.6 [17.6–17.7] | 829.8 [828.4–831.1] | insufficient_rounds ×1.40 | True |
| prefill_heavy | 1 | native_dattn | 2 | 170.3 [169.6–170.9] | 702.4 [698.7–706.0] | 17.1 [17.0–17.1] | 45.8 [45.6–45.9] |  |  |
| prefill_heavy | 1 | native_spec | 2 | 172.0 [171.5–172.5] | 583.7 [583.0–584.4] | 11.6 [11.6–11.6] | 59.5 [59.4–59.6] | insufficient_rounds ×1.30 | True |
| prefill_heavy | 16 | native_dattn | 2 | 767.3 [764.2–770.5] | 3981.7 [3664.8–4298.5] | 71.3 [70.6–72.0] | 165.8 [162.6–169.1] |  |  |
| prefill_heavy | 16 | native_spec | 2 | 889.8 [757.5–1022.1] | 4071.4 [3989.2–4153.6] | 71.0 [68.3–73.7] | 161.8 [160.7–162.9] | insufficient_rounds ×0.98 | True |
| shared_prefix | 1 | native_dattn | 2 | 71.6 [71.3–71.9] | 1192.1 [1188.8–1195.3] | 17.1 [17.0–17.1] | 54.1 [54.0–54.3] |  |  |
| shared_prefix | 1 | native_spec | 2 | 72.5 [71.4–73.6] | 770.3 [770.2–770.4] | 12.6 [12.5–12.6] | 73.8 [73.5–74.1] | insufficient_rounds ×1.36 | True |
| shared_prefix | 16 | native_dattn | 2 | 167.9 [156.6–179.2] | 2996.6 [2665.1–3328.0] | 35.9 [35.9–36.0] | 335.0 [303.7–366.3] |  |  |
| shared_prefix | 16 | native_spec | 2 | 139.2 [138.0–140.4] | 2488.9 [2451.3–2526.4] | 34.2 [34.1–34.2] | 405.7 [401.6–409.7] | insufficient_rounds ×1.21 | True |
