# Engine scorecard

Runs: l4-1794657637 · baseline: `native_budget` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_budget | 2 | 64.0 [62.9–65.1] | 2212.4 [2212.0–2212.9] | 16.9 [16.9–16.9] | 58.0 [58.0–58.0] |  |  |
| interactive | 1 | native_fla | 2 | 63.8 [63.1–64.5] | 2215.1 [2212.9–2217.4] | 16.9 [16.9–16.9] | 58.0 [57.9–58.0] | insufficient_rounds ×1.00 | True |
| interactive | 1 | native_static | 2 | 63.9 [63.4–64.4] | 2208.7 [2208.4–2208.9] | 16.8 [16.8–16.9] | 58.1 [58.1–58.1] | insufficient_rounds ×1.00 | True |
| interactive | 16 | native_budget | 2 | 125.5 [106.2–144.8] | 3620.6 [3311.8–3929.3] | 24.5 [24.4–24.6] | 617.1 [597.9–636.3] |  |  |
| interactive | 16 | native_fla | 2 | 129.8 [107.6–152.1] | 3675.5 [3312.8–4038.1] | 24.5 [24.4–24.5] | 615.7 [595.6–635.7] | insufficient_rounds ×1.00 | True |
| interactive | 16 | native_static | 2 | 118.4 [109.4–127.3] | 4003.9 [3306.9–4700.8] | 24.4 [24.4–24.4] | 604.6 [574.2–635.0] | insufficient_rounds ×0.98 | True |
| moderate | 1 | native_budget | 2 | 123.0 [122.4–123.6] | 4696.7 [4696.1–4697.2] | 17.3 [17.3–17.3] | 56.1 [56.1–56.1] |  |  |
| moderate | 1 | native_fla | 2 | 122.7 [121.8–123.6] | 4547.4 [4542.5–4552.3] | 17.3 [17.3–17.3] | 56.4 [56.4–56.4] | insufficient_rounds ×1.01 | True |
| moderate | 1 | native_static | 2 | 236.6 [235.6–237.6] | 4819.7 [4812.0–4827.5] | 17.3 [17.3–17.3] | 54.7 [54.7–54.8] | insufficient_rounds ×0.98 | True |
| moderate | 16 | native_budget | 2 | 611.2 [596.9–625.6] | 8871.1 [8832.9–8909.4] | 30.2 [29.4–31.1] | 489.5 [483.1–495.9] |  |  |
| moderate | 16 | native_fla | 2 | 640.7 [639.5–641.9] | 8286.7 [8277.7–8295.7] | 29.5 [29.5–29.6] | 501.5 [501.3–501.7] | insufficient_rounds ×1.02 | True |
| moderate | 16 | native_static | 2 | 691.9 [685.6–698.2] | 8873.6 [8683.0–9064.2] | 29.7 [29.6–29.9] | 486.6 [483.8–489.4] | insufficient_rounds ×0.99 | True |
| prefill_heavy | 1 | native_budget | 2 | 345.5 [338.8–352.3] | 965.4 [964.7–966.1] | 17.9 [17.9–17.9] | 36.6 [36.5–36.6] |  |  |
| prefill_heavy | 1 | native_fla | 2 | 184.7 [184.3–185.1] | 742.9 [742.5–743.3] | 17.9 [17.9–17.9] | 43.3 [43.3–43.3] | insufficient_rounds ×1.18 | True |
| prefill_heavy | 1 | native_static | 2 | 917.4 [910.1–924.8] | 1555.2 [1537.7–1572.8] | 17.9 [17.9–18.0] | 22.0 [21.9–22.0] | insufficient_rounds ×0.60 | True |
| prefill_heavy | 16 | native_budget | 2 | 2490.8 [2480.3–2501.2] | 6066.3 [6012.0–6120.5] | 81.9 [81.6–82.1] | 105.2 [105.1–105.3] |  |  |
| prefill_heavy | 16 | native_fla | 2 | 2172.6 [2164.8–2180.5] | 5796.4 [5785.6–5807.3] | 75.0 [74.5–75.5] | 115.0 [114.8–115.2] | insufficient_rounds ×1.09 | True |
| prefill_heavy | 16 | native_static | 2 | 3294.6 [3168.1–3421.1] | 6623.1 [6581.1–6665.2] | 78.7 [78.2–79.2] | 90.2 [88.5–92.0] | insufficient_rounds ×0.86 | True |
| shared_prefix | 1 | native_budget | 2 | 69.8 [69.8–69.9] | 1253.4 [1252.4–1254.3] | 17.9 [17.9–17.9] | 51.6 [51.6–51.7] |  |  |
| shared_prefix | 1 | native_fla | 2 | 72.1 [71.0–73.3] | 1250.4 [1249.3–1251.6] | 17.9 [17.9–17.9] | 51.6 [51.6–51.6] | insufficient_rounds ×1.00 | True |
| shared_prefix | 1 | native_static | 2 | 71.6 [71.5–71.7] | 1541.9 [1530.9–1552.9] | 18.0 [17.9–18.1] | 48.0 [47.7–48.2] | insufficient_rounds ×0.93 | True |
| shared_prefix | 16 | native_budget | 2 | 186.7 [186.0–187.4] | 3247.2 [3232.3–3262.0] | 45.6 [45.4–45.7] | 294.7 [294.4–295.1] |  |  |
| shared_prefix | 16 | native_fla | 2 | 188.2 [186.3–190.1] | 3241.9 [3227.3–3256.5] | 45.6 [45.5–45.7] | 293.7 [293.5–294.0] | insufficient_rounds ×1.00 | True |
| shared_prefix | 16 | native_static | 2 | 191.6 [189.1–194.2] | 3580.1 [3247.6–3912.6] | 45.6 [45.5–45.7] | 267.2 [247.5–286.9] | insufficient_rounds ×0.91 | True |
