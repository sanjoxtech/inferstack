# Engine scorecard

Runs: l4-1794660324 · baseline: `native_budget` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_budget | 2 | 69.9 [69.7–70.2] | 2246.7 [2244.7–2248.8] | 17.1 [17.1–17.1] | 57.2 [57.1–57.2] |  |  |
| interactive | 1 | native_static | 2 | 70.0 [69.6–70.5] | 2240.6 [2230.9–2250.3] | 17.0 [16.9–17.1] | 57.4 [57.1–57.6] | insufficient_rounds ×1.00 | True |
| interactive | 16 | native_budget | 2 | 136.6 [136.1–137.0] | 3398.1 [3395.1–3401.1] | 24.9 [24.9–25.0] | 620.2 [620.0–620.4] |  |  |
| interactive | 16 | native_static | 2 | 137.6 [137.0–138.2] | 3369.3 [3367.0–3371.7] | 24.7 [24.6–24.8] | 624.5 [622.1–626.9] | insufficient_rounds ×1.01 | True |
| moderate | 1 | native_budget | 2 | 161.8 [160.9–162.8] | 4784.3 [4781.0–4787.5] | 17.5 [17.5–17.5] | 55.0 [55.0–55.0] |  |  |
| moderate | 1 | native_static | 2 | 273.4 [272.0–274.9] | 4900.3 [4880.0–4920.6] | 17.5 [17.4–17.5] | 53.8 [53.7–53.9] | insufficient_rounds ×0.98 | True |
| moderate | 16 | native_budget | 2 | 995.7 [967.7–1023.6] | 8959.4 [8914.2–9004.7] | 30.1 [30.1–30.2] | 469.3 [469.0–469.7] |  |  |
| moderate | 16 | native_static | 2 | 971.1 [961.5–980.6] | 9026.4 [9003.5–9049.2] | 30.0 [30.0–30.1] | 471.3 [470.3–472.3] | insufficient_rounds ×1.00 | True |
| prefill_heavy | 1 | native_budget | 2 | 431.4 [425.7–437.2] | 1060.2 [1050.7–1069.6] | 18.1 [18.0–18.2] | 33.0 [32.8–33.3] |  |  |
| prefill_heavy | 1 | native_static | 2 | 1028.2 [1025.0–1031.5] | 1661.4 [1654.8–1668.0] | 18.1 [18.1–18.1] | 20.4 [20.4–20.5] | insufficient_rounds ×0.62 | True |
| prefill_heavy | 16 | native_budget | 2 | 3018.4 [3010.8–3026.0] | 7437.9 [7402.4–7473.5] | 95.3 [95.0–95.5] | 88.4 [88.4–88.5] |  |  |
| prefill_heavy | 16 | native_static | 2 | 3825.3 [3819.3–3831.4] | 7531.6 [7520.0–7543.2] | 88.4 [88.3–88.5] | 80.6 [80.3–80.8] | insufficient_rounds ×0.91 | True |
| shared_prefix | 1 | native_budget | 2 | 83.1 [82.2–84.1] | 1330.6 [1326.0–1335.1] | 18.2 [18.1–18.2] | 49.6 [49.5–49.7] |  |  |
| shared_prefix | 1 | native_static | 2 | 82.3 [82.0–82.6] | 1612.8 [1606.3–1619.3] | 18.1 [18.1–18.1] | 47.0 [47.0–47.1] | insufficient_rounds ×0.95 | True |
| shared_prefix | 16 | native_budget | 2 | 198.1 [196.3–200.0] | 3684.2 [3659.8–3708.5] | 50.1 [49.8–50.3] | 267.3 [265.7–268.9] |  |  |
| shared_prefix | 16 | native_static | 2 | 189.2 [176.7–201.8] | 3359.7 [3277.4–3442.1] | 48.3 [47.2–49.5] | 276.6 [271.8–281.4] | insufficient_rounds ×1.03 | True |
