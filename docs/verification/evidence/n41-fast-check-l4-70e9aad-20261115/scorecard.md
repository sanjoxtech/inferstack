# Engine scorecard

Runs: l4-1794705906 · baseline: `native_backlog` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_backlog | 2 | 62.7 [62.5–62.9] | 2220.8 [2215.8–2225.7] | 16.9 [16.9–17.0] | 57.8 [57.6–57.9] |  |  |
| interactive | 1 | native_fla | 2 | 63.2 [63.1–63.3] | 2206.6 [2198.5–2214.7] | 16.8 [16.8–16.9] | 58.1 [57.9–58.4] | insufficient_rounds ×1.01 | True |
| interactive | 16 | native_backlog | 2 | 127.8 [109.4–146.1] | 3671.3 [3329.1–4013.4] | 24.7 [24.5–24.8] | 598.3 [565.0–631.7] |  |  |
| interactive | 16 | native_fla | 2 | 273.0 [110.7–435.4] | 3970.8 [3311.6–4630.1] | 24.4 [24.2–24.6] | 591.3 [553.0–629.6] | insufficient_rounds ×0.99 | True |
| moderate | 1 | native_backlog | 2 | 121.1 [120.6–121.7] | 4554.0 [4544.9–4563.0] | 17.3 [17.3–17.4] | 56.3 [56.2–56.4] |  |  |
| moderate | 1 | native_fla | 2 | 120.9 [120.9–121.0] | 4544.8 [4538.6–4551.0] | 17.3 [17.3–17.3] | 56.4 [56.4–56.5] | insufficient_rounds ×1.00 | True |
| moderate | 16 | native_backlog | 2 | 630.8 [609.5–652.1] | 8262.7 [8247.5–8277.9] | 29.6 [29.4–29.7] | 502.5 [502.0–503.1] |  |  |
| moderate | 16 | native_fla | 2 | 633.2 [605.3–661.2] | 8562.6 [8161.4–8963.8] | 29.4 [29.3–29.6] | 498.9 [491.1–506.7] | insufficient_rounds ×0.99 | True |
| prefill_heavy | 1 | native_backlog | 2 | 185.7 [185.5–185.8] | 748.0 [747.8–748.3] | 18.0 [18.0–18.1] | 43.0 [42.9–43.1] |  |  |
| prefill_heavy | 1 | native_fla | 2 | 184.8 [184.5–185.0] | 743.9 [742.6–745.2] | 18.0 [17.9–18.0] | 43.2 [43.1–43.2] | insufficient_rounds ×1.00 | True |
| prefill_heavy | 16 | native_backlog | 2 | 1369.9 [1366.2–1373.6] | 5467.8 [5434.1–5501.5] | 87.7 [87.4–88.0] | 124.5 [123.9–125.0] |  |  |
| prefill_heavy | 16 | native_fla | 2 | 2187.9 [2181.4–2194.4] | 6154.5 [5830.5–6478.6] | 75.0 [74.9–75.1] | 112.5 [110.5–114.4] | insufficient_rounds ×0.90 | True |
| shared_prefix | 1 | native_backlog | 2 | 69.2 [69.1–69.2] | 1261.5 [1259.6–1263.3] | 18.0 [18.0–18.0] | 51.6 [51.6–51.7] |  |  |
| shared_prefix | 1 | native_fla | 2 | 69.3 [69.1–69.5] | 1255.4 [1254.1–1256.7] | 18.0 [17.9–18.0] | 51.8 [51.7–51.9] | insufficient_rounds ×1.00 | True |
| shared_prefix | 16 | native_backlog | 2 | 165.7 [164.5–166.9] | 3532.3 [3512.4–3552.2] | 48.5 [48.3–48.8] | 275.3 [274.0–276.5] |  |  |
| shared_prefix | 16 | native_fla | 2 | 177.7 [166.3–189.2] | 3559.1 [3504.6–3613.6] | 49.3 [47.9–50.6] | 258.3 [239.4–277.2] | insufficient_rounds ×0.94 | True |
