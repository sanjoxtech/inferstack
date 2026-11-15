# Engine scorecard

Runs: l4-1794708568 · baseline: `native_backlog` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_backlog | 2 | 66.3 [62.8–69.8] | 2239.0 [2219.6–2258.5] | 17.1 [16.9–17.2] | 57.3 [56.8–57.8] |  |  |
| interactive | 1 | native_conv | 2 | 70.8 [67.4–74.1] | 2247.5 [2226.0–2269.0] | 17.1 [17.0–17.2] | 57.1 [56.5–57.6] | insufficient_rounds ×1.00 | True |
| interactive | 16 | native_backlog | 2 | 118.9 [111.7–126.1] | 4041.3 [3398.0–4684.5] | 24.7 [24.5–24.9] | 596.3 [571.6–621.0] |  |  |
| interactive | 16 | native_conv | 2 | 132.3 [110.6–154.0] | 3390.3 [3375.2–3405.4] | 24.9 [24.6–25.2] | 618.5 [607.9–629.1] | insufficient_rounds ×1.04 | True |
| moderate | 1 | native_backlog | 2 | 126.2 [122.4–130.1] | 4609.4 [4576.8–4641.9] | 17.5 [17.4–17.6] | 55.7 [55.4–56.1] |  |  |
| moderate | 1 | native_conv | 2 | 140.7 [136.7–144.7] | 4640.8 [4625.8–4655.7] | 17.6 [17.6–17.6] | 55.4 [55.2–55.6] | insufficient_rounds ×0.99 | True |
| moderate | 16 | native_backlog | 2 | 652.9 [652.4–653.4] | 8736.6 [8459.6–9013.7] | 29.9 [29.7–30.2] | 485.5 [478.7–492.2] |  |  |
| moderate | 16 | native_conv | 2 | 598.1 [595.2–600.9] | 9755.7 [8412.4–11098.9] | 30.2 [30.1–30.2] | 476.9 [458.0–495.9] | insufficient_rounds ×0.98 | True |
| prefill_heavy | 1 | native_backlog | 2 | 189.7 [186.8–192.6] | 756.2 [749.6–762.7] | 18.1 [18.0–18.3] | 42.6 [42.2–43.0] |  |  |
| prefill_heavy | 1 | native_conv | 2 | 181.1 [179.5–182.7] | 750.2 [746.5–753.9] | 18.2 [18.2–18.3] | 42.9 [42.8–43.1] | insufficient_rounds ×1.01 | True |
| prefill_heavy | 16 | native_backlog | 2 | 1389.4 [1383.8–1394.9] | 5836.5 [5531.0–6142.0] | 87.7 [87.6–87.9] | 121.5 [119.9–123.1] |  |  |
| prefill_heavy | 16 | native_conv | 2 | 1408.0 [1394.0–1422.0] | 5343.2 [5320.4–5366.1] | 87.2 [87.0–87.4] | 124.8 [124.2–125.5] | insufficient_rounds ×1.03 | True |
| shared_prefix | 1 | native_backlog | 2 | 69.9 [69.1–70.7] | 1269.7 [1255.8–1283.6] | 18.1 [17.9–18.2] | 51.1 [50.6–51.6] |  |  |
| shared_prefix | 1 | native_conv | 2 | 80.7 [80.2–81.2] | 1277.5 [1272.6–1282.4] | 18.2 [18.2–18.2] | 50.4 [50.3–50.5] | insufficient_rounds ×0.99 | True |
| shared_prefix | 16 | native_backlog | 2 | 202.6 [199.3–206.0] | 3645.1 [3382.3–3907.9] | 46.4 [45.7–47.1] | 267.8 [251.8–283.8] |  |  |
| shared_prefix | 16 | native_conv | 2 | 178.1 [177.0–179.2] | 3799.6 [3772.4–3826.8] | 52.7 [52.3–53.2] | 259.6 [259.3–259.9] | insufficient_rounds ×0.97 | True |
