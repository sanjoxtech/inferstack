# Engine scorecard

Runs: l4-1794709894 · baseline: `native_backlog` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_backlog | 2 | 66.4 [66.2–66.6] | 2245.6 [2238.2–2253.0] | 17.1 [17.1–17.2] | 57.1 [57.0–57.3] |  |  |
| interactive | 1 | native_conv | 2 | 66.4 [66.2–66.6] | 2253.2 [2253.0–2253.4] | 17.2 [17.2–17.2] | 56.9 [56.9–56.9] | insufficient_rounds ×1.00 | True |
| interactive | 16 | native_backlog | 2 | 279.2 [110.2–448.2] | 4052.0 [3362.4–4741.6] | 24.8 [24.8–24.8] | 582.3 [540.6–623.9] |  |  |
| interactive | 16 | native_conv | 2 | 114.8 [112.5–117.2] | 3412.6 [3412.2–3412.9] | 24.9 [24.9–24.9] | 621.8 [621.3–622.2] | insufficient_rounds ×1.07 | True |
| moderate | 1 | native_backlog | 2 | 128.0 [127.5–128.5] | 4624.3 [4620.7–4627.9] | 17.6 [17.6–17.6] | 55.5 [55.5–55.5] |  |  |
| moderate | 1 | native_conv | 2 | 129.6 [128.5–130.7] | 4627.4 [4619.4–4635.3] | 17.6 [17.6–17.6] | 55.4 [55.4–55.5] | insufficient_rounds ×1.00 | True |
| moderate | 16 | native_backlog | 2 | 650.6 [650.4–650.8] | 8788.5 [8450.9–9126.0] | 30.0 [30.0–30.0] | 489.0 [484.2–493.9] |  |  |
| moderate | 16 | native_conv | 2 | 917.8 [590.1–1245.5] | 16437.3 [8488.4–24386.2] | 30.1 [30.1–30.2] | 406.7 [318.6–494.8] | insufficient_rounds ×0.83 | True |
| prefill_heavy | 1 | native_backlog | 2 | 187.6 [187.4–187.9] | 753.7 [752.8–754.6] | 18.2 [18.2–18.2] | 42.6 [42.6–42.7] |  |  |
| prefill_heavy | 1 | native_conv | 2 | 173.2 [172.0–174.3] | 739.7 [737.6–741.7] | 18.2 [18.2–18.2] | 43.4 [43.3–43.6] | insufficient_rounds ×1.02 | True |
| prefill_heavy | 16 | native_backlog | 2 | 1408.7 [1394.5–1422.9] | 5809.9 [5464.0–6155.7] | 88.3 [87.5–89.2] | 121.4 [118.5–124.4] |  |  |
| prefill_heavy | 16 | native_conv | 2 | 1381.6 [1333.9–1429.4] | 5578.1 [5172.8–5983.5] | 84.2 [84.0–84.3] | 126.1 [122.7–129.5] | insufficient_rounds ×1.04 | True |
| shared_prefix | 1 | native_backlog | 2 | 73.2 [72.3–74.0] | 1281.3 [1279.5–1283.1] | 18.1 [18.1–18.2] | 50.7 [50.6–50.8] |  |  |
| shared_prefix | 1 | native_conv | 2 | 74.0 [73.4–74.5] | 1270.0 [1267.7–1272.2] | 18.2 [18.2–18.2] | 50.7 [50.6–50.7] | insufficient_rounds ×1.00 | True |
| shared_prefix | 16 | native_backlog | 2 | 191.9 [180.7–203.1] | 3611.9 [3275.7–3948.0] | 46.2 [46.2–46.3] | 269.6 [248.5–290.7] |  |  |
| shared_prefix | 16 | native_conv | 2 | 192.1 [191.2–192.9] | 3364.7 [3359.8–3369.6] | 47.6 [47.6–47.6] | 278.0 [277.9–278.0] | insufficient_rounds ×1.03 | True |
