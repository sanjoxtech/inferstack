# Engine scorecard

Runs: l4-1794931158 · baseline: `native_overlap2` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_overlap2 | 3 | 32.0 [31.9–32.0] | 1344.6 [1344.1–1354.3] | 10.2 [10.2–10.2] | 96.9 [96.3–96.9] |  |  |
| interactive | 1 | native_single | 3 | 31.8 [31.8–31.8] | 1345.2 [1343.0–1345.9] | 10.2 [10.2–10.2] | 96.9 [96.8–96.9] | overlap ×1.00 | True |
| interactive | 4 | native_overlap2 | 3 | 47.3 [46.2–50.1] | 1499.9 [1499.2–1513.9] | 11.0 [10.9–11.1] | 349.4 [349.4–350.6] |  |  |
| interactive | 4 | native_single | 3 | 53.1 [49.1–53.6] | 1492.5 [1490.6–1502.8] | 10.9 [10.8–11.0] | 351.1 [350.6–352.4] | ahead ×1.00 | True |
| interactive | 8 | native_overlap2 | 3 | 65.0 [47.3–67.9] | 1743.9 [1737.2–1746.5] | 12.3 [12.1–12.5] | 621.3 [615.5–624.0] |  |  |
| interactive | 8 | native_single | 3 | 65.8 [47.4–72.0] | 1731.3 [1704.7–1752.3] | 12.3 [12.3–12.4] | 622.4 [616.4–623.2] | overlap ×1.00 | True |
| interactive | 16 | native_overlap2 | 3 | 65.1 [64.1–68.4] | 2190.0 [2185.8–2205.8] | 15.5 [15.4–15.5] | 991.6 [988.7–992.9] |  |  |
| interactive | 16 | native_single | 3 | 65.1 [63.9–67.3] | 2199.0 [2184.9–2201.3] | 15.5 [15.4–15.5] | 990.4 [988.1–996.1] | overlap ×1.00 | True |
| moderate | 1 | native_overlap2 | 3 | 74.2 [73.9–75.0] | 2782.4 [2779.3–2791.2] | 10.3 [10.3–10.4] | 94.0 [93.9–94.1] |  |  |
| moderate | 1 | native_single | 3 | 74.9 [74.0–75.3] | 2777.7 [2777.3–2794.7] | 10.4 [10.3–10.5] | 94.1 [93.2–94.1] | overlap ×1.00 | True |
| moderate | 4 | native_overlap2 | 3 | 67.3 [63.1–68.3] | 3110.4 [3072.0–3178.0] | 11.4 [11.4–11.6] | 340.1 [338.3–342.1] |  |  |
| moderate | 4 | native_single | 3 | 67.9 [66.8–68.6] | 3163.2 [3100.5–3176.3] | 11.4 [11.4–11.4] | 340.5 [339.1–342.8] | overlap ×1.00 | True |
| moderate | 8 | native_overlap2 | 3 | 83.4 [81.4–83.7] | 3658.1 [3650.6–3766.3] | 13.1 [12.9–13.1] | 599.1 [592.7–602.4] |  |  |
| moderate | 8 | native_single | 3 | 82.8 [81.3–85.3] | 3642.1 [3627.6–3697.9] | 13.0 [12.9–13.1] | 599.9 [596.5–601.8] | overlap ×1.00 | True |
| moderate | 16 | native_overlap2 | 3 | 113.3 [109.5–114.0] | 4682.1 [4552.8–4696.4] | 16.4 [16.3–16.5] | 945.6 [940.9–951.5] |  |  |
| moderate | 16 | native_single | 3 | 109.7 [109.1–110.4] | 4686.0 [4667.2–4817.4] | 16.5 [16.5–16.9] | 931.5 [929.1–944.1] | overlap ×0.99 | True |
| prefill_heavy | 1 | native_overlap2 | 3 | 127.7 [127.6–127.7] | 544.2 [543.5–544.8] | 11.4 [11.4–11.5] | 65.7 [65.5–65.8] |  |  |
| prefill_heavy | 1 | native_single | 3 | 127.6 [127.4–130.7] | 544.4 [542.1–546.5] | 11.4 [11.4–11.5] | 65.5 [65.3–65.9] | overlap ×1.00 | True |
| prefill_heavy | 4 | native_overlap2 | 3 | 250.5 [224.3–509.1] | 1459.5 [1405.9–2008.2] | 14.7 [14.4–14.9] | 142.6 [102.8–144.4] |  |  |
| prefill_heavy | 4 | native_single | 3 | 102.4 [95.1–102.5] | 906.4 [900.8–939.5] | 18.4 [18.2–18.6] | 182.9 [182.5–183.4] | ahead ×1.28 | True |
| prefill_heavy | 8 | native_overlap2 | 3 | 217.5 [216.7–225.3] | 1935.3 [1695.3–1950.2] | 20.3 [20.0–23.1] | 214.0 [211.7–215.4] |  |  |
| prefill_heavy | 8 | native_single | 3 | 125.2 [124.6–125.6] | 1607.9 [1578.8–11718.8] | 26.0 [26.0–28.0] | 243.6 [71.5–244.6] | overlap ×1.14 | True |
| prefill_heavy | 16 | native_overlap2 | 3 | 262.9 [222.6–444.0] | 1757.4 [1693.7–2989.4] | 28.2 [27.0–41.8] | 300.6 [268.7–301.2] |  |  |
| prefill_heavy | 16 | native_single | 3 | 230.2 [217.6–236.1] | 2766.8 [2761.3–2787.9] | 49.7 [47.9–49.9] | 290.4 [289.0–290.5] | overlap ×0.97 | True |
| shared_prefix | 1 | native_overlap2 | 3 | 39.7 | 718.7 | 12.0 | 80.6 |  |  |
| shared_prefix | 1 | native_single | 3 | 39.7 [39.5–39.7] | 718.7 [716.2–723.3] | 12.0 [12.0–12.0] | 80.5 [80.3–80.6] | insufficient_rounds ×1.00 | True |
| shared_prefix | 4 | native_overlap2 | 3 | 110.3 | 827.2 | 14.4 | 233.7 |  |  |
| shared_prefix | 4 | native_single | 3 | 53.7 [51.7–53.8] | 836.9 [836.8–838.4] | 14.8 [14.8–15.1] | 251.6 [251.3–252.8] | insufficient_rounds ×1.08 | True |
| shared_prefix | 8 | native_overlap2 | 3 | 123.1 | 997.3 | 18.5 | 335.7 |  |  |
| shared_prefix | 8 | native_single | 3 | 52.4 [51.6–53.0] | 1074.4 [1065.3–1075.6] | 21.5 [21.2–21.7] | 331.9 [330.2–334.0] | insufficient_rounds ×0.99 | True |
| shared_prefix | 16 | native_overlap2 | 3 | 173.2 | 1662.5 | 23.5 | 493.2 |  |  |
| shared_prefix | 16 | native_single | 3 | 83.2 [81.9–97.7] | 1826.4 [1809.0–1836.0] | 28.0 [27.8–28.2] | 489.2 [487.9–490.8] | insufficient_rounds ×0.99 | True |
