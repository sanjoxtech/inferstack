# Engine scorecard

Runs: l4-1794961643 · baseline: `native_burstwait` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_burstwait | 3 | 29.1 [28.9–29.3] | 1349.0 [1340.9–1352.9] | 10.2 [10.2–10.2] | 96.6 [96.4–97.1] |  |  |
| interactive | 1 | native_longgraph | 3 | 32.2 [32.0–32.4] | 1352.3 [1348.5–1358.5] | 10.2 [10.1–10.3] | 96.4 [95.7–96.9] | overlap ×1.00 | True |
| interactive | 4 | native_burstwait | 3 | 64.4 [49.8–65.1] | 1502.4 [1499.6–1527.9] | 11.0 [10.9–11.0] | 348.6 [346.5–350.1] |  |  |
| interactive | 4 | native_longgraph | 3 | 50.4 [50.0–53.8] | 1500.7 [1488.3–1522.5] | 10.9 [10.8–11.1] | 349.8 [346.3–352.7] | overlap ×1.00 | True |
| interactive | 8 | native_burstwait | 3 | 50.3 [48.4–51.6] | 1725.8 [1724.7–1745.2] | 12.5 [12.5–12.5] | 613.0 [612.0–615.6] |  |  |
| interactive | 8 | native_longgraph | 3 | 67.5 [43.9–77.4] | 1778.5 [1746.8–1789.0] | 12.5 [12.3–12.7] | 608.6 [606.1–616.6] | overlap ×0.99 | True |
| interactive | 16 | native_burstwait | 3 | 68.8 [68.4–73.1] | 2179.6 [2151.8–2213.0] | 15.5 [15.3–15.7] | 985.9 [983.5–996.1] |  |  |
| interactive | 16 | native_longgraph | 3 | 65.0 [65.0–70.4] | 2200.3 [2164.0–2202.9] | 15.5 [15.5–15.6] | 992.2 [983.3–996.1] | overlap ×1.01 | True |
| moderate | 1 | native_burstwait | 3 | 58.5 [58.1–59.0] | 2740.1 [2726.2–2742.4] | 10.3 [10.3–10.4] | 94.8 [94.4–94.9] |  |  |
| moderate | 1 | native_longgraph | 3 | 61.7 [61.7–61.8] | 2749.0 [2744.2–2755.6] | 10.4 [10.4–10.4] | 94.3 [94.1–94.4] | behind ×1.00 | True |
| moderate | 4 | native_burstwait | 3 | 60.6 [59.7–60.7] | 3152.3 [3147.3–3157.9] | 11.6 [11.6–11.6] | 337.1 [336.4–337.6] |  |  |
| moderate | 4 | native_longgraph | 3 | 59.5 [57.8–61.3] | 3154.0 [3127.4–3180.2] | 11.6 [11.6–11.7] | 335.8 [335.0–338.3] | overlap ×1.00 | True |
| moderate | 8 | native_burstwait | 3 | 67.3 [67.0–67.5] | 3713.7 [3706.5–3730.2] | 13.1 [13.1–13.3] | 588.5 [583.3–589.8] |  |  |
| moderate | 8 | native_longgraph | 3 | 88.2 [88.1–91.3] | 3708.3 [3699.9–3715.9] | 13.2 [13.1–13.2] | 592.7 [587.7–593.4] | overlap ×1.01 | True |
| moderate | 16 | native_burstwait | 3 | 118.5 [118.1–122.2] | 4611.8 [4596.5–4619.4] | 16.4 [16.4–16.4] | 945.7 [944.8–948.2] |  |  |
| moderate | 16 | native_longgraph | 3 | 123.6 [118.5–143.8] | 4730.8 [4581.7–5060.6] | 16.7 [16.5–16.8] | 931.6 [910.1–933.2] | behind ×0.99 | True |
| prefill_heavy | 1 | native_burstwait | 3 | 127.7 [125.2–127.7] | 545.8 [542.4–546.0] | 11.4 [11.4–11.4] | 65.6 [65.5–66.1] |  |  |
| prefill_heavy | 1 | native_longgraph | 3 | 130.9 [130.2–131.1] | 549.7 [546.0–550.9] | 11.4 [11.4–11.5] | 65.1 [64.9–65.4] | behind ×0.99 | True |
| prefill_heavy | 4 | native_burstwait | 3 | 97.2 [96.6–102.3] | 950.5 [907.5–964.7] | 18.5 [18.3–18.5] | 180.3 [179.9–182.6] |  |  |
| prefill_heavy | 4 | native_longgraph | 3 | 96.6 [95.7–102.5] | 953.1 [914.9–958.4] | 18.4 [18.4–18.5] | 179.9 [178.9–182.2] | overlap ×1.00 | True |
| prefill_heavy | 8 | native_burstwait | 3 | 126.5 [126.4–129.8] | 1622.9 [1581.1–1661.4] | 27.0 [25.5–28.3] | 241.4 [238.4–244.9] |  |  |
| prefill_heavy | 8 | native_longgraph | 3 | 125.9 [124.6–128.5] | 1623.4 [1604.9–11550.1] | 28.0 [25.4–28.4] | 240.4 [72.1–243.2] | overlap ×1.00 | True |
| prefill_heavy | 16 | native_burstwait | 3 | 221.5 [221.1–243.9] | 2826.4 [2749.5–2985.3] | 47.6 [46.0–48.6] | 289.2 [286.5–292.2] |  |  |
| prefill_heavy | 16 | native_longgraph | 3 | 222.5 [221.5–237.7] | 2813.9 [2813.5–2862.3] | 49.0 [48.9–49.7] | 286.6 [284.3–290.0] | overlap ×0.99 | True |
| shared_prefix | 1 | native_burstwait | 3 | 37.0 [36.7–37.1] | 719.2 [718.0–722.7] | 12.1 [12.1–12.1] | 80.5 [80.4–80.7] |  |  |
| shared_prefix | 1 | native_longgraph | 3 | 39.9 [39.8–40.0] | 724.7 [722.3–725.5] | 12.0 [12.0–12.1] | 80.2 [80.0–80.4] | behind ×1.00 | True |
| shared_prefix | 4 | native_burstwait | 3 | 51.8 [38.5–53.1] | 843.0 [839.4–866.5] | 15.2 [14.8–16.7] | 251.3 [237.8–251.3] |  |  |
| shared_prefix | 4 | native_longgraph | 3 | 51.5 [51.2–52.3] | 848.4 [846.8–853.0] | 15.3 [15.3–15.4] | 249.2 [248.7–250.6] | overlap ×0.99 | True |
| shared_prefix | 8 | native_burstwait | 3 | 44.9 [42.9–50.7] | 1081.5 [1073.8–1090.4] | 22.6 [21.3–22.8] | 322.3 [318.8–333.6] |  |  |
| shared_prefix | 8 | native_longgraph | 3 | 51.8 [50.8–55.0] | 1082.7 [1075.1–1087.2] | 21.5 [21.4–21.8] | 331.2 [327.7–331.7] | overlap ×1.03 | True |
| shared_prefix | 16 | native_burstwait | 3 | 77.5 [74.1–81.5] | 1872.0 [1855.3–1910.7] | 28.3 [28.1–29.0] | 478.5 [470.5–481.3] |  |  |
| shared_prefix | 16 | native_longgraph | 3 | 84.5 [82.9–88.9] | 1842.8 [1836.0–1881.0] | 28.4 [28.3–28.8] | 483.7 [477.7–483.8] | overlap ×1.01 | True |
