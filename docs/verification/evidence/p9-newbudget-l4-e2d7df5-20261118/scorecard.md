# Engine scorecard

Runs: l4-1794978663 · baseline: `native_longgraph` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_longgraph | 3 | 32.1 [32.0–32.8] | 1359.8 [1351.2–1366.0] | 10.3 [10.2–10.3] | 95.9 [95.4–96.3] |  |  |
| interactive | 1 | native_newbudget | 3 | 32.1 [32.1–32.2] | 1357.5 [1356.8–1361.2] | 10.2 [10.2–10.3] | 95.9 [95.8–96.0] | overlap ×1.00 | True |
| interactive | 4 | native_longgraph | 3 | 51.1 [36.0–53.6] | 1533.7 [1511.5–1553.1] | 11.1 [11.1–11.3] | 344.6 [343.5–347.1] |  |  |
| interactive | 4 | native_newbudget | 3 | 55.4 [50.2–59.1] | 1508.4 [1507.1–1515.2] | 11.0 [11.0–11.0] | 348.1 [346.7–348.8] | overlap ×1.01 | True |
| interactive | 8 | native_longgraph | 3 | 62.1 [46.0–77.2] | 1748.8 [1718.8–1766.8] | 12.3 [12.3–12.4] | 617.6 [615.0–620.4] |  |  |
| interactive | 8 | native_newbudget | 3 | 62.4 [54.1–67.9] | 1730.1 [1719.9–1765.1] | 12.4 [12.3–12.5] | 618.4 [613.4–619.3] | overlap ×1.00 | True |
| interactive | 16 | native_longgraph | 3 | 66.9 [64.5–67.4] | 2216.6 [2195.0–2222.6] | 15.7 [15.6–15.7] | 979.4 [979.3–987.2] |  |  |
| interactive | 16 | native_newbudget | 3 | 70.7 [67.3–74.2] | 2237.5 [2174.4–2256.6] | 15.6 [15.5–15.7] | 979.2 [975.2–989.5] | overlap ×1.00 | True |
| moderate | 1 | native_longgraph | 3 | 60.6 [60.4–60.7] | 2762.1 [2759.4–2762.8] | 10.4 [10.4–10.4] | 94.0 [93.8–94.1] |  |  |
| moderate | 1 | native_newbudget | 3 | 60.6 [60.4–60.8] | 2754.9 [2751.9–2762.5] | 10.4 [10.4–10.4] | 94.1 [93.9–94.1] | overlap ×1.00 | True |
| moderate | 4 | native_longgraph | 3 | 59.7 [58.2–59.8] | 3168.0 [3162.8–3175.1] | 11.7 [11.7–11.7] | 335.5 [335.2–335.7] |  |  |
| moderate | 4 | native_newbudget | 3 | 59.3 [58.9–60.1] | 3172.3 [3159.4–3172.8] | 11.7 [11.7–11.7] | 335.6 [335.2–336.0] | overlap ×1.00 | True |
| moderate | 8 | native_longgraph | 3 | 87.4 [87.1–89.9] | 3699.1 [3692.2–3717.9] | 13.1 [13.0–13.2] | 590.1 [589.4–591.1] |  |  |
| moderate | 8 | native_newbudget | 3 | 91.4 [90.5–91.5] | 3706.7 [3702.4–3712.2] | 13.0 [13.0–13.0] | 590.1 [589.2–590.3] | overlap ×1.00 | True |
| moderate | 16 | native_longgraph | 3 | 141.1 [121.4–143.4] | 4667.5 [4667.2–4844.4] | 16.7 [16.6–17.0] | 916.9 [915.0–927.5] |  |  |
| moderate | 16 | native_newbudget | 3 | 147.0 [140.0–150.3] | 4657.4 [4636.8–4663.7] | 16.7 [16.6–16.7] | 930.6 [929.2–931.2] | ahead ×1.01 | True |
| prefill_heavy | 1 | native_longgraph | 3 | 128.1 [126.6–129.9] | 547.7 [545.8–552.4] | 11.5 [11.5–11.5] | 65.3 [64.9–65.4] |  |  |
| prefill_heavy | 1 | native_newbudget | 3 | 128.5 [128.1–129.1] | 551.0 [550.8–551.2] | 11.5 [11.5–11.5] | 65.0 [65.0–65.1] | overlap ×1.00 | True |
| prefill_heavy | 4 | native_longgraph | 3 | 96.9 [96.9–104.5] | 956.5 [915.1–964.6] | 18.6 [18.5–18.7] | 179.3 [178.4–180.9] |  |  |
| prefill_heavy | 4 | native_newbudget | 3 | 103.1 [96.3–105.9] | 921.5 [907.7–957.4] | 18.6 [18.4–18.7] | 180.4 [180.4–180.9] | overlap ×1.01 | True |
| prefill_heavy | 8 | native_longgraph | 3 | 125.8 [125.5–126.1] | 1621.7 [1612.4–11648.7] | 28.3 [26.1–28.4] | 240.1 [71.6–240.7] |  |  |
| prefill_heavy | 8 | native_newbudget | 3 | 126.2 [124.9–127.7] | 1598.5 [1585.1–1633.2] | 26.4 [25.4–28.3] | 241.3 [239.2–244.8] | overlap ×1.00 | True |
| prefill_heavy | 16 | native_longgraph | 3 | 224.7 [221.3–230.2] | 2780.6 [2746.3–2800.7] | 48.9 [48.2–49.5] | 289.0 [285.6–291.6] |  |  |
| prefill_heavy | 16 | native_newbudget | 3 | 238.7 [226.4–240.3] | 2796.4 [2795.9–2900.4] | 50.1 [48.7–50.5] | 285.3 [285.0–285.9] | overlap ×0.99 | True |
| shared_prefix | 1 | native_longgraph | 3 | 40.0 [39.9–40.1] | 725.5 [724.6–726.3] | 12.1 [12.0–12.1] | 79.9 [79.7–79.9] |  |  |
| shared_prefix | 1 | native_newbudget | 3 | 40.7 [40.5–40.9] | 725.3 [724.8–726.5] | 12.1 [12.1–12.1] | 79.7 [79.7–79.9] | overlap ×1.00 | True |
| shared_prefix | 4 | native_longgraph | 3 | 51.8 [51.1–52.2] | 846.5 [842.4–854.7] | 15.3 [15.3–15.4] | 250.6 [250.2–250.7] |  |  |
| shared_prefix | 4 | native_newbudget | 3 | 53.9 [52.7–54.2] | 858.5 [856.0–866.9] | 15.4 [15.3–15.5] | 247.5 [245.6–247.9] | behind ×0.99 | True |
| shared_prefix | 8 | native_longgraph | 3 | 51.7 [51.6–53.1] | 1080.7 [1077.2–1086.1] | 21.6 [21.4–21.6] | 329.9 [329.6–330.4] |  |  |
| shared_prefix | 8 | native_newbudget | 3 | 52.5 [52.0–56.0] | 1093.2 [1086.6–1101.7] | 21.7 [21.5–22.1] | 326.3 [325.5–330.5] | overlap ×0.99 | True |
| shared_prefix | 16 | native_longgraph | 3 | 83.4 [82.0–89.8] | 1853.0 [1839.6–1893.7] | 28.6 [28.4–29.0] | 481.7 [474.3–482.2] |  |  |
| shared_prefix | 16 | native_newbudget | 3 | 90.8 [84.5–91.9] | 1902.5 [1849.7–1910.8] | 29.1 [28.5–29.3] | 472.5 [471.6–482.0] | overlap ×0.98 | True |
