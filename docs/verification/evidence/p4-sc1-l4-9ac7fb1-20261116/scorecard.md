# Engine scorecard

Runs: l4-1794826562 · baseline: `native_graphs` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_graphs | 1 | 31.4 | 1330.7 | 10.1 | 97.8 |  |  |
| interactive | 1 | native_packed | 1 | 70.5 | 1362.0 | 10.0 | 95.5 | insufficient_rounds ×0.98 | True |
| interactive | 4 | native_graphs | 1 | 57.6 | 1473.7 | 10.7 | 356.3 |  |  |
| interactive | 4 | native_packed | 1 | 70.0 | 1626.7 | 11.7 | 331.4 | insufficient_rounds ×0.93 | True |
| interactive | 8 | native_graphs | 1 | 42.4 | 1702.6 | 12.3 | 626.2 |  |  |
| interactive | 8 | native_packed | 1 | 78.3 | 1821.7 | 12.8 | 598.3 | insufficient_rounds ×0.96 | True |
| interactive | 16 | native_graphs | 1 | 67.1 | 2121.2 | 15.0 | 1017.0 |  |  |
| interactive | 16 | native_packed | 1 | 90.4 | 2256.7 | 15.8 | 973.3 | insufficient_rounds ×0.96 | True |
| moderate | 1 | native_graphs | 1 | 79.3 | 2757.4 | 10.2 | 94.8 |  |  |
| moderate | 1 | native_packed | 1 | 78.9 | 2745.3 | 10.2 | 95.2 | insufficient_rounds ×1.00 | True |
| moderate | 4 | native_graphs | 1 | 98.0 | 3208.6 | 11.7 | 332.6 |  |  |
| moderate | 4 | native_packed | 1 | 132.8 | 3374.9 | 11.5 | 331.3 | insufficient_rounds ×1.00 | True |
| moderate | 8 | native_graphs | 1 | 91.5 | 3677.2 | 13.2 | 588.3 |  |  |
| moderate | 8 | native_packed | 1 | 112.6 | 3956.0 | 13.6 | 567.4 | insufficient_rounds ×0.96 | True |
| moderate | 16 | native_graphs | 1 | 164.0 | 4830.1 | 16.8 | 918.4 |  |  |
| moderate | 16 | native_packed | 1 | 152.0 | 4928.3 | 16.9 | 906.9 | insufficient_rounds ×0.99 | True |
| prefill_heavy | 1 | native_graphs | 1 | 133.7 | 549.6 | 11.2 | 65.4 |  |  |
| prefill_heavy | 1 | native_packed | 1 | 132.1 | 547.9 | 11.2 | 65.6 | insufficient_rounds ×1.00 | True |
| prefill_heavy | 4 | native_graphs | 1 | 244.7 | 1237.4 | 20.0 | 139.8 |  |  |
| prefill_heavy | 4 | native_packed | 1 | 226.5 | 1375.3 | 21.4 | 134.5 | insufficient_rounds ×0.96 | True |
| prefill_heavy | 8 | native_graphs | 1 | 153.1 | 1769.1 | 32.9 | 205.5 |  |  |
| prefill_heavy | 8 | native_packed | 1 | 228.2 | 1757.4 | 33.8 | 197.9 | insufficient_rounds ×0.96 | True |
| prefill_heavy | 16 | native_graphs | 1 | 257.0 | 2918.0 | 59.2 | 243.3 |  |  |
| prefill_heavy | 16 | native_packed | 1 | 293.7 | 3106.4 | 62.5 | 230.4 | insufficient_rounds ×0.95 | True |
| shared_prefix | 1 | native_graphs | 1 | 38.8 | 758.8 | 11.9 | 79.0 |  |  |
| shared_prefix | 1 | native_packed | 1 | 80.8 | 782.2 | 11.8 | 74.6 | insufficient_rounds ×0.94 | True |
| shared_prefix | 4 | native_graphs | 1 | 50.2 | 824.7 | 15.2 | 258.5 |  |  |
| shared_prefix | 4 | native_packed | 1 | 87.6 | 944.0 | 18.0 | 215.0 | insufficient_rounds ×0.83 | True |
| shared_prefix | 8 | native_graphs | 1 | 48.4 | 1044.0 | 20.8 | 340.8 |  |  |
| shared_prefix | 8 | native_packed | 1 | 87.6 | 1243.5 | 27.5 | 275.6 | insufficient_rounds ×0.81 | True |
| shared_prefix | 16 | native_graphs | 1 | 90.5 | 1642.7 | 26.6 | 503.7 |  |  |
| shared_prefix | 16 | native_packed | 1 | 128.0 | 1860.6 | 33.2 | 419.9 | insufficient_rounds ×0.83 | True |
