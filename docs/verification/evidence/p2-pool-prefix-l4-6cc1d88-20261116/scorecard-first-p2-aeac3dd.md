# Engine scorecard

Runs: l4-1794805437 · baseline: `native_packed` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_packed | 1 | 71.4 | 1393.1 | 10.2 | 93.5 |  |  |
| interactive | 1 | native_spec | 1 | 66.4 | 1379.3 | 10.2 | 94.1 | insufficient_rounds ×1.01 | True |
| interactive | 16 | native_packed | 1 | 95.6 | 2331.0 | 16.2 | 943.2 |  |  |
| interactive | 16 | native_spec | 1 | 797.9 | 3490.4 | 15.9 | 684.2 | insufficient_rounds ×0.73 | True |
| moderate | 1 | native_packed | 1 | 144.3 | 2812.0 | 10.3 | 92.0 |  |  |
| moderate | 1 | native_spec | 1 | 125.2 | 2799.1 | 10.4 | 92.2 | insufficient_rounds ×1.00 | True |
| moderate | 16 | native_packed | 1 | 238.2 | 5562.8 | 18.9 | 761.4 |  |  |
| moderate | 16 | native_spec | 1 | 963.6 | 6686.5 | 17.9 | 737.1 | insufficient_rounds ×0.97 | True |
| prefill_heavy | 1 | native_packed | 1 | 196.2 | 604.2 | 11.6 | 57.4 |  |  |
| prefill_heavy | 1 | native_spec | 1 | 171.8 | 582.2 | 11.5 | 59.8 | insufficient_rounds ×1.04 | True |
| prefill_heavy | 16 | native_packed | 1 | 312.7 | 3787.9 | 78.5 | 180.6 |  |  |
| prefill_heavy | 16 | native_spec | 1 | 847.5 | 3451.2 | 64.4 | 172.7 | insufficient_rounds ×0.96 | True |
| shared_prefix | 1 | native_packed | 1 | 83.1 | 788.1 | 12.0 | 73.5 |  |  |
| shared_prefix | 1 | native_spec | 1 | 72.3 | 774.5 | 12.5 | 73.9 | insufficient_rounds ×1.01 | True |
| shared_prefix | 16 | native_packed | 1 | 163.6 | 1977.3 | 35.0 | 392.0 |  |  |
| shared_prefix | 16 | native_spec | 1 | 137.6 | 2472.1 | 33.3 | 413.6 | insufficient_rounds ×1.06 | False |
