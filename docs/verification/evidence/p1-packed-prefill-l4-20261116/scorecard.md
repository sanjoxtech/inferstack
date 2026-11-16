# Engine scorecard

Runs: l4-1794803232 · baseline: `native_packed` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_packed | 1 | 71.3 | 1379.9 | 10.1 | 94.2 |  |  |
| interactive | 1 | native_spec | 1 | 65.4 | 1365.1 | 10.1 | 95.2 | insufficient_rounds ×1.01 | True |
| interactive | 16 | native_packed | 1 | 89.1 | 2637.2 | 16.4 | 912.0 |  |  |
| interactive | 16 | native_spec | 1 | 785.9 | 3485.1 | 15.6 | 690.3 | insufficient_rounds ×0.76 | True |
| moderate | 1 | native_packed | 1 | 72.6 | 2794.6 | 10.4 | 93.7 |  |  |
| moderate | 1 | native_spec | 1 | 125.4 | 2793.4 | 10.3 | 93.0 | insufficient_rounds ×0.99 | True |
| moderate | 16 | native_packed | 1 | 135.9 | 4872.8 | 16.9 | 906.2 |  |  |
| moderate | 16 | native_spec | 1 | 769.2 | 5975.1 | 17.4 | 773.0 | insufficient_rounds ×0.85 | True |
| prefill_heavy | 1 | native_packed | 1 | 126.9 | 540.1 | 11.4 | 66.0 |  |  |
| prefill_heavy | 1 | native_spec | 1 | 172.8 | 579.4 | 11.5 | 59.8 | insufficient_rounds ×0.91 | True |
| prefill_heavy | 16 | native_packed | 1 | 358.8 | 3932.8 | 64.9 | 201.9 |  |  |
| prefill_heavy | 16 | native_spec | 1 | 887.2 | 3782.3 | 65.7 | 167.4 | insufficient_rounds ×0.83 | True |
| shared_prefix | 1 | native_packed | 1 | 97.9 | 734.7 | 12.1 | 73.1 |  |  |
| shared_prefix | 1 | native_spec | 1 | 71.0 | 761.1 | 12.5 | 74.7 | insufficient_rounds ×1.02 | True |
| shared_prefix | 16 | native_packed | 1 | 466.4 | 3841.7 | 48.4 | 245.3 |  |  |
| shared_prefix | 16 | native_spec | 1 | 133.2 | 2441.3 | 32.9 | 418.7 | insufficient_rounds ×1.71 | False |
