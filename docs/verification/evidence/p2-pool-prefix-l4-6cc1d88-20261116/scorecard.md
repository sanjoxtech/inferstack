# Engine scorecard

Runs: l4-1794806529 · baseline: `native_packed` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_packed | 1 | 73.1 | 1395.7 | 10.2 | 93.2 |  |  |
| interactive | 1 | native_spec | 1 | 67.2 | 1384.9 | 10.2 | 93.7 | insufficient_rounds ×1.01 | True |
| interactive | 16 | native_packed | 1 | 98.9 | 2349.0 | 16.3 | 936.0 |  |  |
| interactive | 16 | native_spec | 1 | 805.4 | 3527.1 | 16.0 | 676.2 | insufficient_rounds ×0.72 | True |
| moderate | 1 | native_packed | 1 | 82.0 | 2817.7 | 10.5 | 92.8 |  |  |
| moderate | 1 | native_spec | 1 | 132.1 | 2828.0 | 10.4 | 91.6 | insufficient_rounds ×0.99 | True |
| moderate | 16 | native_packed | 1 | 164.3 | 5763.3 | 17.1 | 836.3 |  |  |
| moderate | 16 | native_spec | 1 | 772.0 | 6090.7 | 17.7 | 754.6 | insufficient_rounds ×0.90 | True |
| prefill_heavy | 1 | native_packed | 1 | 137.9 | 559.1 | 11.5 | 63.9 |  |  |
| prefill_heavy | 1 | native_spec | 1 | 171.5 | 583.2 | 11.5 | 59.7 | insufficient_rounds ×0.93 | True |
| prefill_heavy | 16 | native_packed | 1 | 365.9 | 4139.9 | 67.5 | 196.6 |  |  |
| prefill_heavy | 16 | native_spec | 1 | 864.4 | 3776.6 | 65.9 | 168.4 | insufficient_rounds ×0.86 | True |
| shared_prefix | 1 | native_packed | 1 | 86.3 | 804.9 | 12.1 | 72.7 |  |  |
| shared_prefix | 1 | native_spec | 1 | 75.9 | 772.3 | 12.6 | 73.4 | insufficient_rounds ×1.01 | True |
| shared_prefix | 16 | native_packed | 1 | 166.3 | 2008.1 | 35.5 | 384.7 |  |  |
| shared_prefix | 16 | native_spec | 1 | 140.5 | 2522.9 | 33.7 | 407.4 | insufficient_rounds ×1.06 | False |
