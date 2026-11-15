# Engine scorecard

Runs: l4-1794756180 · baseline: `native_dattn` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| interactive | 1 | native_dattn | 1 | 63.8 | 2174.0 | 16.6 | 59.0 |  |  |
| interactive | 1 | native_spec | 1 | 65.6 | 1379.1 | 10.2 | 94.3 | insufficient_rounds ×1.60 | True |
| interactive | 4 | native_dattn | 1 | 69.6 | 2438.9 | 18.6 | 210.9 |  |  |
| interactive | 4 | native_spec | 1 | 90.8 | 1711.5 | 11.6 | 324.4 | insufficient_rounds ×1.54 | True |
| interactive | 8 | native_dattn | 1 | 101.3 | 3407.2 | 20.3 | 338.3 |  |  |
| interactive | 8 | native_spec | 1 | 110.7 | 1805.6 | 12.4 | 600.4 | insufficient_rounds ×1.77 | True |
| interactive | 16 | native_dattn | 1 | 106.0 | 3774.4 | 22.6 | 615.5 |  |  |
| interactive | 16 | native_spec | 1 | 194.8 | 2961.5 | 15.8 | 858.4 | insufficient_rounds ×1.39 | True |
| moderate | 1 | native_dattn | 1 | 122.4 | 4417.0 | 16.8 | 58.1 |  |  |
| moderate | 1 | native_spec | 1 | 125.9 | 2810.2 | 10.4 | 92.2 | insufficient_rounds ×1.59 | True |
| moderate | 4 | native_dattn | 1 | 123.0 | 4932.3 | 18.8 | 208.0 |  |  |
| moderate | 4 | native_spec | 1 | 138.7 | 3339.3 | 11.8 | 321.1 | insufficient_rounds ×1.54 | True |
| moderate | 8 | native_dattn | 1 | 318.0 | 6234.7 | 20.8 | 349.0 |  |  |
| moderate | 8 | native_spec | 1 | 291.4 | 4106.8 | 13.4 | 538.5 | insufficient_rounds ×1.54 | True |
| moderate | 16 | native_dattn | 1 | 1164.7 | 8006.6 | 23.6 | 570.6 |  |  |
| moderate | 16 | native_spec | 1 | 390.5 | 5106.8 | 17.4 | 850.8 | insufficient_rounds ×1.49 | True |
| prefill_heavy | 1 | native_dattn | 1 | 168.5 | 700.2 | 17.0 | 46.0 |  |  |
| prefill_heavy | 1 | native_spec | 1 | 174.7 | 582.3 | 11.5 | 59.6 | insufficient_rounds ×1.30 | True |
| prefill_heavy | 4 | native_dattn | 1 | 503.1 | 1099.7 | 19.0 | 117.4 |  |  |
| prefill_heavy | 4 | native_spec | 1 | 352.2 | 1321.0 | 21.3 | 126.7 | insufficient_rounds ×1.08 | True |
| prefill_heavy | 8 | native_dattn | 1 | 753.8 | 3025.7 | 33.2 | 135.9 |  |  |
| prefill_heavy | 8 | native_spec | 1 | 523.6 | 2322.5 | 37.5 | 144.8 | insufficient_rounds ×1.07 | True |
| prefill_heavy | 16 | native_dattn | 1 | 755.9 | 3648.5 | 71.5 | 170.2 |  |  |
| prefill_heavy | 16 | native_spec | 1 | 755.1 | 3706.6 | 66.0 | 176.5 | insufficient_rounds ×1.04 | True |
| shared_prefix | 1 | native_dattn | 1 | 70.3 | 1182.5 | 16.9 | 54.7 |  |  |
| shared_prefix | 1 | native_spec | 1 | 72.0 | 763.3 | 12.5 | 74.5 | insufficient_rounds ×1.36 | True |
| shared_prefix | 4 | native_dattn | 1 | 87.9 | 1402.9 | 22.4 | 163.5 |  |  |
| shared_prefix | 4 | native_spec | 1 | 134.3 | 869.9 | 13.6 | 232.9 | insufficient_rounds ×1.42 | True |
| shared_prefix | 8 | native_dattn | 1 | 98.7 | 2286.9 | 30.0 | 195.8 |  |  |
| shared_prefix | 8 | native_spec | 1 | 103.3 | 1050.3 | 22.3 | 305.2 | insufficient_rounds ×1.56 | True |
| shared_prefix | 16 | native_dattn | 1 | 163.1 | 2653.2 | 35.9 | 368.2 |  |  |
| shared_prefix | 16 | native_spec | 1 | 140.0 | 2261.2 | 33.1 | 417.7 | insufficient_rounds ×1.13 | True |
