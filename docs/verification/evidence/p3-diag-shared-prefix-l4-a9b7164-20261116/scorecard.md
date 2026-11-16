# Engine scorecard

Runs: l4-1794816143 · baseline: `native_packed` · measurement only: relations need >= 3 rounds with non-overlapping ranges; not a release decision; quality is evaluated separately

| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | vs baseline (tok/s) | equal work |
|---|---|---|---|---|---|---|---|---|---|
| shared_prefix | 4 | native_packed | 1 | 96.2 | 1064.7 | 18.1 | 203.2 |  |  |
| shared_prefix | 4 | native_spec | 1 | 93.9 | 1427.8 | 17.9 | 154.6 | insufficient_rounds ×0.76 | True |
| shared_prefix | 8 | native_packed | 1 | 98.4 | 1320.4 | 28.6 | 263.1 |  |  |
| shared_prefix | 8 | native_spec | 1 | 100.5 | 2034.1 | 22.3 | 224.8 | insufficient_rounds ×0.85 | True |
