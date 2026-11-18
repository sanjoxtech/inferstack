# Cost at SLO (closed loop), GPU $0.80/h

| case | profile | highest conc. meeting SLO | out tok/s | req/s | $ / 1k requests | $ / 1M tokens |
|---|---|---|---|---|---|---|
| interactive | native_bb4k | 16 | 984 | 7.69 | 0.0289 | 0.226 |
| moderate | native_bb4k | 16 | 926 | 3.62 | 0.0614 | 0.240 |
| prefill_heavy | native_bb4k | 4 | 182 | 5.67 | 0.0392 | 1.224 |
| shared_prefix | native_bb4k | 16 | 523 | 14.35 | 0.0155 | 0.425 |
| interactive | native_bb8k | 16 | 985 | 7.70 | 0.0289 | 0.226 |
| moderate | native_bb8k | 16 | 924 | 3.61 | 0.0616 | 0.240 |
| prefill_heavy | native_bb8k | 4 | 182 | 5.68 | 0.0391 | 1.223 |
| shared_prefix | native_bb8k | 16 | 532 | 14.61 | 0.0152 | 0.418 |
