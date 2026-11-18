# Cost at SLO (closed loop), GPU $0.80/h

| case | profile | highest conc. meeting SLO | out tok/s | req/s | $ / 1k requests | $ / 1M tokens |
|---|---|---|---|---|---|---|
| interactive | native_bb8k | 16 | 996 | 7.78 | 0.0286 | 0.223 |
| moderate | native_bb8k | 16 | 942 | 3.68 | 0.0604 | 0.236 |
| prefill_heavy | native_bb8k | 4 | 181 | 5.64 | 0.0394 | 1.230 |
| shared_prefix | native_bb8k | 16 | 533 | 14.62 | 0.0152 | 0.417 |
| interactive | native_longgraph | 16 | 999 | 7.80 | 0.0285 | 0.222 |
| moderate | native_longgraph | 16 | 938 | 3.66 | 0.0606 | 0.237 |
| prefill_heavy | native_longgraph | 4 | 184 | 5.74 | 0.0387 | 1.210 |
| shared_prefix | native_longgraph | 8 | 335 | 9.51 | 0.0234 | 0.663 |
