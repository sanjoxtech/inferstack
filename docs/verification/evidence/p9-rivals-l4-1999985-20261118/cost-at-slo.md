# Cost at SLO (closed loop), GPU $0.80/h

| case | profile | highest conc. meeting SLO | out tok/s | req/s | $ / 1k requests | $ / 1M tokens |
|---|---|---|---|---|---|---|
| interactive | native_longgraph | 16 | 989 | 7.73 | 0.0288 | 0.225 |
| moderate | native_longgraph | 16 | 925 | 3.61 | 0.0615 | 0.240 |
| prefill_heavy | native_longgraph | 4 | 181 | 5.66 | 0.0393 | 1.227 |
| shared_prefix | native_longgraph | 8 | 331 | 9.39 | 0.0237 | 0.671 |
| interactive | sglang | 16 | 805 | 6.29 | 0.0353 | 0.276 |
| moderate | sglang | 4 | 232 | 0.91 | 0.2451 | 0.957 |
| prefill_heavy | sglang | 4 | 158 | 4.95 | 0.0449 | 1.402 |
| shared_prefix | sglang | 16 | 561 | 15.04 | 0.0148 | 0.396 |
