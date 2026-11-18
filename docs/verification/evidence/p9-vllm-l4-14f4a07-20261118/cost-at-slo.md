# Cost at SLO (closed loop), GPU $0.80/h

| case | profile | highest conc. meeting SLO | out tok/s | req/s | $ / 1k requests | $ / 1M tokens |
|---|---|---|---|---|---|---|
| interactive | native_longgraph | 16 | 982 | 7.68 | 0.0290 | 0.226 |
| moderate | native_longgraph | 16 | 931 | 3.64 | 0.0611 | 0.239 |
| prefill_heavy | native_longgraph | 4 | 179 | 5.59 | 0.0398 | 1.243 |
| shared_prefix | native_longgraph | 8 | 329 | 9.33 | 0.0238 | 0.675 |
| interactive | vllm | 16 | 796 | 6.22 | 0.0358 | 0.279 |
| moderate | vllm | 8 | 421 | 1.65 | 0.1351 | 0.528 |
| prefill_heavy | vllm | 4 | 150 | 4.67 | 0.0475 | 1.486 |
| shared_prefix | vllm | 16 | 538 | 14.53 | 0.0153 | 0.413 |
