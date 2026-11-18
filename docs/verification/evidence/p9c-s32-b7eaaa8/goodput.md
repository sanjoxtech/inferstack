# Open-loop goodput and cost (Poisson arrivals), GPU $0.80/h

| case | profile | goodput@99 req/s | $ / 1k req @99 | goodput@90 req/s | $ / 1k req @90 |
|---|---|---|---|---|---|
| interactive | native_wait | 6 | 0.0370 | 6 | 0.0370 |
| moderate | native_wait | 2 | 0.1111 | 2 | 0.1111 |
| prefill_heavy | native_wait | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | native_wait | 6 | 0.0370 | 6 | 0.0370 |
| interactive | native_s32 | 6 | 0.0370 | 6 | 0.0370 |
| moderate | native_s32 | 3 | 0.0741 | 3 | 0.0741 |
| prefill_heavy | native_s32 | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | native_s32 | 6 | 0.0370 | 6 | 0.0370 |
