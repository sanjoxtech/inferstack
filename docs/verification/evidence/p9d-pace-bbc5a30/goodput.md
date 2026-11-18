# Open-loop goodput and cost (Poisson arrivals), GPU $0.80/h

| case | profile | goodput@99 req/s | $ / 1k req @99 | goodput@90 req/s | $ / 1k req @90 |
|---|---|---|---|---|---|
| interactive | native_s32 | 6 | 0.0370 | 6 | 0.0370 |
| moderate | native_s32 | 3 | 0.0741 | 3 | 0.0741 |
| prefill_heavy | native_s32 | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | native_s32 | 6 | 0.0370 | 6 | 0.0370 |
| interactive | native_pace3 | 6 | 0.0370 | 6 | 0.0370 |
| moderate | native_pace3 | 3 | 0.0741 | 3 | 0.0741 |
| prefill_heavy | native_pace3 | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | native_pace3 | 6 | 0.0370 | 6 | 0.0370 |
| interactive | native_pace6 | 8 | 0.0278 | 8 | 0.0278 |
| moderate | native_pace6 | 3 | 0.0741 | 3 | 0.0741 |
| prefill_heavy | native_pace6 | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | native_pace6 | 6 | 0.0370 | 8 | 0.0278 |
