# Open-loop goodput and cost (Poisson arrivals), GPU $0.80/h

| case | profile | goodput@99 req/s | $ / 1k req @99 | goodput@90 req/s | $ / 1k req @90 |
|---|---|---|---|---|---|
| interactive | native_pace6m8 | 8 | 0.0278 | 8 | 0.0278 |
| moderate | native_pace6m8 | 3 | 0.0741 | 3 | 0.0741 |
| prefill_heavy | native_pace6m8 | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | native_pace6m8 | 8 | 0.0278 | 8 | 0.0278 |
| interactive | native_pipe | 8 | 0.0278 | 8 | 0.0278 |
| moderate | native_pipe | 3 | 0.0741 | 4 | 0.0556 |
| prefill_heavy | native_pipe | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | native_pipe | 8 | 0.0278 | 8 | 0.0278 |
