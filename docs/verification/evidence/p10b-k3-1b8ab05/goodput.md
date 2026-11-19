# Open-loop goodput and cost (Poisson arrivals), GPU $0.80/h

| case | profile | goodput@99 req/s | $ / 1k req @99 | goodput@90 req/s | $ / 1k req @90 |
|---|---|---|---|---|---|
| interactive | native_lazy | 8 | 0.0278 | 8 | 0.0278 |
| moderate | native_lazy | 3 | 0.0741 | 4 | 0.0556 |
| prefill_heavy | native_lazy | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | native_lazy | 8 | 0.0278 | 8 | 0.0278 |
| interactive | native_lazy_k3 | 8 | 0.0278 | 8 | 0.0278 |
| moderate | native_lazy_k3 | 4 | 0.0556 | 4 | 0.0556 |
| prefill_heavy | native_lazy_k3 | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | native_lazy_k3 | 6 | 0.0370 | 8 | 0.0278 |
