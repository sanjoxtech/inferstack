# Open-loop goodput and cost (Poisson arrivals), GPU $0.80/h

| case | profile | goodput@99 req/s | $ / 1k req @99 | goodput@90 req/s | $ / 1k req @90 |
|---|---|---|---|---|---|
| interactive | native_bb4k | 4 | 0.0556 | 4 | 0.0556 |
| moderate | native_bb4k | 2 | 0.1111 | 2 | 0.1111 |
| prefill_heavy | native_bb4k | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | native_bb4k | 6 | 0.0370 | 6 | 0.0370 |
| interactive | vllm | 12 | 0.0185 | 12 | 0.0185 |
| moderate | vllm | 4 | 0.0556 | 4 | 0.0556 |
| prefill_heavy | vllm | 2 | 0.1111 | 4 | 0.0556 |
| shared_prefix | vllm | 12 | 0.0185 | 12 | 0.0185 |
| interactive | sglang | 6 | 0.0370 | 6 | 0.0370 |
| moderate | sglang | 3 | 0.0741 | 3 | 0.0741 |
| prefill_heavy | sglang | 1 | 0.2222 | 2 | 0.1111 |
| shared_prefix | sglang | 6 | 0.0370 | 8 | 0.0278 |
