# SLO tune

Method: optuna 4.5.0 TPE (constraints) · trials: 7 · feasible: 0
SLO: `{"objective": "output_tps", "cells": [["interactive", 1], ["interactive", 4], ["interactive", 8], ["interactive", 16], ["moderate", 1], ["moderate", 4], ["moderate", 8], ["moderate", 16], ["prefill_heavy", 1], ["prefill_heavy", 4], ["prefill_heavy", 8], ["prefill_heavy", 16], ["shared_prefix", 1], ["shared_prefix", 4], ["shared_prefix", 8], ["shared_prefix", 16]], "min_success_rate": 1.0, "cell_limits": {"interactive@1": {"ttft_p95_ms": 43.2}, "interactive@4": {"ttft_p95_ms": 92.3}, "interactive@8": {"ttft_p95_ms": 91.4}, "interactive@16": {"ttft_p95_ms": 131.1}, "moderate@1": {"ttft_p95_ms": 46.6}, "moderate@4": {"ttft_p95_ms": 137.1}, "moderate@8": {"ttft_p95_ms": 253.0}, "moderate@16": {"ttft_p95_ms": 401.4}, "prefill_heavy@1": {"ttft_p95_ms": 122.0}, "prefill_heavy@4": {"ttft_p95_ms": 426.2}, "prefill_heavy@8": {"ttft_p95_ms": 794.9}, "prefill_heavy@16": {"ttft_p95_ms": 1471.9}, "shared_prefix@1": {"ttft_p95_ms": 77.8}, "shared_prefix@4": {"ttft_p95_ms": 92.6}, "shared_prefix@8": {"ttft_p95_ms": 109.5}, "shared_prefix@16": {"ttft_p95_ms": 150.5}}, "cell_floors": {"interactive@1": {"output_tps": 63.4}, "interactive@4": {"output_tps": 231.0}, "interactive@8": {"output_tps": 440.4}, "interactive@16": {"output_tps": 805.1}, "moderate@1": {"output_tps": 63.7}, "moderate@4": {"output_tps": 232.4}, "moderate@8": {"output_tps": 434.8}, "moderate@16": {"output_tps": 776.9}, "prefill_heavy@1": {"output_tps": 52.7}, "prefill_heavy@4": {"output_tps": 159.6}, "prefill_heavy@8": {"output_tps": 241.5}, "prefill_heavy@16": {"output_tps": 323.9}, "shared_prefix@1": {"output_tps": 60.5}, "shared_prefix@4": {"output_tps": 195.3}, "shared_prefix@8": {"output_tps": 295.4}, "shared_prefix@16": {"output_tps": 556.4}}, "notes": "Dominance SLO: in every cell beat the best of vLLM 0.33.0 and SGLang 0.5.23 on both TTFT p95 (<= best rival) and output tok/s (>= best rival); within that, maximise geometric-mean tok/s. Rival values: medians of 3 rounds, L4, Qwen3.5-2B-Base BF16, default workload, run p4-full-l4-3d08aa0-20261116. Re-measure rivals before claiming."}`

| trial | feasible | objective tok/s | params |
|---|---|---|---|
| 0 | False | 257.9 | `{"native.max_requests_per_batch": 4, "native.batch_wait_ms": 6.0, "native.static_decode_compact_from_tail": true, "native.decode_quantum": 1, "native.prefix_state_cache_max_entries": 128, "native.static_decode_spec": true}` |
| 1 | False | 197.1 | `{"native.max_requests_per_batch": 16, "native.batch_wait_ms": 0.0, "native.static_decode_compact_from_tail": true, "native.decode_quantum": 1, "native.prefix_state_cache_max_entries": 256, "native.static_decode_spec": false}` |
| 2 | False | 196.4 | `{"native.max_requests_per_batch": 4, "native.batch_wait_ms": 0.0, "native.static_decode_compact_from_tail": true, "native.decode_quantum": 2, "native.prefix_state_cache_max_entries": 256, "native.static_decode_spec": false}` |
| 3 | False | 276.8 | `{"native.max_requests_per_batch": 4, "native.batch_wait_ms": 1.0, "native.static_decode_compact_from_tail": true, "native.decode_quantum": 2, "native.prefix_state_cache_max_entries": 256, "native.static_decode_spec": true}` |
| 4 | False | 280.0 | `{"native.max_requests_per_batch": 8, "native.batch_wait_ms": 0.0, "native.static_decode_compact_from_tail": true, "native.decode_quantum": 1, "native.prefix_state_cache_max_entries": 128, "native.static_decode_spec": true}` |
| 5 | False | 279.1 | `{"native.max_requests_per_batch": 16, "native.batch_wait_ms": 0.0, "native.static_decode_compact_from_tail": true, "native.decode_quantum": 2, "native.prefix_state_cache_max_entries": 128, "native.static_decode_spec": true}` |
| 6 | False | 196.8 | `{"native.max_requests_per_batch": 4, "native.batch_wait_ms": 6.0, "native.static_decode_compact_from_tail": false, "native.decode_quantum": 1, "native.prefix_state_cache_max_entries": 256, "native.static_decode_spec": false}` |

No feasible trial: the SLO is not met anywhere in this space.
Closest: trial 4 → closest.toml; cells still missing the SLO:
- interactive@1: `{"output_tps": 97.40028755517358, "success_rate": 1.0, "ttft_p95_ms": 31.199474850029674}`
- interactive@4: `{"output_tps": 349.01958836582503, "success_rate": 1.0, "ttft_p95_ms": 80.60135750002928}`
- interactive@8: `{"output_tps": 621.3894624009289, "success_rate": 1.0, "ttft_p95_ms": 78.29938380001522}`
- interactive@16: `{"output_tps": 1000.6935519700111, "success_rate": 1.0, "ttft_p95_ms": 170.6973258498806}`
- moderate@1: `{"output_tps": 94.42154014754007, "success_rate": 1.0, "ttft_p95_ms": 73.49652965006044}`
- moderate@4: `{"output_tps": 340.0112645300312, "success_rate": 1.0, "ttft_p95_ms": 90.78704699993523}`
- moderate@8: `{"output_tps": 591.7645417004139, "success_rate": 1.0, "ttft_p95_ms": 137.98450904993157}`
- moderate@16: `{"output_tps": 959.1470180429784, "success_rate": 1.0, "ttft_p95_ms": 221.30037890003678}`
- prefill_heavy@1: `{"output_tps": 66.57188638235341, "success_rate": 1.0, "ttft_p95_ms": 124.63547495000284}`
- prefill_heavy@4: `{"output_tps": 183.32104803034002, "success_rate": 1.0, "ttft_p95_ms": 254.24666699996123}`
- prefill_heavy@8: `{"output_tps": 253.4650925676418, "success_rate": 1.0, "ttft_p95_ms": 482.7949117999651}`
- prefill_heavy@16: `{"output_tps": 309.3379576957386, "success_rate": 1.0, "ttft_p95_ms": 736.7401545999769}`
- shared_prefix@1: `{"output_tps": 81.43584643535819, "success_rate": 1.0, "ttft_p95_ms": 74.19742980000589}`
- shared_prefix@4: `{"output_tps": 254.9432382862651, "success_rate": 1.0, "ttft_p95_ms": 91.39985849986942}`
- shared_prefix@8: `{"output_tps": 353.9723888700828, "success_rate": 1.0, "ttft_p95_ms": 150.82298745005568}`
- shared_prefix@16: `{"output_tps": 525.5592217001026, "success_rate": 1.0, "ttft_p95_ms": 287.7304148500457}`
