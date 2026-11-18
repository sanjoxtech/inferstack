# P9: cost per 1,000 requests at the SLO, InferStack v1.23 vs vLLM vs SGLang (L4, 2026-11-18)

Qwen3.5-2B-Base@b1485b2f, BF16, one L4 (Colab), default workload, 3 rounds per engine.
Two sessions, each with InferStack v1.23 as the in-session baseline:
- SGLang 0.5.23: run 1999985 (evidence `p9-rivals-l4-1999985-20261118/`; vLLM 0.34.0 failed to start there, OOM).
- vLLM 0.33.0 (pinned): run 14f4a07 (evidence `p9-vllm-l4-14f4a07-20261118/`).
InferStack's own numbers agree across both sessions within 1%.

Method (`scripts/cost_at_slo.py`): per engine and case, the highest concurrency (1/4/8/16) whose
3-round medians meet the SLO v2 **industry** latency limits (same bar for every engine: TTFT
p95/p99 and per-token p95 by workload class; no "beat the rival" floor) with 100% success. Cost =
GPU $/h / requests per second, at an assumed **$0.80/h** for an L4 (relative, not a bill).
Closed-loop approximation, not open-loop goodput.

| case | InferStack v1.23 | vLLM 0.33.0 | SGLang 0.5.23 | cheapest |
|---|---|---|---|---|
| interactive (chat) | **$0.029** /1k req (c16) | $0.036 (c16) | $0.035 (c16) | InferStack, -18% vs best rival |
| moderate (~500 tok prompts) | **$0.061** (c16) | $0.135 (c8) | $0.245 (c4) | InferStack, 2.2x cheaper than vLLM |
| prefill_heavy (~2k tok prompts) | **$0.040** (c4) | $0.048 (c4) | $0.045 (c4) | InferStack, -12% |
| shared_prefix (RAG) | $0.024 (c8) | $0.015 (c16) | **$0.015** (c16) | rivals, 37% cheaper |

Per 1M output tokens at the same points: InferStack 0.23 / 0.24 / 1.24 / 0.68; vLLM 0.28 / 0.53 / 1.49 / 0.41;
SGLang 0.28 / 0.96 / 1.40 / 0.40 (interactive / moderate / prefill_heavy / shared_prefix).

## Reading

- We are the cheapest engine at the SLO in 3 of 4 workload classes. Moderate is the big one: the
  rivals miss the TTFT limit beyond 4-8 users, we hold it at 16.
- Shared-prefix (RAG) is our clear loss: at 16 users our TTFT p95 (~350 ms) is over the 300 ms
  class limit, so we are priced at 8 users; both rivals meet it at 16. Next target.
- prefill_heavy: every engine is capped at 4 users by TTFT p95 / per-token p95 limits.

## Caveats

Fixed concurrency, not Poisson arrivals; one GPU type; assumed price; scorecard prompts are reused
across cells (prefix caches matter, for all engines). Open-loop goodput is the next step of P9.

## Update: v1.28 closes the shared-prefix gap (run e8c590e, 3 rounds vs v1.23)

The shared-prefix c16 burst was split 6+7+3 by the engine gatherer (rows x longest prompt <=
max_batch_tokens 8192), not by the lane. v1.28 = v1.23 + max_batch_tokens 32768, a packed budget
that counts only uncached tokens (8192), and re-queueing of rows over the budget (instead of the
Transformers path).

| shared_prefix | v1.23 | v1.28 |
|---|---|---|
| c16 TTFT p95 | 354 ms | **257 ms** (now under the 300 ms class limit) |
| c16 output tok/s | 488 | **533** (+9%) |
| c8 output tok/s | 335 | 345 |
| $ / 1k requests at the SLO | $0.0234 (c8) | **$0.0152 (c16)**, level with vLLM ($0.0153) and SGLang ($0.0148) |

Other cells: throughput overlaps in 13 of 16, SLO v2 cells met 9 / 9. Watch: prefill_heavy c8 TTFT
p95 242 -> 480 ms (also seen with v1.27); it does not change the cost table (prefill_heavy is priced at
c4 for every engine, c8 fails the per-token limit), but it is the next tail to explain.

**Result: InferStack v1.28 is the cheapest engine at the SLO in 3 of 4 workload classes and level
in the 4th (shared_prefix).** Decision: KEEP v1.28 as the candidate.

## Update: v1.29 removes the prefill_heavy c8 tail (run 3a53316, 3 rounds vs v1.28)

v1.28's 8,192 uncached-token budget let 3-4 cold ~2k-token prompts share one prefill (TTFT 383-484 ms
for the whole group). v1.29 lowers it to 4,096 (at most 2 cold rows; a cache-hit burst of ~100 new
tokens per row still goes in one): prefill_heavy c8 TTFT p95 381 -> 246 ms, shared_prefix c16 kept
(259 -> 257 ms, $0.0152/1k), throughput overlaps in all 16 cells. prefill_heavy c16 TTFT p95 467 -> 527 ms
(that cell fails the per-token limit for every engine, so the cost table is unchanged).
**Decision: v1.29 is the candidate.**

## Open-loop goodput (Poisson arrivals), run faeed5f

`scripts/goodput.py`, v1.29 (`native_bb4k`) vs vLLM 0.33.0 vs SGLang 0.5.23 on one L4. A request is good
when TTFT <= the class p99 limit and per-token time <= the class limit; goodput@99 = highest arrival
rate with >= 99% good requests. Evidence: `docs/verification/evidence/p9-goodput-faeed5f/`.

| case | v1.29 req/s (@99) | vLLM | SGLang |
|---|---|---|---|
| interactive | 4 | **12** | 6 |
| moderate | 2 | **4** | 3 |
| prefill_heavy | 2 | 2 | 1 |
| shared_prefix | 6 | **12** | 6 |

This reverses the closed-loop ranking: under random arrivals vLLM serves 2-3x our rate. The closed-loop
table above stays valid for fixed concurrency, but it is not the headline.

Cause (from per-request rows): below the cliff v1.29 is faster per token (interactive 11.5 ms vs
vLLM 17.5 ms at 2 req/s), then collapses in one step (interactive 4 -> 6 req/s: per-token 17 -> 94 ms
median, TTFT p99 79 ms -> 15 s). When all 16 slots are taken, `admit_packed` returned rows that fit the
lane to the Transformers prefill path, so a slow side cohort decoded beside the lane and stretched every
step. vLLM keeps 15-48 requests in flight at the same rates with per-token <= 31 ms.

Fix under test (P9b, v1.30): `static_decode_wait_for_slot` keeps such rows queued until a slot frees.
Expected: no collapse, but a ceiling of about 16 slots / request time (interactive ~6 req/s), so
beating vLLM's 12 also needs more slots (P9c: 32 slots, decode graph bucket 32).

### P9b result (run 7834d40, same L4 session A/B)

| case | v1.29 req/s (@99) | v1.30 (wait for slot) |
|---|---|---|
| interactive | 4 | **6** |
| moderate | 2 | 2 |
| prefill_heavy | 2 | 2 |
| shared_prefix | 6 | 6 (8 req/s: 81% good vs 9%) |

The collapse is gone: past capacity the misses are now queue time (TTFT), not slow decode. Per-token
stays at 18-19 ms (limit 30) while TTFT queueing starts once in-flight requests reach the 16 slots
(moderate 3 req/s: 15 in flight, mean queue 307 ms; 4 req/s: 2.1 s). Engine counters: each arrival
still gets its own packed prefill (~55 ms), and MTP acceptance is 96%. Kept as v1.30.

Next, P9c (v1.31): 32 slots and decode graph buckets up to 32, to use the per-token headroom.
