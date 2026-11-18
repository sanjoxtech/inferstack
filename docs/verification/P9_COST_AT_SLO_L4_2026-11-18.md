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
