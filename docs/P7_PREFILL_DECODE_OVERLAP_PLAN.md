# P7 plan: prefill and decode at the same time on one L4 (SLO v2 long-prompt cells)

Status: design, 2026-11-17. Not started in code.

## Why

SLO v2 (`configs/tune/slo-v2-l4.json`) asks, for prefill_heavy c16 (~2k-token prompts, 16 users):
TTFT p95 <= 500 ms **and** per-token p95 <= 40 ms. Measured on L4 (run p4-full-l4-3d08aa0, 3 rounds):

| engine | TTFT p50 / p95 | per-token p50 / p95 | e2e p50 |
|---|---|---|---|
| SGLang 0.5.23 | 177 / 1720 ms | 21.6 / 61.9 ms | 2352 ms |
| vLLM 0.33.0 | 460 / 1502 ms | 30.5 / 61.4 ms | 1309 ms |
| InferStack v1.16 | 273 / 476 ms | 61.2 / 86.1 ms | 2288 ms |

The rivals favour decode (new requests wait for the first token, then stream fast). We favour
prefill (first token fast, then running requests stall ~150 ms behind each packed prefill).
Both are taking turns on the GPU, so the stall moves between TTFT and per-token latency but the
sum does not shrink. Meeting both limits needs the decode steps to keep running *while* a
prefill runs. Nobody in the scorecard does that today.

## Options (rival and paper study before code)

1. **Mixed batch** (vLLM V1 chunked prefill): decode rows ride in the prefill forward as 1-token
   segments. Our `prefill_packed(starts=...)` already continues rows from slot state, so a live row
   is just a segment with start = its position. Blocker: the spec lane (MTP draft, two recurrent
   banks, MTP KV cache) would not advance through this path. Works only with spec off, or with a
   spec-aware mixed step (large change).
2. **Two streams, one GPU** (NanoFlow / POD-Attention style intra-device overlap): launch the
   packed prefill on a second CUDA stream into slots the decode graph never touches, and keep
   replaying decode graphs on the main stream. Decode is memory-bound and prefill compute-bound,
   so they share the SMs reasonably. Requirements:
   - new rows prefill into **top slots** (>= the decode batch bucket), then one `move_row` each when
     done (~1 ms each, measured P6); padding rows in the decode graph must never reach them;
   - the prefill's host work must not block the decode loop: today it runs on the single GPU
     thread (FLA host overhead ~1.2-1.6 ms per GDN layer call). Needs a second launch thread or
     a CUDA-graphed prefill (P4 graphs cover small buckets only);
   - completion is an event, not a sync: the lane polls `event.query()` between decode steps,
     then samples the first token and loads spec rows.
3. **Chunked prefill alone**: does not help this cell. Total prefill time per request lifetime is
   unchanged; it only spreads the same stall (n44 measured it worse for TTFT).

## Proposed order

1. Measure the ceiling cheaply: a script that replays a decode graph on stream A while a
   1×2k packed prefill runs on stream B (L4), reporting decode step time and prefill time
   alone vs together. If decode steps stay < ~35 ms during prefill and prefill slows < ~1.5×,
   option 2 is worth building. (< $0.5)
2. Top-slot admission + event completion in the lane, behind a flag (`static_decode_overlap_prefill`).
3. Second launch thread for the prefill, or graph more prefill shapes.
4. A/B vs v1.18 on SLO v2 with the tuner's scorer (cells passing, then geo tok/s).

Risk: GIL contention between two launch threads; FLA autotune on a second stream; correctness of
slot isolation (tests: greedy equality with overlap on, rows finishing mid-prefill).
