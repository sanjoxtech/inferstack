# P13 plan: piecewise CUDA graphs and mixed prefill+decode steps (2026-11-19)

Status: plan, nothing built. Target: Qwen3.5-2B-Base@b1485b2f, BF16, one L4 (Colab), open-loop goodput
(`scripts/goodput.py`, SLO `configs/tune/slo-v2-l4.json`).

## Why

Open-loop goodput@99 (req/s), v1.36 against the rivals (P9/P10 runs, see
`docs/verification/P9_COST_AT_SLO_L4_2026-11-18.md`):

| case | v1.36 | vLLM 0.33.0 | SGLang 0.5.23 |
|---|---|---|---|
| interactive | 8 (12 req/s: 66-77% good) | 12 | 6 |
| moderate | 3-4 | 4 | 3 |
| prefill_heavy | 2 | 2 | 1 |
| shared_prefix | 8 | 12 | 6 |

Per-token decode is not the gap: at 34-38 live rows a spec step emits 1.97 tokens per row in 37-41 ms
(~20 ms per token). The gap is how arrivals enter:

- every arrival needs its own packed prefill pass (interactive 12 req/s: 69 prefills x 37.8 ms, 12% of
  GPU time), which re-reads all 4.15 GB of weights and stalls every decoding row for that pass;
- to amortise that pass, admission waits up to 6 decode steps (~234 ms), the main source of TTFT misses
  (56 of 240 at 12 req/s); waiting less (K=3) makes more prefills and collapses earlier (P10b).

vLLM serves 12 req/s by putting prefill tokens (chunked) into the same forward as decode tokens, so the
weights are read once per step and nothing waits for a separate pass. It keeps that cheap with piecewise
CUDA graphs: the dense parts are captured per total-token count, attention runs outside the graphs.
Our engine captures whole steps (one graph per batch bucket x length bucket), so a mixed step today
would run eager. Eager is host-bound here (a small packed prefill: ~21 ms of GPU work inside ~65 ms wall;
FLA's varlen chunk call ~1.2 ms of Python per GDN layer, 18 layers).

Ideas only from vLLM / SGLang (Apache-2.0); no code copied.

## Design

One step = a flat token batch: the decode rows' spec tokens first (2 per row: last accepted token, MTP
draft), then prefill tokens (whole short prompts, or a chunk of a long one), N tokens in all.

Per layer:

1. dense-in (graphed): input RMSNorm + the mixer's input projections (GDN: qkvz + ba; attention: qkv).
2. mixer (eager, split by token range):
   - decode range [0, 2B): today's kernels unchanged (banked GDN with lazy banks; decode attention
     over the slot pool), viewed as [B, 2, ...];
   - prefill range [2B, N): FLA varlen conv + chunk delta rule from the slot's state (the existing
     `starts` continuation path), varlen attention writing K/V into the slots.
3. dense-out (graphed): output projection + residual + post-norm + MLP + residual, fused with the next
   layer's dense-in, so there are 25 pieces per step (24 mixers).

Head: final norm, lm_head only on the 2B decode positions plus each prefill row's last position (a
gather, not all N); verify/accept, then the MTP draft for decode rows (small, eager or one graph) and
the MTP window load for rows whose prompt finished in this step (`load_spec_rows`, as now).

Graphs: one per (piece, token bucket N), with N in {16, 32, 64, 96, 128, 192, 256, 384, 512}: 25 x 9 =
225 small graphs with static input/output buffers per bucket, one memory pool. A pure-decode step can
still use today's whole-step graph if it stays faster (gate in S2).

Scheduler: a token budget per step (e.g. 256 prefill tokens). Arrivals join the next step at once (no
pacing); a long prompt is chunked across steps and decodes only after its last chunk. Prefix-cache hits
and snapshot recording keep the current packed path at first (fallback flag), then move over.

## Stages and gates

| stage | work | evidence | gate to continue |
|---|---|---|---|
| S0 | measure, no engine change: (a) host and GPU time of eager FLA varlen conv + chunk per layer at N = 32..512, cold and warm; (b) one dense piece captured per bucket, replay vs the same ops inside today's whole-step graph; (c) projected mixed-step time = sum of pieces + mixers | `scripts/piecewise_probe.py` on L4, ~$0.50 | projected mixed step (34 decode rows + ~250 prefill tokens) <= decode step + 60% of today's separate prefill, and host time < GPU time |
| S1 | refactor the spec step into dense pieces + mixers over a flat token batch, eager only; decode-only and mixed variants | CPU greedy-equality tests vs today's lane (`tests/test_qwen_static_lane.py` style), Triton tests on GPU | outputs identical; no change to served profiles |
| S2 | piecewise capture per token bucket; pure-decode A/B vs whole-step graphs | `decode_step_profile.py` + goodput A/B, ~$1.50 | pure decode no more than 3% slower (else keep whole-step graphs for decode-only steps) |
| S3 | mixed scheduler: token budget, chunked prefill, admission without pacing; prefix-hit rows on the old path | lane tests (arrivals mid-decode, chunk boundaries, cancellation, row moves, pipeline) | greedy-equal; no regression in the light-load TTFT (45-55 ms) |
| S4 | goodput A/B vs v1.36 (3 runs each, the run-to-run spread is ~10 points at the knee) | `scripts/goodput.py`, ~$1.50 per run | keep if interactive or shared_prefix reaches 12 req/s at 99%, or a clear gain on both with no loss elsewhere |

Each run stays under the $10 campaign gate; S0 to S4 together are expected near $8 if nothing repeats.

## Risks

- Host overhead of FLA's varlen calls in eager mixers (S0a decides; fallback: graph the prefill mixer for
  fixed chunk shapes, as P4 does for small prefills, or a lighter wrapper around FLA's kernels).
- Graph count and memory (225 graphs plus static buffers; measure in S2, trim buckets if needed).
- GDN state hand-off when a prompt is chunked across steps (the `starts` path already continues from the
  slot's conv/recurrent state; needs tests at chunk sizes that are not multiples of 64).
- Interactions to keep working: lazy banks and the reject fix-up, the pipelined step (P9f), row
  compaction, the pool prefix cache, overlap prefill (P7). S1 keeps all of them behind the old path until
  each is covered by a test.

## Out of scope

FP8 or other weight formats (changes numerics; a separate, labelled result), multi-GPU, sampling (the
spec lane is greedy-only), new models.

## S0 result (run 2e5c80c, L4): gate failed, P13 stopped in this form

| piece | measured |
|---|---|
| today's decode step (graph), 16 / 32 / 48 rows | 27.0 / 33.9 / 41.4 ms |
| same step eager (host time to launch) | ~39-40 ms host at any row count |
| today's separate prefill (graphed), 64 / 128 / 256 / 512 tokens | 24.1 / 27.9 / 39.8 / 64.9 ms |
| dense ops of a step on N tokens, 25 piece graphs vs one graph | N=64: 13.6 vs 13.4; N=320: 26.1 vs 25.3; N=576: 44.9 vs 43.1 ms |
| prefill mixers eager (FLA varlen conv + chunk, SDPA), 64..512 tokens | 29.9-32.6 ms, equal to their host time |

Projected mixed step vs today's two steps: slower by 3-10 ms for 64-256 prefill tokens at every row count,
about even at 512; the gate (save >= 40% of the separate prefill) fails in all 12 cells.

What it shows:
- Piecewise graphs themselves are cheap: 25 pieces cost 0-2 ms over one graph, with ~0.2 ms host.
- Eager mixers are host-bound: the FLA varlen calls take ~30 ms of Python per step whatever the size (the
  GPU waits on the host), and an eager spec step takes ~39 ms of host launches. A mixed step only pays if
  every mixer is graphed too (prefill mixers per chunk shape, decode mixers per batch x length bucket),
  hundreds to over a thousand graphs.
- Sharing the weight read helps less than assumed on L4: dense cost is flat to ~128 tokens (12.9-14.6 ms)
  and then grows with tokens (24.7 ms at 288, 42 ms at 544): past ~128 tokens the step is compute-bound,
  so prefill tokens are not free riders on a decode step's weight read.
- Even with all mixers graphed, the best case (32 rows + 64 prefill tokens: ~46 vs 58 ms separate) saves
  ~12 ms per prefill, about 4% of GPU time at interactive 12 req/s (69 prefills in 20 s), plus shorter
  admission waits.

Decision: do not build P13 as planned. v1.36 stays the served profile.
