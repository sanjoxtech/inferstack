# Why vLLM and SGLang beat native, and what to change — 13 November 2026

Inputs: the L4 scorecard (docs/verification/COLAB_L4_SCORECARD_2026-11-13.md), the N2
prefill diagnosis (docs/verification/N2_PREFILL_DIAGNOSIS_L4_2026-11-13.md) and the
vLLM 0.33.0 / SGLang 0.5.23 server logs from the same scorecard run (their own startup
configuration, read from `round-0/<engine>/server.log` in the evidence ZIP).

Goal set by the owner: native must improve **on every configuration**, through engine
design, not by tuning one knob for one benchmark. Each item below is judged against that.

## What the three engines actually run (same GPU, model, weights)

| | native (production-v1.6.0) | vLLM 0.33.0 | SGLang 0.5.23 |
|---|---|---|---|
| Model code | Hugging Face Transformers modules + our Triton fusions | own model code, torch.compile (inductor) | own model code |
| Decode launch overhead | eager: every kernel launched from Python each step (`cuda_graphs = false`, `explicit_cuda_graph = false`) | CUDA graphs FULL + PIECEWISE, 51 batch sizes 1–512 | decode CUDA graphs, batch sizes 1–48 |
| Prefill granularity | 128-token chunks, 1 chunk per scheduler turn | chunked prefill with a token budget per step (prefill and decode share one forward) | `chunked_prefill_size` 4096: a 1570-token prompt is one forward (log: `#new-token: 1570`) |
| Gated DeltaNet prefill | upstream PyTorch reference (`chunk_kernel: upstream/reference`) | "Triton/FLA GDN prefill kernel" | `linear_attn_backend: triton` |
| Gated DeltaNet decode | our Triton kernel | Triton | Triton |
| Full attention | Transformers attention interface | FlashAttention 2 | FlashInfer |
| KV / state memory | per-cohort Transformers cache, padded batches | paged KV + paged Mamba/GDN state | paged KV + Mamba state pool, radix cache |
| Sampling | our fused sampler | FlashInfer top-k/top-p | — |

## Where the gap comes from

**Decode (all workloads).** Client decode at c=1: vLLM/SGLang ~16 ms/token, native_full
~29–30, native_unfused ~45. Reading the 2B BF16 weights once is roughly 4–5 GB, about
15 ms on an L4, so vLLM and SGLang decode at close to the memory-bandwidth floor. The
remaining ~14 ms (native_full) is overhead outside the GPU math: the N2 trace counts
~2,600 kernel launches per forward pass, all issued from Python. CUDA graphs replay a whole
step as one launch; that is the main thing both rivals do and native does not.
(Inference from the trace and the bandwidth floor; a decode-only trace would confirm it.)

**Prefill, single long prompt.** Native runs 13 forwards of 128 tokens. Each extra forward
re-reads all weights and repeats the launch overhead (~60 ms each at cohort 1). SGLang does
the same prompt in one forward. Measured native ceiling with one forward: 978 → 214 ms.

**Prefill, several prompts at once.** Compute-bound on native (~7k tokens/s at cohort 4,
chunk size barely matters). GPU time goes to GEMMs and to the Gated DeltaNet prefill
running as hundreds of small PyTorch ops (bmm, FP32 triangular solves, FP32 GEMMs,
copies). Both rivals use a fused Triton (FLA-style) chunked kernel here.

**High concurrency.** Native pads every cohort to its longest prompt, caps at 4 active
cohorts, and keeps one cache per cohort; rivals pack variable-length requests with paged
memory. This is a likely contributor at c=8/16 that this run did not isolate.

## Native improvement plan (engine design, applies to every configuration)

Ordered by expected gain per unit of work. Each step: one candidate, H3 handoff,
numerical/quality gates, then the full L4 scorecard against the recorded baseline; keep
it only if it is better across the scorecard, not on one cell.

1. **Decode CUDA graphs that actually replay.** Capture the decode step per batch-size
   bucket with a static cache. The repo has the switches (`explicit_cuda_graph`,
   `graph_static_cache`) but replay was never proven (CURRENT_STATE_AUDIT A6). Expected:
   decode ms/token from ~29 toward ~16–20 on every case and concurrency. Biggest, broadest win.
2. **Token-budget prefill scheduling.** Replace "one 128-token chunk per turn" with a
   per-step token budget shared by prefill and decode (the vLLM/SGLang design), so a long
   prompt runs in few forwards when the GPU is idle and is split only when decode needs
   the step. A design change, not a chunk-size tweak: it adapts to load.
3. **Fused GDN chunked-prefill kernel.** Bring in an FLA-style Triton kernel (pinned,
   provenance recorded) behind the existing `custom_gdn_triton` switch with the reference
   path as the numerical oracle. Targets the compute-bound multi-prompt prefill.
4. **Packed (varlen) batches and paged state.** Remove padding and the per-cohort cache
   limit; largest change, needed for c=16-class load. After 1–3 show their effect.

Not pursued: raising `prefill_chunk_tokens` alone (helps one case, hurts decode
interleaving), prefix-cache snapshot changes (1.3% of prefill).

Constraint: items 1–4 change native runtime or kernels. The owner's standing rule says no
new native kernels or optimization version yet, so each needs an explicit go-ahead.
