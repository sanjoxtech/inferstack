# vLLM 0.33.0 and SGLang 0.5.23: how they run Qwen3.5 prefill and decode together

Source read: shallow clones of the exact tags we benchmark (`vllm@v0.33.0`, `sglang@v0.5.23`).
Purpose: inputs for N5.2 (mixed prefill+decode) and the TTFT work. Ideas only, no copied code.

## vLLM: one packed token batch per step

- Every step is one flat batch of tokens with no padding. Decode tokens come first, then prefill
  chunk tokens.
- The dense parts (projections, MLP, norms) run once over all tokens. Full attention uses
  varlen/paged kernels over the same packed batch.
- Only the GDN layer splits the batch
  (`model_executor/layers/mamba/gdn/qwen_gdn_linear_attn.py`, `_forward_core`):
  - **Conv:**
    - Decode tokens use `causal_conv1d_update`, with a per-row slot index (`state_indices`).
    - Prefill segments use `causal_conv1d_fn`, with `query_start_loc` (cu_seqlens), `cache_indices` and
      `has_initial_state`.
  - **Recurrent state:**
    - Decode rows use one recurrent-update kernel, `fused_sigmoid_gating_delta_rule_update`, reading
      and writing `ssm_state[slot]` in place.
    - Prefill segments use the chunked kernel, `chunk_gated_delta_rule`, with `cu_seqlens`. Its initial
      state is gathered from `ssm_state[prefill_state_indices]` and zeroed for fresh requests. The
      final state is written back to the same slots.
  - **Output:** the outputs are stitched back in decode-first order.
- **Spec decode:** spec rows get their own index set (`spec_state_indices_tensor`
  [batch, num_spec+1]). The state is kept per draft position. There is also a fused MTP decode path.
- **Metadata:** the batch metadata (`v1/attention/backends/gdn_attn.py`) holds the counts `num_prefills`,
  `num_decodes` and `num_spec_decodes`, plus the token-index splits. It is built once per step and
  shared by all GDN layers.

## SGLang: separate prefill and decode batches, fast prefill

- **Mixed mode is off:** mixing prefill and decode in one batch needs `--enable-mixed-chunk`. The
  scorecard launches SGLang without it (`configs/scorecard/*.json`). The test helpers default it to
  False, so in our runs SGLang very likely ran prefill-only and decode-only batches.
  - Not verified: the flag's default value in the server arg definition.
- **Prefill pacing:** a scheduler interval protects decode once the running batch is large enough
  (`managers/scheduler.py`, `_should_defer_prefill`). Small running batches admit prefill every
  round.
- **Conclusion for us:** SGLang beats us on prefill_heavy c4–c16 and on TTFT (122–168 ms vs our
  350–765 ms) **without** mixed batches. Its edge is a fast packed prefill (varlen attention plus
  the chunked GDN kernel over many prompts at once) and scheduling, not prefill+decode fusion.

## What this means for InferStack

1. **Prefill first, not the fusion step.** Our prefill still runs the Transformers modules on
   padded cohorts and writes an HF cache that is then copied into the static pool. A packed
   prefill straight into the pool has three parts:
   - flat tokens plus cu_seqlens;
   - the chunked GDN kernel with initial state gathered by slot;
   - varlen attention writing the pool's KV.

   This targets TTFT and prefill_heavy directly, and it is the base N5.2 needs anyway.
2. **N5.2 shape:** follow vLLM. One packed step = [spec decode rows (2 tokens each) | prefill chunk
   tokens], one pass for the dense layers, with GDN and attention split by segment. Our banked spec
   GDN kernel becomes the decode branch.
3. **Order check:** the TTFT split run (queue / wait / prefill) decides between "packed prefill
   first" and "scheduler first".
