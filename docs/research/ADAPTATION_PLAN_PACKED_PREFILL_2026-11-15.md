# Adaptation plan: what to take from vLLM, SGLang and mini-sglang, compared with InferStack v1.14

Sources read: vLLM v0.33.0, SGLang v0.5.23 and mini-sglang @9a91cfa (MIT, ~9k lines), plus the
landscape survey `INFERENCE_ENGINE_LANDSCAPE_2026-11-15.md`. We take ideas only; no code is copied.

## Side by side

| Concern | vLLM / SGLang / mini-sglang | InferStack v1.14 |
|---|---|---|
| Prefill input | Flat tokens of many requests, no padding, positions per request | Padded cohort `[B, W]`, left padding plus attention_mask |
| Prefill writes into | The same KV/state pool that decode reads (page table or slot index) | A Transformers cache, then `load_rows` copies it into the static pool, with one `nonzero` host sync per row |
| GDN in prefill | Chunked kernel with `cu_seqlens`; initial state gathered by slot, final state written to slot | FLA chunked kernel on padded `[B, W]` through the Transformers module |
| Full attention in prefill | Varlen flash/flashinfer kernels | Transformers sdpa over padded rows |
| Prefill vs decode order | Prefill batch first when any is runnable (mini-sglang); SGLang paces prefill by an interval once the batch is large | Decode turn first, then one prefill job per loop iteration (`decode_priority = true`) |
| CPU/GPU overlap | Next tokens stay on the GPU (token pool); host results processed one step late with a non-blocking copy and an event | First token: `.tolist()` sync inside `_accept` in prefill; spec step: one sync per step |
| Decode | CUDA graphs per batch size | CUDA graphs per (batch, length) bucket, spec verify+draft in one graph (already at parity or ahead) |
| Mixed prefill+decode | vLLM always; SGLang only with `--enable-mixed-chunk` (off in our runs) | None |

We are ahead on decode (10–12 ms/token against ~15.6). We are behind on the prefill side, which
covers both the forward itself and the hand-off into the decode pool.

## Plan (each step must be greedy-equal to Transformers, then one L4 fast check under $1)

**P0: TTFT split (running).** Queue, wait and prefill per request, v1.13 against v1.14. If "wait"
dominates, P3 moves ahead of P1.

**P1: packed prefill straight into the static pool.** This is the core change.

- New `QwenStaticDecoder.prefill_packed(st, slots, prompts)`:
  - **Inputs and dense layers:** flat tokens `[1, T]` with `cu_seqlens` and per-request positions.
    The dense layers run once over the T tokens.
  - **GDN conv:** FLA's varlen causal conv. The last K-1 inputs per request go into
    `st.conv[slot]`.
  - **GDN recurrent:** FLA `chunk_gated_delta_rule(cu_seqlens=…, initial_state=0)`. The final
    states go into `st.rec[slot]`, which is bank 0 when spec is on.
  - **Full attention:**
    - K/V are written straight into `st.kv[i][slot, :, pos]` and the mask columns are set.
    - Attention v1 runs sdpa per request: only 6 of the 24 layers are full attention, and it is
      simple and exact.
    - A varlen kernel comes later, only if the profile asks for it.
  - **Logits:** only at each request's last token. The post-norm hidden states of the last W
    tokens feed `load_spec_rows` directly.
- **Removes:** padding work, the Transformers cache build, `load_rows` copies and per-row syncs.
- **Fallbacks:** prefix-cache hits and rows that don't fit the pool stay on today's path.
- **Targets:** prefill_heavy c4–c16 and TTFT at every load.

**P2: prefix cache in pool format.** Snapshot the KV columns and the GDN conv/rec state at
64-token block boundaries; this follows vLLM's "align" mode and LMDeploy #4465. A hit copies the
snapshot into the slot, and P1 continues from `cached_len`. Target: shared_prefix c16.

**P3: scheduling and overlap (mini-sglang style).**

- Prefill first when requests are waiting, with a token budget. After a prefill the decode rows run
  next, so they never starve.
- The first token stays on the GPU and is emitted after the next launch. The host step lags by one
  step, with a non-blocking copy and an event.
- Target: TTFT and c1 latency.

**P4: prefill CUDA graphs by token bucket** (vLLM piecewise). Dense layers in the graph, attention
and GDN outside. Target: c1 TTFT (65 against 41 ms).

**P5: N5.2 mixed step.** The P1 forward with spec decode rows prepended (2 tokens each). The GDN
split follows vLLM: decode rows go to the banked kernel, prefill segments to the chunked kernel.

## Not adopting now

- RDMA/host KV mobility: needs more than one node.
- FP8 GEMMs: changes numerics, and the comparison is unfair unless the rivals run FP8 too.
- Megakernels: decode is already ahead.
- Rust/C++ rewrite: no profile shows Python as the blocker.
