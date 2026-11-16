# Rival study: how vLLM graphs Gated DeltaNet (P4), 16 November 2026

Source: vLLM `vllm/v1/attention/backends/gdn_attn.py` on main (Apache-2.0), read for design
ideas only. No code was copied.

## What vLLM does

- **CUDA graphs cover decode only.** That includes speculative decode. A batch with any prefill
  runs eagerly. Prefill-side metadata (cu_seqlens, FLA chunk indices and offsets) is rebuilt for
  every batch and is never inside a graph.
- **Persistent buffers.** All metadata a captured graph reads lives in tensors allocated once at
  init, sized for the largest capture: state indices, query start locations, spec masks and
  accepted counts. Each batch copies into them in place, so a graph never points at memory it
  does not own. Our P4 crash was exactly this failure (FLA's cache freed tensors a graph read).
- **Chunk indices are built on the host** from a CPU copy of the query start locations, then
  copied to the GPU asynchronously, which avoids a device-to-host sync.
- **Padding is inert by construction.** Padded request slots get a null state index, zero query
  length (the start-location tail repeats the final count), a false spec mask and an accepted
  count of 1.

## What we do differently, and what to take

- **We go further.** v1.16 also graphs small prefills: k rows padded to a fixed split, made
  inert with beta = 0 and g = 0. vLLM does not graph GDN prefill at all, so this is where our
  c1 TTFT lead comes from (31 ms vs ~43 ms, 1-round evidence).
- **Take: own every graph input.** Chunk indices could be precomputed per (k, L) bucket into
  buffers the graph owns, instead of relying on FLA's identity cache. The current fix (keeping
  cached tensors alive) works, but it still depends on FLA internals.
- **Take: build host-side metadata from CPU copies.** Our eager path builds cu_seqlens on the
  host already. Check that no `.tolist()` on device tensors remains on the eager prefill path.
