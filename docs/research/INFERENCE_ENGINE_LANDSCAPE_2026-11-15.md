# Inference engine landscape, November 2026: what helps InferStack on one L4

Single-pass web and GitHub survey: about 13 searches; licenses checked through the GitHub API.
Source reading of vLLM 0.33.0 and SGLang 0.5.23 is in `VLLM_SGLANG_MIXED_BATCH_STUDY_2026-11-15.md`.
Performance numbers below are vendor or PR claims that we have not reproduced. Ideas only, no copied code.

## Headline

None of the 2026 GDN prefill kernels run on the L4 (sm89):

| Kernel | Supported hardware |
|---|---|
| FlashQLA (Qwen, TileLang) | SM90+ |
| cuLA | SM90/SM10x |
| FlashInfer `gdn_prefill` | SM100-class (per its docs) |

On sm89 the GDN kernel is FLA's Triton `chunk_gated_delta_rule`, which we have already used since v1.9.
Any lead on prefill and TTFT has to come from packing, scheduling and launch overhead.

## Engines and libraries

| Project | Relevant to us | License |
|---|---|---|
| vLLM | Packed token batch, chunked prefill, piecewise CUDA graphs, GDN prefix "align" mode | Apache-2.0 |
| SGLang | chunked_prefill_size 4096, radix cache, chunk ends aligned to 64-token GDN blocks (PR #44076) | Apache-2.0 |
| mini-sglang | ~5k-line reference: radix cache, overlap scheduler, chunked prefill, graphs | MIT |
| nano-vllm (+ qwen3.5 fork) | Minimal vLLM; Qwen3.5 support in the fork not verified | MIT |
| TokenSpeed (LightSeek, preview) | C++ scheduler, per-step GDN state snapshots for spec rollback | MIT (one listing conflicts) |
| LMDeploy TurboMind | CUDA GDN for Qwen3.5 (#4389); GDN state saved at each KV-block boundary (#4465) | Apache-2.0 |
| llama.cpp | Fused GDN path (see LLAMA_CPP_V1_CODE_STUDY) | MIT |
| ExLlamaV3 | Qwen3.5 GDN on consumer GPUs | MIT |
| TensorRT-LLM | Qwen3.5 TTFT p90 reported worse than SGLang; ~28 min engine build | Apache-2.0 |
| Mirage MPK, ThunderKittens, Ada-MK | Decode megakernels; Ada-MK keeps plain kernels for prefill | Apache-2.0 / MIT / paper |
| POD-Attention | Fused prefill+decode attention (full-attention layers only) | paper |
| LightLLM, xLLM, KTransformers, MLC-LLM, mistral.rs | Nothing Qwen3.5-GDN-specific for sm89 | Apache/MIT |

License flags:

- Aphrodite is AGPL-3.0.
- Modular MAX uses its own community license.
- EAGLE and TileLang report NOASSERTION. Read their license files before using anything from them.
- HF TGI was archived in March 2026.

## Techniques, checked against what InferStack already has

| # | Technique | InferStack status |
|---|---|---|
| 1 | FLA chunked GDN prefill kernel | **Done** (v1.9 `gdn_prefill_backend = "fla"`, v1.11 FLA conv) |
| 2 | Token-budget prefill, a long prompt in one forward | **Done** (v1.8/v1.12 budgets 16384 idle / 1024 busy). 64-token-aligned chunk ends: open, small |
| 3 | Packed varlen prefill across requests (cu_seqlens into FLA GDN + varlen attention), writing the static pool directly | **Open.** The rivals run it; we pad cohorts and copy an HF cache into the pool. Top candidate |
| 4 | Prefill CUDA graphs bucketed by token count, attention/GDN outside the graph (vLLM piecewise) | **Open.** Targets c1 TTFT (65 vs 41 ms) |
| 5 | Overlap scheduling: CPU plans batch k+1 while the GPU runs batch k | **Open** (N7) |
| 6 | Hybrid prefix cache with GDN state at block boundaries | **Have** a prefix state cache. SGLang's radix tree is still ahead on shared_prefix c16 |
| 7 | FP8 W8A8 GEMMs on Ada | Changes numerics; only fair if the rivals run FP8 too. Parked |
| 8 | More GDN-layer fusion (conv, gating, norm, output gate) | Partly done; open for prefill |
| 9 | POD-style fused prefill+decode attention | Small: full-attention layers only |
| 10 | Decode megakernel; multi-draft MTP | Decode only. MTP one-draft done (N6.2); 2+ drafts = N6.5 |

Recommended order after the TTFT split run: 3 (packed prefill into the pool), then 4 (prefill
graphs), then N5.2 mixed steps built on 3, then 5.

Sources: see the links in the Research run summary of 2026-11-15:

- vLLM RFC on hybrid prefix checkpoints (L40S numbers)
- FlashInfer gdn_prefill docs
- SGLang PR #44076
- fla-org/flash-linear-attention, QwenLM/FlashQLA, inclusionAI/cuLA
- mirage-project/mirage, HazyResearch/ThunderKittens, arXiv 2605.11581 (Ada-MK)
- sgl-project/mini-sglang, GeeeekExplorer/nano-vllm, lightseekorg/tokenspeed
- InternLM/lmdeploy, turboderp-org/exllamav3, NVIDIA/TensorRT-LLM
- arXiv 2410.18038 (POD-Attention)
