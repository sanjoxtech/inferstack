# llama.cpp v1.0 integration and research plan

Status: approved roadmap addition, 14 November 2026.

This document adds `ggml-org/llama.cpp` to the InferStack research and engine roadmap without changing the current ownership boundary or interrupting the active native-static qualification.

## 1. Why this belongs in InferStack

llama.cpp is not a new project; the important current event is the **llama.cpp / ggml v1.0.0 release on 13 November 2026**. v1.0.0 establishes a stable public C API/ABI direction and includes inference techniques directly relevant to InferStack's measured bottlenecks.

Authoritative sources for this research item:

- Repository: https://github.com/ggml-org/llama.cpp
- v1.0.0 release: https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0
- Hugging Face announcement: https://huggingface.co/blog/ggml-org/llama-cpp-v1
- Fused GDN cleanup: https://github.com/ggml-org/llama.cpp/pull/32078
- Backend scheduler reuse: https://github.com/ggml-org/llama.cpp/pull/32009
- MoE CPU/GPU expert cache: https://github.com/ggml-org/llama.cpp/pull/30966
- FP8 KV cache: https://github.com/ggml-org/llama.cpp/pull/28392

llama.cpp is relevant in two distinct roles:

1. **Reference implementation / technique source** for native InferStack research.
2. **Future engine backend** for portable, local, edge, CPU+GPU hybrid and highly quantized deployments.

Do not conflate those roles. A technique may be studied without adding llama.cpp as a serving engine, and adding llama.cpp as an engine does not mean copying its internals into native InferStack.

## 2. Position in the engine portfolio

InferStack's intended engine matrix becomes:

| Engine | Primary role |
|---|---|
| InferStack Native | research backend; custom kernels/runtime; controlled experimentation |
| vLLM | high-throughput GPU serving reference/backend |
| SGLang | high-performance GPU/agent-serving reference/backend |
| llama.cpp | portable/local/edge/private inference; CPU+GPU hybrid; broad hardware and quantized GGUF serving |

llama.cpp should be added only after a capability profile and conformance contract exist. Until then it is `planned`, not `supported`.

## 3. Immediate relevance to the current native bottleneck

The current L4 evidence shows the largest native gap is prefill. Multi-prompt Qwen3.5 prefill still uses the upstream/reference Gated DeltaNet path and produces thousands of small operations, including bmm, triangular solve, FP32 GEMM and copy operations.

llama.cpp v1.0.0 removed its old non-fused GDN paths because fused GDN is now the standard execution path. This makes llama.cpp a high-value implementation reference before InferStack N3.

### Mandatory ordering

Do not start a new native GDN prefill kernel before these steps:

1. Finish and record the full serving scorecard for `production-v1.7.0` / `native_static`.
2. Preserve the result as the new native decode baseline if the whole scorecard improves.
3. Perform a bounded llama.cpp Qwen3.5/GDN code study.
4. Compare llama.cpp's algorithm/state/layout choices with:
   - Transformers reference GDN,
   - InferStack decode Triton GDN,
   - InferStack current prefill path,
   - relevant FLA/FlashInfer/SGLang/vLLM implementations where provenance permits.
5. Write a design note for InferStack N3.
6. Only then implement the fused/chunked prefill candidate.

The purpose is not to port blindly. It is to identify reusable algorithmic and systems choices, then implement an independently testable InferStack candidate.

## 4. Technique cards to ingest

Add the following as external evidence/technique candidates in the inference knowledge system. Each remains a hypothesis until tested on an applicable InferStack environment.

### LCPP-T1 — fused GDN

Source: llama.cpp PR #32078 and Qwen3.5 model implementation.

Questions:

- What recurrence/chunk formulation is used?
- Which state tensors are kept persistent?
- Which tensors are fused or prepacked?
- How does graph topology remain independent of ubatch size?
- Which precision is used for state/update math?
- How are Qwen3.5 head layouts handled?
- How does the fused path differ between prefill and autoregressive decode?

Target use: InferStack N3 prefill design.

### LCPP-T2 — backend scheduler and graph reuse

Source: PR #32009 and v1.0.0 scheduler changes.

Questions:

- Which scheduler parameters define graph compatibility?
- Which buffers and graph results are reused?
- When is reuse invalidated?
- Can the same principle reduce InferStack recapture/reallocation for static decode and future prefill graphs?

Target use: native graph/runtime lifecycle and H5 experimentation efficiency.

### LCPP-T3 — FP8 KV cache

Source: PR #28392 / v1.0.0.

Questions:

- hardware/backend requirements;
- static scale representation;
- attention-kernel compatibility;
- memory reduction;
- quality impact;
- conversion overhead;
- interaction with long-context and concurrency.

Target use: a future memory-capacity technique card. It is not automatically enabled on T4/A100 or any model; CompatibilityResolver must decide support.

### LCPP-T4 — MoE expert cache

Source: PR #30966.

Concept:

- hot experts cached on GPU;
- cold experts resident in host memory;
- cache miss can upload to GPU or execute a configurable share on CPU;
- tuning searches cache size and CPU execution share.

Target use: future large-MoE placement/cost optimization. This is especially relevant to InferStack's longer-term hardware-aware execution planner.

### LCPP-T5 — NextN / speculative API

v1.0.0 promotes NextN/speculative APIs into the public `llama.h` surface.

Target use: speculative/MTP backend study and a future engine capability. Do not claim equivalence to InferStack's speculative methods without a matched acceptance contract.

### LCPP-T6 — multi-GPU placement

llama.cpp documents:

- single-GPU;
- layer/pipeline split;
- experimental tensor split;
- NCCL-assisted reductions;
- per-device tensor split;
- CPU/GPU offload.

Target use: expand the knowledge used by `CompatibilityResolver` and future placement planning. Preserve the existing rule: multiple GPUs are not one pooled VRAM device unless an explicit supported sharding strategy exists.

### LCPP-T7 — GGUF serving representation and tensor packing

Study:

- GGUF immutable serving artifacts;
- conversion-time tensor merges for Qwen3.5;
- quantized weight layouts;
- mmap/DirectIO/lazy loading;
- CPU/GPU hybrid placement.

Target use: inform a future InferStack serving-pack representation. Model Foundry remains the owner of source model artifacts; InferStack may create immutable serving representations without changing training weights.

## 5. llama.cpp engine adapter

Future implementation name: `LlamaCppAdapter`.

The first adapter should use `llama-server` through the same engine contract used by existing third-party engines. Do not embed libllama into the control plane in the first iteration.

### Required capabilities

The adapter should be able to:

- pin an exact llama.cpp release/commit or immutable container image;
- pin GGUF model artifact and digest;
- start/stop `llama-server`;
- use OpenAI-compatible endpoints where applicable;
- report readiness and engine identity;
- expose normalized metrics required by the scorecard;
- record context size, batch/ubatch, parallel slots, flash-attention state, KV types, device/split mode and quantization;
- fail closed on unsupported model/backend combinations;
- cleanly release GPU/CPU resources.

### Configuration identity

llama.cpp configuration fields that materially affect evidence must be part of the experiment identity, including at least:

- model GGUF digest and quantization;
- context size;
- logical batch / ubatch;
- parallel slots / continuous batching;
- KV cache K/V types;
- GPU layer count;
- device list;
- split mode and tensor split;
- flash-attention setting;
- speculative/NextN mode if enabled;
- backend build identity (CUDA/Metal/HIP/etc.).

## 6. Fair-comparison rule

A llama.cpp scorecard must not be presented as a fair engine comparison merely because it serves the same named model.

GGUF conversion and quantization can change the serving representation.

For a fair comparison:

1. pin the same source model revision and tokenizer;
2. record the exact GGUF conversion commit/tool/version;
3. prefer F16/BF16 GGUF for the first engine comparison where feasible;
4. if a quantized GGUF is used, quantization is an explicit treatment, not an engine-only comparison;
5. run the same workload and client-side measurement policy;
6. run quality/numerical gates appropriate to the representation;
7. compare only on the same hardware or clearly separate hardware-specific results.

A local/edge scorecard may intentionally use quantized GGUF, but it must answer a different question: best deployment under that device/memory/cost constraint.

## 7. Roadmap packages

### L0 — evidence ingestion and design study

Build/record:

- v1.0.0 evidence source;
- technique cards LCPP-T1 through LCPP-T7;
- Qwen3.5 fused-GDN design study;
- serving-representation note.

Acceptance:

- every claim has a source;
- no technique is marked proven by external evidence alone;
- N3 design explicitly compares algorithms/state/layouts rather than copying code.

### L1 — engine capability contract

Build:

- llama.cpp engine capability profile;
- profile schema for the first supported backend;
- conformance command and normalized identity.

Acceptance:

- supported/unsupported/unknown states are explicit;
- exact llama.cpp and GGUF identity is captured;
- unsupported options fail before benchmark work.

### L2 — LlamaCppAdapter

Build:

- launch/ready/stop;
- OpenAI-compatible request adapter;
- metric normalization;
- cleanup;
- Docker/local execution first.

Acceptance:

- conformance succeeds on one pinned small GGUF model;
- engine shutdown releases resources;
- exact build/model/config appears in evidence.

### L3 — portable/local scorecard

Run on a selected device/hardware target where llama.cpp is meaningful.

Potential targets:

- Apple Silicon local;
- CPU-only x86;
- NVIDIA GPU with CUDA;
- memory-constrained hybrid CPU+GPU.

Acceptance:

- at least three measured rounds;
- quality gates;
- no cross-hardware engine claims;
- quantization treatment clearly separated.

### L4 — matched GPU engine comparison

Only if the same source model can be represented and qualified fairly enough for the question being asked.

Compare:

- Native;
- vLLM;
- SGLang;
- llama.cpp.

This is secondary to the portable/local scorecard because llama.cpp's primary value proposition differs from vLLM/SGLang.

## 8. Interaction with E1 portable execution

llama.cpp fits naturally into E1:

```text
ExecutionManifest
      |
CompatibilityResolver
      |
   engine=llama.cpp
      |
local / Docker / notebook / paid GPU / K8s
      |
llama-server
      |
ResultBundle
      |
Trial / Evidence / Memory
```

The provider layer remains independent. RunPod/Modal/Kubernetes are places where an engine can run; llama.cpp is an engine.

## 9. Interaction with the evolution loop

External llama.cpp changes should enter InferStack as evidence, not automatic code changes.

```text
llama.cpp release / PR
       |
Evidence intake
       |
Technique applicability
       |
Current profiler + model + GPU + SLO
       |
Planner
       |
bounded experiment
       |
Verifier
       |
keep / reject / inconclusive
       |
memory
```

This is a concrete example of the continuous evolution architecture InferStack is intended to support.

## 10. Immediate next actions

Priority order from the current repository state:

1. **Run and record the full L4/A100 scorecard with the new `native_static` lane.**
2. If the whole scorecard improves, make that result the decode baseline.
3. Execute L0: deep-read llama.cpp Qwen3.5 fused GDN and scheduler/graph implementation.
4. Write the InferStack N3 GDN-prefill design note.
5. Implement/test token-budget prefill scheduling and the N3 fused GDN candidate according to the measured ordering.
6. In parallel after the native scorecard, start the small L1 capability-profile task for a future `LlamaCppAdapter`.
7. Do not delay E1.3 paid-provider work solely for llama.cpp; engine and provider tracks remain orthogonal.

## 11. Non-goals

Do not:

- replace vLLM or SGLang with llama.cpp;
- rewrite InferStack native around ggml;
- automatically convert every Model Foundry artifact to GGUF;
- claim an external llama.cpp speedup as an InferStack speedup;
- copy code without preserving license/provenance obligations;
- compare quantized GGUF with BF16 engines and call it a pure engine benchmark;
- let a new engine bypass ExecutionManifest, CompatibilityResolver, quality gates or ResultBundle;
- interrupt the current native-static full-scorecard qualification.

## 12. Definition of success

This roadmap addition succeeds when InferStack can use llama.cpp in two disciplined ways:

1. **Learn from it:** externally discovered techniques become provenance-bearing hypotheses that can improve native InferStack through bounded experiments.
2. **Run it:** llama.cpp becomes a qualified engine adapter for workloads where its portability, quantization and CPU/GPU flexibility are advantageous.

The first concrete value should come from the fused-GDN study feeding the measured Qwen3.5 prefill bottleneck, before any broad adapter expansion.
