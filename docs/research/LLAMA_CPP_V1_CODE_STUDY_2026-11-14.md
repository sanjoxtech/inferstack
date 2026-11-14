# llama.cpp v1.0 / Qwen3.5 code study — 14 November 2026

Status: **L0 code study complete; implementation not started.**

Purpose: turn the llama.cpp v1.0.0 release from general external evidence into a concrete, provenance-bearing design input for InferStack's measured Qwen3.5 prefill bottleneck.

## Reproducibility pins

Two upstream identities are intentionally recorded:

- llama.cpp v1.0.0 release target: 2d8a32502909ca065c41c4cf5f2fb1c427c047fc
- code-study snapshot on 14 November 2026: 61dbf49228c8a96b72e40c89d93cb7c003411f8c

The later snapshot is used where post-v1 source makes the Qwen3.5 implementation easier to inspect. Claims tied to v1.0.0 are checked against the release notes or merged PRs.

Primary upstream sources:

- https://github.com/ggml-org/llama.cpp/releases/tag/v1.0.0
- https://github.com/ggml-org/llama.cpp/pull/32078
- https://github.com/ggml-org/llama.cpp/pull/32009
- https://github.com/ggml-org/llama.cpp/pull/32089
- https://github.com/ggml-org/llama.cpp/pull/32102
- https://github.com/ggml-org/llama.cpp/pull/28392
- https://github.com/ggml-org/llama.cpp/pull/30966
- https://github.com/ggml-org/llama.cpp/pull/31888
- https://github.com/ggml-org/llama.cpp/blob/61dbf49228c8a96b72e40c89d93cb7c003411f8c/src/models/qwen35.cpp
- https://github.com/ggml-org/llama.cpp/blob/61dbf49228c8a96b72e40c89d93cb7c003411f8c/src/models/delta-net-base.cpp
- https://github.com/ggml-org/llama.cpp/blob/61dbf49228c8a96b72e40c89d93cb7c003411f8c/ggml/src/ggml-cuda/gated_delta_net.cu
- https://github.com/ggml-org/llama.cpp/blob/61dbf49228c8a96b72e40c89d93cb7c003411f8c/ggml/src/ggml-cuda/gated-delta-net-mma.cu
- https://github.com/ggml-org/llama.cpp/blob/61dbf49228c8a96b72e40c89d93cb7c003411f8c/docs/multi-gpu.md

This document describes ideas and interfaces, not copied implementation code. Any native InferStack implementation must be independently written and verified against our own numerical and serving acceptance gates.

## 1. Main conclusion

The most useful llama.cpp lesson for InferStack N3 is **not merely "write a fused GDN kernel."**

The stronger pattern is:

~~~
Qwen3.5 recurrent layer
        |
        v
one logical GDN backend primitive
        |
        +--> specialized high-token / MMA path when shape + GPU are qualified
        |
        +--> recurrent fallback path otherwise
        |
        v
write recurrent state directly to its destination when possible
~~~

The graph sees one GDN operation rather than hundreds or thousands of chunk-level algebra nodes. This keeps graph topology largely independent of ubatch token count and lets the backend select the correct implementation.

That maps directly to the problem measured in InferStack: our decode GDN is custom/Triton, but chunked prefill still falls through to the upstream/reference path.

## 2. What llama.cpp's Qwen3.5 graph actually does

The Qwen3.5 graph inherits llm_build_delta_net_base.

For a recurrent/linear-attention layer the high-level flow is:

~~~
input
  |
  +--> Q/K/V(+z) projection
  |
  +--> beta / alpha projection
  |
  +--> causal convolution + SiLU
  |
  +--> Q/K L2 normalization
  |
  +--> recurrent GDN primitive
  |
  +--> gated normalization
  |
  +--> output projection
~~~

The key boundary is build_recurrent_attn(...), which eventually emits one ggml_gated_delta_net graph op.

The GDN op consumes Q, K, V, gate, beta and the recurrent state. Its output contains attention output plus recurrent-state output/snapshots. Decode and prefill therefore share the same logical primitive rather than maintaining two unrelated graph-level algorithms.

### InferStack implication

N3 should introduce a clear native GDN primitive boundary instead of allowing the scheduler/model layer to know the internal chunk algebra.

Proposed interface conceptually:

~~~
gdn(
    q, k, v,
    gate, beta,
    state_in,
    snapshot_policy,
    execution_shape
) -> output, state_out
~~~

The implementation may use Triton/CUDA internally, but the runtime should reason about GDN as one operation.

## 3. Why PR #32078 matters more than its title suggests

PR #32078 removes the old graph-level autoregressive and chunked GDN implementations because the fused backend path is now always used.

Before that cleanup, chunked GDN expanded recurrence math into a token/chunk-dependent graph. The old implementation padded and reshaped chunks, built decay masks, matrix products and triangular operations at the graph level.

After the fused-op transition:

- graph node budget no longer needs token-count-proportional GDN terms;
- graph topology is no longer proportional to ubatch token count for GDN;
- the backend owns the recurrent algorithm;
- Q/K head-repeat branches needed only by the unfused implementation disappear.

### InferStack implication

Do **not** implement N3 by translating the current Transformers/reference prefill expression one operator at a time into Triton launches.

That would preserve the wrong abstraction.

Target:

~~~
bad N3
scheduler -> bmm -> solve -> GEMM -> copies -> ... -> state

preferred N3
scheduler -> GDN primitive -> state
~~~

This is important for both raw kernel cost and future graph capture/replay.

## 4. CUDA implementation: two execution regimes

The CUDA GDN implementation first attempts a specialized MMA path and falls back to a recurrent CUDA kernel if the specialized path is not eligible.

### 4.1 Recurrent fallback

The fallback kernel specializes common state widths (16/32/64/128), partitions state columns across warps/CTAs, and loops over the token dimension inside the kernel.

The important property for us is that a long token sequence still appears as one backend operation to the graph.

### 4.2 High-token MMA path

The MMA implementation is deliberately gated. The current source checks model shape, token count, sequence count, head geometry and GPU architecture before selecting it.

Notable source-level constraints in the studied snapshot include:

- minimum token count of 64;
- specific Q/GDN geometry, including the Qwen-style 128 state-width path;
- head-count constraints;
- positive/valid sequence and grouping geometry;
- only GPU families benchmarked for the schedule are enabled.

The source explicitly documents the principle: **MMA capability alone does not imply a speedup.**

It chooses different slice/whole-head schedules based on occupancy, including whether head x sequence work would leave too many SMs idle.

### Hardware caveat for our lab

The studied CUDA whitelist includes Ada-class and selected Ampere/Blackwell cases but does not list every Ampere product. Therefore we must not infer that an A100 will use the same specialized path merely because it supports tensor cores.

For InferStack:

- L4 and A100 must be independently qualified;
- eligibility must be telemetry, not assumption;
- a fallback recurrent kernel must remain correct and fast enough;
- kernel selection should be evidence-driven per hardware/shape family.

## 5. Recurrent-state copy fusion

llama.cpp has a backend graph pattern that recognizes the GDN output followed by the strided copy that stores recurrent snapshots.

When the pattern is safe, the GDN kernel writes state snapshots directly into the recurrent cache destination and the separate copy node can be skipped.

Conceptually:

~~~
without fusion

GDN kernel
   |
temporary state tail
   |
copy/scatter
   |
recurrent cache


with fusion

GDN kernel ----------------> recurrent cache
   |
attention output
~~~

### InferStack implication

Our N3 acceptance profiler should separate:

- GDN arithmetic time;
- state-output materialization;
- state-cache copy/scatter time.

A direct-state-write candidate should be tested after numerical equivalence is proven. This is particularly relevant because our existing prefill trace contains a large number of copy operations.

## 6. Scheduler and graph arena reuse

PR #32009 formalizes the parameters that define backend-scheduler compatibility:

- maximum graph nodes;
- parallel scheduling mode;
- operation offload mode.

A current scheduler can be reused when its behavior is compatible and its reserved capacity is at least as large as the new request.

PR #32089 similarly reuses graph-result arenas instead of allocating new instances on every reserve.

### Transferable design

InferStack should separate request identity from execution-arena identity.

A future prefill/static-graph arena can be keyed by the minimum properties that determine compatibility rather than request IDs.

Candidate compatibility dimensions:

- model/serving-pack digest;
- device + kernel build identity;
- dtype;
- execution mode;
- batch/token bucket capacity;
- recurrent-state geometry;
- attention/GDN backend;
- graph-capture-safe feature set.

If a new workload fits an existing arena, reset metadata and reuse buffers rather than recreating them.

## 7. Qwen3.5 conversion-time tensor merging

PR #32102 extends conversion-time shared-input merging to Qwen3.5.

Two important merged weights are:

~~~
ATTN_QKV_GATE
    = [ATTN_QKV | ATTN_GATE]

SSM_BETA_ALPHA_CAT
    = [SSM_BETA | SSM_ALPHA]
~~~

At runtime this changes two pairs of projections into one matmul plus cheap views.

This validates the direction of InferStack's v1.6 forward-fusion experiments, but llama.cpp moves part of the optimization into the **serving representation** rather than rebuilding it dynamically at runtime.

### InferStack serving-pack implication

A future immutable serving pack may contain:

- merged shared-input projections;
- hardware/backend-friendly tensor order;
- quantized serving tensors when explicitly qualified;
- source-model digest and transformation provenance;
- transformation version;
- numerical equivalence evidence.

Model Foundry still owns source/training weights. InferStack owns a derived serving representation.

## 8. FP8 KV: compatibility is multidimensional

PR #28392 adds OCP F8_E4M3 KV cache with optional static per-tensor scales.

The implementation is useful because it does not reduce compatibility to one boolean.

For K scaling the graph can algebraically move the scale onto Q:

~~~
stored K' = K / s
Q'        = Q * s

Q' dot K' = Q dot K
~~~

For V it stores V/s and restores the scale after the weighted sum.

The implementation also:

- stores optional per-layer F32 K/V scale tensors;
- validates scales are finite and non-zero;
- rejects architecture paths that bypass shared scaling helpers;
- constrains combinations involving tensor split and quantized KV;
- requires compatible attention/cache-write backends;
- tests conversion, context shifting and numerical behavior.

### InferStack implication

Our FP8-KV technique record should eventually contain a compatibility matrix over:

~~~
GPU
x architecture
x attention backend
x engine build
x K/V cache type
x split mode
x context operation
~~~

not only supports_fp8=true.

## 9. MoE cache: a strong H5/placement reference

llama.cpp's MoE cache is not an immediate Qwen3.5-dense optimization, but it is a strong example of the optimization loop we want InferStack to become.

The system can keep experts in host memory, cache hot experts on device, and on a miss decide how much work is computed by CPU versus uploaded to GPU.

The tuning script measures a few configurations, estimates platform transfer/CPU characteristics, replays routing traces through the cache model and recommends a cache size / CPU-share combination.

PR #31888 then shows another systems lesson: CPU work only helps if scheduling allows CPU and GPU work to overlap. Removing an unnecessary dependency improved token generation in the reported Gemma 4 tests.

### InferStack implication

Future placement optimization should reason from:

- host-memory bandwidth;
- PCIe/NVLink/interconnect bandwidth;
- GPU compute/bandwidth;
- expert hit rate;
- pinned vs mmap'd host memory;
- model routing trace;
- overlap opportunities.

This belongs in hardware-aware optimization memory, not in the immediate N3 implementation.

## 10. Multi-GPU lessons for CompatibilityResolver

Current llama.cpp exposes multiple split strategies: single device, layer, row and experimental tensor split.

The documentation/source also makes limitations explicit:

- tensor split has additional attention requirements;
- some KV quantization combinations are unsupported;
- automatic fit is not universal;
- performance depends heavily on interconnect/NCCL;
- some features are disabled or unsupported under tensor split;
- MoE cache and tensor parallelism are not universally composable.

### InferStack implication

2 x 16 GB remains a topology, not a 32 GB device.

A portable manifest must validate:

~~~
model
+ engine
+ architecture
+ split strategy
+ feature combination
+ topology/interconnect
~~~

before allocating GPU time.

## 11. N3 architecture recommendation

The code study changes N3 from "write a fused prefill kernel" into a staged engine-design task.

### N3.0 — keep the current gate

First run the full serving scorecard with native_static.

Do not mix decode qualification and a new prefill implementation in one treatment.

### N3.1 — token-budget prefill scheduling

Remove the avoidable repeated whole-model forwards caused by 128-token/one-chunk-per-turn policy.

This is orthogonal to GDN kernel speed and should be measured independently.

### N3.2 — define one native GDN primitive

Create a stable internal contract for Q/K/V/gate/beta/state and output/state snapshots.

Acceptance:

- reference equivalence on small deterministic cases;
- decode and prefill use the same semantic contract;
- state layout and snapshot behavior are explicit;
- no request-level scheduler code depends on internal GDN algebra.

### N3.3 — dual-path GPU implementation

Build/qualify:

1. **recurrent path** for decode/small token counts/fallback hardware;
2. **high-token prefill path** that uses tensor-core/MMA-style tiled computation where the target GPU/shape has measured benefit.

Do not hard-code one global crossover. Measure it by hardware + shape.

### N3.4 — direct recurrent-state write

Test a kernel path that can write state snapshots directly to the InferStack state cache when layout/alignment permit.

Acceptance includes proving that a removed copy is real in profiler evidence.

### N3.5 — serving-pack projection merge

Separately test conversion/build-time merging of:

- QKV + z/gate projection;
- beta + alpha projection;

with immutable transformation identity and numerical equivalence.

### N3.6 — full serving acceptance

For every retained treatment run:

- numerical/quality gates;
- short and long prompt workloads;
- c=1/4/8/16;
- TTFT/TPOT/output throughput;
- VRAM;
- kernel/trace evidence;
- no-regression check on prefix cache/refill/compaction/cancellation;
- full Native vs vLLM vs SGLang scorecard.

Only whole-server evidence can promote N3.

## 12. Experiments to run, in order

| Experiment | Question | Main metric |
|---|---|---|
| E-N3-0 | Does native_static improve the complete server? | full scorecard |
| E-N3-1 | How much is repeated prefill-forward overhead? | TTFT / forward count |
| E-N3-2 | Does one GDN primitive match reference numerically? | max/relative error |
| E-N3-3 | recurrent vs high-token path crossover by GPU | GDN kernel ms |
| E-N3-4 | does direct state write remove copy cost? | trace + kernel time |
| E-N3-5 | do serving-pack projection merges reduce forward time? | model GPU ms |
| E-N3-6 | do combined retained treatments improve all workloads? | Pareto/full scorecard |

Run L4 and A100 qualification separately. Never transfer a kernel-selection result between them without evidence.

## 13. What we should not copy

Do not port ggml graph code or CUDA code line-for-line.

What is transferable is the systems design:

- one backend GDN primitive;
- graph topology independent of token count;
- specialized + fallback execution paths;
- hardware/shape gating based on measured support;
- state-write fusion;
- reusable scheduler/graph arenas;
- serving-time tensor packing.

InferStack should implement these ideas under its own runtime contracts, tests and profiling evidence.

## 14. Decision from this study

**LCPP-T1 (fused GDN): high-priority, directly applicable.**

It should influence the N3 design immediately after the native_static whole-server scorecard.

**LCPP-T2 (scheduler/graph reuse): applicable, medium/high priority.**

Use it in static decode/prefill arena lifecycle, but do not combine it with the first N3 kernel treatment.

**LCPP-T7 (serving representation/tensor packing): high-value follow-up.**

The Qwen3.5 conversion-time merges provide a concrete serving-pack experiment.

**LCPP-T3 (FP8 KV), T4 (MoE cache), T5 (NextN), T6 (multi-GPU): retain as qualified future techniques.**

They are valuable, but none should delay the measured Qwen3.5 prefill path.

## 15. Next owner/agent handoff

Owner:

1. run the latest full native_static L4/A100 scorecard and preserve the evidence bundle.

Implementation/research agent after that result:

1. use this document as the N3 source note;
2. write the N3 internal GDN contract and numerical harness;
3. implement token-budget prefill as a separate treatment;
4. build a recurrent GDN primitive candidate;
5. prototype a high-token GPU path only after hardware-specific profiling;
6. keep every change behind a profile/feature treatment until the full scorecard proves it.

The llama.cpp adapter remains a separate engine-integration track and should not block N3.
