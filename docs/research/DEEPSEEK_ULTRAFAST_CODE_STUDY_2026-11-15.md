# DeepSeek Ultrafast integration and code-study plan — 15 November 2026

Upstream:
- repository: https://github.com/basetenlabs/deepseek-ultrafast
- inspected commit: 76766a97308bfd1658fb055e835d0f92de737dc3
- initial public engine commit: 257c4bbb494907e2ba6d5bd2544fff0bfcb0abb8
- license: Elastic License 2.0

This document records DeepSeek Ultrafast as external inference evidence for InferStack and defines the narrow conditions under which it may become an external engine target.

## 1. What the project actually is

DeepSeek Ultrafast is not a generic vLLM/SGLang replacement. The current repository is a highly specialized Rust + CUDA inference server for DeepSeek-V4.1-Flash on Blackwell-class `sm_100a` / `sm_103a` systems.

The upstream README reports:

| target | decode | step | accepted tokens/step | 1M-token prefill |
|---|---:|---:|---:|---:|
| 4 x GB300 / TP4 | about 1,740 tok/s | 3.30 ms | about 5.6 | 13.2 s / 76k tok/s |
| 8 x B300 / TP8 | about 2,060 tok/s | 2.71 ms | about 5.7 | 11.7 s / 85k tok/s |

These are upstream results, not InferStack evidence.

The architecture is intentionally narrow:
- Rust owns server/control execution;
- hand-written CUDA owns the hot model path;
- one TP group executes a request;
- multiple replicas provide service-level concurrency;
- speculative draft/verify work is captured into CUDA graphs;
- session KV can move between replicas through host-memory RDMA;
- Redis is an ownership directory, not the KV payload store.

## 2. Immediate InferStack decision

Add DeepSeek Ultrafast in **three distinct roles**.

### Role A — external technique/reference source: YES, now

Treat the implementation as provenance-bearing external evidence for:
- framework-free specialized native execution;
- captured speculative/block-verification graphs;
- custom TP communication and NVLS-aware collectives;
- model-specific tensor/weight packing;
- compressed KV snapshots for transport;
- cross-replica KV movement over RDMA;
- cache-ownership directories that decouple routing from cache locality;
- long-context bounded-replay prefill;
- agent-driven profiling/hill-climbing workflows.

No external performance claim is treated as an InferStack result.

### Role B — external engine endpoint: YES, later

InferStack already has `generic_openai`, so an operator-controlled DeepSeek Ultrafast endpoint can be exercised without embedding its source.

A future specialized `DeepSeekUltrafastAdapter` is justified only when we have supported hardware or a managed endpoint and need extra identity/telemetry:
- exact upstream commit/image;
- DeepSeek model revision;
- TP width;
- `sm_100a` / `sm_103a` hardware;
- context/chunk settings;
- DSpark block/verification configuration;
- average iteration time;
- average accepted draft length;
- KV transfer hit/restore bytes;
- RDMA/Redis directory identity.

### Role C — copy/port its implementation into InferStack Native: NO

Do not copy source or make InferStack Native a derivative of this engine.

Reasons:
1. current InferStack Native serves Qwen3.5 on L4/A100/T4-class hardware, while this engine is DeepSeek-V4.1-Flash + B200/B300/GB300 specific;
2. several optimizations depend on DeepSeek architecture details and Blackwell features;
3. the repository is Elastic License 2.0, not Apache/MIT. Its license prohibits providing the software to third parties as a hosted/managed service when users get access to a substantial set of its functionality;
4. our product direction includes inference serving, so implementation reuse needs a separate license/legal decision.

InferStack may study public architecture and independently implement techniques. Keep provenance, do not mechanically translate upstream kernels.

## 3. Code-level findings

### DSUF-T1 — captured speculative decode graph

The server initializes a speculative state, captures the speculative path once, then replays the captured graph inside decode.

Relevant upstream paths:
- `crates/server/src/engine.rs`
- `crates/engine/src/model.rs`
- `crates/kernels/cu/draft_markov.cu`
- `crates/kernels/cu/draft_head8.cu`

The observed performance relationship is important:

```text
effective decode throughput
    ~= accepted tokens per step / step time
```

Using the upstream TP4 medians:
- ~5.6 accepted tokens / 3.30 ms ~= ~1.7k tok/s.

This shows that the headline TPS is not only a faster one-token decode step. It combines:
- low step latency;
- multi-token speculative acceptance.

InferStack implication:
- keep `ms/step`, `accepted_tokens/step`, verification cost and final tok/s as separate evidence dimensions;
- never compare speculative TPS with one-token decode without recording acceptance semantics.

### DSUF-T2 — TP scaling has diminishing returns

Doubling from TP4 to TP8 changes upstream decode from ~1,740 to ~2,060 tok/s: only ~18% more output throughput.

Step time changes from 3.30 to 2.71 ms, also roughly an 18% improvement.

1M prefill improves from 13.2 s to 11.7 s, around 11%.

InferStack implication:
- hardware selection must optimize cost/goodput/latency rather than maximize GPU count;
- E1 CompatibilityResolver should keep topology/parallel strategy in experiment identity;
- H5 should be allowed to reject a wider TP configuration when communication cost dominates.

### DSUF-T3 — Rust control plane + hand-written CUDA hot path

The model hot path does not depend on PyTorch/SGLang/Triton DSL execution. Rust owns rank threads, request lifecycle and server logic; CUDA implements attention, GEMV/GEMM, MoE, sampling, communication, draft and KV transforms.

InferStack implication:
- this is evidence for a future **Native-v2 specialized backend** boundary, not a reason to rewrite the current Python/Triton Native engine now;
- a future backend may expose the same InferStack engine contract while moving hot execution into a compiled Rust/C++/CUDA runtime.

Gate:
- only start this track if profiler evidence shows framework/launch/control overhead remains material after N4 packed/varlen work.

### DSUF-T4 — custom communication / NVLS

The engine creates one worker/rank per GPU, enables peer access, shares rank communication buffers and conditionally sets up NVLS on supported Blackwell configurations. CUDA communication is fused into model-specific kernels rather than delegated wholesale to a generic framework layer.

InferStack implication:
- use as reference evidence for a future TP communication research card;
- do not apply to Qwen3.5 single-GPU work;
- qualify on exact multi-GPU hardware/topology.

### DSUF-T5 — compressed KV snapshot transport

The engine does not simply copy the whole live GPU cache between replicas.

Its snapshot path:
- selects reusable segments;
- retains rollback/tail coverage;
- validates model/cache geometry and fingerprints;
- optionally FP4-encodes compressible cache segments;
- copies the encoded snapshot into host memory;
- transfers it peer-to-peer;
- reconstructs only the requested reusable prefix on the destination.

Relevant upstream:
- `crates/engine/src/kvstore.rs`
- `crates/engine/src/fp4.rs`
- `crates/kernels/cu/kv_fp4.cu`
- `crates/kv-transfer/src/*`

Important distinction:
- this FP4 path is a **wire/snapshot representation for transferable state**. Do not equate it with enabling FP4 for all live attention KV computation.

InferStack implication:
- add a future `cache_snapshot_encoding` treatment distinct from live `kv_cache_dtype`;
- record restore cost, transfer bytes, numerical/quality gate and reuse coverage.

### DSUF-T6 — RDMA KV sharing decouples cache locality from routing

Each replica can retain recent conversation cache state in host memory. Redis records ownership/revision. If a request reaches another replica, that replica can discover the owner and retrieve the state host-to-host over RDMA.

Conceptually:

```text
request
   |
load-aware router
   |
replica B
   |
Redis: cache currently owned by A
   |
A host memory --RDMA--> B host memory
   |
restore reusable prefix
   |
continue generation
```

This creates a different serving design from sticky/prefix-affinity routing.

InferStack implication:
- current prefix/cache routing and future distributed cache should be separate policy choices;
- create a future experiment comparing:
  1. cache-affinity routing,
  2. load-only routing + state transfer,
  3. hybrid routing using estimated transfer cost.

This belongs after real GPU Kubernetes/remote multi-node execution exists.

### DSUF-T7 — ownership directory and revision fencing

Redis stores ownership metadata, not model state. The directory uses revisions/epochs so stale cache ownership can be fenced after restart or transfer.

InferStack implication:
- reuse the concept in future distributed-state contracts:
  `state_id + model identity + revision + owner + geometry + checksum`;
- a cache location hint must never be trusted without identity/revision validation.

### DSUF-T8 — prepare/execute overlap

The server has a bounded preparation queue. Cache lookup/remote-state preparation occurs on CPU/runtime work while the TP workers execute prior work.

InferStack implication:
- this is useful for our scheduler even before RDMA: tokenization, cache lookup, artifact prep and route decisions should overlap GPU work where safe;
- measure queue-prep wall time separately from GPU execution.

### DSUF-T9 — bounded-replay long-context prefill

The server documents CED bounded SWA replay: rather than replaying full decoder state, it reconstructs a bounded decoder window and reuses retained encoder outputs where applicable.

This is DeepSeek-V4.1-specific and is not directly portable to Qwen3.5.

InferStack implication:
- technique card only;
- applicability must be architecture-aware.

### DSUF-T10 — agent-driven engine optimization

Phil Howes described the project as mostly agent-driven after agents were allowed to study SGLang; the reported TP4 step time moved from roughly 5 ms to 3.5 ms to 3.1 ms during rapid iteration.

The upstream README also states that coding agents optimized kernels while reading FlashInfer and SGLang source.

InferStack implication:
- this is direct external evidence for H5's bounded optimization loop:
  source/reference study -> profiler -> candidate -> compile -> numerical gate -> benchmark -> keep/reject.
- the agent does not get authority to merge/promote; the controller/verifier still owns acceptance.

## 4. Where this fits relative to current InferStack

Current repository evidence has already advanced past the earlier N3 plan:
- production-v1.7 static decode was kept;
- v1.8 token-budget prefill was tested;
- v1.9 FLA prefill was kept;
- the next native step is N4 packed/varlen prefill.

Therefore DeepSeek Ultrafast should **not** reset our roadmap to another decode rewrite.

Use it like this:

```text
CURRENT
  |
  +-- v1.9 Qwen3.5 Native
  |      |
  |      +-- N4 packed/varlen prefill          <- immediate
  |
  +-- DeepSeek Ultrafast external evidence
         |
         +-- speculative acceptance metrics    <- H5 evidence schema
         +-- Rust/CUDA native-v2               <- later
         +-- TP/NVLS communication             <- later multi-GPU
         +-- RDMA cache sharing                <- later H4/distributed
         +-- snapshot FP4 transport            <- later distributed cache
         +-- load-aware routing                <- later routing experiment
```

## 5. Proposed InferStack packages

### DU0 — evidence ingestion (now)

Record upstream commit, license, benchmark claims and technique cards DSUF-T1..T10 as untrusted external evidence.

Acceptance:
- upstream results cannot satisfy an InferStack performance gate;
- source/commit/license are present;
- every card has explicit applicability.

### DU1 — experimental metric schema

Extend the benchmark evidence model, when speculative methods are evaluated, with:
- decode step ms;
- drafts proposed;
- tokens accepted;
- acceptance length;
- verification ms;
- output tok/s.

This prevents speculative throughput from hiding a slow verifier or different acceptance policy.

### DU2 — external endpoint capability (later)

Use `generic_openai` first. Add a specialized adapter only when supported hardware/endpoint exists and extra DeepSeek Ultrafast telemetry is needed.

Do not vendor upstream source.

### DU3 — distributed KV mobility experiment (later)

Prerequisites:
- real multi-node GPU execution;
- provider with RDMA-capable networking;
- state snapshot identity contract;
- distributed cache telemetry.

Compare affinity vs transfer-aware routing under the same workload.

### DU4 — Native-v2 compiled backend research (later)

Question:
Can a model-specialized compiled backend materially beat the current Python/Triton runtime after N4?

Do not begin without profiler evidence.

## 6. Hardware applicability

The current upstream build targets `sm_100a` and `sm_103a`, and its README names B200/B300/GB300-class configurations.

Our current common lab GPUs are not direct targets:
- T4: sm75;
- A100: sm80;
- L4: sm89.

Therefore:
- do not spend Colab/Kaggle time trying to build the upstream engine;
- a live qualification needs an external B200/B300/GB300 environment or the Baseten API;
- E1 must record that hardware requirement before launch.

## 7. License boundary

The inspected upstream repository uses Elastic License 2.0.

Operational rule for InferStack:
- external study and internal benchmarking may be useful, subject to the license;
- do not vendor/copy upstream implementation into InferStack Native;
- do not offer the upstream software as a hosted InferStack backend without confirming the intended use is permitted;
- independently implemented techniques should retain provenance showing the source of the idea, while the implementation must be ours;
- this document is engineering guidance, not legal advice.

## 8. Immediate actions

1. Keep N4 packed/varlen prefill as the immediate native runtime task.
2. Add DU0 to research memory/technique registry.
3. Add speculative acceptance metrics before our next speculative/MTP campaign.
4. Add "distributed KV mobility" as an H4/E1 multi-node milestone after RunPod/real GPU Kubernetes exists.
5. Add a future model-specialized compiled-backend research track rather than rewriting Native now.
6. If we later obtain B200/B300 capacity, run DeepSeek Ultrafast as an external endpoint through the same ExecutionManifest/ResultBundle contract before writing a specialized adapter.

## 9. Main lesson

DeepSeek Ultrafast is valuable to InferStack less because it is another server and more because it demonstrates a production pattern:

```text
model-specific architecture
      +
static/captured execution
      +
speculative multi-token progress
      +
hardware-specific kernels
      +
communication-aware TP
      +
movable distributed cache state
      +
simple load-aware replicas
      +
bounded agent-driven optimization
      =
very high specialized inference performance
```

InferStack's job is to make those ideas **conditional, measurable techniques**, not fixed assumptions.


## 10. Additional code-level findings from the Baseten engineering write-up

These findings refine the earlier study and are especially relevant to InferStack's future H5 optimizer.

### DSUF-T11 — Programmatic Dependent Launch (PDL)

Baseten's engineering write-up identifies PDL as one of the three hardware-level optimization levers used for DeepSeek-V4.1-Flash. The upstream CUDA wrapper exposes PDL launch configuration, so dependent kernels can begin work with less host-side serialization when the dependency can be expressed to CUDA.

InferStack implication:
- add `programmatic_dependent_launch` as a future kernel-launch treatment;
- only apply it when a measured producer/consumer chain leaves a launch bubble;
- record GPU architecture, kernel pair, overlap evidence and end-to-end impact;
- do not assume PDL helps on L4/A100 simply because the API exists.

This is a launch/overlap optimization, not a model-semantic optimization.

### DSUF-T12 — residual-stream fusion

Phil Howes reports that the specialized engine fused DeepSeek's four residual streams into one kernel instead of many separate operations. The key transferable idea is not the exact DeepSeek kernel but **cross-operation fusion around a model-specific residual topology**.

InferStack implication:
- add a future `residual_stream_fusion` technique card;
- use profiler evidence to identify repeated residual/norm/merge traffic;
- treat it as architecture-specific: Qwen3.5 does not inherit DeepSeek's exact four-stream topology;
- validate memory traffic and kernel-count reduction separately from numerical correctness.

This reinforces the existing v1.6 serving-pack/fusion direction: the best fusion boundary may come from model topology rather than generic operator names.

### DSUF-T13 — fused router + expert execution

The upstream CUDA MoE path contains fused routing/expert machinery, and Howes explicitly describes combining expert selection and expert execution to avoid an intermediate write/read round trip.

Conceptually:

~~~text
generic MoE

router
  -> write selected experts
  -> read selected experts
  -> launch expert GEMM


specialized path

router + expert plan + expert execution
  -> one fused/pipelined path
~~~

InferStack implication:
- record `router_expert_fusion` as a MoE-only technique;
- measure saved HBM traffic, routing latency and expert GEMM utilization;
- keep it out of the current dense Qwen3.5 N4 path;
- make it available when InferStack later serves an MoE model.

### DSUF-T14 — host-loop minimization / whole-step graphing

Baseten's broader DeepSeek optimization work describes minimizing the Python/host loop by grouping kernels into a small number of breakable CUDA graphs. DeepSeek Ultrafast takes the specialized case further: its speculative draft + verify + accept path can be captured as one graph, including variable verification branches through CUDA graph conditional nodes.

This is materially different from "capture one-token decode."

The important evidence dimensions are:

- kernels/graph replay;
- host launches/step;
- graph nodes;
- accepted tokens/step;
- step latency;
- verification work;
- final output tokens/s.

InferStack implication:
- extend future speculative experiments so graph scope is part of treatment identity;
- compare one-token graph, multi-token speculative graph and breakable/piecewise graph separately;
- never attribute speculative throughput to CUDA graphing alone.

### DSUF-T15 — model-specialized optimization sandbox

The strongest product lesson from this project is the development workflow.

The team's first agent-driven attempt from scratch struggled. Allowing agents to inspect SGLang/FlashInfer gave them a working reference surface; they then rapidly profiled and hill-climbed a narrow engine, and successful ideas were ported back into SGLang.

That suggests a new InferStack H5 execution mode:

~~~text
measured bottleneck
      |
      v
Model-Specific Sandbox
      |
      +--> reference source checkout(s)
      +--> exact model + hardware
      +--> profiler
      +--> isolated kernel/engine candidate
      +--> numerical oracle
      +--> benchmark
      |
      v
KEEP / REJECT
      |
      +--> keep as specialized backend
      |
      +--> upstream proven technique into Native / adapter
~~~

The sandbox is not production by default. It is an **optimization laboratory** with hard verification boundaries.

Required controls:

- immutable reference commits and licenses;
- no unreviewed source copying across incompatible licenses;
- isolated workspace/container;
- explicit experiment budget;
- compiler/profiler output retained as evidence;
- numerical/quality gates before performance acceptance;
- exact hardware/model/workload identity;
- controller-owned promotion.

This is a direct refinement of H5: the optimizer agent should be allowed to read strong external implementations, but only the verifier/controller decides whether a candidate becomes part of InferStack.

## 11. Recommended integration into InferStack

DeepSeek Ultrafast should be represented in four layers:

~~~text
External Evidence
   |
   +-- DSUF technique cards
   |     captured speculative graph
   |     PDL
   |     residual fusion
   |     router+expert fusion
   |     TP/NVLS
   |     RDMA KV mobility
   |
   +-- Model-Specific Sandbox reference
   |
   +-- generic_openai external endpoint
   |
   +-- specialized adapter later, only when telemetry/hardware justify it
~~~

Do **not** add its ELv2 source tree as a vendored InferStack engine.

For current Qwen3.5 work:

~~~text
NOW
  v1.9 FLA prefill
      |
      v
  N4 packed/varlen prefill

LATER
  H5 model-specific sandbox
      |
      +--> DeepSeek/MoE experiments on suitable hardware
      +--> compiled Native-v2 experiments
      +--> speculative whole-step graphs
~~~

## 12. Revised immediate decision

1. **Keep N4 packed/varlen prefill as the next Native task.**
2. **Add DSUF-T11..T15 to research memory/technique registry as external evidence.**
3. **Add speculative-step metrics before the next MTP/speculative campaign.**
4. **Add the Model-Specific Sandbox as an H5 design target, not as a production backend today.**
5. **Use DeepSeek Ultrafast through `generic_openai` first if we obtain a managed endpoint.**
6. **Only create `DeepSeekUltrafastAdapter` when we need engine-specific telemetry.**
7. **Do not vendor or port ELv2 source into InferStack Native.**
