# InferStack cloud serving: independent research and implementation plan

Research date: 9 November 2026 (Asia/Kolkata). Recommendation, not a production qualification.
Repository inspected: local 857769d4331af88c370a25ea80ad3d5f84440aa4.
GitHub main observed during research: ff777007e05abcb991e84094ea8e4f68336f627e.
Read alongside ARCHITECTURE.md, V2_AGENT_HANDOFF.md, CONTROL_RUNBOOK.md and the three verification reports. This document adds the next serving-infrastructure plan; it does not replace the ownership boundaries or the integrated-delivery requirement in the handoff.

## Decision

Product scope clarification: [INFERENCE_HARNESS_PLAN.md](INFERENCE_HARNESS_PLAN.md)
defines the integrated research → implementation → measurement → diagnosis →
deployment feedback loop. This cloud plan supplies its serving infrastructure.
InferStack remains multi-engine; references to vLLM below describe the first
Kubernetes integration profile, not a requirement for every deployment. Native,
vLLM and SGLang profiles each need their own compatibility and runtime evidence.

Use InferStack as the verification and optimization control plane over a standard serving data plane. Start with one qualified vLLM deployment profile on an existing GPU Kubernetes cluster. Integrate llm-d's Gateway API Inference Extension implementation for replica selection, Prometheus for operational evidence, and KEDA with inference metrics for autoscaling. Add immutable weight caching before enabling automatic scaling.

InferStack continues to own candidate isolation, frozen release identity, trusted evaluation, promotion eligibility, rollout state, rollback obligations and experiment budgets. The external router owns replica selection within a release. KEDA owns its generated HPA and the replica count. Do not install overlapping serving controllers for the same workload.

This is the best fit for this repository's current architecture and Kubernetes provider, subject to a GPU compatibility spike. It is not a claim that Kubernetes or vLLM is best for every team. An operator already using KServe can expose the same contract through LLMInferenceService; an NVIDIA fleet needing disaggregation can qualify Dynamo as an alternative profile.

## Findings that change the supplied assessment

1. **The engine and platform versions are broadly supported by GitHub, but latest is not a compatibility matrix.** vLLM's newest listed tag is v0.33.0rc3 even though GitHub marks prerelease=false; treat its RC name explicitly. Dynamo's newer model-specific development release does not supersede general stable v1.6.2. KAITO's latest guardrail component is not its main operator release. See the snapshot below.
2. **Do not build new scaling integration around WVA.** The release-tagged llm-d v1.0 documentation deprecates Workload Variant Autoscaler and recommends KEDA + EPP metrics. Its default in-flight signal requires no Flow Control; gateway queue metrics require Flow Control. Cost-aware placement across heterogeneous variants is not replaced feature for feature. [Tagged architecture](https://github.com/llm-d/llm-d/blob/v1.0.0/docs/architecture/autoscaling/README.md), [tagged deployment guide](https://github.com/llm-d/llm-d/blob/v1.0.0/guides/workload-autoscaling/README.md).
3. **The uniqueness claim is not established.** Argo Rollouts and Flagger already implement substantial metric-gated progressive delivery. llm-d v1.0 itself has blue/green InferencePool rollout, traffic-splitting, optional shadowing and disaggregated-rollout guidance. These are considerably more relevant than tutorial repository star counts. Their documentation does not establish equivalence to InferStack's entire verification workflow. [Argo](https://argoproj.github.io/argo-rollouts/), [Flagger](https://docs.flagger.app/usage/deployment-strategies), [llm-d rollouts](https://github.com/llm-d/llm-d/blob/v1.0.0/guides/rollouts/README.md).
4. **There is an existing runtime-inventory integration.** llm-d documents k8s-aibom, which observes model declarations, runtimes and resolved images with evidence/confidence tiers. Sigstore policy-controller checks image signatures/attestations at admission. Model Transparency signs and verifies model-file hashes. None of these observations alone proves the tensors currently loaded in accelerator memory. InferStack's own attestation currently compares endpoint/provider observations, revisions, arguments and artifact digests; it is also not hardware-rooted proof of loaded tensors. [llm-d inventory](https://github.com/llm-d/llm-d/blob/v1.0.0/guides/runtime-inventory/README.md), [k8s-aibom](https://github.com/GoogleCloudPlatform/k8s-aibom), [Sigstore admission](https://docs.sigstore.dev/policy-controller/overview/), [model signing](https://github.com/sigstore/model-transparency).
5. **Autotuning has broader precedents.** modelsphere/llm-autotune describes platform-owned benchmarks, production baselines, remeasurement, quality/latency redlines and PostgreSQL recovery. NVIDIA Triton Model Analyzer searches serving configurations. These are adjacent competitors or sources of techniques, regardless of stars. Their descriptions are not independent validation of their reliability. [LLM AutoTune](https://github.com/modelsphere/llm-autotune), [Triton Model Analyzer](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/model_analyzer/README.html).
6. **Stronger rollout gates require measurement work first.** The live gateway measures first upstream bytes; non-streamed requests put full completion latency in the same ttfb field. AdmissionController retains at most 20,000 in-memory events, and RolloutController compares median TTFB/error rate. EndpointVerifier already has content-aware client TTFT, p95 TTFT and chunk-gap measurements, but that does not make them available as durable live rollout signals. evaluation/judge.py compares repeated aggregates; it is not a semantic evaluator or a tail-sample sufficiency test. Adding a p95 calculation alone would not close the gap.
7. **The weight-cache gap is real, with a qualification.** KubernetesProvider only mounts 2Gi /tmp scratch; an operator can still bake weights into an image, but no managed model-cache volume is exposed. vLLM supports the optional runai_streamer loader for Safetensors. Faster weight reads do not necessarily halve full time-to-ready, which includes scheduling, image pull, runtime initialization and compilation. Benchmark the complete path. [Pinned vLLM loader documentation](https://github.com/vllm-project/vllm/blob/v0.32.0/docs/models/extensions/runai_model_streamer.md), [streamer release](https://github.com/dsx-ai-factory/model-streamer/releases/tag/0.17.1).

The statement that most teams use precisely these three engines is a plausible ecosystem overview, not a conclusion supported by this search: GitHub stars and project documentation do not measure deployment market share. Ray Serve LLM is another material serving alternative with distributed placement, autoscaling and initialization support. [Ray documentation](https://docs.ray.io/en/latest/serve/llm/index.html).

## GitHub release snapshot

Read public repository metadata and release collections through the GitHub connector. Counts are observations, not selection scores. Exact metadata is retained in research/CLOUD_SERVING_GITHUB_2026-11-09.json. A release tag is a research candidate, not an InferStack-supported combination.

| Project | Observed stars | Release to distinguish | Published UTC |
|---|---:|---|---|
| [vLLM](https://github.com/vllm-project/vllm/releases) | 96,569 | v0.33.0rc3; preceding non-RC v0.32.0 | 2 Nov; 18 Oct 2026 |
| [SGLang](https://github.com/sgl-project/sglang/releases/tag/v0.5.23) | 38,591 | v0.5.23 | 2 Nov 2026 |
| [TensorRT-LLM](https://github.com/NVIDIA/TensorRT-LLM/releases) | 15,040 | v1.4.0rc3 prerelease; releases/latest returned v1.2.1 | 6 Nov; 20 Apr 2026 |
| [llm-d](https://github.com/llm-d/llm-d/releases/tag/v1.0.0) | 5,179 | v1.0.0 | 20 Oct 2026 |
| [Dynamo](https://github.com/ai-dynamo/dynamo/releases/tag/v1.6.2) | 8,459 | v1.6.2 general stable; newer model-specific prerelease exists | 5 Nov 2026 |
| [KServe](https://github.com/kserve/kserve/releases/tag/v0.21.0) | 6,201 | v0.21.0 | 25 Sep 2026 |
| [AIBrix](https://github.com/vllm-project/aibrix/releases/tag/v0.8.0) | 5,182 | v0.8.0 | 13 Oct 2026 |
| [GAIE](https://github.com/kubernetes-sigs/gateway-api-inference-extension/releases/tag/v1.6.2) | 801 | v1.6.2 | 17 Sep 2026 |
| [LeaderWorkerSet](https://github.com/kubernetes-sigs/lws/releases/tag/v1.0.0) | 903 | v1.0.0 | 16 Oct 2026 |
| [KAITO](https://github.com/kaito-project/kaito/releases/tag/v0.13.0) | 1,039 | operator v0.13.0; guardrail-core-v0.1.1 is separate | 24 Oct 2026 |
| [Model Streamer](https://github.com/dsx-ai-factory/model-streamer/releases/tag/0.17.1) | 407 | 0.17.1, without a v prefix | 29 Oct 2026 |
| [Model Transparency](https://github.com/sigstore/model-transparency/releases/tag/v1.1.1) | 249 | v1.1.1 | 10 Oct 2025 |

CNCF lists llm-d at Sandbox maturity, accepted 12 March 2026. CNCF membership is governance context, not evidence that a particular deployment is qualified. [CNCF project record](https://www.cncf.io/projects/llm-d/).

GitHub repository searches included inference attestation rollout, model runtime attestation, vllm-bayesian-autotuner, llm-autotune and llm-d workload variant autoscaler. The first returned no repository-name/description matches; that is not proof of absence. The runtime search also found AWS's Guardian attested-inference sample, which explicitly labels itself for learning/prototyping. [AWS sample](https://github.com/aws-samples/sample-guarded-user-controlled-attested-runtime-deployment). This was a bounded public-source review, not a full audit of competing codebases.

## Build versus integrate

| Concern | First choice | Selection reason and alternative |
|---|---|---|
| Engine | vLLM, exact GPU/model-compatible image digest | Already an adapter; broad common API. Keep SGLang as a separately qualified challenger. TensorRT-LLM is later and requires its own build/runtime capability work. |
| Replica routing | llm-d router implementing GAIE | Avoid custom cache indexing/scheduling. Start from optimized baseline; qualify precise KV-event routing only when needed. GAIE alone is an API/extension framework, not a complete deployment. |
| Autoscaling | KEDA + Prometheus + release-scoped EPP metrics | Follows llm-d's current supported direction. In-flight first; queue policy after Flow Control is qualified. Dynamo Planner advisory mode is a useful alternative within a Dynamo deployment, not a drop-in controller for the existing provider. |
| Weight startup | Immutable PVC/node cache; optional streamer | Separate artifact integrity from read optimization. Use KServe LocalModelCache when using KServe, instead of duplicating that controller. |
| Release/evaluation policy | InferStack | Preserve trusted evaluator isolation, frozen identity and crash recovery. Reuse Argo/Flagger analysis ideas; a future Argo actuator must replace, rather than compete with, route ownership. |
| Full serving operator | None additional initially | KServe is appropriate if the operator needs its CRDs/lifecycle ecosystem. Dynamo fits NVIDIA disaggregation; AIBrix already covers routing, metrics scaling, model lifecycle and LoRA. Select one profile, rather than installing all three. |
| Managed alternative | Capability-limited external endpoint profile | Useful when avoiding cluster operations matters more than modifying engines. Do not advertise kernel optimization or full attestation where the provider exposes only an API. |

KServe's LLMInferenceService path wraps llm-d routing, while its traditional InferenceService supports other modes; saying all KServe is built on that router is too broad. [KServe LLM gateway guide](https://kserve.github.io/website/docs/model-serving/generative-inference/llmisvc/llmisvc-envoy-ai-gateway). AIBrix is also a substantive alternative rather than a missing component to install alongside llm-d. [Tagged AIBrix capabilities](https://github.com/vllm-project/aibrix/blob/v0.8.0/README.md).

Managed platforms are not interchangeable: Bedrock custom import restricts supported model configurations and charges via Custom Model Units/copies and active-time rules; the specific Google Model Garden custom-weights offering remains Preview; Modal GPU snapshots have compatibility requirements. No managed vendor is selected on price here because workload, region, availability and sustained volume are unspecified. Generic Vertex custom-container serving must not be confused with the particular Preview custom-weights feature. [Bedrock import](https://docs.aws.amazon.com/us_en/bedrock/latest/userguide/model-customization-import-model.html), [AWS pricing](https://aws.amazon.com/bedrock/pricing/), [Google feature](https://docs.cloud.google.com/gemini-enterprise-agent-platform/models/model-garden/deploy-models-with-custom-weights), [Modal snapshots](https://modal.com/docs/guide/memory-snapshots).

## Target integration contract

Client → InferStack authentication/admission and immutable release choice → release-scoped upstream route → GAIE/llm-d EPP chooses a ready, verified replica → vLLM.

There must be exactly one stable/candidate traffic splitter. Initially retain InferStack's existing per-request release choice and route each selected release to a separate upstream pool. If a later profile delegates splitting to HTTPRoute, make InferStack reconcile that route's desired state and remove the first splitter for that profile. Preserve streaming cancellation, request IDs, model-alias rewriting and session behavior through both hops. Verify actual selected-pod attribution, not just the pool address.

External infrastructure objects bind to release ID, frozen identity, allocation ID, namespace and owner. Record the observed Pod UID, imageID, node and device identity when available. Replacement or scale-out pods are new runtime instances: readiness alone must not admit them before required identity checks. Cached weights, KV reuse and adapters must respect model/tokenizer/config/tenant boundaries.

One owner per field/action:

- InferStack: release eligibility, desired route policy, rollout journal, runtime admission evidence, budgets and experiment allocations.
- llm-d/GAIE: endpoint picking, queue/admission features explicitly enabled for that profile, cache index and replica drain membership.
- KEDA: ScaledObject-driven HPA and replica target. InferStack's scale() is reserved for manual mode or explicit controller handoff.
- Kubernetes: pod scheduling, GPU device assignment and termination. InferStack observes and reconciles allocation accounting against actual replicas.

The existing gateway max_inflight defaults must be reconciled with EPP queue limits: otherwise it can reject traffic before the autoscaler's queue sees demand. Record every rejection source, bound queues/deadlines, and avoid retry amplification.

## Ordered work packages

### P0 — Close the existing qualification boundary

The previous background run has since been recorded as complete by commit 857769d. Its third-review report records 260 suite passes, 26 PostgreSQL passes, two Docker passes, one kind pass and all 36 probes passing. The CPU qualification remains not_qualified because GPU gates are unavailable. These are repository-reported results, not rerun results from this research task. Some older runbook/status tables still describe earlier implementation states.

Reconcile qualification artifact hashes and commit IDs; independently rerun the required checks in an isolated environment after the concurrent run is settled. Do not infer GitHub validation from local results: remote main still points to ff777007 during this research. Reconcile documentation against actual behavior and retain old results as dated history.

Acceptance: third-review isolation/replacement/retirement probes pass unchanged; required PostgreSQL/Docker/Kubernetes checks have no unexplained skips; actual GPU native/vLLM/SGLang capability claims match live conformance; unsupported profiles remain explicit. Produce one qualification manifest for the integrated candidate and the target hardware. Do not tag v2.0.0 based on CPU evidence.

Touchpoints: scripts/v2_qualify.py, scripts/engine_conformance.py, docs/verification/, docs/V2_AGENT_HANDOFF.md and docs/CONTROL_RUNBOOK.md. Documentation updates must preserve concurrent changes.

### P1 — Live evidence and defensible rollout gates

Depends on P0's identity and recovery baseline. Extend serving/gateway.py and serving/health.py to separate upstream first byte, client first content, full completion, content-event gaps, engine token timing, errors, cancellation and timeout. Never label SSE chunks as tokens. Export release/pool/replica-scoped operational measurements to Prometheus; persist the rollout's query, closed evidence window, sample counts, policy revision and result in the existing durable store. Raw prompts need not become metric labels or database rows.

Add p95 client TTFT, optional p99 with adequate evidence, E2E and engine ITL/TPOT gates where actually observable. Define absolute SLO and relative-regression limits. Handle stalled/open streams, telemetry loss/restart and incomplete windows as explicit outcomes. Keep error/cancellation denominators separate; missing data cannot pass or become zero. Do not average percentiles across replicas.

Bind a trusted held-out evaluation suite and evaluator revision to the release. Reuse EndpointVerifier and judge.py for supported repeated metrics, but add an explicit live/offline evaluation-result contract and sample policy. Task-normalized refusal/length checks are diagnostics or configured gates, not universal indicators that shorter/less-refusing is better. Candidate code cannot change test cases, evaluator or thresholds.

Acceptance: injected tail-only and token-stall regressions roll back despite unchanged p50; metadata-only SSE does not count as first content; corrupt/missing evaluation holds promotion; known quality regression rejects. Restart and multiple gateway workers preserve complete evidence accounting. Minimum samples, maximum stage duration and sequential-testing policy are frozen before the canary. Low volume holds or ends inconclusive; it does not pass on a timer. At 100% candidate traffic, use retained comparable baseline evidence or bounded independent probes and record that basis.

Touchpoints: serving/gateway.py, serving/health.py, telemetry/, contracts/metrics.py, optimization/experiment.py, evaluation/judge.py and releases/rollout.py. Add targeted failure tests and a rollout-evidence runbook. This is a medium integration package, not just adding percentile arithmetic.

### P2 — Artifact cache and measured startup

Can be prepared alongside P1 after P0; must finish before autonomous scaling. Add an operator-configured model volume/storage profile separate from scratch, compatible with non-root/read-only serving containers. Pin immutable model/tokenizer/config revisions and file manifests. Downloader/init jobs own artifact credentials and verify integrity before publishing a cache entry atomically. Runtime readers cannot overwrite shared trusted artifacts. Define cache size, eviction, ownership, permissions and node-affinity behavior; never use arbitrary hostPath access by default.

Add a versioned engine loader option; qualify runai_streamer for the selected vLLM/Safetensors/image combination with its optional dependency installed. Keep the current default loader available. Validate tokenizer/config downloads and object-store access separately. Cache/PVC access modes and zone binding must permit the intended replica topology. Ensure DNS/object-store egress is allowed narrowly through the existing policy, and compilation caches have bounded writable locations.

Acceptance: a model larger than 2Gi starts without manual staging; cache hit/miss, incomplete/corrupt cache, disk-full, credential expiry and node change produce correct readiness/failure behavior. Compare at least cold download, warm disk cache and restarted runtime with the same pinned artifact. Report scheduling, image pull, artifact fetch, GPU load, compilation and first successful generation separately, plus full time-to-ready and cost. Claim speedup only on measured profiles.

Touchpoints: infrastructure/remote.py, contracts/records.py, engines/managed.py, engine launch profiles and new deployment manifests. KServe deployments should map to its existing cache resources where applicable. [KServe cache semantics](https://kserve.github.io/website/docs/model-serving/generative-inference/modelcache/localmodel).

### P3 — Release-scoped cache-aware routing

Depends on P1 attribution and P2 launchability. Qualify the llm-d optimized-baseline profile with two real replicas. Generate separate InferencePools per immutable release and preserve a warm rollback pool within quota. Start with supported approximate prefix affinity/load selection; add precise cache-event indexing only when workload evidence warrants its extra components. The precise mode requires engine KV-event support and router subscriptions. [Tagged precise routing guide](https://github.com/llm-d/llm-d/blob/v1.0.0/guides/precise-prefix-cache-routing/README.md).

Extend provider observation and runtime admission to enumerate every serving Pod. A single request to a Service and a shared command-line declaration are not sufficient to establish all replicas' identity. Fail closed or quarantine unknown/mismatched pods according to the bound policy; recheck replacements and scale-out members. Document the trust level of reported identity versus verified artifact files.

Acceptance: shared-prefix and unique-prefix workloads demonstrate measured routing behavior against service/random routing at equal capacity; no presumed benefit for non-repeating prompts. Wrong-release, stale-cache, failed metrics, router restart, mismatched replica and draining replica scenarios cannot cross release boundaries. A long stream remains pinned and finishes across route changes. Correlate selected-pod identity and end-to-end latency through the extra hop. Cache affinity yields to safety and overload constraints.

Touchpoints: infrastructure/remote.py, releases/attestation.py, releases/runtime.py, serving/gateway.py and a new deployment-profile directory. Add real-cluster routing/failure qualification. Do not implement a second cache scheduler inside InferStack.

### P4 — Inference-driven scaling with one actuator

Depends on P1–P3 and measured startup distributions. Start with recorded recommendations/offline trace replay; KEDA does not supply Dynamo's advisory interface, so keep actuation disabled through an explicit profile state until replay and bounds are accepted. Then enable one ScaledObject/HPA per release Deployment. Begin with supported in-flight metrics; enable bounded Flow Control before depending on gateway queue metrics. Add token backlog or SLO signals after calibration for the actual prompt/output mix.

Configure warm minimums, maximum replicas, spend/GPU quotas, pending-pod handling, scale-up rate, scale-down stabilization and graceful drain from measured workload/startup behavior. GPU utilization remains diagnostic. Autoscaling cannot create cluster GPU capacity by itself; insufficient nodes/quota must be surfaced. Reconcile resource-time accounting from observed replica counts, including surge and stable rollback capacity. Freeze or separate capacity during paired optimization trials.

Acceptance: burst, long-prompt, long-generation, idle, metrics-outage, unschedulable-pod and GPU-quota scenarios stay within configured bounds. No competing HPA or manual reconciliation loop fights replica ownership. Downscale removes membership before termination and respects stream deadlines. Report time-to-capacity, tail latency, queue time, rejection rate, oscillation and spend against fixed capacity. Start with minReplicas at least one; scale-to-zero is a separate opt-in qualification requiring a measured wake-up path that meets the chosen service objective.

Touchpoints: infrastructure/allocations.py, infrastructure/remote.py, telemetry/, control/maintenance.py and new KEDA/Prometheus manifests. Keep optimization triggers distinct from serving autoscaling. [Pinned scaling ownership and metric rules](https://github.com/llm-d/llm-d/blob/v1.0.0/docs/architecture/autoscaling/keda-epp.md). Dynamo's v1.6.2 Planner has genuine advisory mode, but only use it under a separately qualified Dynamo profile. [Pinned Planner configuration](https://github.com/ai-dynamo/dynamo/blob/v1.6.2/docs/fern/pages/reference/components/planner-configuration.mdx).

### P5 — Full serving cost attribution and integrated qualification

Basic allocation pricing and a cost-per-million metric definition already exist. Complete the serving-side connection: billable replica time, idle/warm/rollback capacity, failed requests, storage/egress/CPU where known, and authoritative input/output token usage. Publish gross infrastructure cost and SLO-qualified goodput cost separately, with scope, currency, price provenance and observation interval. Do not allocate the entire bill only to successful request durations or count SSE events as tokens. Unknown prices/token totals make the ratio unavailable.

Acceptance: reconcile one bounded deployment interval against provider billing/price evidence; include idle and canary/shadow overhead; prevent double counting shared allocations. Compare fixed capacity against P4 for equal workload and quality/SLO objectives. Execute shadow → bounded canary → promotion → induced regression → rollback → crash/restart → retirement on the integrated profile. Shadow copies are explicitly sampled, budgeted and isolated from client-visible/tool side effects; confirm that the chosen gateway actually supports mirroring to InferencePools.

Deliver one integrated candidate, a compatibility manifest, measured evidence, runbooks and a concise known-limitations report. User installation/benchmarking of every intermediate package is not required.

## Deferred until measurements justify them

- **Multi-node and prefill/decode separation:** use LeaderWorkerSet/DisaggregatedSet or Dynamo's existing topology. Require model-size/network/SLO evidence, same-version P/D pairing, KV-transfer failure handling, coordinated retirement and topology-aware budgets before declaring support.
- **LoRA:** qualify immutable adapter hashes, base-model compatibility, tenant isolation, per-adapter evaluation, routing, cache identity, load/unload races and rollback. Do not treat the adapter name as sufficient identity.
- **Spot and cloud provisioning:** begin with resumable experiments; durable serving needs an explicit interruption/capacity strategy, credentials, provider quotas and a teardown ledger. Automatic replica scaling and node provisioning are separate capabilities.
- **Advanced startup:** GPU snapshots, P2P weight distribution, llm-d fast actuation and distributed KV offload are separate experiments. Their lifecycle/device-access assumptions may conflict with InferStack's current isolation guarantees; do not enable them by default just for a cold-start headline.
- **Heterogeneous cost optimization:** KEDA does not replace WVA's global placement optimizer. Qualify a separate placement recommendation mechanism if this becomes a real workload requirement; keep one owner for each resulting action.

## Compatibility and release checklist

The first implementation task is a compatibility spike, not installing every newest release in the table. Start from the tagged llm-d guide and lock its chart/operator/CRD/router/engine dependencies together. Its common-versions file even pins v0.30.0 for some accelerator paths, demonstrating why the latest engine tag is insufficient. [Tagged version configuration](https://github.com/llm-d/llm-d/blob/v1.0.0/docker/common-versions).

Record Kubernetes, Gateway API, InferencePool CRD and gateway implementation versions; llm-d router/EPP/chart and KEDA versions; image digests; Python/CUDA/driver/device-plugin versions; model architecture, revisions, tokenizer and loader; GPU model/count/topology; storage class/access modes; network policy; feature flags; evaluator/workload/policy hashes. Verify selected component licenses from their actual files; GitHub's NOASSERTION metadata for some NVIDIA/KAITO repositories is not a license conclusion.

Use CPU/kind tests for object reconciliation and failures, a real GPU for model behavior/startup/performance, and an actual multi-replica GPU cluster for routing and scaling. Simulation/emulated metrics validate control logic only. Every package's acceptance evidence must distinguish those scopes. A known incompatibility selects an older supported tuple or an explicitly unsupported state; it never silently drops a release gate.

No serving code, cluster resources, credentials or existing qualification process were modified by this research. New work from this task consists only of this plan and its research snapshot.

## Subsequent research addendum

The [Vercel / Fireworks Ultrafast review](research/VERCEL_ULTRAFAST_RESEARCH_2026-11-09.md) verifies the 7 November 1,000+ tokens/s announcement, captures public endpoint metrics and pricing, and proposes external-route qualification plus an engine-supported speculative-decoding experiment. It supplements P0–P5; it does not establish production qualification or replace the serving-stack plan.

The [logged-in X scan](research/X_INFERENCE_SCAN_2026-11-09.md) screens 143 unique posts (116 retained), then checks selected leads against first-party projects. It adds Together Compiler/JITEngine to the optimization comparison, DSpine/NF12 as experimental techniques, and Kinference/DeepInfer/Lebrel as further integration or competitive-review leads. It corrects the earlier narrow autotuning comparison; reported speedups and trust guarantees remain unqualified on InferStack hardware.


## Implementation status — 9 November 2026 (implementation agent)

Appended by the implementation agent; the plan above is unchanged. Each package
lists what is implemented and what was verified on this CPU-only host. Nothing
below is GPU or production-cluster qualification.

| Package | Commit | Implemented | Verified here | Still needs |
|---|---|---|---|---|
| P0 | this commit | `scripts/compatibility_manifest.py` (exact component tuple; absent = null), written into every `v2_qualify.py` run; runbook and handoff status reconciled | Manifest on kind v1.37: no Gateway API / InferencePool / KEDA CRDs present (recorded as null) | Colab A100 qualification; push so GitHub CI and a fresh checkout see this code |
| P1 | 06aace9 | Per-request durable evidence (migration 6) with separate phases and outcomes; Prometheus `/metrics`; closed-window, release-attributed observation; p95/p99 TTFT, E2E, ITL, stall, cancel and empty-stream gates; min samples; `max_duration_s` ends inconclusive; telemetry loss holds; frozen policy digest; decisive evidence artifacts; bound held-out evaluation | Tail-only and stall regressions roll back with an unchanged p50; metadata-only SSE gives no first content; missing/corrupt evaluation holds, failed rolls back; restart and two gateway workers keep complete accounting; real-gateway phases | Live traffic volumes; sequential-testing policy beyond fixed minimum samples |
| P2 | f0c808c | Immutable checksum-keyed model cache with staged, verified, atomic publish; read-only PVC mount, verify init-container, bounded compile cache; credentialed downloader Job; LRU eviction honouring leases; allowlisted `--load-format`; measured startup (per-pod scheduled/initialized/started/ready) | Corrupt, incomplete, tampered, disk-full, credential-failure, concurrent-publish and eviction cases; manifests | A real >2Gi model and the streamer loader on GPU; cold/warm/restart timing profiles; storage access modes |
| P3 | e7c1312 | Per-release InferencePool + HTTPRoute keyed on `x-inferstack-release`; membership gated by per-pod identity admission (image digest, `--revision`), quarantine on mismatch; pooled releases ready only after admission; pool retired before pods; selected-endpoint attribution (migration 7) | Admission labelling of real pods on kind (2 admitted; wrong pin quarantines both); release header and endpoint recorded through the gateway | llm-d/GAIE installed; measured shared-prefix vs unique-prefix routing; router restart and drain on a GPU cluster |
| P4 | ca35f44 | Modes manual / advisory / keda on the Deployment; advisory trace replay; one ScaledObject on in-flight demand only after an accepted replay of the same policy; manual scale refused while KEDA owns replicas; quota caps; no scale-to-zero; pending pods; GPU accounting from live pods | Burst, idle, noise, outage, quota and sustained-load replays; manual-scale refusal on a real kind Deployment | KEDA + Prometheus scaling GPU replicas; Flow Control queue signals; measured time-to-capacity |
| P5 | 663439a | Sampled, budgeted, tool-free shadow traffic; input tokens (migration 8); gross vs goodput cost per million tokens with provenance; unavailable instead of guessed; bill reconciliation; like-for-like capacity comparison | Integrated real-process lifecycle: shadow, canary, induced regression, rollback across a controller restart, promotion, retirement, cost; evidence tables on hosted PostgreSQL 18.6 | Reconciliation against a real provider bill; shadow mirroring through the chosen data-plane gateway |

Deferred items (multi-node, prefill/decode disaggregation, LoRA, spot, advanced
startup, heterogeneous placement) remain deferred as the plan specifies.


## Portable experiment execution bridge

The cloud-serving packages remain the deployment/scaling foundation. The new [portable execution plan](PORTABLE_EXPERIMENT_EXECUTION_PLAN.md) defines how a pinned experiment reaches those substrates without creating a second control plane.

Provider order for the first portable-execution qualification is intentionally different from the eventual Kubernetes production stack:

- reuse Colab/Kaggle first through generated notebook/result bundles;
- add RunPod first as the paid experiment provider behind AllocationManager;
- consider Verda next for low-level VM/host-control experiments;
- use Modal first as an autoscaling endpoint for an already-verified release;
- use Kubernetes GPU nodes later for real cluster placement, routing, scaling and rollout qualification.

Kubernetes remains an orchestrator, not a source of GPUs. kind/CPU evidence validates object lifecycle only. Any GPU-performance or engine claim still requires a recorded target-GPU environment and the same pinned model/workload/evaluator identity.
