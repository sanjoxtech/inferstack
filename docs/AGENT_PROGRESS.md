# Agent progress log (live)

Shared, append-only status for every agent working in this repository. Update at least
every ~5 minutes while working and at every milestone (newest entry at the top of the
log). Each entry: time (UTC), agent, current task, state, files touched, next step.
Check `git log origin/main` for the authoritative history; this file is coordination.

## Current state (summary)

- 2026-11-13: HEAD before this entry `c2d534d`. E1.0–E1.2 implemented (3516693, 484e5ed). Fair engine scorecard (N0/N1) completed on Colab **L4**: all 4 engines passed, 3 rounds, 5760/5760 requests ok. Output tok/s vs native_unfused: native_full 1.15–1.52×, vLLM 2.53–4.19×, SGLang 2.56–4.69×. Largest native gap: prefill. See docs/verification/COLAB_L4_SCORECARD_2026-11-13.md. No A100 scorecard yet.
- Older summary (2026-11-09) below is historical.
- Reviewed HEAD `main`: c460793 (H3 step 1). CI: five successful jobs; manual provider smoke skipped.
- Harness plan: H0 accepted; H1/H2 implemented; H3 step 1 implemented (bounded implementer handoff, immutable validated patch and resume). H3 refinement/stop policy/demo and H4/H5 remain.
- Independent focused check on c460793: 39 H1/H2/H3 tests passed, no failures/skips (PostgreSQL parametrizations excluded).
- GPU: Colab A100 run at d33b470 **qualified** (with same-commit CI sandbox evidence,
  docs/verification/COLAB_A100_2026-11-09.md). Speed numbers single-run only.
- Qualification scope: the recorded A100 qualification covers the older d33b470 native/Qwen3.5-2B-Base environment, not latest H3, other engines, T4 or production GPU Kubernetes. Repeated speed/task-quality validation and final integrated qualification remain.
- In-progress work observed: uncommitted CI workflow changes for hosted PostgreSQL/provider checks; preserve the implementation agent's edits. No result from the running hosted PostgreSQL verification is claimed here.

## Log

### 2026-11-15 — DeepSeek Ultrafast follow-up research
- Verified upstream `basetenlabs/deepseek-ultrafast` at commit `76766a97308bfd1658fb055e835d0f92de737dc3` against its README, Rust/CUDA implementation and Baseten engineering write-up.
- Extended the study with PDL, model-topology residual fusion, fused router+expert execution, whole-speculative-step graphing and the model-specific optimization-sandbox pattern.
- Product decision: use DeepSeek Ultrafast as external evidence and future external endpoint/reference; do not vendor its ELv2 source into InferStack Native. `generic_openai` is sufficient before a specialized adapter is justified.
- Immediate Native ordering unchanged: v1.9 kept; N4 packed/varlen prefill remains next. H5 gains a future model-specific sandbox mode with pinned reference sources + profiler + numerical oracle + benchmark + controller-owned promotion.


### 2026-11-15 — DeepSeek Ultrafast external architecture study

- Studied `basetenlabs/deepseek-ultrafast` at upstream commit `76766a97308bfd1658fb055e835d0f92de737dc3`; added `docs/research/DEEPSEEK_ULTRAFAST_CODE_STUDY_2026-11-15.md`.
- Upstream is a DeepSeek-V4.1-Flash-specific Rust/CUDA engine for B200/B300/GB300-class `sm_100a/sm_103a`, not a generic replacement for vLLM/SGLang and not runnable on our T4/A100/L4 lab hardware.
- Technique handoff: captured speculative decode + acceptance metrics, custom TP/NVLS, compressed FP4 cache snapshots for transfer, RDMA cache mobility + Redis ownership/revision directory, bounded-replay long-context prefill, and agent-driven profiling/optimization.
- License boundary: upstream is Elastic License 2.0. Do not vendor or mechanically port its code into Native; use provenance-bearing external evidence and independently implement applicable ideas.
- Current ordering unchanged: v1.9 is kept; N4 packed/varlen prefill remains the immediate native task. DeepSeek Ultrafast informs later distributed-cache/routing, speculative/MTP metrics, multi-GPU TP and possible compiled Native-v2 work.


### 2026-11-14 UTC — ChatGPT (llama.cpp v1.0 roadmap integration)

- Reviewed ggml-org/llama.cpp v1.0.0 (13 Nov 2026) and current repo. Added `docs/LLAMA_CPP_INTEGRATION_PLAN.md` as the authoritative track for llama.cpp as (a) an external technique/reference source and (b) a future fourth engine backend for portable/local/edge/quantized inference.
- Immediate native dependency: finish the full `native_static` scorecard first; before N3 fused GDN prefill implementation, study llama.cpp's Qwen3.5 fused-GDN path (PR #32078), scheduler/graph reuse (#32009), and relevant tensor/layout choices. External code is evidence, not InferStack proof.
- Technique candidates recorded in the plan: fused GDN, graph/scheduler reuse, FP8 KV, MoE CPU/GPU expert cache, NextN/speculative API, multi-GPU placement, and GGUF/serving-pack representation.
- Updated `INFERENCE_HARNESS_PLAN.md`, `PORTABLE_EXPERIMENT_EXECUTION_PLAN.md`, and `NATIVE_GAP_ANALYSIS_2026-11-13.md`. llama.cpp is an engine, not a provider; E1.3 RunPod/other provider work remains orthogonal.
- Future adapter: `LlamaCppAdapter` via pinned `llama-server` first, with exact GGUF/build/config identity and fair-comparison rules. No adapter code or performance claim was added in this docs-only change.


### 2026-11-13 UTC — Claude Code (implementation agent, step 1 wired into the server)

- a96cfd6: static decode lane in QwenHybridContinuousEngine (runtime/qwen_static_lane.py). After prefill, rows move into a 16-slot pool (per-row positions, packed live rows, graphs per batch x length bucket captured at start, attention over the live length bucket). Rows that do not fit (prompt + max_new > 4096 or pool full) stay on the Transformers path. Profile production-v1.7.0 (= v1.6.0 + static_decode, strict); scorecard lane native_static.
- Tests (Docker, transformers 5.22.2, CPU eager): 11 decoder tests (mixed lengths, row moves, mask holes, length buckets) and 3 engine tests (lane on == lane off greedy output with early finishes, late arrivals and a long-row fallback; prefix-cache hit; cancellation). Local suite green.
- Scorecard notebook repinned to a96cfd6 with a native_static stage. Needs push, then a full L4 scorecard run; keep v1.7 only if the whole scorecard improves.
- Also: official Colab MCP server (googlecolab/colab-mcp) registered at user scope as colab-proxy-mcp; it drives a Colab tab the owner opens in the browser.

### 2026-11-13 UTC — Claude Code (implementation agent, step 1 second GPU run)

- L4 (ff92334): all 24 cells ran. native_full graph + Triton GDN: 16.4 / 18.3 / 22.2 ms per step at batch 1/4/16 (short prompts), 26.4 at batch 16 with 400-word prompts; vLLM 19.3 at batch 16. Triton GDN beats the torch-GDN graph by 17-20% at batch 16. Agreement 0.984-1.000. Record: docs/verification/STATIC_DECODE_L4_2026-11-13.md (second run section).
- Remaining batch-16 gap grows with prompt length, so full attention over the padded bucket is the next suspect. Next: wire the static decoder into the served engine and run the full L4 scorecard.

### 2026-11-13 UTC — Claude Code (implementation agent, step 1 first GPU numbers)

- L4 (405692e): static graph decode native_unfused 18.4 ms/step vs 43.3 now at batch 1 (2.35x); 1.38-2.35x over batch 1/4/16; greedy agreement 0.996-1.000. Eager static = same speed as now, so the gain is the graph. native_full failed (fusion removes projection modules). Record: docs/verification/STATIC_DECODE_L4_2026-11-13.md.
- ff92334: fused-weight support (test reproduces and fixes the error), Triton grouped GDN decode inside the graph (targets batch 16: 33.5 ms vs vLLM 19.3). Notebook repinned to ff92334; needs push and an L4 rerun.

### 2026-11-13 UTC — Claude Code (implementation agent, native step 1: static decode + CUDA graph)

- Owner decision (replaces the handoff rule "no new native kernels or another optimization version yet"): native optimization is now in scope; goal is a world-class native engine; a change is kept only if it improves the whole scorecard, no single-knob hacks.
- Plan: docs/NATIVE_GAP_ANALYSIS_2026-11-13.md (1 decode CUDA graphs, 2 token-budget prefill scheduling, 3 fused GDN prefill kernel, 4 packed/paged batches).
- Step 1 component done (405692e): src/inferstack/runtime/qwen_static_decode.py; tests/test_qwen_static_decode.py 6 passed in Docker (python:3.12 + torch CPU + transformers 5.22.2), skip locally/CI (no transformers); two injected faults caught.
- GPU check prepared: docs/notebooks/colab_static_decode_bench.ipynb (pins 405692e). Not yet wired into the serving engine; that follows the GPU numbers.

### 2026-11-13 UTC — Claude Code (implementation agent, N2 prefill diagnosis recorded)

- Owner ran colab_n2_prefill_diagnosis.ipynb on L4 (pin 476bf3f); CI for 286d0fa green. Report: docs/verification/N2_PREFILL_DIAGNOSIS_L4_2026-11-13.md, JSON under docs/verification/evidence/n2-l4-20261113/.
- Ranked: (1) 128-token prefill chunks: cohort-1 prefill 978 ms -> 214 ms at one 2048 chunk (4.6x), config-only lever, decode-stall trade-off unmeasured; (2) cohort 4 compute-bound (~7k tok/s), GDN prefill runs the upstream/reference PyTorch chain (bmm, FP32 trsm/sgemm, copies), fused GDN prefill kernel = N3 kernel work; (3) prefix snapshots 1.3%, rejected; (4) served overhead ~160 ms at c=1, ~0 at c=4.
- Next: owner decision. Cheapest: scorecard treatment with larger prefill_chunk_tokens (config only). Larger: N3 GDN prefill kernel (blocked by owner rule).

### 2026-11-13 UTC — Claude Code (implementation agent, N2 prefill diagnosis prepared)

- N2 (NATIVE_ENGINE_OPTIMIZATION_PLAN) started on the largest measured gap, prefill. Diagnosis only: no source/profile change.
- Added scripts/n2_prefill_diagnosis.py (phases: model / prefix-snapshot / other split across cohort 1,4 × chunk 128,512,2048 × prefix cache on/off; ops: torch.profiler table + trace) and tests/test_n2_prefill_diagnosis.py (5 passed; with scheduler tests 16 passed, CPU fake model).
- Notebook docs/notebooks/colab_n2_prefill_diagnosis.ipynb pins 476bf3f; Colab clones from GitHub, so it needs a push before running.
- Code-read suspects to confirm or reject with the run: 128-token prefill chunks, one chunk per scheduler turn with decode priority, full KV snapshot every 128 tokens for single-request cohorts.

### 2026-11-13 UTC — Claude Code (implementation agent, L4 scorecard recorded)

- Owner ran the scorecard notebook on Colab L4 (pin 787c567, run l4-1794553433). Stages: all 4 passed; vLLM metric-only waiver. Fair run: 3 rotated rounds, 12 × 480 requests, 0 failed.
- Recorded: docs/verification/COLAB_L4_SCORECARD_2026-11-13.md plus scorecard summary and manifest under docs/verification/evidence/. Raw ZIP stays a user download (sha256 in the doc).
- Next per PORTABLE_EXPERIMENT_EXECUTION_PLAN §14 step 4: import this result through the E1.2 ResultBundle path without rewriting the raw evidence. RunPod/Modal/K8s and native optimization (N2) wait for owner go-ahead.

### 2026-11-11 UTC — Claude Code (implementation agent, leftover-engine guard)

- Owner interrupted a final run started without vLLM; a server survived holding 6126 MiB, and the next vLLM smoke failed in 59 s (needs 0.92 of the GPU free). Cleared with pkill; GPU back to 0 MiB.
- Notebook: `require_empty_gpu()` runs before every smoke and the final run, killing leftover engine processes and refusing to continue if memory stays above 1 GiB. Final run id is now `<gpu>-<unix time>` so a re-run never collides with an interrupted one; final-run wall cap 12 h on L4.

### 2026-11-11 UTC — Claude Code (implementation agent, vLLM conformance log reviewed)

- conformance-vllm.log (L4, vLLM 0.33.0, max_model_len 8192): healthy. Requests 200, clean stop. Ready only after 857.8 s: vLLM profiled the unused vision encoder with a video item and retried OOM allocations (3.5 GiB) for ~10 min; peak activation 11.57 GiB, KV cache 4.16 GiB = 248,366 tokens (30x at 8192), enough for the workload.
- Notebook: scorecard ready timeout raised from 1200 s to 2400 s (smoke and final run).
- Open fairness/time question for the owner: run vLLM text-only (`--limit-mm-per-prompt {"image":0,"video":0}`) like the native engine. Not applied; it changes the pinned vLLM profile.
- The log does not list /metrics names, so the vLLM inter-token metric name is still unverified.

### 2026-11-11 UTC — Claude Code (implementation agent, vLLM conformance waiver)

- Staged L4 run: native_unfused and native_full smoke PASS. vLLM 0.33.0 with the 8192 cap: ready, deterministic, usage, abandoned-stream and clean-stop checks all pass; only `metric_families_present` failed (missing `engine_tpot_ms`).
- Notebook: a conformance whose only failed check is missing metric families is waived and recorded in waivers.json (the scorecard measures latency client-side). Any other failed check still fails the stage.
- Source: vLLM `engine_tpot_ms` now also accepts `vllm:inter_token_latency_seconds` (assumed new name; not yet verified against a vLLM 0.33 /metrics dump). Takes effect only when the notebook pin moves past 787c567. Related tests: 22 passed.

### 2026-11-11 UTC — Claude Code (implementation agent, scorecard notebook staged one engine at a time)

- Owner request: in one session, bring up and test one engine, then the next. Notebook restructured: stage helpers cell, then 4a native_unfused, 4b native_full, 4c vLLM (install + qualify + smoke), 4d SGLang (same), then 5 = fair rotated 3-round measurement over the profiles that passed, then download.
- Smoke = the exact scorecard argv/env for one profile with a small workload (2 cases x concurrency 1/4, 16 requests); prints PASS/FAIL with the server-log tail, and GPU memory before/after stop to catch leftover processes. A failed stage only needs that cell re-run; stages.json is saved in the ZIP.
- Tested locally against fake engines (pass, failed requests, exit-before-ready, final measurement cell). Pin unchanged (787c567). Not yet run on GPU.

### 2026-11-11 UTC — Claude Code (implementation agent, first L4 scorecard ZIP reviewed)

- ZIP `inferstack-colab-l4-evidence-20261110T160619Z.zip` (sha256 fec73fe2…2cee): NVIDIA L4, commit 787c567, 3 rounds. Only the two native profiles produced measurements.
- Result (L4, measurement only): native_full beats native_unfused in all 16 cells, out tok/s x1.16-x1.51, non-overlapping ranges; decode ~30 vs ~45 ms/tok at concurrency 1. shared_prefix c1 equal-work = False.
- vLLM 0.33.0: conformance failed, CUDA OOM during startup. Conformance launched without a context cap, so vLLM sized for 262144 tokens. Never reached the scorecard.
- SGLang 0.5.23: the Python 3.12 venv fix worked and conformance passed, but every scorecard launch died with `FileNotFoundError: 'ninja'` (FlashInfer JIT). The scorecard PATH lacked the SGLang venv bin; conformance had it.
- Notebook fixes (no runtime code change, pin still 787c567): conformance passes `--max-model-len=8192` / `--context-length=8192` via `--extra`; scorecard PATH = native venv, then engine venv bins. Not yet re-run on GPU.

### 2026-11-10 UTC — Claude Code (implementation agent, SGLang install fix for Colab Python 3.13)

- Owner's L4 run: vLLM installed; SGLang failed building `outlines_core` from source. Cause verified on PyPI: sglang 0.5.23 pins `outlines==0.1.11` (outlines_core 0.1.x), whose latest 0.1.27 has no cp313 wheel; Colab is Python 3.13.
- Fix: the engine-install cell creates the SGLang venv with uv-managed Python 3.12 (replacing a stale 3.13 venv); vLLM unchanged. Cell dry-run locally with installs stubbed: stale venv replaced, 3.12 venv created. Not yet run on Colab.

### 2026-11-10 UTC — Claude Code (implementation agent, scorecard notebook accepts L4)

- Owner could not get an A100 on Colab (only T4/L4). `docs/notebooks/colab_a100_engine_scorecard.ipynb` now accepts A100 or L4 (both BF16, CC >= 8.0); T4 is still refused (no BF16).
- The GPU label (`a100`/`l4`) is written into the output dir, run-context.json, scorecard run id and ZIP name, so an L4 run can never be filed as A100. Scorecard wall-time cap is 10 h on L4 (5 h on A100).
- Source pin unchanged (787c567); no runtime code changed. Nothing A100-specific found in the profile config or native loader (no quantization in either profile). Not yet run on an L4.

### 2026-11-10 UTC — Claude Code (implementation agent, E1.2 notebook round trip)
- `execution/runner.py`: ordered contract `python -m inferstack.execution.runner --manifest --workload --out`: discover environment -> workload fixture sha256 must equal the manifest's -> CompatibilityResolver (unsupported/declared-fallback-only = refused, nothing launched) -> probe_required runs a launch+readiness+largest-case probe only when every probe is one the runner performs -> measured rounds through the N0 scorecard measurement code -> engine version observed vs pinned -> ResultBundle. Gates the runner does not execute (quality/correctness) are `missing`. `--preflight-only` exits 2 on unsupported. Derived vLLM/SGLang launch pins revision, tokenizer revision, dtype, context, TP and profile flags; native needs an explicit template.
- `execution/results.py`: ResultBundle v1 zip (`bundle.json` + raw files). `open_bundle` refuses unsafe names, unlisted/missing members, checksum mismatch, manifest digest/experiment-id mismatch, unexpected manifest and credential-looking text. `import_bundle` stores the zip as an artifact and records HardwareEnvironment, Evidence and (completed + deployment-bound) one Trial per round x case x concurrency; non-passed gates go into trial errors so they can never be proof. Idempotent.
- `execution/notebooks.py` + CLI `python -m inferstack.control experiment-export|experiment-import`: Colab/Kaggle notebook embedding the canonical manifest (digest re-checked in the first cell), private pinned clone, runner venv, exact `engine==version` venv, preflight cell that stops before model download, run, download.
- Tests: `tests/test_portable_roundtrip.py` 11 passed (real fake-engine processes); E1 total 55. Full local suite: 453 passed, 20 skipped (PostgreSQL/Docker/Kubernetes live only), 0 failed. No GPU, no provider.
- Not yet: import of the owner's A100 scorecard ZIP (different, older format; next step once it arrives: preserve raw, wrap into a bundle without rewriting), quality/correctness gates inside the runner, H1 experience reading imported trials.

### 2026-11-09 13:40 UTC — Claude Code (implementation agent, E1.0 + E1.1)
- E1.0 `src/inferstack/execution/manifest.py`: frozen `ExecutionManifestV1` binding pinned record refs (`id@N`) plus a resolved identity snapshot. Two digests: `experiment_id` (model/tokenizer/config revisions, engine version/image, native `source_revision` + H3 `candidate_id`/`patch_sha256`, dtype, KV dtype, profile, hardware requirement, workload fixture/context/concurrency, evaluator, technique refs) and `digest` (also limits, declared fallbacks, lineage). Exact engine versions only; credentials refused; `derive()` makes a new manifest, never edits one; `from_records()` binds existing deployment/model/workload/objective/technique records.
- E1.1 `src/inferstack/execution/compatibility.py`: `resolve(manifest, HardwareEnvironment)` gives supported / supported_with_fallback (only for fallbacks the manifest pre-declared) / probe_required / unsupported with reasons, checked facts, unknowns, probes and alternatives (each alternative names whether it is a new experiment identity). Per-device memory (2x16 GB is never 32 GB without declared TP), precision compute-capability floors (BF16 8.0, FP8 8.9), static model-fit estimate, engine x model support only from conformance evidence, capability matrix, profile schema, technique engine/dtype/CC/VRAM/pinned deps, installed engine version. Proof reuse untouched (stays with `research.techniques.proof_of`).
- Tests: `tests/test_portable_execution.py` 44 passed. Full local suite: 444 passed, 18 skipped, 0 failed. No provider contacted.
- Stashed (not mine, not lost): the other agent's uncommitted edits to `docs/INFERENCE_HARNESS_PLAN.md` and `docs/CLOUD_SERVING_RESEARCH_PLAN.md` conflicted with dc31ae7; kept in `git stash` "other-agent uncommitted doc edits before E1 pull". Untracked research docs left as they were.
- Next: E1.2 notebook round trip (ResultBundle contract, export notebook from a manifest, import with digest/checksum verification). Owner: A100 scorecard run still pending.

### 2026-11-09 12:52 UTC — ChatGPT (portable execution planning)
- Added `docs/PORTABLE_EXPERIMENT_EXECUTION_PLAN.md`: E1 one-manifest/many-substrates plan, CompatibilityResolver, notebook ResultBundle round trip, RunPod-first paid adapter, Modal verified endpoint role, later GPU Kubernetes qualification.
- Added `docs/PORTABLE_EXECUTION_AGENT_HANDOFF.md`: another-agent execution contract for E1.0/E1.1 first, exact suggested files/tests/negative cases and commit sequence.
- Linked E1 into the master harness and cloud-serving plans. E1 is cross-cutting to H3/H4; it does not replace H1-H5 or existing AllocationManager/provider/release infrastructure.
- Immediate owner action remains: run `docs/notebooks/colab_a100_engine_scorecard.ipynb` on A100 and preserve the ZIP. Immediate implementation action: E1.0 ExecutionManifest v1, then E1.1 CompatibilityResolver, with no paid provider contacted yet.
- Public planning prices observed 9 Nov 2026: RunPod A100 80 GB ~US$1.79/h; Verda A100 SXM4 40 GB ~US$1.33/h; Modal A100 40 GB ~US$0.000583/s (~US$2.10/GPU-h) before CPU/RAM. Prices are provenance-bearing planning inputs, never hard-coded compatibility evidence.


### 2026-11-09 11:43 UTC — Claude Code (implementation agent)
- N0 (native campaign): scripts/engine_scorecard.py + configs/scorecard/a100-qwen3.5-2b.json committed (787c567); tests/test_engine_scorecard.py 5 passed (CPU, fake engines).
- Notebook docs/notebooks/colab_a100_engine_scorecard.ipynb pinned to 787c567: native + vLLM + SGLang in separate venvs, conformance first, 3 rotated rounds, ZIP download.
- Waiting on owner: run the notebook on a Colab A100 (~2-4 h) and send the ZIP.
- Next for me while waiting: N2 prep (profiling flow for the largest measured native gap) once the scorecard shows where native loses.

### 2026-11-09 11:25 UTC — Claude Code (implementation agent)
- H3 step 2 committed (acf3848); CI green (6 jobs; provider smoke manual/skipped). Full suite 395 passed / 18 skipped locally.
- Left untouched (another agent's, uncommitted): docs/INFERENCE_HARNESS_PLAN.md native-order paragraphs, docs/NATIVE_ENGINE_OPTIMIZATION_PLAN.md, docs/research/GPU_INFRA_*.json.
- Next: native A100 campaign prep (N0/N1 fair native/vLLM/SGLang comparison notebook) per the plan's execution order.

### 2026-11-09 11:10 UTC — Claude Code (implementation agent)
- Task: H3 step 2 (campaign control) + "investigate this claim" action. In progress, uncommitted.
- Built: optimization/campaign.py (diagnose_outcome, CampaignState, targets); worker loop now plan -> execute -> verify -> diagnose -> replan with lineage; stops on target_met / kept / iteration_limit / budget_exhausted / stagnation / needs_implementation; failed builds are diagnosed outcomes; best verified candidate kept; ProblemDesk.investigate + POST /control/problems/{id}/investigate.
- Evidence so far: tests/test_harness_h3_campaign.py 15 passed.
- Files I am editing: control/worker.py, optimization/{campaign,planner,tools}.py, diagnosis/intake.py, control/api.py.
- CI note: provider smoke is manual-only again (owner decision: paid calls).

### 2026-11-09 10:40 UTC — Claude Code (implementation agent)
- CI fixes. Neon job: run only the PostgreSQL test modules (torch was not installed there); Neon passed 15/15 in CI.
- Live provider smoke failed in CI (model returned non-JSON / invalid plans). Planner: tolerant JSON extraction, explicit stop_rule values, `profile_options` (real allowed keys per engine) in the context, clearer config_change error. Local live runs: OpenRouter and DeepSeek both plan_validated with 0 corrections.
- Flaky crash test: durable-execution test lease 1.0 s -> 2.0 s (a 1.6 s CI stall lost the lease before the crash point).
- Not mine, left uncommitted: docs/INFERENCE_HARNESS_PLAN.md native-engine paragraph (another agent).

### 2026-11-09 10:27 UTC — Claude Code (implementation agent)
- Task: CI uses hosted services. GitHub secrets set: NEON_DATABASE_URL, OPENROUTER_API_KEY, DEEPSEEK_API_KEY (values never printed).
- ci.yml: new job "PostgreSQL tests on hosted Neon" (push/manual; -k postgres; each test creates/drops its own DB); provider smoke now runs on push to main too.
- Note: Neon is us-east-2; from India each query is ~250 ms, so run Neon tests in CI, not locally.
- H3 step 1 CI (c460793): green.

### 2026-11-09 10:21 UTC — Codex (independent progress review)
- Reviewed c460793; production source unchanged by this review. Read H1 memory/context, H2 intake/diagnosis, H3 handoff contracts and recorded A100 qualification scope.
- Independently ran tests/test_harness_h1.py, test_harness_h2.py and test_harness_h3.py with PostgreSQL parametrizations excluded: 39 passed, 0 failed, 0 skipped, 27.628s. JUnit/log: results/progress-check-h1-h3-c460793.*. This is focused CPU/local sandbox verification, not a full source audit or GPU rerun.
- Confirmed exact-commit CI: five successful jobs and one skipped manual provider job (run 42598807688).
- A100 evidence: d33b470 has 27 passing kernel-equivalence checks and native smoke; original report blocked on sandbox skips, amended report qualified using same-commit CI under the recorded policy. All performance lanes remain inconclusive_single_run. Raw archives were user downloads, not independently rerun here.
- Next: finish H3 verifier-driven refinement, controller stop policy and induced-bottleneck demonstration; then H4 FactoryOS-to-release integration and qualification of the final integrated GPU candidate. H5 remains later.

### 2026-11-09 10:09 UTC — Claude Code (implementation agent)
- Task: H3 step 1 (implementer handoff) complete and committed.
- Evidence: tests/test_harness_h3.py 18 passed; full suite 380 passed / 18 skipped; probes 36 passed (macOS, CPU).
- Added: control API /control/implementations/{id} (GET, /patch, /revise, /cancel, /resume); CLI `implementation --id [--diff --source-root]`; serve uses INFERSTACK_SOURCE_ROOT for validation.
- Next: wait for CI, then H3 step 2 (verifier-driven parent/child hypotheses + controller stop policy).

### 2026-11-09 10:03 UTC — Claude Code (implementation agent)
- Task: H3 step 1, implementer handoff (uncommitted, in progress).
- Done so far: new `implementation` record + `src/inferstack/implementation/handoff.py` (ImplementationDesk: controller-owned handoff, journaled attempts, isolated validation, immutable content-addressed candidate; CommandImplementer = coding-agent path in the sandbox). Worker opens a handoff for a needs_implementation plan and resumes from `implementation_ids` with the exact patch. dag binds `implementation_id` to the validated patch. Executor and desk share `materialize_source`. Sandbox.run gained `readable_files`.
- Evidence: tests/test_harness_h3.py 17 passed (macOS Seatbelt).
- Files I am editing (please avoid): implementation/, control/worker.py, optimization/dag.py, execution/{patches,executor,sandbox}.py, contracts/records.py.
- Next: control API + CLI for implementations, full suite, plan status, commit, CI.

### 2026-11-09 09:53 UTC — Claude Code (implementation agent)
- Task: H2 verified on CI (8dfea02 green: py3.11, py3.12, Docker image, kind, lock).
- State: H2 complete. Waiting for owner go-ahead on H3.
- Next: H3 step 1 = implementer handoff contract (stored patch artifact + build manifest, journaled attempts), before any autonomous coding.

### 2026-11-09 09:48 UTC — Claude Code (implementation agent)
- Task: H2 committed (44c9ac9): diagnosis/ intake + problem records; created this file.
- State: verifying CI for 44c9ac9.
- Files: src/inferstack/diagnosis/*, contracts/records.py (Problem), control/api.py, tests/test_harness_h2.py.
- Next: confirm CI green, then start H3 design (implementer handoff contract first).

### 2026-11-14 — llama.cpp L0 implementation study completed

Upstream study pins:

- v1.0.0 release target: 2d8a32502909ca065c41c4cf5f2fb1c427c047fc
- inspected master snapshot: 61dbf49228c8a96b72e40c89d93cb7c003411f8c

Added docs/research/LLAMA_CPP_V1_CODE_STUDY_2026-11-14.md.

Key handoff: N3 is now defined as token-budget scheduling + a single GDN primitive with
separate recurrent/high-token paths, optional direct state-cache writes, then a separate
serving-pack projection-merge treatment. L4/A100 kernel selection must be qualified
independently.

No runtime code was changed. The next owner action remains the complete
native_static serving scorecard; only then should the N3 implementation treatment start.


### 2026-11-14 — production-v1.7.0 full L4 scorecard: keep static decode

- Run `l4-1794639267` at `a96cfd6`, driven with the Google Colab CLI (no browser, no token
  on the runtime: repo shipped as a git bundle). All 5 profiles passed; 3 rotated rounds.
- native_static vs native_full: ahead in 16/16 cells on output tok/s (×1.03–×1.85), decode
  ms/token, E2E p50/p95. TTFT 1 cell behind (prefill_heavy c16, +3%). Peak memory +0.6–1.7 GB.
- Single-stream decode 16.9 ms/token (vLLM 15.8). Remaining gap is prefill: TTFT 4–8× vLLM.
- Decision: keep v1.7. Next: N3 prefill. Writeup: docs/verification/V17_SCORECARD_L4_2026-11-14.md.

### 2026-11-14 — N3 (v1.8 token budget, v1.9 FLA prefill) full L4 scorecard: keep v1.9

- Run `l4-1794662024` at `9979a30`; native_static/native_budget/native_fla/vLLM/SGLang, 3 rounds,
  all stages passed. Caveat: static/budget lanes used FLA through the Transformers wrapper
  (fixed in `65eb66c`), so fla vs budget measures direct vs dispatched FLA.
- v1.9 vs v1.8: prefill_heavy c1 TTFT 344 → 187 ms, c16 tok/s ×1.11; no regression beyond noise.
  v1.8 vs v1.7: TTFT down in all prefill-bound cells; shared_prefix c16 E2E p95 +18% (busy budget).
- vs vLLM: tok/s ×0.79–0.93 interactive, ×0.41 prefill_heavy c16. Gaps: batched prefill, batched decode.
- Decision: keep v1.9. Next: N4 packed/varlen prefill. Writeup: docs/verification/V19_SCORECARD_L4_2026-11-14.md.

### 2026-11-15 — N4.1 backlog-aware busy prefill budget (production-v1.10.0): fast check, promising

- Cause from the v1.9 run: prefill_heavy c16 TTFT rose in ~370 ms steps of 4-5 requests; while
  decoding, prefill was capped at 1024 tokens per forward. v1.10 lets a forward take up to
  ceil(queued prefill tokens / 4), between the busy and idle budgets.
- First on-runtime test run found a real bug: with FLA installed, transformers swapped FLA into
  the recurrent reference GDN (and into CPU test tensors). Fixed in `70e9aad`; 42 CPU engine tests
  + 2 CUDA FLA tests pass on the L4 (first time these engine tests ran anywhere: no local torch).
- Fast check `70e9aad` (c1/c16, 2 rounds) v1.10 vs v1.9: prefill_heavy c16 TTFT 2188 → 1370 ms,
  tok/s ×1.11, E2E p95 6155 → 5468 ms; decode 75 → 88 ms/tok there (prefill now takes the step).
  shared_prefix c16 tok/s ×1.07; every other cell within noise. Evidence:
  docs/verification/evidence/n41-fast-check-l4-70e9aad-20261115/.
- Decision: candidate; full 3-round scorecard together with N4.2 (packed varlen prefill).

### 2026-11-15 — N4.2 fused GDN conv (v1.11), N4.3 prefill-first (v1.12), startup warmup

- Prefill diagnosis (L4, B=16 x ~1560 tokens): pure compute, linear in batch (~11.6k tok/s);
  GEMMs ~half and already ~66 TFLOPS; the rest overhead (copies 269 ms, torch conv 162 + SiLU 48 ms,
  padded mem-efficient attention 117 ms). Evidence: n42-prefill-diagnosis-l4-89ae361-20261115/.
- v1.11: GDN prefill conv on FLA's fused conv+SiLU over [B, T, D] (kernels/gdn_conv.py, reference
  oracle). Pure prefill B=16 2203 -> 1918 ms; copies -> ~77 ms. Only for forwards >= 1024 tokens
  (wrapper cost hurt short-prompt TTFT by 4-15 ms). GPU engine test: greedy outputs identical.
- v1.12: prefill-first draining (idle budget 16384, drain_steps 1). n44 vs v1.11: prefill_heavy c16
  TTFT 1313 -> 778 ms, tok/s x1.15, E2E p95 5416 -> 4098; shared_prefix c16 x1.07; c1 unchanged.
- Cold-start stall: first large prefill after startup paid ~15 s of Triton/FLA compile+autotune
  (n42b/n43 moderate c16 TTFT max 16 s). prefill_warmup at startup removed it (n44 worst TTFT 2.3-3.4 s,
  prefill_heavy c16 tails); ready time 17 -> 18 s.
- Next: full 3-round scorecard v1.12 vs v1.9 vs vLLM vs SGLang.

### 2026-11-15 — production-v1.12.0 full L4 scorecard: keep v1.12

- Run `l4-1794715458` at `ff48390`: native_fla (clean v1.9) / native_drain (v1.12) / vLLM / SGLang,
  3 rounds, all passed, engine tests passed on the runtime.
- v1.12 vs v1.9: prefill_heavy c16 tok/s x1.28, TTFT 2210 -> 794 ms, E2E p95 5867 -> 4161; c8 x1.14;
  no cell worse outside the round ranges.
- vs vLLM: x0.92 interactive c1 down to x0.51 prefill_heavy/shared_prefix c16. Largest gap now
  batched decode (interactive c16 24.7 vs 19.4 ms/tok; shared_prefix c16 50.5 vs 21.3).
- Decision: keep v1.12. Next: decode step at batch 16, then mixed prefill+decode forwards.
  Writeup: docs/verification/V112_SCORECARD_L4_2026-11-15.md.

### 2026-11-15 — N5.0 decode step + N6.0 MTP acceptance (parallel L4s)

- N5.0 (static decode step, v1.12): B1 16.3-17.3 ms (~bandwidth floor); B16 22.0 / 24.1 / 34.3 ms at
  55 / 420 / 1250 tokens of context. Attention ran masked mem-efficient SDPA over the whole length
  bucket with GQA re-reads -> N5.1 split-KV decode attention kernel (140752c), GPU tests pass.
- N6.0 (Qwen3.5-2B's own MTP head, 15 mtp.* tensors transformers drops; teacher-forced, greedy):
  draft-1 acceptance 0.96 interactive / 0.93 moderate / 0.92 prefill_heavy / 0.86 shared_prefix
  with the post-final-norm hidden state (pre-norm: 0.17-0.70). Chained drafts 2-3: 0.15-0.50.
  Expected tokens per verify step with 1 draft: 1.86-1.96. Caveat: scorecard prompts are word lists,
  continuations may be easier than chat. Evidence: n60-mtp-acceptance-l4-7280c6a-20261115/.
- Decision: build N6.1 (one-draft MTP speculative decode in the static lane, lossless greedy).

### 2026-11-15 — N5.1 decode attention (production-v1.13.0): served fast check, strong candidate

- n52 (3d3a884, L4, 2 rounds) v1.13 vs v1.12 tok/s: shared_prefix c16 x1.37 (269 -> 370; decode
  47.4 -> 35.6 ms/tok), moderate c16 x1.23 (decode 29.7 -> 23.5), prefill_heavy c16 x1.20 (E2E p95
  4523 -> 3655), interactive c16 x1.12; c1 cells x1.01-1.06. No cell worse outside its range.
- Against the v1.12 full run's rivals (same L4 image), v1.13 c16 tok/s vs vLLM: shared_prefix ~0.70
  (was 0.51), moderate ~0.83 (0.67), interactive ~0.80 (0.75), prefill_heavy ~0.60 (0.51).
- Next: N6.1 MTP speculative decode; full scorecard for v1.13 (+N6.1 if ready).

### 2026-11-15 — N6.1 MTP speculative decode lane: model-step bench, x1.19-x1.42 tok/s

- Code 0e334d1: runtime/qwen_spec_decode.py (verify [last, draft] in one graph; GDN state after
  token 1 saved and restored on rejected rows; MTP layer with its own KV cache drafts from the last
  accepted position; the graph writes the next step's inputs). L4 run: engine tests incl. 5 new
  forced right/wrong/mixed-draft tests (equal to Transformers greedy) passed; FLA tests passed.
- scripts/spec_decode_bench.py, v1.13 kernels (triton attention + GDN), 128 new tokens, plain static
  lane vs spec lane, both graph replay:

  | batch | words | plain ms/step | spec ms/step | tokens/step | tok/s plain -> spec | x |
  |---|---|---|---|---|---|---|
  | 1 | 48 | 16.3 | 22.8 | 1.97 | 61 -> 86 | 1.40 |
  | 4 | 48 | 17.9 | 24.9 | 1.97 | 223 -> 315 | 1.42 |
  | 16 | 48 | 20.7 | 33.8 | 1.97 | 772 -> 932 | 1.21 |
  | 1 | 400 | 16.4 | 23.0 | 1.92 | 61 -> 83 | 1.37 |
  | 4 | 400 | 18.1 | 25.1 | 1.92 | 221 -> 305 | 1.38 |
  | 16 | 400 | 21.1 | 34.4 | 1.94 | 758 -> 905 | 1.19 |

- Acceptance 0.92-0.97 in-lane (N6.0 offline estimate 0.86-0.96).
- Agreement with the plain lane: 38/42 rows identical over 128 tokens. 4 rows differ from token ~6
  (bf16 2-token vs 1-token forward; to be checked against Transformers greedy).
- The step costs +6.5 ms at B1 and +13 ms at B16 over a plain step. Suspects:
  - MTP lm_head read: 1 GB, ~3.4 ms;
  - GDN state save/restore traffic: ~0.9 GB at B16;
  - two attention calls per layer.
- Evidence: docs/verification/evidence/n61-spec-l4-0e334d1-20261115/.
- Next:
  - N6.1b: cut step overhead (MTP draft vocab, bank-swapped GDN state instead of copy+where,
    2-query attention kernel);
  - check divergence vs Transformers;
  - N6.2: wire into the served static lane behind a profile flag.

### 2026-11-15 — N6.1b/c: banked GDN state + pruned draft vocab; spec output = Transformers greedy

- n61b (17de60a, L4): tests passed (banked GDN kernel vs reference, spec lane, static decode).
  The spec lane matched Transformers greedy on **42/42 rows** over 128 tokens.
- The earlier "drift" was a bench artifact in the *plain* lane. Attention scratch was keyed by
  id(st) and never released, so a later state reused a stale smaller scratch (plain lane at B16
  w400 differed from Transformers at token 1 on 11/16 rows). Fixed: release() drops that state's
  scratch, with a regression test.
- Step overhead vs plain (B1 / B16): full-vocab draft +6.0 / +8.0 ms; 65536-id draft +3.0 / +4.9 ms.
  Banked GDN removed most of the B16 cost (was +13 ms in N6.1).
- Draft vocab: 65536 keeps acceptance equal to the full vocab (0.92-0.97). 32768 / 16384 drop to
  0.48 on the 400-word prompts.

  | batch | words | plain tok/s | spec (draft vocab 65536) tok/s | x |
  |---|---|---|---|---|
  | 1 | 48 | 61 | 102 | 1.65 |
  | 4 | 48 | 222 | 381 | 1.71 |
  | 16 | 48 | 767 | 1227 | 1.60 |
  | 1 | 400 | 61 | 98 | 1.61 |
  | 4 | 400 | 220 | 368 | 1.67 |
  | 16 | 400 | 756 | 1186 | 1.57 |

- n61c (11aab11, second L4 in parallel): the MTP history window barely matters. With MTP entries
  only for the last 4 prompt positions, acceptance stays within 0.00-0.07 of the full history.
  So serving can draft from the last prefill forward's hidden states alone.
- Evidence: n61b-spec-l4-17de60a-20261115/, n61c-window-l4-11aab11-20261115/.
- Next: N6.2 served spec lane (profile flag, draft vocab 65536, window 256), then the full scorecard.

### 2026-11-15 — N6.2 served spec decode (production-v1.14.0): fast check x1.21-x1.61 tok/s

- n62 (c891340, L4, 2 rounds, workload-fast) v1.14 `native_spec` vs v1.13 `native_dattn`. All 34
  engine tests passed on the runtime, including 3 new engine-level spec-lane tests.

  | case | c1 tok/s | c16 tok/s | c1 decode ms/tok |
  |---|---|---|---|
  | interactive | 58.2 -> 93.8 (x1.61) | 638 -> 867 (x1.36) | 16.8 -> 10.3 |
  | moderate | 57.4 -> 92.1 (x1.61) | 591 -> 830 (x1.40) | 17.0 -> 10.4 |
  | prefill_heavy | 45.8 -> 59.5 (x1.30) | 166 -> 162 (x0.98) | 17.1 -> 11.6 |
  | shared_prefix | 54.1 -> 73.8 (x1.36) | 335 -> 406 (x1.21) | 17.1 -> 12.6 |

- c1 decode is now 10.3 ms/token. vLLM was 15.8 in the v1.12 full run.
- Watch:
  - interactive c16 TTFT 110 -> 193 ms. Per-row spec admission is eager, with syncs.
  - prefill_heavy c16 is unchanged; it is prefill-bound (N5.2 mixed forwards).
- Evidence: n62-served-l4-c891340-20261115/. Next: full 3-round scorecard with vLLM/SGLang.

### 2026-11-15 — v1.14 full scorecard: native leads output tok/s in 12 of 16 cells

- Run `l4-1794740111` at `08578c4`: native_dattn / native_spec / vLLM / SGLang, 3 rounds, all passed.
- v1.14 vs the best rival tok/s:
  - interactive x1.17-x1.48 (all c);
  - moderate x1.08-x1.44 (all c);
  - prefill_heavy c1 x1.14, but c4-c16 x0.55-x0.79;
  - shared_prefix c1-c4 x1.14-x1.23, c8 x1.01 (tie), c16 x0.73.
- v1.14 vs v1.13: tok/s up in every cell (x1.02-x1.62).
- Regressions: TTFT at interactive c4-c16 (c16: 107 -> 193 ms), from eager per-row spec admission.
- Decision: keep v1.14. Writeup: docs/verification/V114_SCORECARD_L4_2026-11-15.md.
- Next:
  - N6.3: batched spec admission (TTFT);
  - N5.2: mixed prefill+decode (prefill_heavy / shared_prefix c16);
  - c1 TTFT (65 vs 41 ms).

## 2026-11-15 — N6.3 batched spec admission: lossless, no TTFT win

- Code: `QwenSpecDecoder.load_spec_rows` (one right-padded MTP forward for all admitted rows, first
  tokens stay on GPU), used by `StaticDecodeLane.try_admit`. Commits a20368b, b1e7a36.
- L4 fast check `l4-1794754092` @b1e7a36 (native_spec, 2 rounds, default workload): all engine tests
  passed (spec decode incl. batched == per-row load, static lane, banked GDN).
- Result vs v1.14 run @08578c4: output tok/s and TTFT unchanged within noise in 15/16 cells
  (interactive c16 TTFT 485 ms [179–791], one bad round). Per-row admission syncs were not the
  TTFT cause. Evidence: docs/verification/evidence/n63-batched-admit-l4-b1e7a36-20261115/.
- Next: instrument the request timeline (queue wait, prefill, first emit, admission) before
  another TTFT change.

## 2026-11-15 — TTFT split (queue / wait / prefill), v1.13 vs v1.14, L4 @77880b5

- One round per profile. Means per request (not p50). Evidence:
  docs/verification/evidence/ttft-split-l4-77880b5-20261115/.
- **Prefill dominates.** At c1 the prefill stage alone is 57 ms (interactive, ~60-token
  prompt) and 117 ms (moderate), while the rivals' whole TTFT is 41–48 ms. That is per-forward
  overhead, not compute.
- **Under load, queue + wait grow in v1.14**, e.g. prefill_heavy c16: queue 204, wait 275,
  prefill 416 ms. The extra decode steps between prepare and the first token are 1.5–2.6.
- **Decision:** P1 (packed prefill into the pool, own forward) is the main lever. It also cuts
  launches, the c1 cost. P3 scheduling follows for queue/wait.

## 2026-11-16 — P1 packed prefill into the static pool (v1.15 `native_packed`), L4

- Code: `QwenStaticDecoder.prefill_packed` (flat tokens; FLA varlen conv + chunked delta rule
  with cu_seqlens; per-segment sdpa writing pool K/V) and `StaticDecodeLane.admit_packed`,
  behind `static_decode_packed_prefill`. Profile `production-v1.15.0.toml` = v1.14 + P1.
- Tests (L4): CPU suite 23 passed (tiny model, packed vs Transformers logits within 1e-4,
  plain and spec engine paths greedy-equal); FLA suite 21 passed incl. FLA varlen vs
  per-segment reference.
- Fast scorecard, 1 round, @b3490e9 (measurement only). TTFT p50 native_spec → native_packed:
  interactive c16 786 → 89 ms, moderate c16 769 → 136, prefill_heavy c16 887 → 359,
  moderate c1 125 → 73. Out tok/s at c16 +19–32%. **Regression:** shared_prefix c16 TTFT
  133 → 466 ms, tok/s 419 → 245, because packed rows store no prefix snapshots (P2).
- shared_prefix outputs differed in length (equal_work False at c16). Real-model check
  (`scripts/packed_prefill_check.py`, 64 greedy tokens vs unpadded Transformers): rows fully
  matching, old vs packed: interactive 7/8 vs 7/8, shared_prefix 6/8 vs 5/8, prefill_heavy
  3/4 vs 2/4; first-token max |logit diff| 0.13–0.14 vs 0.13–0.16. Same error scale: bf16
  near-tie flips, not a bug. Packed may drift slightly earlier on long prompts (small sample).
- Evidence: docs/verification/evidence/p1-packed-prefill-l4-20261116/.
- Next: P2 (prefix cache in pool format) to recover shared_prefix, then a 3-round scorecard.

## 2026-11-16 — P2 prefix cache in pool format, L4 @6cc1d88

- Code: `runtime/pool_prefix_cache.py` (slot K/V + GDN conv/rec snapshot at 128-token block
  boundaries), `prefill_packed(starts=)` continuation, cohort prefix sharing in
  `StaticDecodeLane._prefix_plan`. A lone prompt stores its prefix only if one of the last 64
  packed prompts shared it (first P2 @aeac3dd stored every cold prompt: moderate c1 TTFT 73 →
  144 ms; fixed).
- Tests (L4): CPU suite and FLA suite pass (continuation vs from-scratch, snapshot restore,
  slot copy, cohort sharing, plain and spec engine paths greedy-equal).
- Fast scorecard, 1 round, TTFT p50 native_spec → native_packed (v1.15 = P1+P2):
  interactive c1 67 → 73, c16 805 → 99; moderate c1 132 → 82, c16 772 → 164;
  prefill_heavy c1 172 → 138, c16 864 → 366; shared_prefix c1 76 → 86, c16 141 → 166
  (was 466 with P1 alone). Out tok/s c16: +38%, +11%, +17%, −6% (shared_prefix).
- Evidence: docs/verification/evidence/p2-pool-prefix-l4-6cc1d88-20261116/.
- Next: 3-round scorecard with vLLM/SGLang to decide keep v1.15.

## 2026-11-16 — v1.15 full L4 scorecard (3 rounds, with vLLM/SGLang) @6186a56: keep v1.15

- v1.15 vs v1.14: TTFT better in 13/16 cells (prefill_heavy c16 894 → 350 ms, moderate c16
  434 → 157); tok/s better in 6, equal in 7, worse on shared_prefix c4/c8/c16 (−12/−15/−5%,
  decode interrupted by admission-time prefills).
- vs rivals: v1.15 leads output tok/s in 11/16 cells; SGLang still leads prefill-bound TTFT and
  prefill_heavy c4–c16 throughput; c1 TTFT 73–137 vs 43–54 ms.
- Doc: docs/verification/V115_SCORECARD_L4_2026-11-16.md. Next: P3 pacing for shared_prefix,
  then P4 prefill graphs.

## 2026-11-16 — P3 diagnosis: shared_prefix c4/c8, L4 @a9b7164 (1 round, shared_prefix only)

- Prefix cache works: native_packed c8 had 32/32 pool-cache hits (36,864 prompt tokens skipped);
  TTFT mean 105 ms vs 328 ms for native_spec; decode ms/token equal at c4 (18.1 vs 17.9).
- In this run native_packed out-tok/s *beat* native_spec (c4 203 vs 155, c8 263 vs 225), the
  opposite of the full run (205 vs 232, 258 vs 302). shared_prefix c4–c8 throughput swings with
  run conditions; the full-run "regression" is not reproduced in isolation.
- Clear cost: a packed admission takes 79–87 ms even on a cache hit with a short suffix. That is
  per-forward overhead (Python + per-layer launches + per-segment conv calls), the same thing
  behind the c1 TTFT gap (73 vs 43 ms). Target: P4 (cheaper small prefills), before pacing.
- Evidence: docs/verification/evidence/p3-diag-shared-prefix-l4-a9b7164-20261116/.

## 2026-11-16 P4 profile (L4, e836a32 then f6164b6)
- Small packed prefill is host-bound: c1 cache hit (48 new tokens) 68 ms wall, ~21 ms GPU kernel time, ~1200 launches; FLA chunk_gated_delta_rule ~1.2 ms Python per call x 18 layers.
- FLA varlen causal_conv1d takes per-sequence initial_state (equal to per-segment calls, diff 0.0). Continuation path now one conv call: c4 hit 96 -> 74 ms, c8 hit 133 -> 83 ms. FLA tests pass on L4.
- FLA under torch.cuda.graph: capture works, replay 0.09 ms vs eager 1.58 ms, but cu_seqlens is read on host (ops/utils/index.py tolist) so the segment split is frozen at capture. Graphs need a fixed split per bucket (rows padded to a bucket length).
- Evidence: docs/verification/evidence/p4-profile-l4-f6164b6-20261116/

## 2026-11-16 P4 graphed small prefills (L4, d069bca)
- Inert padding (beta=0, g=0) probe: outputs equal, final state diff <= 3e-8.
- prefill_graphed: k<=8 rows x L in {16,32,64,128}, captured at warm-up. Tests incl. graphed-vs-eager (logits, hidden, K/V, conv/rec, mask; fresh, continuation, 2nd replay) pass on L4.
- Wall ms eager -> graphed: c1 40 tok 62 -> 24; c1 hit 1152+48 68 -> 25; c4 40 tok 67 -> 40; c4 hit 74 -> 41; c8 hit 83 -> 66. Unbucketed shapes unchanged.
- Next: 1-round scorecard native_packed vs native_graphs (equal-work check), then full scorecard.

## 2026-11-16 P4 crash fix + 1-round scorecard (L4, 9ac7fb1)
- v1.16 smoke crashed (illegal address on the first graphed prefill). Cause: FLA's identity tensor_cache (4 entries) held cu_seqlens-derived index tensors the graphs read but did not own; eager calls evicted them, memory was reused. Fix: each graph keeps the cached tensors alive. Test now captures 2 graphs, churns caches/allocator, then compares graphed vs eager. Scheduler crashes now print a traceback.
- 1 round, native_packed (v1.15) -> native_graphs (v1.16), equal work in all 16 cells, 100% success:
  TTFT p50 interactive c1 71->31, c8 78->42; shared_prefix c1 81->39, c4 88->50, c8 88->48, c16 128->90; prefill_heavy c8 228->153.
  tok/s up in 14/16 cells, shared_prefix c4/c8/c16 +20/+24/+20%. moderate c1 unchanged (prompts > 128-token bucket).
- Next: full 3-round scorecard with vLLM/SGLang (needs approval, ~10 CU); rival study (how vLLM/SGLang graph GDN prefill).

## 2026-11-16 18:45 IST — v1.16 full scorecard (3d08aa0) done: KEEP
- 3 rounds, equal work, 100% success. v1.16 ≥ v1.15 tok/s in all 16 cells (13 ahead, 3 overlap).
- vs vLLM/SGLang: ahead 12/16; behind prefill_heavy c4/c8/c16 + shared_prefix c16 (decode stalls behind long prefill: 61 vs 21.6 ms/tok at prefill_heavy c16).
- Next: P5 chunked prefill interleaved with decode (rival study first). Doc: docs/verification/V116_SCORECARD_L4_2026-11-16.md

## 2026-11-16 ~22:50 IST — P5 v1.17 (125e923): KEEP as candidate
- Rival study: SGLang offline prefill within 5% of ours; prefill_heavy gap = cross-cell prompt repeats served by rivals' radix/APC caches.
- v1.17 = per-layer sync removal + first-sight prefix snapshots (128 entries / 3 GB). vs v1.16: prefill_heavy c4/c8/c16 +28/+19/+20% tok/s; prefill_heavy c1 −3%, cold c1 TTFT +16–24 ms (two-pass snapshot).
- Next: single-pass snapshot (record GDN state at block boundary inside the one prefill). Doc: docs/verification/V117_RADIX_SCORECARD_L4_2026-11-16.md

## 2026-11-17 — P5 v1.18 single-pass snapshot (134f3d1): on L4
- Pushed v1.17 (26bed35). v1.18 = v1.17 + `prefix_cache_single_pass_snapshot`: a lone cold prompt's snapshot is recorded inside its one prefill (FLA chunk-start state h at the block boundary; `kernels/fla_gdr_states.py` mirrors fla-core 0.6.0 forward, MIT).
- Local suite: 476 passed (Qwen3.5 tests skip locally). Colab run `p5-sp-134f3d1`: FLA tests + native_radix vs native_single, 3 rounds.
- 2026-11-17 ~08:00 IST — v1.18 run `p5-sp-0aed0e2` done: FLA tests pass on L4; vs v1.17 TTFT down in 9 cells (moderate c1 95→75, prefill_heavy c16 262→219), tok/s ahead 7 / overlap 8 / behind ×0.99 1. KEEP. Doc: docs/verification/V118_SINGLE_PASS_SNAPSHOT_L4_2026-11-17.md
- 2026-11-17 — P6 host-time work. Diag f9103d1: host after each spec sync = 676 ms of 7.0 s in prefill_heavy c16 (~9 ms/step). emit_batched (v1.19, 3278e44): no effect, all 16 cells overlap — DROP. Tail-fill compaction (v1.20, 40cff0e): host/step 8.9→2.0 ms (prefill_heavy c16), tok/s ahead ×1.02–1.07 in prefill_heavy c8/c16 + shared_prefix c8/c16, rest overlap; BUT TTFT p50 up in 6 cells (shared_prefix c16 82→160 ms, prefill_heavy c16 233→294) via queue time (39→89 ms). HOLD: find why queue grows before keeping. Evidence: docs/verification/evidence/p6-{emit,tail}-l4-*.
- 2026-11-17 — 3-way (821c688): v1.20.1 (tail-fill + max 4 rows/prefill) fixes shared_prefix c16 TTFT p50 (153→85 ms) keeping +6% tok/s vs v1.18; prefill_heavy c16 TTFT still 279 vs 221 ms (tok/s +4%); interactive c16 tok/s −3%. No single winner: tuner (phase 2, scripts/slo_tune.py) should pick per SLO.
- 2026-11-17 15:05 — first tuner run (91161cb, 7 trials, Optuna TPE, from v1.18): none meets SLO; best trial 4 (8 rows/prefill, wait 0, tail-fill, spec on) 280 geo tok/s, 9/16 cells on SLO v2 (= v1.18). Spec off costs ~30%; 6 ms batch wait hurts. Settings alone cannot close the long-prompt cells -> P7 probe next.
- 2026-11-17 — P7 probe (0a485bd): decode step 34 ms alone; during a 1x2k prefill on a 2nd stream/thread 87 ms (2 steps), 2x2k 123 ms (3 steps); prefill slows 1.15-1.28x. Below gate (35 ms). Testing high-priority decode stream next.
- 2026-11-17 — P7 probe 2 (944135b): high-priority decode stream keeps decode at 39.6/41.4 ms p50 (alone 34.8) during a 1x2k / 2x2k prefill on a 2nd stream; 5 and 11 decode steps served during the prefill; prefill slows 1.55x / 1.64x. Default priority: 87/121 ms. Combined work ~30% faster than taking turns. GO for P7 build behind a flag.
- 2026-11-17 16:30 — P7 v1 A/B (6970392, 3 rounds, tests pass): decode p95 down (prefill_heavy c16 80.6->48.8 ms, c8 46.9->30.6) but TTFT p95 up (c16 480->2139 ms) and tok/s down (prefill_heavy c4/c16 -24%/-17%, shared_prefix c4/c16 -8%/-7%). SLO v2: 7/16 vs v1.18 9/16. Cause: one job at a time into 2 spare slots throttles admission (queue builds). DROP v1 config; v2 = more spare slots, multi-row jobs, sync admission when queue grows.
- 2026-11-17 19:30 — pushed 873a424 (P7 v2 code, driver reclaimed-VM fix). P7 v2 A/B blocked: Colab reclaimed the VM mid-run (18:25), then 'Service Unavailable' for L4 (19:15). AutoKernel study written (docs/research/AUTOKERNEL_STUDY.md). test_durable_execution remains timing-flaky on a loaded Mac (lease 2 s); not touched.
- 2026-11-17 22:00 — P7 v2 A/B (b1af725, tests pass): prefill_heavy c16 per-token p95 82->51 ms, tok/s +4%, but TTFT p95 499->947; prefill_heavy c4 TTFT 240->751, tok/s -22%; shared_prefix rounds 0-1 crashed (Triton autotuner not thread-safe: sync short-prompt prefill ran concurrently with the overlap job) -> 503s. SLO v2: v1.18 8/16, v1.22 6/16. NOT kept. Fix: no sync prefill while a job runs. P7 parked behind its flag; next P8/P9.
- 2026-11-17 23:20 — P8 A/B (66b0848, tests pass incl. long-graph snapshot test): moderate c1 TTFT p95 76->60 ms (rivals 47), moderate c4 TTFT p50 67->59; tok/s overlap in 14 cells, moderate c1 ahead x1.01, moderate c4 behind x0.99; moderate c16 TTFT p50 112->142 to watch. SLO v2 9/16 both. KEEP v1.23 as candidate; next: profile the 500-token graph replay to find the remaining ~13 ms.
- 2026-11-18 — P8 admit-stage diag (3db7e87, 1 round fast workload, synced stages): moderate c1 TTFT 56 ms = queue 3.6 (batch_wait_ms 3) + prefill graph 45.5 + spec rows 6.2 + ~1 rest; interactive c1 graph prefill 23 ms for ~60 tokens (kernel-latency floor). Next: batch_wait_ms 0 (tuner trial 4 agreed) and graph the MTP spec-row load (~6 ms).
- 2026-11-18 06:05 — v1.24 (batch_wait 0) vs v1.23: c1 TTFT -3 ms (interactive 32->29, moderate 61->58) but burst cells worse (interactive c16 TTFT p95 107->146, c4 67->93); SLO v2 9->7. NOT kept. Next: v1.25 burst-aware wait (wait only if previous arrival < 20 ms ago).
- 2026-11-18 07:00 — v1.25 burst-aware wait vs v1.23: c1 TTFT -3 ms kept (interactive 33->29, moderate 63->59) but the first arrival of a burst is a 'lone' request, so bursts split into two prefills: interactive c16 TTFT p95 108->166, prefill_heavy c4/c8 p95 135->240 / 247->368. SLO v2 8/16 both. NOT kept; the 3 ms wait stays (cheap insurance). Next c1 lever: spec-row load on a graph (~6 ms).
- 2026-11-18 08:40 — P9 cost at SLO (closed loop, $0.80/h L4): InferStack cheapest in 3/4 classes (interactive -18%, moderate 2.2x vs vLLM, prefill_heavy -12%); shared_prefix rivals 37% cheaper (we are capped at c8 by TTFT p95). Doc: docs/verification/P9_COST_AT_SLO_L4_2026-11-18.md
- 2026-11-18 11:20 — v1.26 (lane budget counts uncached tokens) vs v1.23: no change (shared c16 TTFT p95 358->360); burst still split 6+7+3 because the engine gatherer caps rows x longest prompt <= max_batch_tokens (8192). Next: v1.27 = max_batch_tokens 32768 + lane re-queues budget overflow instead of the Transformers path.
- 2026-11-18 12:10 — v1.27 vs v1.23: shared_prefix c16 TTFT p95 352->255 ms, tok/s +9% (now priced at c16: $0.0152/1k = rivals); side effect prefill_heavy c8 TTFT p95 240->380 (cold bursts in one 16k-token prefill). Next v1.28: uncached budget 8192.
- 2026-11-18 12:25 — v1.28 vs v1.23: shared_prefix c16 TTFT p95 354->257, +9% tok/s, $0.0152/1k (= rivals); prefill_heavy c4 TTFT p95 235->133; prefill_heavy c8 p95 242->480 (watch). KEEP v1.28 as candidate. Cheapest-or-level in all 4 classes.
- 2026-11-18 12:45 — prefill_heavy c8 tail in v1.28: cold ~2k-token prompts admitted 3-4 per prefill under the 8192 uncached budget (TTFT 383/480 ms per group; v1.23 groups of 1-2). v1.29: budget 4096 (<=2 cold rows; hit bursts still one prefill).
