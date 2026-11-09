# Agent handoff — E1 Portable Experiment Execution

Purpose: give another implementation agent enough context to execute the next InferStack infrastructure milestone without re-planning the product.

Date: 9 November 2026.
Authoritative product plans:
- `docs/INFERENCE_HARNESS_PLAN.md`
- `docs/PORTABLE_EXPERIMENT_EXECUTION_PLAN.md`
- `docs/CLOUD_SERVING_RESEARCH_PLAN.md`
- `docs/ARCHITECTURE.md`

## 1. Current repository state you must respect

At the time this handoff was written, the repository already has:

- H0 accepted.
- H1 experiment memory implemented.
- H2 typed intake/diagnosis implemented.
- H3 step 1 implementer handoff implemented.
- H3 step 2 bounded campaign control implemented.
- A recorded Colab A100 qualification for an older pinned native/Qwen3.5 environment.
- A newer fair-engine scorecard notebook waiting for a real A100 run:
  `docs/notebooks/colab_a100_engine_scorecard.ipynb`.
- Docker and Kubernetes providers behind the infrastructure allocation contract.
- GPU inventory collection, allocation ownership, quotas, reconciliation, pricing records, model-cache support and release/rollout infrastructure.

Do not recreate those systems.

Before changing code:
1. read `docs/AGENT_PROGRESS.md`;
2. inspect HEAD and CI;
3. check for active/uncommitted ownership notes;
4. preserve the existing acceptance probes and H0/H1/H2/H3 contracts.

## 2. Goal

Build **E1 — Portable Experiment Execution**.

The same pinned inference experiment must be describable once, checked against actual hardware, exported to a notebook/provider, executed, returned as a verifiable ResultBundle, and imported into the existing Trial/Evidence/Memory/Verifier path.

The first implementation slice is **E1.0 + E1.1 only**:

1. ExecutionManifest v1
2. CompatibilityResolver

Do not start paid-provider code before these contracts pass.

## 3. Architectural rule

Do not create a second remote-GPU control plane.

Reuse:

- `contracts/records.py`
- `infrastructure/inventory.py`
- `infrastructure/allocations.py`
- `infrastructure/remote.py`
- `engines/capabilities.py`
- `engines/profiles.py`
- the existing artifact/store/control contracts

The ExecutionManifest is a **resolved binding layer over existing records**, not a replacement for them.

Recommended new files:

```text
src/inferstack/execution/manifest.py
src/inferstack/execution/compatibility.py
src/inferstack/execution/results.py        # contract shell only if useful for E1.0
tests/test_portable_execution.py
```

Avoid schema migration unless persistence cannot be achieved through the current artifact/reference system. Prefer a versioned Pydantic artifact contract first.

## 4. E1.0 — ExecutionManifest v1

### Required properties

The manifest contains:

- schema version
- model/deployment/workload/objective/technique refs
- resolved immutable model/config/tokenizer identity
- engine name/version/image/plugin identity
- dtype/precision
- max context and generation limits
- concurrency/workload fixture identity
- engine profile/options
- hardware requirements
- evaluator/measurement policy identity
- runtime/GPU/token/cost limits
- optional campaign/plan/implementation lineage

It must not contain credentials.

### Canonical digest

Implement deterministic canonical JSON.

Acceptance:

- same input -> same digest;
- map ordering does not alter digest;
- model revision change -> new digest;
- tokenizer revision change -> new digest;
- engine version/image change -> new digest;
- dtype change -> new digest;
- workload fixture/evaluator/hardware requirement change -> new digest;
- credentials cannot be serialized.

### Do not duplicate mutable truth

References remain references. The resolved snapshot exists to prove what was executed.

If an existing record has a newer revision after manifest creation, the old manifest remains unchanged.

## 5. E1.1 — CompatibilityResolver

Recommended API shape (adjust names to repository conventions):

```python
resolve_compatibility(
    manifest: ExecutionManifestV1,
    environment: HardwareEnvironment,
    *,
    capabilities=...,
    profile_schema=...,
) -> CompatibilityResult
```

Use states:

```text
supported
supported_with_fallback
unsupported
probe_required
```

A result contains:

- status
- reasons
- checked facts
- unknown facts
- required probes
- explicit alternatives
- any fallback treatment that would create a different manifest identity

### Mandatory checks

1. GPU count.
2. Per-GPU memory requirement.
3. Compute capability where the technique declares one.
4. Required parallelism/topology.
5. Engine/model capability state.
6. Engine profile option validity.
7. Precision/technique compatibility.
8. Required dependency/plugin capability where known.
9. Proof reuse applicability: never broaden H0/H1 identity rules.
10. Model-fit uncertainty: return `probe_required` if reliable fit cannot be established statically.

### Required negative cases

#### T4 FP8-like request

A fixture that requests a technique requiring an unestablished native FP8 capability on a T4 profile must not pass.

Expected: unsupported or probe_required with FP16 / different-hardware alternatives.

#### Two T4s are not one 32 GB GPU

A manifest requiring one >=32 GB device must not pass against two 16 GB T4s.

A separate manifest with a declared TP=2 strategy may be considered only if the engine/model capability is supported or probe_required.

#### Unknown engine/model intersection

Must not become supported.

Expected: probe_required or unsupported, with the reason.

#### Fallback identity

If a resolver proposes BF16/FP16 instead of FP8, or a different engine, that is a new treatment. The resolver returns it as an alternative; it must not mutate the original manifest in place.

## 6. Tests for the first implementation slice

Add focused tests, then run the broader affected suite.

Minimum tests:

- canonical digest determinism;
- digest changes on every important identity field;
- secret-field rejection;
- frozen resolved snapshot;
- A100 BF16 supported fixture;
- T4 FP16 supported fixture when declared;
- T4 FP8-like negative fixture;
- two-T4 single-device-memory negative fixture;
- explicit TP=2 case;
- unknown capability -> probe_required;
- invalid profile option -> unsupported;
- TechniqueV2 min CC / min VRAM enforcement;
- fallback does not mutate manifest;
- proof applicability is not weakened.

Run:

```bash
pytest -q tests/test_portable_execution.py
pytest -q tests/test_harness_h1.py tests/test_harness_h2.py tests/test_harness_h3.py tests/test_harness_h3_campaign.py
pytest -q
```

If the repository's full suite requires optional services, preserve the established skip/environment behavior and report exactly what ran.

## 7. E1.2 — next slice after E1.0/E1.1

Do not implement this until the first slice is green.

Build notebook export/import:

```text
ExecutionManifest
  -> generated Colab/Kaggle notebook
  -> hardware discovery
  -> compatibility preflight
  -> engine run
  -> benchmark/quality/profile
  -> ResultBundle
  -> import
  -> Trial/Evidence/Hardware
```

Refactor rather than discard the current notebook fixtures:

- `docs/notebooks/colab_a100_h0_qualification.ipynb`
- `docs/notebooks/colab_a100_engine_scorecard.ipynb`
- `docs/notebooks/kaggle_t4x2_serving_smoke.ipynb`

The import path must verify manifest digest and artifact checksums.

## 8. E1.3 and later

After E1.2:

### RunPod

First paid provider adapter.

It must use the existing provider/allocation lifecycle and budgets.

Do not call RunPod directly from campaign logic.

### Modal

First autoscaling endpoint/deployment adapter after a candidate is verified.

### GPU Kubernetes

Later qualification on a real GPU node pool. The existing kind/CPU lifecycle evidence does not qualify GPU performance.

## 9. Parallel owner action

The current fair-engine A100 scorecard is already prepared.

Owner action:

```text
run docs/notebooks/colab_a100_engine_scorecard.ipynb
return/preserve the result ZIP
```

Do not wait for E1 to collect that evidence. E1.2 can later import/preserve it using the new ResultBundle path without rewriting the original raw evidence.

## 10. Provider planning facts

Planning observations on 9 November 2026:

- RunPod A100 80 GB Pod: about $1.79/hour.
- Verda A100 SXM4 40 GB VM: about $1.33/hour on-demand.
- Modal A100 40 GB: about $0.000583/sec (~$2.10/GPU-hour) before CPU/RAM.

These are not constants in code. Prices require provenance/time and operator spend limits.

## 11. Hard non-goals

Do not:

- add a second planner;
- add a second allocation database;
- make FactoryOS manage GPU providers;
- make Model Foundry own inference placement;
- auto-spend;
- call unknown capability supported;
- sum GPU VRAM without a sharding plan;
- weaken quality/numerical gates;
- compare different model/hardware identities as an engine benchmark;
- put optimizer-LLM calls in the serving hot path;
- make Kubernetes required for E1.

## 12. First PR/commit definition of done

The first implementation change is done when:

1. `ExecutionManifestV1` can be created from test fixtures and hashed deterministically.
2. `CompatibilityResolver` produces the four states with reasons.
3. A100 BF16/T4 FP16/negative T4 FP8/2xT4 fixtures are covered.
4. Existing H1/H2/H3 tests still pass.
5. No paid provider is contacted.
6. Documentation points to the next E1.2 notebook round-trip step.
7. `docs/AGENT_PROGRESS.md` is updated with exact tests and commit.

## 13. Suggested commit sequence

Keep commits reviewable:

```text
E1.0: add portable execution manifest and identity tests
E1.1: add hardware/engine compatibility resolver and fixtures
E1.2a: add ResultBundle contract
E1.2b: generate notebook from manifest
E1.2c: import result bundle into experiment evidence
E1.3: add RunPod provider behind AllocationManager
E1.4: add verified endpoint provider
E1.6: qualify real GPU Kubernetes
```

Do not combine provider billing, notebook generation and manifest identity in one giant change.

## 14. Success criterion

The important behavior is not "remote GPU launched."

It is:

> A pinned experiment is rejected before GPU spend when incompatible, executes reproducibly when compatible, returns evidence bound to the exact environment, and becomes reusable inference memory without changing the experiment's identity.

That is the foundation required before adding more GPU providers.
