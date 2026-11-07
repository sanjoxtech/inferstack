# InferStack ownership and evolution roadmap

InferStack owns inference execution and its own optimization control plane.
FactoryOS and Model Foundry remain independent systems.

| System | Owns | Exchanges with InferStack |
| --- | --- | --- |
| FactoryOS | Agents, tools, workflows, harness evolution | Inference requests, workload signals, incidents and SLOs |
| InferStack | Loading, kernels, scheduling, caches, serving representations, benchmarking and inference evolution | Responses, metrics, capacity and validated execution profiles |
| Model Foundry | Training, adaptation, datasets, model weights and checkpoint lifecycle | Immutable model artifacts and model issue reports |

InferStack may pack, transpose or quantize a derived serving representation.
It preserves the original model artifact and validates numerical/quality changes.
It does not train, fine-tune or modify the source checkpoint.

## Current imported baseline

This repository begins with the saved v1.6.0 forward-fusion candidate.
The runtime, kernel experiments, benchmark scripts and reference tests are present.
The evolution control plane described below is a roadmap, not implemented functionality.
The `foundry/` package is a model-artifact loader/integration, not a training system.

## Planned control plane

1. Define evidence, technique, experiment and execution-profile schemas with provenance.
2. Persist model/hardware/workload fingerprints, evidence and experiment outcomes.
3. Establish model intake with capability checks, a correct baseline and fallback paths.
4. Ingest internal profiling, consumer signals and external inference research.
5. Generate provider-independent optimization hypotheses and scoped experiment plans.
6. Run isolated candidates through correctness, numerical and workload regression gates.
7. Use repeated trials, noise checks, SLO feasibility and multi-objective comparisons
   to keep, reject or retest candidates. Preserve unsuccessful experiments too.
8. Promote validated profiles through staged rollout with rollback.

These stages are InferStack responsibilities. FactoryOS may supply signals but does
not own or direct InferStack's optimization loop. LLM proposals are hypotheses;
measured gates determine eligibility for promotion.

The serving data plane uses validated execution profiles. Research, planning and
experimentation run outside the token generation path. A new architecture or GPU
requires capability checks and measurement; resemblance to a known model alone
does not establish kernel compatibility.
