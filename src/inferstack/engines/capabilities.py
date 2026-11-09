"""Engine capability matrix with explicit supported / unsupported / unknown states.

A capability is only `supported` when there is a named basis for it. Declarations
are claims (`basis='declared'`); `tested` means an automated test in this repo
exercises it; `observed` means a runtime observation on a specific deployment.
`unknown` is never treated as supported: `require()` refuses both unsupported and
unknown, so callers must obtain evidence before depending on a capability.

Runtime observations (Package C) refine these defaults per deployment; they never
upgrade an `unsupported` default without new evidence.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum


class Support(str, Enum):
    SUPPORTED = 'supported'
    UNSUPPORTED = 'unsupported'
    UNKNOWN = 'unknown'


class UnsupportedCapability(RuntimeError):
    pass


@dataclass(frozen=True)
class Capability:
    engine: str
    operation: str
    state: Support
    basis: str          # declared | tested | observed | none
    evidence: str       # file/test/doc reference or reason

    def as_dict(self):
        d = asdict(self)
        d['state'] = self.state.value
        return d


ENGINES = ('inferstack_native', 'vllm', 'sglang', 'generic_openai')
OPERATIONS = ('launch_local', 'attach', 'ready_probe', 'complete', 'stream', 'cancel', 'usage_tokens',
              'engine_token_timing', 'prometheus_metrics', 'graceful_drain', 'continuous_batching',
              'chunked_prefill', 'prefix_cache', 'paged_kv', 'cuda_graph_replay', 'kernel_patch',
              'gpu_telemetry', 'request_trace')

S, U, K = Support.SUPPORTED, Support.UNSUPPORTED, Support.UNKNOWN
_DEFAULTS = {
    'inferstack_native': {
        'launch_local': (S, 'declared', 'ManagedEngine subprocess supervision; no automated launch test yet'),
        'attach': (S, 'tested', 'engines/managed.py ManagedEngine.ready'),
        'ready_probe': (S, 'declared', 'api/app.py readiness route'),
        'complete': (S, 'tested', 'tests/test_native_api_metrics.py'),
        'stream': (S, 'tested', 'tests/test_native_api_metrics.py'),
        'cancel': (S, 'tested', 'tests/test_qwen_scheduler.py::test_cancellation_compacts'),
        'usage_tokens': (S, 'tested', 'native streaming usage, tests/test_native_api_metrics.py'),
        'engine_token_timing': (S, 'tested', 'inferstack_timing events, evaluation/metrics.py'),
        'continuous_batching': (S, 'tested', 'tests/test_qwen_scheduler.py'),
        'chunked_prefill': (S, 'tested', 'tests/test_qwen_scheduler.py'),
        'prefix_cache': (S, 'tested', 'runtime/hybrid_prefix_cache.py, tests/test_cache_sampler.py'),
        'paged_kv': (U, 'declared', 'runtime reports paged_kv_active=False (CURRENT_STATE_AUDIT A6)'),
        'cuda_graph_replay': (K, 'none', 'actual_cuda_graph_replay_proven=False; needs GPU evidence'),
        # No repository artifact backs the earlier A100 claim: declared until the GPU
        # kernel_equivalence gate (scripts/v160_kernel_equivalence.py) records evidence.
        'kernel_patch': (S, 'declared', 'fused norm/silu kernels; CPU emulation in tests/test_kernels_reference.py; '
                                        'A100 kernel_equivalence qualification pending'),
        'prometheus_metrics': (U, 'declared', 'native API exposes status JSON, not Prometheus text'),
        'graceful_drain': (U, 'declared', 'not implemented (Package I)'),
        'gpu_telemetry': (K, 'none', 'nvidia-smi polling scripts only; no per-deployment pipeline'),
        'request_trace': (U, 'declared', 'no trace propagation yet (Package C)'),
    },
    'vllm': {
        'launch_local': (S, 'declared', 'engines/catalog.py launch_plan; no live run in this repo'),
        'attach': (S, 'tested', 'OpenAI-compatible HTTP adapter with mock transport; tests/test_quality_phases.py'),
        'ready_probe': (S, 'tested', 'GET /v1/models; tests/test_managed_control.py::test_engine_readiness_and_model_override'),
        'complete': (S, 'tested', 'mock transport only; tests/test_quality_phases.py, tests/test_engines.py'),
        'stream': (S, 'declared', 'adapter forwards SSE; not exercised live'),
        'continuous_batching': (S, 'declared', 'vLLM documentation'),
        'paged_kv': (S, 'declared', 'vLLM documentation'),
        'prefix_cache': (S, 'declared', 'vLLM documentation; version-dependent flag'),
        'prometheus_metrics': (S, 'declared', 'https://docs.vllm.ai/en/latest/usage/metrics/'),
        'kernel_patch': (U, 'declared', 'only via supported plugin system or a versioned fork'),
    },
    'sglang': {
        'launch_local': (S, 'declared', 'engines/catalog.py launch_plan; no live run in this repo'),
        'attach': (S, 'declared', 'shared OpenAI-compatible adapter; no SGLang-specific test'),
        'ready_probe': (S, 'declared', 'GET /v1/models via shared adapter'),
        'complete': (S, 'declared', 'shared adapter; untested against SGLang'),
        'stream': (S, 'declared', 'adapter forwards SSE; not exercised live'),
        'continuous_batching': (S, 'declared', 'SGLang documentation'),
        'paged_kv': (S, 'declared', 'SGLang documentation'),
        'prefix_cache': (S, 'declared', 'radix cache, SGLang documentation'),
        'prometheus_metrics': (S, 'declared', 'https://docs.sglang.io/docs/references/production_metrics'),
        'request_trace': (S, 'declared', 'https://docs.sglang.io/docs/references/production_request_trace'),
        'kernel_patch': (U, 'declared', 'only via backend plugins or a versioned fork'),
    },
    'generic_openai': {
        'launch_local': (U, 'declared', 'remote API; InferStack does not own the process'),
        'attach': (S, 'tested', 'OpenAI-compatible HTTP adapter with mock transport; tests/test_quality_phases.py'),
        'ready_probe': (S, 'tested', 'GET /v1/models; tests/test_managed_control.py::test_engine_readiness_and_model_override'),
        'complete': (S, 'tested', 'mock transport only; tests/test_quality_phases.py, tests/test_engines.py'),
        'stream': (S, 'declared', 'adapter forwards SSE'),
        'kernel_patch': (U, 'declared', 'no access to remote kernels'),
        'gpu_telemetry': (U, 'declared', 'remote API exposes no GPU telemetry'),
        'cuda_graph_replay': (U, 'declared', 'not observable on a remote API'),
        'graceful_drain': (U, 'declared', 'remote lifecycle not owned'),
    },
}


def matrix() -> list[Capability]:
    out = []
    for engine in ENGINES:
        for op in OPERATIONS:
            state, basis, evidence = _DEFAULTS[engine].get(op, (K, 'none', 'no evidence recorded'))
            out.append(Capability(engine, op, state, basis, evidence))
    return out


def lookup(engine: str, operation: str) -> Capability:
    if engine not in ENGINES:
        raise ValueError(f'unknown engine: {engine}')
    if operation not in OPERATIONS:
        raise ValueError(f'unknown operation: {operation}')
    state, basis, evidence = _DEFAULTS[engine].get(operation, (K, 'none', 'no evidence recorded'))
    return Capability(engine, operation, state, basis, evidence)


def require(engine: str, operation: str) -> Capability:
    cap = lookup(engine, operation)
    if cap.state is not Support.SUPPORTED:
        raise UnsupportedCapability(f'{engine} {operation} is {cap.state.value}: {cap.evidence}')
    return cap
