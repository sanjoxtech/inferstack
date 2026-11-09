"""CompatibilityResolver (E1.1): can this exact manifest run on this exact machine?

Checked before any GPU time is spent. Deterministic; no model, no network.

Outcomes:

- `supported`: every requirement is established from recorded facts.
- `supported_with_fallback`: the manifest itself cannot run here, but one of the
  fallbacks *the manifest pre-declared* can. The result names that fallback as a new
  manifest with its own identity; the original manifest is never changed.
- `probe_required`: nothing blocks it, but at least one fact is unknown (engine x
  model support, compute capability, whether the model fits); run the listed probes
  first.
- `unsupported`: a requirement is violated. `alternatives` lists changes that might
  work; each one is a different experiment.

Rules that are easy to get wrong and are enforced here:

- GPU memory is per device. Two 16 GB GPUs are not one 32 GB GPU; memory is only
  divided across devices when the manifest declares tensor_parallel and the engine
  supports that for this model.
- `unknown` is never `supported` (engine capabilities, engine x model support,
  compute capability, model fit).
- A technique marked `proven` elsewhere proves nothing here; proof reuse stays with
  the H0/H1 applicability rules (`research.techniques.proof_of`).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

from inferstack.contracts.records import HardwareEnvironment
from inferstack.engines import capabilities as capability_matrix
from inferstack.engines.capabilities import Support
from inferstack.engines.profiles import ProfileError
from inferstack.engines.profiles import check as check_profile

from .manifest import DTYPE_BYTES, ExecutionManifestV1

SUPPORTED, WITH_FALLBACK, PROBE, UNSUPPORTED = 'supported', 'supported_with_fallback', 'probe_required', 'unsupported'
GIB = 1 << 30
# Minimum compute capability for a native precision path (BF16 tensor cores: Ampere 8.0;
# FP8: Ada 8.9 / Hopper 9.0). Weight-only quantized paths are separate qualified techniques.
DTYPE_MIN_CC = {'bf16': '8.0', 'fp8_e4m3': '8.9', 'fp8_e5m2': '8.9'}
# Static fit: weights / tensor_parallel must stay under this share of one device's memory
# to be called fitting; above it (KV cache, activations, CUDA context) only a probe decides.
FIT_SAFE_SHARE = 0.6
# Whether an engine can shard one model across GPUs at all (engine x model support is separate).
TENSOR_PARALLEL = {'inferstack_native': Support.UNSUPPORTED, 'vllm': Support.SUPPORTED,
                   'sglang': Support.SUPPORTED, 'generic_openai': Support.UNSUPPORTED}


@dataclass
class CompatibilityResult:
    status: str
    experiment_id: str
    reasons: list[str] = field(default_factory=list)
    checked: dict = field(default_factory=dict)
    unknown: list[str] = field(default_factory=list)
    probes: list[str] = field(default_factory=list)
    alternatives: list[dict] = field(default_factory=list)
    fallback: dict | None = None      # {'change', 'manifest': ExecutionManifestV1-as-json, 'experiment_id'}

    def as_dict(self):
        return asdict(self)


def _cc(value: str) -> tuple[int, int]:
    major, minor = value.split('.')
    return int(major), int(minor)


def _environment(environment) -> HardwareEnvironment:
    if isinstance(environment, HardwareEnvironment):
        return environment
    return HardwareEnvironment.model_validate(environment)


class _Check:
    def __init__(self):
        self.blockers, self.unknown, self.probes, self.checked, self.alternatives = [], [], [], {}, []

    def block(self, reason, *alternatives):
        self.blockers.append(reason)
        for alt in alternatives:
            if alt not in self.alternatives:
                self.alternatives.append(alt)

    def maybe(self, reason, probe):
        self.unknown.append(reason)
        if probe not in self.probes:
            self.probes.append(probe)


def _hardware_alternative(reason):
    return {'kind': 'environment', 'change': None, 'reason': reason, 'new_identity': False}


def _derived_alternative(manifest, change, reason):
    try:
        derived = manifest.derive(change)
    except ValueError as exc:
        return {'kind': 'manifest', 'change': change, 'reason': f'{reason} (not derivable: {exc})',
                'new_identity': True, 'experiment_id': None}
    return {'kind': 'manifest', 'change': change, 'reason': reason, 'new_identity': True,
            'experiment_id': derived.experiment_id}


def _evaluate(manifest: ExecutionManifestV1, env: HardwareEnvironment, techniques, model_support,
              capability_lookup) -> _Check:
    c = _Check()
    engine = manifest.engine.name
    need = manifest.hardware
    gpus = list(env.gpus)
    c.checked['gpus'] = [{'name': g.name, 'memory_bytes': g.memory_bytes, 'compute_capability': g.compute_capability}
                         for g in gpus]

    # 1. accepted GPU products
    if need.gpu_names:
        gpus = [g for g in gpus if any(n.lower() in g.name.lower() for n in need.gpu_names)]
        if not gpus:
            c.block(f'no GPU matches the required products {list(need.gpu_names)}',
                    _hardware_alternative(f'run on {list(need.gpu_names)}'))

    # 2. GPU count
    c.checked['gpu_count'] = {'required': need.gpu_count, 'available': len(gpus)}
    if len(gpus) < need.gpu_count:
        c.block(f'needs {need.gpu_count} GPU(s), environment has {len(gpus)}',
                _hardware_alternative(f'an environment with >= {need.gpu_count} GPUs'))

    # 3. per-device memory: model weights / TP, and the declared per-device minimum
    tp = need.tensor_parallel
    weights = None
    if manifest.model.parameter_count:
        weights = manifest.model.parameter_count * DTYPE_BYTES[manifest.dtype]
    per_device_floor = max(need.min_gpu_memory_bytes or 0, 0)
    for technique in techniques:
        per_device_floor = max(per_device_floor, technique.min_gpu_memory_bytes or 0)
    eligible = sorted((g for g in gpus if g.memory_bytes >= per_device_floor), key=lambda g: -g.memory_bytes)
    c.checked['memory'] = {'per_device_required_bytes': per_device_floor or None, 'tensor_parallel': tp,
                           'eligible_devices': len(eligible), 'weights_bytes_estimate': weights}
    if per_device_floor and len(eligible) < need.gpu_count:
        biggest = max((g.memory_bytes for g in gpus), default=0)
        total = sum(g.memory_bytes for g in gpus)
        reason = (f'needs {need.gpu_count} device(s) with >= {per_device_floor / GIB:.1f} GiB each; largest device '
                  f'has {biggest / GIB:.1f} GiB')
        alternatives = [_hardware_alternative(f'a GPU with >= {per_device_floor / GIB:.1f} GiB')]
        if total >= per_device_floor and len(gpus) > 1:
            reason += (f' (total {total / GIB:.1f} GiB across {len(gpus)} devices does not count: memory is per '
                       'device unless the model is sharded with tensor_parallel)')
            if TENSOR_PARALLEL.get(engine) is not Support.UNSUPPORTED and tp == 1:
                alternatives.append(_derived_alternative(
                    manifest, {'tensor_parallel': len(gpus), 'gpu_count': len(gpus)},
                    f'shard across {len(gpus)} GPUs with tensor_parallel={len(gpus)} '
                    '(a different experiment; engine x model TP support must be established)'))
        c.block(reason, *alternatives)
    if weights is not None and gpus:
        devices = eligible[:need.gpu_count] or sorted(gpus, key=lambda g: -g.memory_bytes)[:need.gpu_count]
        smallest = min(g.memory_bytes for g in devices)
        share = weights / tp / smallest
        c.checked['memory']['weights_share_of_device'] = round(share, 3)
        if share >= 1.0:
            alternatives = [_hardware_alternative('a GPU with more memory')]
            if TENSOR_PARALLEL.get(engine) is not Support.UNSUPPORTED and len(gpus) > tp:
                alternatives.append(_derived_alternative(
                    manifest, {'tensor_parallel': len(gpus), 'gpu_count': len(gpus)},
                    f'shard across {len(gpus)} GPUs (a different experiment)'))
            if DTYPE_BYTES[manifest.dtype] > 1:
                alternatives.append({'kind': 'technique', 'change': None, 'new_identity': True,
                                     'reason': 'a separately qualified weight-only quantization technique'})
            c.block(f'model weights (~{weights / GIB:.1f} GiB / tp={tp}) exceed one device '
                    f'({smallest / GIB:.1f} GiB)', *alternatives)
        elif share > FIT_SAFE_SHARE:
            c.maybe(f'weights use {share:.0%} of a device; KV cache and activations may not fit',
                    'load the model and run one max-context request (fit probe)')
    elif weights is None:
        c.maybe('model size unknown: fit cannot be established statically',
                'load the model and run one max-context request (fit probe)')

    # 4. compute capability: declared minimum, technique minimums and the precision path
    floors = []
    if need.min_compute_capability:
        floors.append((need.min_compute_capability, 'hardware requirement'))
    if manifest.dtype in DTYPE_MIN_CC:
        floors.append((DTYPE_MIN_CC[manifest.dtype], f'native {manifest.dtype} path'))
    for technique in techniques:
        if technique.min_compute_capability:
            floors.append((technique.min_compute_capability, f'technique {technique.name}'))
    ccs = [g.compute_capability for g in gpus]
    c.checked['compute_capability'] = {'devices': ccs, 'floors': [f for f, _ in floors]}
    if floors and gpus:
        if any(cc is None for cc in ccs):
            c.maybe('GPU compute capability not recorded', 'record nvidia-smi compute_cap for every device')
        else:
            lowest = min(ccs, key=_cc)
            for floor, why in floors:
                if _cc(lowest) < _cc(floor):
                    alternatives = [_hardware_alternative(f'a GPU with compute capability >= {floor}')]
                    if why.startswith('native') and _cc(lowest) >= (7, 0):
                        alternatives.insert(0, _derived_alternative(
                            manifest, {'dtype': 'fp16', **({'kv_cache_dtype': None} if manifest.kv_cache_dtype
                                                            and manifest.kv_cache_dtype.startswith('fp8') else {})},
                            f'FP16 runs on compute capability {lowest} (a different experiment)'))
                        if manifest.dtype.startswith('fp8'):
                            alternatives.append({'kind': 'technique', 'change': None, 'new_identity': True,
                                                 'reason': 'a separately qualified weight-only FP8/INT8 path'})
                    c.block(f'{why} needs compute capability >= {floor}; device has {lowest}', *alternatives)
    if manifest.kv_cache_dtype and manifest.kv_cache_dtype.startswith('fp8') and gpus and all(ccs):
        lowest = min(ccs, key=_cc)
        if _cc(lowest) < (8, 9):
            c.maybe(f'FP8 KV cache on compute capability {lowest} relies on an engine software path',
                    'run the quality gate with the FP8 KV cache on this device')

    # 5. tensor parallel: engine ability, then engine x model support
    if tp > 1:
        state = TENSOR_PARALLEL.get(engine, Support.UNKNOWN)
        if state is Support.UNSUPPORTED:
            c.block(f'{engine} cannot shard a model across GPUs (tensor_parallel={tp})',
                    _hardware_alternative('one GPU large enough for the whole model'))
        else:
            c.checked['tensor_parallel'] = {'engine': state.value}

    # 6. engine x model support (never assumed)
    key = (engine, manifest.model.repo_id)
    arch_key = (engine, manifest.model.architecture) if manifest.model.architecture else None
    state = model_support.get(key) or (model_support.get(arch_key) if arch_key else None) or 'unknown'
    if tp > 1:
        state = model_support.get((engine, manifest.model.repo_id, f'tp{tp}'), 'unknown')
    c.checked['engine_model_support'] = state
    if state == 'unsupported':
        c.block(f'{engine} does not support {manifest.model.repo_id}' + (f' with tensor_parallel={tp}' if tp > 1 else ''))
    elif state != 'supported':
        c.maybe(f'{engine} x {manifest.model.repo_id}' + (f' (tp={tp})' if tp > 1 else '') + ' support is unknown',
                f'engine conformance run: {engine} on {manifest.model.repo_id}' + (f' tp={tp}' if tp > 1 else ''))

    # 7. engine version already installed in the environment must match
    installed = env.libraries.get(engine) or (env.libraries.get('inferstack') if engine == 'inferstack_native' else None)
    if installed and engine != 'inferstack_native' and installed != manifest.engine.version:
        c.block(f'environment has {engine} {installed}; manifest pins {manifest.engine.version}',
                _hardware_alternative(f'a fresh environment to install {engine}=={manifest.engine.version}'))

    # 8. engine capabilities (manifest + techniques)
    required = list(manifest.required_capabilities)
    for technique in techniques:
        required += [r for r in technique.requires_capabilities if r not in required]
    caps = {}
    for op in required:
        try:
            cap = capability_lookup(engine, op)
        except ValueError:
            caps[op] = 'unknown'
            c.maybe(f'capability {op} is not in the capability matrix', f'establish {engine} {op}')
            continue
        caps[op] = cap.state.value
        if cap.state is Support.UNSUPPORTED:
            c.block(f'{engine} does not support {op}: {cap.evidence}')
        elif cap.state is Support.UNKNOWN:
            c.maybe(f'{engine} {op} is unknown: {cap.evidence}', f'establish {engine} {op} on this device')
    c.checked['capabilities'] = caps

    # 9. profile options
    if manifest.profile:
        try:
            check_profile(engine, dict(manifest.profile))
            c.checked['profile'] = 'valid'
        except ProfileError as exc:
            c.block(f'invalid profile: {exc}')
    if manifest.kv_cache_dtype and manifest.kv_cache_dtype != 'auto':
        try:
            check_profile(engine, {'kv_cache_dtype': manifest.kv_cache_dtype})
        except ProfileError as exc:
            c.block(f'invalid kv_cache_dtype: {exc}')

    # 10. techniques: engine/version, precision, hardware names, pinned dependencies
    for technique in techniques:
        requirement = next((e for e in technique.engines if e.engine == engine), None)
        if requirement is None:
            c.block(f'technique {technique.name} does not target {engine}')
        if technique.dtypes and manifest.dtype not in technique.dtypes:
            c.block(f'technique {technique.name} supports {technique.dtypes}, manifest is {manifest.dtype}')
        if technique.hardware and gpus and not any(any(h.lower() in g.name.lower() for h in technique.hardware)
                                                   for g in gpus):
            c.block(f'technique {technique.name} targets {technique.hardware}',
                    _hardware_alternative(f'run on {technique.hardware}'))
        unpinned = [d.name for d in technique.dependencies if not d.version.startswith('==')]
        if unpinned:
            c.block(f'technique {technique.name} dependencies are not pinned: {unpinned}')
    c.checked['proof'] = 'not evaluated here; proof reuse follows research.techniques.proof_of identity rules'
    return c


def resolve(manifest: ExecutionManifestV1, environment, *, techniques=(), model_support=None,
            capability_lookup=capability_matrix.lookup) -> CompatibilityResult:
    """Decide whether `manifest` can run on `environment` (a HardwareEnvironment or its body).

    techniques: Technique/TechniqueV2 models the manifest references.
    model_support: {(engine, repo_id_or_architecture): 'supported'|'unsupported'|'unknown'} from
      engine conformance evidence; (engine, repo_id, 'tpN') for sharded support. Missing = unknown.
    """
    env = _environment(environment)
    techniques = list(techniques)
    model_support = dict(model_support or {})
    primary = _evaluate(manifest, env, techniques, model_support, capability_lookup)
    result = CompatibilityResult(status=SUPPORTED, experiment_id=manifest.experiment_id, checked=primary.checked,
                                 unknown=primary.unknown, probes=primary.probes, alternatives=primary.alternatives)
    if not primary.blockers:
        result.status = PROBE if primary.unknown else SUPPORTED
        result.reasons = ['all requirements established'] if not primary.unknown else list(primary.unknown)
        return result
    result.status, result.reasons = UNSUPPORTED, list(primary.blockers)
    for change in manifest.fallbacks:
        try:
            derived = manifest.derive(dict(change))
        except ValueError:
            continue
        attempt = _evaluate(derived, env, techniques, model_support, capability_lookup)
        if not attempt.blockers and not attempt.unknown:
            result.status = WITH_FALLBACK
            result.fallback = {'change': dict(change), 'experiment_id': derived.experiment_id,
                               'manifest': derived.model_dump(mode='json', by_alias=True)}
            result.reasons = [f'primary: {r}' for r in primary.blockers] + [
                f'declared fallback {dict(change)} is supported as a separate experiment']
            break
    return result
