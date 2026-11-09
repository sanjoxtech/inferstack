"""One profile-change contract shared by the planner (validation) and the executor (materialization).

A profile key is executable only if it appears here: the planner accepts exactly
these keys with these value types, and the executor materializes exactly these
keys. vLLM and SGLang keys map to command-line flags; InferStack-native keys map to
fields of its TOML runtime profile, written as an overlay in the candidate workspace
(the native server has no per-key flags).

Some options are conditional on the deployment's pinned artifact: `load_format`
other than `auto` is accepted only when the deployment record declares
`profile.supported_load_formats` containing that value (the operator attests that the
pinned image carries the loader's dependency and the weights are in its format).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Option:
    kind: type                     # bool, int, float or str
    flag: str | tuple | None = None   # '--name' (valued) or ('--on', '--off'|None) for booleans
    section: str | None = None     # native TOML section
    values: frozenset | None = None   # allowed values for str options
    minimum: float | None = None
    maximum: float | None = None
    requires_declared: str | None = None  # deployment profile key listing supported values


PROFILES = {
    'vllm': {
        'max_num_seqs': Option(int, '--max-num-seqs', minimum=1, maximum=4096),
        'max_num_batched_tokens': Option(int, '--max-num-batched-tokens', minimum=1, maximum=1 << 20),
        'gpu_memory_utilization': Option(float, '--gpu-memory-utilization', minimum=0.05, maximum=0.99),
        'block_size': Option(int, '--block-size', minimum=1, maximum=256),
        'kv_cache_dtype': Option(str, '--kv-cache-dtype', values=frozenset({'auto', 'fp8', 'fp8_e4m3', 'fp8_e5m2'})),
        'enable_prefix_caching': Option(bool, ('--enable-prefix-caching', '--no-enable-prefix-caching')),
        'enable_chunked_prefill': Option(bool, ('--enable-chunked-prefill', '--no-enable-chunked-prefill')),
        'load_format': Option(str, '--load-format', values=frozenset({'auto', 'safetensors', 'runai_streamer'}),
                              requires_declared='supported_load_formats'),
    },
    'sglang': {
        'max_running_requests': Option(int, '--max-running-requests', minimum=1, maximum=4096),
        'chunked_prefill_size': Option(int, '--chunked-prefill-size', minimum=-1, maximum=1 << 20),
        'mem_fraction_static': Option(float, '--mem-fraction-static', minimum=0.05, maximum=0.99),
        'schedule_policy': Option(str, '--schedule-policy', values=frozenset({'lpm', 'random', 'fcfs', 'dfs-weight'})),
        'kv_cache_dtype': Option(str, '--kv-cache-dtype', values=frozenset({'auto', 'fp8_e5m2', 'fp8_e4m3'})),
        'disable_radix_cache': Option(bool, ('--disable-radix-cache', None)),
    },
    'inferstack_native': {
        'max_num_batched_tokens': Option(int, section='scheduler', minimum=1, maximum=1 << 20),
        'max_running_requests': Option(int, section='scheduler', minimum=1, maximum=4096),
        'max_prefill_chunk_tokens': Option(int, section='scheduler', minimum=1, maximum=1 << 16),
        'decode_priority': Option(bool, section='scheduler'),
        'forward_fusion': Option(bool, section='native'),
    },
}
DEFAULT_NATIVE_PROFILE = 'configs/profiles/production-v1.6.0.toml'


class ProfileError(ValueError):
    pass


def check(engine, patch, deployment=None):
    """Validate a profile patch for an engine (and, when given, the target deployment)."""
    schema = PROFILES.get(engine)
    if schema is None:
        raise ProfileError(f'profile changes are not supported for engine {engine}')
    if not isinstance(patch, dict) or not patch:
        raise ProfileError('profile_patch must be a non-empty object')
    unknown = sorted(set(patch) - set(schema))
    if unknown:
        raise ProfileError(f'profile keys not allowed for {engine}: {unknown}')
    for key, value in patch.items():
        option = schema[key]
        ok = isinstance(value, option.kind) and not (option.kind is not bool and isinstance(value, bool))
        if option.kind is float and isinstance(value, int) and not isinstance(value, bool):
            ok = True
        if not ok:
            raise ProfileError(f'{key} must be {option.kind.__name__}')
        if option.values is not None and value not in option.values:
            raise ProfileError(f'{key}={value!r} is not one of {sorted(option.values)}')
        if option.minimum is not None and value < option.minimum or option.maximum is not None and value > option.maximum:
            raise ProfileError(f'{key}={value!r} outside [{option.minimum}, {option.maximum}]')
        if option.requires_declared and value != 'auto':
            declared = ((deployment or {}).get('profile') or {}).get(option.requires_declared) or []
            if value not in declared:
                raise ProfileError(f'{key}={value!r} requires the deployment to declare it in '
                                   f'profile.{option.requires_declared} for its pinned image/weights')
    return patch


def to_flags(engine, argv, patch):
    """Apply a validated patch to a vLLM/SGLang argv (replacing existing occurrences)."""
    schema = PROFILES[engine]
    argv = list(argv)
    for key, value in patch.items():
        flag = schema[key].flag
        if flag is None:
            raise ProfileError(f'{engine} option {key} is not a command-line flag')
        names = [f for f in (flag if isinstance(flag, tuple) else (flag,)) if f]
        for name in names:
            while name in argv:
                i = argv.index(name)
                del argv[i:i + (1 if isinstance(flag, tuple) else 2)]
        if isinstance(flag, tuple):
            chosen = flag[0] if value else flag[1]
            if chosen:
                argv.append(chosen)
        else:
            argv += [flag, str(value)]
    return argv


def _toml_value(value):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, (int, float)):
        return repr(value)
    if isinstance(value, (list, tuple)):
        return '[' + ', '.join(_toml_value(v) for v in value) + ']'
    return '"' + str(value).replace('\\', '\\\\').replace('"', '\\"') + '"'


def native_overlay(argv, patch, workspace, *, source_root='.'):
    """Write a native runtime profile = base profile + patch into the workspace; point argv at it.

    Returns (argv, overlay_path). The base is the `--profile` already in argv (or the
    default), read from the pinned source tree; only schema keys are changed."""
    from pathlib import Path
    try:
        import tomllib
    except ModuleNotFoundError:  # pragma: no cover
        import tomli as tomllib
    schema = PROFILES['inferstack_native']
    argv = list(argv)
    base = argv[argv.index('--profile') + 1] if '--profile' in argv else DEFAULT_NATIVE_PROFILE
    base_path = Path(base) if Path(base).is_absolute() else Path(source_root) / base
    data = tomllib.loads(base_path.read_text()) if base_path.is_file() else {}
    for key, value in patch.items():
        data.setdefault(schema[key].section, {})[key] = value
    lines = []
    scalars = {k: v for k, v in data.items() if not isinstance(v, dict)}
    for key, value in scalars.items():
        lines.append(f'{key} = {_toml_value(value)}')
    for section, values in data.items():
        if isinstance(values, dict):
            lines.append(f'\n[{section}]')
            lines += [f'{k} = {_toml_value(v)}' for k, v in values.items() if not isinstance(v, dict)]
    overlay = Path(workspace) / 'native-profile.toml'
    overlay.write_text('\n'.join(lines) + '\n')
    if '--profile' in argv:
        argv[argv.index('--profile') + 1] = str(overlay)
    else:
        argv += ['--profile', str(overlay)]
    return argv, overlay
