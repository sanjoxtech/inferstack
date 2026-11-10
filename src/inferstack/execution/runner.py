"""Manifest runner (E1.2): the ordered preflight -> execute -> package contract.

Runs inside a generated notebook (Colab/Kaggle) or any machine with the pinned
checkout. Order, each recorded as a step in the ResultBundle:

1. discover the actual environment (nvidia-smi + libraries), so even a refused run
   records where it was attempted;
2. refuse a workload fixture whose canonical sha256 differs from the manifest's
   `workload.fixture_sha256`;
3. CompatibilityResolver; stop on `unsupported` without launching anything;
4. `probe_required` proceeds only when every listed probe is one this runner
   performs (engine conformance, model fit): launch, readiness and one request of
   the largest workload case; otherwise stop;
5. measured rounds through the scorecard measurement code (same metric rules as
   N0), with engine identity from the served endpoints;
6. package everything as a ResultBundle zip. Gates this runner does not execute
   (quality, correctness) are recorded as `missing`, never passed.

Usage:
    python -m inferstack.execution.runner --manifest m.json --workload w.json --out results/
"""
from __future__ import annotations

import argparse
import asyncio
import importlib.util
import json
import sys
import time
from pathlib import Path

from .compatibility import PROBE, UNSUPPORTED, resolve
from .manifest import ExecutionManifestV1, sha256_canonical
from .results import write_bundle

ROOT = Path(__file__).resolve().parents[3]
RUNNER_PROBES = ('engine conformance run', 'load the model and run one max-context request')
RUNNER_GATES = ('readiness', 'requests_succeeded')
_SCORECARD = None


def scorecard_module():
    """The N0 measurement code (scripts/engine_scorecard.py) from the pinned checkout."""
    global _SCORECARD
    if _SCORECARD is None:
        spec = importlib.util.spec_from_file_location('inferstack_engine_scorecard',
                                                      ROOT / 'scripts' / 'engine_scorecard.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        _SCORECARD = module
    return _SCORECARD


def launch_argv(manifest: ExecutionManifestV1, engine_bin: str | None = None) -> list[str]:
    """Derived launch command for vLLM/SGLang (with {port}); other engines need an explicit template."""
    from inferstack.engines.profiles import to_flags
    m, engine = manifest.model, manifest.engine.name
    dtype = {'bf16': 'bfloat16', 'fp16': 'float16', 'fp32': 'float32'}.get(manifest.dtype)
    if dtype is None:
        raise ValueError(f'no derived launch for dtype {manifest.dtype}; pass an explicit launch template')
    tp = manifest.hardware.tensor_parallel
    if engine == 'vllm':
        argv = [engine_bin or 'vllm', 'serve', m.repo_id, '--revision', m.revision, '--host', '127.0.0.1',
                '--port', '{port}', '--dtype', dtype, '--max-model-len', str(manifest.workload.max_context_tokens)]
        if m.tokenizer_revision:
            argv += ['--tokenizer-revision', m.tokenizer_revision]
        if tp > 1:
            argv += ['--tensor-parallel-size', str(tp)]
    elif engine == 'sglang':
        argv = [engine_bin or 'python', '-m', 'sglang.launch_server', '--model-path', m.repo_id, '--revision',
                m.revision, '--host', '127.0.0.1', '--port', '{port}', '--dtype', dtype, '--context-length',
                str(manifest.workload.max_context_tokens), '--enable-metrics']
        if tp > 1:
            argv += ['--tp', str(tp)]
    else:
        raise ValueError(f'no derived launch for {engine}; pass an explicit launch template')
    patch = dict(manifest.profile)
    if manifest.kv_cache_dtype and manifest.kv_cache_dtype != 'auto':
        patch['kv_cache_dtype'] = manifest.kv_cache_dtype
    return to_flags(engine, argv, patch) if patch else argv


def _version_observed(engine, identity):
    if engine == 'vllm':
        return (identity.get('/version') or {}).get('version')
    if engine == 'sglang':
        return (identity.get('/get_server_info') or {}).get('version')
    return None


class _Steps:
    def __init__(self):
        self.items = []

    def add(self, name, ok, started, detail=None):
        self.items.append({'name': name, 'ok': bool(ok), 'started_at': started, 'ended_at': time.time(),
                           'detail': None if detail is None else str(detail)[:4000]})
        return ok


async def run(manifest_path, workload_path, out_dir, *, environment=None, provider='attached',
              model_support=None, launch=None, engine_bin=None, served_model=None, ready_timeout=900,
              request_timeout=600, runner_meta=None) -> Path:
    """Execute one manifest and return the ResultBundle path. Never raises for a refused or failed run."""
    started = time.time()
    steps = _Steps()
    out_dir = Path(out_dir)
    manifest = ExecutionManifestV1.from_json(Path(manifest_path).read_text())
    workload = json.loads(Path(workload_path).read_text())
    work = out_dir / f'run-{manifest.experiment_id[:12]}-{time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())}'
    work.mkdir(parents=True, exist_ok=False)
    index = {'experiment_id': manifest.experiment_id, 'manifest_digest': manifest.digest,
             'manifest': manifest.model_dump(mode='json', by_alias=True), 'environment': {}, 'preflight': {},
             'status': 'failed', 'runner': {'kind': provider, **(runner_meta or {})}, 'run_dir': 'run',
             'started_at': started, 'gates': {}}
    run_dir = work / 'payload' / 'run'
    run_dir.mkdir(parents=True)

    def finish(status, reason=None):
        gates = {g: 'missing' for g in manifest.measurement.gates}
        gates.update(index['gates'])
        index.update(status=status, failure_reason=reason, steps=steps.items, ended_at=time.time(), gates=gates)
        return write_bundle(work / 'payload', index, out_dir / f'{work.name}.zip')

    t = time.time()
    if environment is None:
        from inferstack.infrastructure.inventory import collect
        environment, _ = collect(provider=provider)
    env_body = environment if isinstance(environment, dict) else environment.model_dump(mode='json')
    env_body = {k: v for k, v in env_body.items() if k in {
        'provider', 'region', 'gpus', 'topology', 'cpu_model', 'cpu_count', 'ram_bytes', 'driver_version',
        'cuda_version', 'libraries', 'observed_at'}}
    index['environment'] = env_body
    steps.add('discover_environment', True, t, f"{len(env_body.get('gpus') or [])} GPU(s)")

    t = time.time()
    fixture = sha256_canonical(workload)
    if not steps.add('workload_fixture', fixture == manifest.workload.fixture_sha256, t,
                     f'fixture sha256 {fixture}'):
        return finish('refused', 'workload fixture does not match the manifest')

    t = time.time()
    result = resolve(manifest, env_body, model_support=model_support)
    index['preflight'] = {'status': result.status, 'reasons': result.reasons, 'unknown': result.unknown,
                          'probes': result.probes, 'alternatives': result.alternatives, 'checked': result.checked}
    if result.status == UNSUPPORTED or result.status == 'supported_with_fallback':
        steps.add('compatibility', False, t, '; '.join(result.reasons))
        reason = ('unsupported: ' + '; '.join(result.reasons) if result.status == UNSUPPORTED else
                  'only a declared fallback is supported; export the fallback manifest as its own experiment')
        return finish('refused', reason)
    uncovered = [p for p in result.probes if not p.startswith(RUNNER_PROBES)]
    if result.status == PROBE and uncovered:
        steps.add('compatibility', False, t, 'unresolved: ' + '; '.join(uncovered))
        return finish('refused', 'probe_required with probes this runner cannot perform: ' + '; '.join(uncovered))
    steps.add('compatibility', True, t, result.status)

    try:
        template = list(launch) if launch else launch_argv(manifest, engine_bin)
    except ValueError as exc:
        return finish('refused', str(exc))
    model_name = served_model or manifest.model.repo_id
    profile = {'name': manifest.engine.name, 'engine': manifest.engine.name, 'model': model_name, 'argv': template}
    index['launch'] = {'argv_template': template, 'served_model': model_name, 'explicit_template': bool(launch)}
    sc = scorecard_module()
    sc_workload = {**workload, 'concurrency': list(manifest.workload.concurrency),
                   'warmup_requests': manifest.measurement.warmup_requests}

    if result.status == PROBE:
        t = time.time()
        largest = max(workload['cases'], key=lambda c: c.get('prompt_words', 0))
        probe_workload = {**sc_workload, 'cases': [largest], 'concurrency': [1], 'warmup_requests': 0,
                          'requests_per_cell': {'min': 1, 'per_concurrency': 0}}
        status = await sc.run_profile(profile, probe_workload, run_dir / 'probe' / profile['name'],
                                      ready_timeout, request_timeout)
        rows = (run_dir / 'probe' / profile['name'] / 'requests.jsonl')
        ok = status.get('ready') and rows.exists() and all(json.loads(r)['success'] for r in
                                                            rows.read_text().splitlines() if r)
        if not steps.add('probe', ok, t, status.get('error') or f'largest case {largest["name"]} served'):
            index['gates']['readiness'] = 'passed' if status.get('ready') else 'failed'
            return finish('probe_failed', status.get('error') or 'probe request failed')

    sc_manifest = {'run_id': 'run', 'created_at': time.time(), 'host': sc.host_identity(),
                   'rounds': manifest.measurement.rounds, 'workload': sc_workload,
                   'workload_sha256': sc.sha(sc_workload), 'profiles': [profile], 'profiles_sha256': sc.sha([profile]),
                   'model': {'uri': manifest.model.repo_id, 'revision': manifest.model.revision},
                   'experiment_id': manifest.experiment_id}
    (run_dir / 'manifest.json').write_text(json.dumps(sc_manifest, indent=2, default=str))
    statuses = []
    for r in range(manifest.measurement.rounds):
        t = time.time()
        status = await sc.run_profile(profile, sc_workload, run_dir / f'round-{r}' / profile['name'],
                                      ready_timeout, request_timeout)
        statuses.append(status)
        steps.add(f'round-{r}', status.get('ready'), t, status.get('error'))
        if r == 0 and status.get('ready'):
            observed = _version_observed(manifest.engine.name, status.get('identity') or {})
            index['launch']['engine_version_observed'] = observed
            if observed is not None and observed != manifest.engine.version:
                index['gates']['readiness'] = 'passed'
                return finish('failed', f'engine reports version {observed}; manifest pins '
                                        f'{manifest.engine.version}')
    ready = [s.get('ready') for s in statuses]
    index['gates']['readiness'] = 'passed' if all(ready) else 'failed'
    rows = [json.loads(line) for r, s in enumerate(statuses) if s.get('ready')
            for line in (run_dir / f'round-{r}' / profile['name'] / 'requests.jsonl').read_text().splitlines()
            if line]
    index['gates']['requests_succeeded'] = 'passed' if rows and all(r['success'] for r in rows) else 'failed'
    card = sc.aggregate([run_dir], min_rounds=min(3, manifest.measurement.rounds))
    (run_dir / 'scorecard.json').write_text(json.dumps(card, indent=2, default=str))
    if not any(ready):
        return finish('failed', 'engine never became ready')
    return finish('completed' if all(ready) else 'partial')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--workload', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--provider', default='attached', help='colab | kaggle | attached | ...')
    parser.add_argument('--environment', help='recorded HardwareEnvironment JSON (default: discover)')
    parser.add_argument('--model-support', help='JSON list of [engine, model, state] from conformance evidence')
    parser.add_argument('--launch', help='JSON argv template with {port} (required for inferstack_native)')
    parser.add_argument('--engine-bin', help='engine executable (vllm binary or python of the engine venv)')
    parser.add_argument('--served-model', help='model id the server reports (default: manifest repo id)')
    parser.add_argument('--preflight-only', action='store_true', help='print the compatibility result and exit')
    parser.add_argument('--ready-timeout', type=float, default=900)
    parser.add_argument('--request-timeout', type=float, default=600)
    parser.add_argument('--export-id', help='id of the generated notebook that invoked the runner')
    args = parser.parse_args(argv)
    support = {tuple(item[:-1]): item[-1] for item in json.loads(Path(args.model_support).read_text())} \
        if args.model_support else None
    environment = json.loads(Path(args.environment).read_text()) if args.environment else None
    if args.preflight_only:
        manifest = ExecutionManifestV1.from_json(Path(args.manifest).read_text())
        if environment is None:
            from inferstack.infrastructure.inventory import collect
            environment, _ = collect(provider=args.provider)
        result = resolve(manifest, environment, model_support=support)
        print(json.dumps(result.as_dict(), indent=2, default=str))
        return 0 if result.status != UNSUPPORTED else 2
    bundle = asyncio.run(run(args.manifest, args.workload, args.out, environment=environment,
                             provider=args.provider, model_support=support,
                             launch=json.loads(args.launch) if args.launch else None, engine_bin=args.engine_bin,
                             served_model=args.served_model, ready_timeout=args.ready_timeout,
                             request_timeout=args.request_timeout,
                             runner_meta={'export_id': args.export_id} if args.export_id else None))
    print(json.dumps({'bundle': str(bundle)}))
    return 0


if __name__ == '__main__':
    sys.exit(main())
