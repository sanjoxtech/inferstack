"""Generate a Colab/Kaggle notebook from an ExecutionManifest (E1.2 export).

The notebook is thin on purpose: it pins the source checkout, installs the pinned
engine, and calls `python -m inferstack.execution.runner`, which owns the ordered
preflight -> execute -> package contract. The manifest and workload are embedded
verbatim; the first code cell refuses to continue if the embedded manifest's
canonical digest differs from the one this export was made for.

Cells:
  1. identity (markdown)
  2. pin: embedded manifest/workload, digest check, private clone at the pinned commit
  3. InferStack runner environment (core lock + package; native extras for the native engine)
  4. engine environment (exact `engine==version` in its own venv) for vLLM/SGLang
  5. preflight only: CompatibilityResolver on the discovered GPU; stops on unsupported
  6. run: measured rounds -> ResultBundle zip
  7. download the bundle
"""
from __future__ import annotations

import json
import re

from .manifest import ExecutionManifestV1, canonical_json, sha256_canonical

TEMPLATE_VERSION = 1
REPO_URL = 'https://github.com/sanjoxtech/inferstack.git'
ENGINE_PACKAGES = {'vllm': 'vllm', 'sglang': 'sglang[all]'}


def _code(source: str) -> dict:
    return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [],
            'source': source.strip('\n').splitlines(keepends=True)}


def _md(source: str) -> dict:
    return {'cell_type': 'markdown', 'metadata': {}, 'source': source.strip('\n').splitlines(keepends=True)}


def export_id(manifest: ExecutionManifestV1, workload: dict, *, provider: str, source_revision: str,
              launch=None, served_model=None) -> str:
    return sha256_canonical({'template': TEMPLATE_VERSION, 'manifest_digest': manifest.digest,
                             'workload': workload, 'provider': provider, 'source_revision': source_revision,
                             'launch': launch, 'served_model': served_model})


def export_notebook(manifest: ExecutionManifestV1, workload: dict, *, provider: str = 'colab',
                    source_revision: str, repo_url: str = REPO_URL, launch: list | None = None,
                    served_model: str | None = None) -> dict:
    """Return an nbformat-4 notebook dict for this exact experiment."""
    if provider not in ('colab', 'kaggle'):
        raise ValueError('provider must be colab or kaggle')
    if not re.fullmatch(r'[0-9a-f]{40}', source_revision or ''):
        raise ValueError('source_revision must be the 40-hex commit of the InferStack checkout to run')
    if sha256_canonical(workload) != manifest.workload.fixture_sha256:
        raise ValueError('workload does not match manifest.workload.fixture_sha256')
    engine = manifest.engine.name
    if engine == 'inferstack_native':
        if manifest.engine.source_revision != source_revision:
            raise ValueError('a native experiment runs the engine source it names: source_revision must equal '
                             'manifest.engine.source_revision')
        if not launch:
            raise ValueError('inferstack_native needs an explicit launch template (argv with {port})')
    elif engine not in ENGINE_PACKAGES:
        raise ValueError(f'no notebook execution path for engine {engine}')
    eid = export_id(manifest, workload, provider=provider, source_revision=source_revision, launch=launch,
                    served_model=served_model)
    m = manifest
    need = m.hardware
    mem = f'{need.min_gpu_memory_bytes / (1 << 30):.0f} GiB per GPU' if need.min_gpu_memory_bytes else 'not declared'
    runtime = 'A100 GPU' if provider == 'colab' else 'the GPU accelerator the manifest needs'
    identity = f"""
# InferStack experiment `{m.experiment_id[:12]}`

Generated from an ExecutionManifest. Runtime: **{runtime}**. Run the cells in order; send back the ZIP from the last cell.

| | |
|---|---|
| experiment id | `{m.experiment_id}` |
| manifest digest | `{m.digest}` |
| model | `{m.model.repo_id}@{m.model.revision[:12]}` |
| engine | `{engine} {m.engine.version}` |
| precision | `{m.dtype}` |
| hardware | {need.gpu_count} GPU(s), {mem}, TP={need.tensor_parallel} |
| workload | `{m.workload.name}` · context {m.workload.max_context_tokens} · concurrency {list(m.workload.concurrency)} |
| rounds | {m.measurement.rounds} |
| source | `{source_revision}` |
| export id | `{eid}` |

The compatibility check runs on the GPU you actually got **before** any model is downloaded. If it says
unsupported, the notebook stops and prints alternatives; nothing is weakened to get a pass. Gates this
runner does not execute (quality, correctness) come back as `missing`.
"""
    manifest_text = canonical_json(m.model_dump(mode='json', by_alias=True))
    pin = f"""
import base64, getpass, hashlib, json, os, signal, subprocess, sys, time
from pathlib import Path
PLATFORM = {provider!r}
COMMIT = {source_revision!r}
EXPORT_ID = {eid!r}
EXPERIMENT_ID = {m.experiment_id!r}
MANIFEST_DIGEST = {m.digest!r}
MANIFEST_JSON = {manifest_text!r}
WORKLOAD_JSON = {canonical_json(workload)!r}
BASE = Path('/content' if PLATFORM == 'colab' else '/kaggle/working')
REPO = BASE / 'inferstack'
OUT = BASE / f'inferstack-experiment-{{EXPERIMENT_ID[:12]}}'
canon = lambda v: json.dumps(v, sort_keys=True, separators=(',', ':'), ensure_ascii=False, allow_nan=False)
assert hashlib.sha256(canon(json.loads(MANIFEST_JSON)).encode()).hexdigest() == MANIFEST_DIGEST, 'manifest altered'
assert (3, 11) <= sys.version_info[:2] < (3, 14), 'This runner needs Python 3.11-3.13.'
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total,compute_cap', '--format=csv,noheader'],
                     capture_output=True, text=True).stdout)

def clone_pinned():
    if not (REPO / '.git').is_dir():
        if PLATFORM == 'kaggle':
            from kaggle_secrets import UserSecretsClient
            token = UserSecretsClient().get_secret('GITHUB_TOKEN').strip()
        else:
            token = getpass.getpass('Read-only GitHub token for this private repo: ').strip()
        env = os.environ.copy()
        env.update(GIT_TERMINAL_PROMPT='0', GIT_CONFIG_COUNT='1',
                   GIT_CONFIG_KEY_0='http.https://github.com/.extraheader',
                   GIT_CONFIG_VALUE_0='AUTHORIZATION: basic '+base64.b64encode(('x-access-token:'+token).encode()).decode())
        result = subprocess.run(['git', 'clone', '-q', {repo_url!r}, str(REPO)], env=env, capture_output=True, text=True)
        del token, env
        if result.returncode:
            raise RuntimeError('Clone failed. Check read-only repository access. Credentials were not printed or stored.')
    subprocess.run(['git', 'checkout', '--detach', COMMIT], cwd=REPO, check=True, capture_output=True, text=True)
    assert subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=REPO, text=True).strip() == COMMIT
    print('Pinned source:', COMMIT)
clone_pinned()
OUT.mkdir(parents=True, exist_ok=True)
(OUT / 'manifest.json').write_text(MANIFEST_JSON)
(OUT / 'workload.json').write_text(WORKLOAD_JSON)

def logged(argv, filename, *, env=None, timeout=3600):
    path = OUT / filename
    start = time.monotonic()
    with path.open('w') as log:
        proc = subprocess.Popen(list(map(str, argv)), cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT,
                                start_new_session=True)
        try:
            while proc.poll() is None:
                if time.monotonic() - start > timeout:
                    raise TimeoutError(f'Wall-time cap reached: {{filename}}')
                print(f'{{filename}}: running, {{int(time.monotonic() - start)}} s', flush=True)
                time.sleep(20)
        except BaseException:
            try:
                os.killpg(proc.pid, signal.SIGTERM); proc.wait(timeout=20)
            except (ProcessLookupError, subprocess.TimeoutExpired):
                try: os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError: pass
            raise
    print(filename, 'exit:', proc.returncode)
    if proc.returncode:
        print(path.read_text(errors='replace')[-3000:])
    return proc.returncode
"""
    extras = '.[dev,native,hf]' if engine == 'inferstack_native' else '.'
    runner_env = f"""
# InferStack itself (the runner and, for the native engine, the engine) in its own virtualenv.
probe = subprocess.run([sys.executable, '-m', 'venv', '--help'], capture_output=True).returncode == 0 and \\
    subprocess.run([sys.executable, '-c', 'import ensurepip'], capture_output=True).returncode == 0
if not probe:
    subprocess.run(['apt-get', 'update', '-qq'], check=True)
    subprocess.run(['apt-get', 'install', '-y', '-qq', f'python{{sys.version_info.major}}.{{sys.version_info.minor}}-venv'], check=True)
VENV = Path('/tmp/inferstack-runner-env')
subprocess.run([sys.executable, '-m', 'venv', '--system-site-packages', str(VENV)], check=True)
PY = VENV / 'bin/python'
ENV = {{**os.environ, 'PATH': str(VENV / 'bin') + os.pathsep + os.environ['PATH'], 'PYTHONPATH': str(REPO / 'src')}}
assert logged([PY, '-m', 'pip', 'install', '--require-hashes', '-r', 'requirements-dev.lock'], 'install-core.log',
              env=ENV, timeout=2700) == 0
assert logged([PY, '-m', 'pip', 'install', '--no-cache-dir', '-e', {extras!r}], 'install-runner.log',
              env=ENV, timeout=2700) == 0
(OUT / 'pip-freeze-runner.txt').write_text(subprocess.check_output([str(PY), '-m', 'pip', 'freeze'], env=ENV, text=True))
"""
    if engine in ENGINE_PACKAGES:
        spec = f'{ENGINE_PACKAGES[engine]}=={m.engine.version}'
        bin_expr = "ENGINE_VENV / 'bin/vllm'" if engine == 'vllm' else "ENGINE_VENV / 'bin/python'"
        engine_env = f"""
# {engine} in its own virtualenv at the exact version the manifest pins.
ENGINE_VENV = Path('/content/venv-{engine}' if PLATFORM == 'colab' else '/kaggle/working/venv-{engine}')
if not (ENGINE_VENV / 'bin/python').exists():
    subprocess.run([sys.executable, '-m', 'venv', str(ENGINE_VENV)], check=True)
assert logged([ENGINE_VENV / 'bin/python', '-m', 'pip', 'install', '--upgrade', 'pip', {spec!r}],
              'install-engine.log', timeout=3600) == 0, 'pinned engine install failed: record this as a failed run'
(OUT / 'pip-freeze-engine.txt').write_text(subprocess.check_output([str(ENGINE_VENV / 'bin/python'), '-m', 'pip', 'freeze'], text=True))
ENGINE_BIN = str({bin_expr})
"""
    else:
        engine_env = """
# The native engine runs from the pinned checkout installed above.
ENGINE_BIN = None
"""
    extra = []
    if launch:
        extra += ['--launch', json.dumps(launch)]
    if served_model:
        extra += ['--served-model', served_model]
    common = f"""
RUNNER = [PY, '-m', 'inferstack.execution.runner', '--manifest', OUT / 'manifest.json',
          '--workload', OUT / 'workload.json', '--provider', PLATFORM, '--export-id', EXPORT_ID] + {extra!r}
if ENGINE_BIN:
    RUNNER += ['--engine-bin', ENGINE_BIN]
"""
    preflight = common + """
# Compatibility on the GPU this session actually has. Nothing large is downloaded before this passes.
code = logged(RUNNER + ['--preflight-only'], 'preflight.log', env=ENV, timeout=600)
print((OUT / 'preflight.log').read_text())
if code == 2:
    raise SystemExit('UNSUPPORTED on this hardware: see reasons and alternatives above. Stopping before spend.')
assert code == 0, 'preflight failed'
"""
    run = """
# Measured run: probes (if required), rounds, and the ResultBundle.
assert logged(RUNNER + ['--out', OUT / 'bundle'], 'runner.log', env=ENV, timeout=6 * 3600) == 0
BUNDLE = Path(json.loads((OUT / 'runner.log').read_text().strip().splitlines()[-1])['bundle'])
print('ResultBundle:', BUNDLE)
"""
    download = """
if PLATFORM == 'colab':
    from google.colab import files
    files.download(str(BUNDLE))
else:
    from IPython.display import FileLink, display
    display(FileLink(str(BUNDLE)))
print('Import with: python -m inferstack.control experiment-import --bundle', BUNDLE.name)
"""
    cells = [_md(identity), _code(pin), _code(runner_env), _code(engine_env), _code(preflight), _code(run),
             _code(download)]
    return {'nbformat': 4, 'nbformat_minor': 5,
            'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3', 'language': 'python'},
                         'accelerator': 'GPU',
                         'inferstack': {'export_id': eid, 'experiment_id': m.experiment_id,
                                        'manifest_digest': m.digest, 'template_version': TEMPLATE_VERSION,
                                        'source_revision': source_revision, 'provider': provider}},
            'cells': cells}
