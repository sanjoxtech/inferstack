# InferStack v2 — Colab qualification run

One consolidated run on an A100 (or L4) runtime. T4 is not a supported
qualification target for the BF16 fused kernels.

## Cell 1 — clone (private repo; token never echoed or stored)

```python
import getpass, subprocess, os, base64
tok = getpass.getpass('GitHub token (read-only, this repo): ').strip()
env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_CONFIG_COUNT': '1',
       'GIT_CONFIG_KEY_0': 'http.https://github.com/.extraheader',
       'GIT_CONFIG_VALUE_0': 'AUTHORIZATION: basic ' + base64.b64encode(f'x-access-token:{tok}'.encode()).decode()}
p = subprocess.run(['git', 'clone', '-q', 'https://github.com/sanjoxtech/inferstack.git', '/content/inferstack'],
                   env=env, capture_output=True, text=True)
del tok, env
print('clone exit', p.returncode, (p.stdout + p.stderr)[-300:])
os.chdir('/content/inferstack')
print(subprocess.run(['git', 'log', '-1', '--format=%H %s'], capture_output=True, text=True).stdout)
```

## Cell 2 — install

```bash
%%bash
cd /content/inferstack
pip install -q -e ".[dev,native,hf,postgres]" > /tmp/install.log 2>&1; echo "install exit=$?"
```

## Cell 2b — engines for conformance (optional; separate pins per runtime)

vLLM and SGLang pin their own torch/CUDA builds. Install only the engines you
qualify, in this order, and record the versions printed by the conformance run.

```bash
%%bash
pip install -q "vllm==0.11.0" > /tmp/vllm.log 2>&1; echo "vllm exit=$?"
# SGLang usually needs its own runtime (conflicting torch pins); qualify it in a fresh one:
# pip install -q "sglang[all]==<pinned>" > /tmp/sglang.log 2>&1
```

## Cell 3 — qualify (≈30–90 min; downloads the pinned Qwen3.5-2B-Base commit)

```bash
%%bash
cd /content/inferstack
# Optional live gates; each one you pass with --require becomes blocking:
#   export DEEPSEEK_API_KEY=...; EXTRA="--deepseek-model deepseek-chat --require provider_deepseek"
#   export INFERSTACK_TEST_POSTGRES_URL=postgresql://...  (direct, non-pooled); EXTRA="$EXTRA --require postgresql"
#   vLLM installed (cell 2b):                              EXTRA="$EXTRA --require vllm_conformance"
python scripts/v2_qualify.py --out results/v2-qualification $EXTRA; echo "qualify exit=$?"
```

`vllm_conformance` / `sglang_conformance` run `scripts/engine_conformance.py`:
launch through the real adapter with the pinned revision, readiness, runtime
version, revision present in the running command line, deterministic completion
with authoritative usage, an abandoned stream with the engine still live, the
Prometheus families the telemetry mapping declares, and a verified stop.

## Cell 4 — read the verdict

```python
import json
q = json.load(open('/content/inferstack/results/v2-qualification/qualification.json'))
print(q['verdict'], 'blocking:', q['blocking'])
for name, g in q['gates'].items():
    print(f"{name:22s} {g['status']:8s}", {k: v for k, v in g.items() if k in ('summary', 'failed', 'reason', 'lane_verdicts')})
```

The result records `gpu_profile` (`a100`, `l4`, `t4`, ...). Profiles are separate
qualification evidence: an A100 run never qualifies a T4 deployment or vice versa.
T4 (compute 7.5) has no native BF16, so the fused BF16 kernels are not expected to
qualify there; run it as its own profile only for engines/dtypes it supports.

`qualified` requires: environment (GPU inventory), model_identity (pinned
manifests), test_suite (full pytest including GPU tests), kernel_equivalence
(FP32-oracle ULP checks) and native_smoke (engine matrix liveness). Optional
gates — vLLM/SGLang conformance, DeepSeek/OpenRouter planning smoke, PostgreSQL,
Docker and Kubernetes (`INFERSTACK_TEST_K8S_CONTEXT`) — are reported and only
block when named with `--require`.
Native lane performance verdicts are informational: a `keep` needs three runs
per side. Zip `results/v2-qualification` and attach it to the release ledger.

Skips are classified by the integration that owns them: a Docker/Kubernetes/PostgreSQL
test skipped because that integration is absent is recorded under
`skipped_optional` and does not fail `test_suite`, unless that integration was
named with `--require` (then its skip blocks).

GPU placement for workers and release runtimes: GPU-engine launches (native, vLLM,
SGLang) reserve specific devices and run with `CUDA_VISIBLE_DEVICES` set to them.
The local worker takes device UUIDs from the `nvidia-smi` inventory automatically;
set `infrastructure.gpu_devices` in the worker config (or `INFERSTACK_GPU_DEVICES`
for the control server) to restrict them. Without identified devices a GPU launch
is refused rather than sharing an unknown device.

## Kaggle (GPU T4 x2) — separate `t4` profile

Use this when no A100 is available. It produces `t4` evidence only, which never
qualifies an A100 deployment (and vice versa).

What to expect on T4 (compute 7.5, no native BF16):
- `test_suite`, `model_identity` and the integration gates run normally.
- `vllm_conformance` runs with `float16` (selected automatically on T4;
  override with `--conformance-dtype`).
- `kernel_equivalence` / `native_smoke` exercise the BF16 fused kernels and are
  expected to fail or be unsupported on T4. A `not_qualified` verdict blocked
  only by those gates is the correct T4 result, not a bug to work around.
- SGLang generally targets newer GPUs; skip it on T4.

Notebook settings: Accelerator **GPU T4 x2**, Internet **on**. Add a Kaggle
secret named `GITHUB_TOKEN` (read-only token for this repo) under
Add-ons → Secrets. Optional secrets: `DEEPSEEK_API_KEY`,
`INFERSTACK_TEST_POSTGRES_URL` (direct, non-pooled host).

Cell 1 — clone (token read from Kaggle Secrets, never printed):

```python
import base64, os, subprocess
from kaggle_secrets import UserSecretsClient
tok = UserSecretsClient().get_secret('GITHUB_TOKEN')
env = {**os.environ, 'GIT_TERMINAL_PROMPT': '0', 'GIT_CONFIG_COUNT': '1',
       'GIT_CONFIG_KEY_0': 'http.https://github.com/.extraheader',
       'GIT_CONFIG_VALUE_0': 'AUTHORIZATION: basic ' + base64.b64encode(f'x-access-token:{tok}'.encode()).decode()}
p = subprocess.run(['git', 'clone', '-q', 'https://github.com/sanjoxtech/inferstack.git', '/kaggle/working/inferstack'],
                   env=env, capture_output=True, text=True)
del tok, env
print('clone exit', p.returncode, (p.stdout + p.stderr)[-300:])
os.chdir('/kaggle/working/inferstack')
print(subprocess.run(['git', 'log', '-1', '--format=%H %s'], capture_output=True, text=True).stdout)
print(subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv'], capture_output=True, text=True).stdout)
```

Cell 2 — install (same as Colab cell 2/2b, from `/kaggle/working/inferstack`):

```bash
%%bash
cd /kaggle/working/inferstack
pip install -q -e ".[dev,native,hf,postgres]" > /tmp/install.log 2>&1; echo "install exit=$?"
pip install -q "vllm==0.11.0" > /tmp/vllm.log 2>&1; echo "vllm exit=$?"
python -c "import torch; print(torch.__version__, torch.cuda.device_count(), torch.cuda.get_device_name(0))"
```

Cell 3 — qualify:

```python
import os, subprocess
from kaggle_secrets import UserSecretsClient
extra = ['--require', 'vllm_conformance']
try:
    os.environ['DEEPSEEK_API_KEY'] = UserSecretsClient().get_secret('DEEPSEEK_API_KEY')
    extra += ['--deepseek-model', 'deepseek-chat', '--require', 'provider_deepseek']
except Exception:
    pass  # optional secret not configured
r = subprocess.run(['python', 'scripts/v2_qualify.py', '--out', 'results/v2-qualification-t4', *extra])
print('qualify exit', r.returncode)
```

Then read the verdict exactly as in Cell 4, with
`/kaggle/working/inferstack/results/v2-qualification-t4/qualification.json`.
Download that directory (it includes `compatibility.json`) as the run's evidence.
