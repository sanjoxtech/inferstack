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
