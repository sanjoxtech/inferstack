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

## Cell 3 — qualify (≈30–90 min; downloads the pinned Qwen3.5-2B-Base commit)

```bash
%%bash
cd /content/inferstack
# Optional live provider gates:
#   export DEEPSEEK_API_KEY=...; EXTRA="--deepseek-model <id>"
python scripts/v2_qualify.py --out results/v2-qualification $EXTRA; echo "qualify exit=$?"
```

## Cell 4 — read the verdict

```python
import json
q = json.load(open('/content/inferstack/results/v2-qualification/qualification.json'))
print(q['verdict'], 'blocking:', q['blocking'])
for name, g in q['gates'].items():
    print(f"{name:22s} {g['status']:8s}", {k: v for k, v in g.items() if k in ('summary', 'failed', 'reason', 'lane_verdicts')})
```

`qualified` requires: environment (GPU inventory), model_identity (pinned
manifests), test_suite (full pytest including GPU tests), kernel_equivalence
(FP32-oracle ULP checks) and native_smoke (engine matrix liveness). Optional
gates — vLLM conformance, DeepSeek/OpenRouter planning smoke, PostgreSQL (needs
server binaries; skipped on Colab) — are reported but never change the verdict.
Native lane performance verdicts are informational: a `keep` needs three runs
per side. Zip `results/v2-qualification` and attach it to the release ledger.
