# InferStack v2 — Colab qualification run

One consolidated run on an A100 (or L4) runtime. T4 is not a supported
qualification target for the BF16 fused kernels.

## Recommended first run: upload a prepared notebook

- [Colab A100 native qualification](notebooks/colab_a100_h0_qualification.ipynb): select an A100 runtime and run cells in order. Pins source `d33b470` (H1, beta_g precision fix, Linux 3.12 process fix), uses the existing pinned `Qwen/Qwen3.5-2B-Base` Foundry artifact, records the environment and executes the consolidated qualification/matrix. Keeps vLLM/SGLang out of this environment.
- [Kaggle T4 x2 FP16 serving smoke](notebooks/kaggle_t4x2_serving_smoke.ipynb): select T4 x2, enable Internet and grant access to a read-only `GITHUB_TOKEN` secret. Uses a separate vLLM 0.11.0 environment and pinned `Qwen/Qwen3-0.6B` revision `c1899de289a04d12100db370d81485cdf75e47ca`. Runs the existing engine conformance function once per GPU with FP16, short context and bounded memory. It does not run native BF16 qualification or split one model across both GPUs.

Download either notebook from this checkout and upload it to the corresponding service. Both save logs and offer an evidence ZIP. The notebook code was locally syntax/schema checked, not GPU-executed here. Their first real run establishes whether the selected environment works; a failure remains recorded evidence. An unavailable sandbox backend can block full qualification even if numerical/serving checks execute. Do not disable mandatory checks to obtain a pass.

The native notebook reuses the host's CUDA torch in a separate virtualenv and records the observed torch/CUDA/Triton versions. The Kaggle notebook lets the pinned vLLM package select matching torch/Transformers dependencies in a fresh environment. These are separate profiles; do not combine their installations. No optimizer-provider API key is required for either first run.

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
assert p.returncode == 0, 'Check read-only repository access and Internet connectivity'
os.chdir('/content/inferstack')
subprocess.run(['git', 'checkout', '--detach', 'd33b47032de1427a45d662fae05fffb09d5dcfea'], check=True)
print(subprocess.run(['git', 'log', '-1', '--format=%H %s'], capture_output=True, text=True).stdout)
```

## Cell 2 — install

```bash
%%bash
cd /content/inferstack
pip install -q -e ".[dev,native,hf,postgres]" > /tmp/install.log 2>&1; echo "install exit=$?"
```

## Additional engines: a separate qualification environment

For the first A100/native run, omit vLLM and SGLang. They select their own
torch/CUDA/Transformers dependencies and need independently pinned environments.

The previous guide's vLLM 0.11.0 example must not be used for Qwen3.5. The
prepared Kaggle notebook uses that version only with the older Qwen3-0.6B
architecture. Qwen3.5 engine conformance needs a newer checked model/engine/backend
combination; the [official model card](https://huggingface.co/Qwen/Qwen3.5-2B)
describes development-build requirements, and the
[upstream recipe](https://github.com/vllm-project/recipes/blob/main/Qwen/Qwen3.5.md)
has separate current installation guidance. Pin the chosen build and prove it on
the target hardware before treating the combination as supported.

## Cell 3 — qualify (≈30–90 min; downloads the pinned Qwen3.5-2B-Base commit)

```bash
%%bash
cd /content/inferstack
# Optional live gates; each one you pass with --require becomes blocking:
#   export DEEPSEEK_API_KEY=...; EXTRA="--deepseek-model deepseek-chat --require provider_deepseek"
#   export INFERSTACK_TEST_POSTGRES_URL=postgresql://...  (direct, non-pooled); EXTRA="$EXTRA --require postgresql"
#   In a separately qualified vLLM environment:          EXTRA="$EXTRA --require vllm_conformance"
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

## Kaggle (GPU T4 x2) — separate serving profile

Use the prepared [T4 x2 notebook](notebooks/kaggle_t4x2_serving_smoke.ipynb) above.
It tests readiness, immutable launch revision, deterministic completion,
authoritative usage, abandoned-stream liveness, declared telemetry and clean stop
through InferStack's real vLLM adapter on each GPU. A failed check stays failed.

The two reports describe two independent replicas. They are not baseline versus
an optimized candidate until an actual treatment and repeated comparison are
introduced, and they do not establish tensor-parallel speedup. Larger-model,
multi-replica traffic and optimization experiments follow these initial checks.

Do not run the full BF16/native runner expecting a `qualified` T4 release. Its
mandatory native/kernel gates target a different hardware/dtype profile. The
small-model FP16 evidence cannot qualify the Qwen3.5 native implementation or an
A100 deployment.
