# Colab A100 — InferStack v1.6.0 Forward Fusion

Use one A100 session. In the signed-in GitHub browser, open
`https://github.com/sanjoxtech/inferstack`, select the commit to test, and download
**Code → Download ZIP**. Record the commit SHA so each experiment identifies its source.

In Colab, select **Runtime → Change runtime type → A100 GPU**, then upload and extract
the GitHub archive in a Python cell:

```python
from google.colab import files
from pathlib import Path
from zipfile import ZipFile
import os

uploaded = files.upload()  # Choose the ZIP downloaded from GitHub.
archives = [name for name in uploaded if name.endswith('.zip')]
assert len(archives) == 1, 'Upload exactly one GitHub repository ZIP.'
destination = Path('/content/inferstack-source')
assert not destination.exists(), 'Use a fresh destination for each source import.'
destination.mkdir()
with ZipFile(archives[0]) as archive:
    for entry in archive.infolist():
        target = (destination / entry.filename).resolve()
        assert target.is_relative_to(destination.resolve()), 'Unsafe archive path.'
    archive.extractall(destination)
roots = list(destination.glob('*/pyproject.toml'))
assert len(roots) == 1, 'Expected one InferStack project root.'
os.chdir(roots[0].parent)
print('Project:', Path.cwd())
```

Install in a new Colab cell. The `%%bash` lines below make each block runnable
as its own cell:

```bash
%%bash
set -e
python -m pip install -U -e ".[dev,native]"
```

Run the cheap gates first:

```bash
%%bash
set -e
pytest -q
python scripts/v160_static_audit.py
inferstack native plan --profile configs/profiles/production-v1.6.0.toml
```

Then run the complete forward-fusion experiment:

```bash
%%bash
set -e
python scripts/v160_colab_matrix.py \
  --foundry-root examples/model_foundry \
  --model-ref foundry://factorylm-2b@candidate \
  --api-model factorylm-2b \
  --tokenizer Qwen/Qwen3.5-2B-Base
```

The matrix begins with `scripts/v160_kernel_equivalence.py`, using actual Qwen3.5 config dimensions on the A100.

Main outputs:

```text
results/v160-matrix/summary.json
results/v160-matrix/analysis.json
results/v160-matrix/kernel-equivalence.log
results/v160-matrix/full-feature-probe.json
results/v160-matrix/runtime-profile.json
results/v160-matrix/baseline-operator-profile.json
results/v160-matrix/full-operator-profile.json
results/v160-matrix/baseline-decode-trace.json
results/v160-matrix/full-decode-trace.json
```

### The key question

The v1.5.5 forward cost was ~44 ms/decode at C8. v1.6 succeeds only if the full-fusion lane measurably reduces that number and the C8 benchmark beats the unfused baseline.

Save the tested commit SHA, GPU details and installed package versions with the
result bundle. Download `results/v160-matrix/` before ending the runtime. CPU/static
checks do not establish A100 numerical correctness or performance.
