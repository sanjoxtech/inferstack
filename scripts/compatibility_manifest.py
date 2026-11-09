"""Record the exact component tuple a qualification ran against (CLOUD_SERVING_RESEARCH_PLAN.md checklist).

Writes JSON with: InferStack version and commit, database schema version, Python
and key library versions, Docker, kubectl client/server, the inference CRDs and
controllers present in the chosen cluster context (Gateway API, InferencePool,
KEDA), GPU inventory and driver/CUDA identity. Absent components are recorded as
absent (null), never assumed. A known incompatibility must select an older
supported tuple or an explicit unsupported state; this script only observes.

    python scripts/compatibility_manifest.py --out results/compatibility.json [--context kind-inferstack]
"""
from __future__ import annotations

import argparse
import json
import platform
import shutil
import subprocess
import sys
from importlib import metadata
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CRDS = {'gateway_api_httproutes': 'httproutes.gateway.networking.k8s.io',
        'inference_pools': 'inferencepools.inference.networking.k8s.io',
        'keda_scaledobjects': 'scaledobjects.keda.sh',
        'leaderworkersets': 'leaderworkersets.leaderworkerset.x-k8s.io'}
PACKAGES = ('fastapi', 'httpx', 'pydantic', 'uvicorn', 'psycopg', 'torch', 'vllm', 'sglang', 'huggingface_hub')


def run(argv, timeout=20):
    if shutil.which(argv[0]) is None:
        return None
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() if out.returncode == 0 else None


def version(package):
    try:
        return metadata.version(package)
    except metadata.PackageNotFoundError:
        return None


def cluster(context):
    if not context:
        return None
    base = ['kubectl', '--context', context]
    raw = run(base + ['version', '-o', 'json'])
    server = (json.loads(raw).get('serverVersion') or {}).get('gitVersion') if raw else None
    crds = {}
    for key, name in CRDS.items():
        found = run(base + ['get', 'crd', name, '-o', 'json'])
        if found:
            body = json.loads(found)
            crds[key] = sorted(v['name'] for v in body['spec'].get('versions', []) if v.get('served'))
        else:
            crds[key] = None
    nodes = run(base + ['get', 'nodes', '-o', 'json'])
    gpu_nodes = None
    if nodes:
        gpu_nodes = [{'name': n['metadata']['name'],
                      'gpus': (n.get('status') or {}).get('allocatable', {}).get('nvidia.com/gpu'),
                      'product': n['metadata'].get('labels', {}).get('nvidia.com/gpu.product')}
                     for n in json.loads(nodes).get('items', [])]
    return {'context': context, 'server_version': server, 'crds': crds, 'nodes': gpu_nodes}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--out', default='results/compatibility.json')
    parser.add_argument('--context', help='kubectl context to inspect (optional)')
    args = parser.parse_args()
    sys.path.insert(0, str(ROOT / 'src'))
    import inferstack
    from inferstack.storage.migrations import LATEST
    kubectl = run(['kubectl', 'version', '--client', '-o', 'json'])
    manifest = {
        'inferstack': {'version': inferstack.__version__, 'commit': run(['git', '-C', str(ROOT), 'rev-parse', 'HEAD']),
                       'dirty': bool(run(['git', '-C', str(ROOT), 'status', '--porcelain'])),
                       'database_schema_version': LATEST},
        'python': {'version': platform.python_version(), 'implementation': platform.python_implementation(),
                   'platform': platform.platform()},
        'packages': {p: version(p) for p in PACKAGES},
        'docker': run(['docker', 'version', '--format', '{{.Server.Version}}']),
        'kubectl_client': (json.loads(kubectl).get('clientVersion') or {}).get('gitVersion') if kubectl else None,
        'cluster': cluster(args.context),
        'gpu': run(['nvidia-smi', '--query-gpu=name,driver_version,memory.total', '--format=csv,noheader']),
        'cuda': run(['nvcc', '--version']),
    }
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2))


if __name__ == '__main__':
    main()
