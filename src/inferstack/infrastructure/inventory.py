"""Hardware and software inventory for a HardwareEnvironment record.

GPU facts come from `nvidia-smi` (one row per GPU UUID; never just the first GPU).
Missing tools produce an empty GPU list and a recorded reason, not invented data.
"""
from __future__ import annotations

import importlib.metadata
import os
import platform
import shutil
import subprocess
import time

GPU_FIELDS = 'uuid,name,memory.total,driver_version,compute_cap,pci.bus_id'
LIBRARIES = ('torch', 'triton', 'transformers', 'vllm', 'sglang', 'flashinfer-python', 'inferstack')


def _run(argv, timeout=15):
    try:
        out = subprocess.run(argv, capture_output=True, text=True, timeout=timeout)
        return (out.stdout, None) if out.returncode == 0 else (None, f'{argv[0]} exit {out.returncode}')
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f'{argv[0]}: {type(exc).__name__}'


def gpus(nvidia_smi='nvidia-smi'):
    if shutil.which(nvidia_smi) is None:
        return [], {'gpu': 'nvidia-smi not found'}
    text, error = _run([nvidia_smi, f'--query-gpu={GPU_FIELDS}', '--format=csv,noheader,nounits'])
    if error:
        return [], {'gpu': error}
    found, driver = [], None
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(',')]
        if len(parts) < 6:
            continue
        uuid, name, mem_mib, driver, cc, bus = parts[:6]
        found.append({'uuid': uuid, 'name': name, 'memory_bytes': int(float(mem_mib)) * 2**20,
                      'compute_capability': cc if cc and cc != '[N/A]' else None, 'pci_bus_id': bus})
    topo, topo_error = _run([nvidia_smi, 'topo', '-m']) if len(found) > 1 else (None, None)
    return found, {'driver_version': driver, 'topology': topo.strip() if topo else None,
                   **({'topology_error': topo_error} if topo_error else {})}


def _memory_bytes():
    try:
        with open('/proc/meminfo') as f:
            for line in f:
                if line.startswith('MemTotal:'):
                    return int(line.split()[1]) * 1024
    except OSError:
        pass
    text, _ = _run(['sysctl', '-n', 'hw.memsize'])
    return int(text) if text and text.strip().isdigit() else None


def _cpu_model():
    try:
        with open('/proc/cpuinfo') as f:
            for line in f:
                if line.startswith('model name'):
                    return line.split(':', 1)[1].strip()
    except OSError:
        pass
    text, _ = _run(['sysctl', '-n', 'machdep.cpu.brand_string'])
    return text.strip() if text else platform.processor() or None


def libraries():
    versions = {}
    for name in LIBRARIES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            continue
    try:
        import torch
        versions['torch.cuda'] = torch.version.cuda or 'none'
    except Exception:
        pass
    return versions


def collect(*, provider='attached', region=None, nvidia_smi='nvidia-smi', clock=time.time):
    """Return a body valid for the `hardware` record schema plus collection notes."""
    gpu_list, gpu_meta = gpus(nvidia_smi)
    body = {'provider': provider, 'region': region,
            'gpus': [{k: g[k] for k in ('uuid', 'name', 'memory_bytes', 'compute_capability')} for g in gpu_list],
            'topology': gpu_meta.get('topology'), 'cpu_model': _cpu_model(), 'cpu_count': os.cpu_count(),
            'ram_bytes': _memory_bytes(), 'driver_version': gpu_meta.get('driver_version'),
            'cuda_version': libraries().get('torch.cuda'), 'libraries': libraries(), 'observed_at': clock()}
    notes = {k: v for k, v in gpu_meta.items() if k.endswith('error') or k == 'gpu'}
    return body, notes
