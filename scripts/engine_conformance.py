#!/usr/bin/env python
"""Live engine conformance (vLLM / SGLang / native) on a declared GPU host.

Launches the engine through its real adapter with a pinned model revision, waits
for readiness, checks runtime identity, a deterministic completion, a stream that
the client abandons (engine must stay live), the Prometheus metric families the
telemetry mapping declares, then stops the process group and confirms nothing is
left. Prints one JSON report; exit 0 only when every check passed.

  python scripts/engine_conformance.py --engine vllm --model Qwen/Qwen3.5-2B-Base \
      --revision b1485b2fa6dfa1287294f269f5fb618e03d52d7c --out results/conformance-vllm.json
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import socket
import sys
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inferstack.engines.adapters import adapter_for  # noqa: E402
from inferstack.infrastructure.local import LocalProcessProvider  # noqa: E402
from inferstack.telemetry.normalize import MAPPINGS, normalize  # noqa: E402


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


async def run(args):
    port = free_port()
    base = f'http://127.0.0.1:{port}'
    checks, report = {}, {'engine': args.engine, 'model': args.model, 'revision': args.revision}
    async with httpx.AsyncClient(base_url=base + '/', timeout=60) as client:
        adapter = adapter_for(args.engine, client)
        argv = adapter.launch_argv(model=args.model, revision=args.revision, port=port, dtype=args.dtype,
                                   extra=tuple(args.extra))
        report['argv'] = argv
        provider = LocalProcessProvider()
        log_path = Path(args.out).with_suffix('.log')
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open('ab') as log:
            try:
                process = await provider.spawn('conformance-' + args.engine, argv, log)
            except OSError as exc:
                report.update(checks={'launched': False}, error=f'{type(exc).__name__}: {exc}'[:500], **{'pass': False})
                Path(args.out).write_text(json.dumps(report, indent=2, default=str))
                print(json.dumps(report, indent=2, default=str))
                return 1
            external = {**provider.external_for(process.pid), 'allocation_id': 'conformance-' + args.engine}
            try:
                deadline = time.monotonic() + args.ready_timeout
                ready = False
                while time.monotonic() < deadline and process.returncode is None:
                    identity = await adapter.runtime_identity()
                    if args.model in identity['served_models']:
                        ready = True
                        break
                    await asyncio.sleep(3)
                checks['ready'] = ready
                if not ready:
                    raise RuntimeError('engine did not become ready (see log)')
                report['identity'] = identity
                checks['version_reported'] = identity.get('version') is not None
                launched = provider.observed_argv(external) or []
                checks['revision_in_running_argv'] = args.revision in launched
                body = {'model': args.model, 'prompt': 'The capital of France is', 'max_tokens': 8, 'temperature': 0}
                first = (await client.post('v1/completions', json=body)).json()
                second = (await client.post('v1/completions', json=body)).json()
                checks['deterministic_completion'] = first['choices'][0]['text'] == second['choices'][0]['text']
                checks['usage_reported'] = isinstance((first.get('usage') or {}).get('completion_tokens'), int)
                async with client.stream('POST', 'v1/completions', json={**body, 'max_tokens': 256,
                                                                          'stream': True}) as stream:
                    async for _ in stream.aiter_bytes():
                        break  # client abandons the stream after the first chunk
                checks['live_after_abandoned_stream'] = args.model in (await adapter.runtime_identity())['served_models']
                scrape = await adapter.scrape()
                if args.engine in MAPPINGS:
                    snapshot = normalize(args.engine, scrape or '', engine_version=identity.get('version'))
                    missing = sorted(k for k, v in snapshot['metrics'].items()
                                     if v.get('value') is None and 'absent' in (v.get('reason') or ''))
                    report['metric_families_missing'] = missing
                    checks['metric_families_present'] = scrape is not None and not missing
                report['features'] = await adapter.observed_features(identity, scrape)
            except Exception as exc:  # noqa: BLE001 - recorded as a failed check with its type
                report['error'] = f'{type(exc).__name__}: {exc}'[:500]
            finally:
                checks['stopped_cleanly'] = provider.stop(external, grace_seconds=30)
    report['checks'] = checks
    report['pass'] = bool(checks) and all(checks.values()) and 'error' not in report
    Path(args.out).write_text(json.dumps(report, indent=2, default=str))
    print(json.dumps(report, indent=2, default=str))
    return 0 if report['pass'] else 1


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--engine', required=True, choices=['vllm', 'sglang'])
    p.add_argument('--model', required=True)
    p.add_argument('--revision', required=True)
    p.add_argument('--dtype', default='bfloat16')
    p.add_argument('--ready-timeout', type=float, default=1200)
    p.add_argument('--out', required=True)
    p.add_argument('--extra', nargs='*', default=[])
    raise SystemExit(asyncio.run(run(p.parse_args())))


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    main()
