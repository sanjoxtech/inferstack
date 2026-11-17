#!/usr/bin/env python
"""Fair multi-engine serving comparison (native campaign N0/N1).

`run` launches each declared engine profile in turn on the same host, sends the
same deterministic workload matrix (cases x closed-loop concurrency) to each, and
stores every raw request record. Every (round, profile) writes into its own
directory and nothing is overwritten. The profile order rotates each round, so
no engine always runs first or last.

`aggregate` reads one or more run directories and writes `scorecard.json` and
`scorecard.md`. These give per-cell medians per round, then the median and range
across rounds, attained concurrency, an equal-work check and explicit missing or
unsupported entries. A comparison says `ahead`/`behind` only with at least 3
rounds and non-overlapping ranges; otherwise it says `overlap` or
`insufficient_rounds`. It is a measurement, not a release decision.

Metric definitions (all client-side, identical for every engine):
  ttft_ms            request start -> first non-empty streamed content
  e2e_ms             request start -> terminal stream event
  client_decode_ms   (e2e_ms - ttft_ms) / (output_tokens - 1), only with engine usage counts
  output_tps         sum(output_tokens) / cell wall time, only when every request reported usage
Engine-internal timings (e.g. native TPOT) are never compared with client timings.

  python scripts/engine_scorecard.py run --profiles configs/scorecard/a100-qwen3.5-2b.json \\
      --out results/scorecard --rounds 3
  python scripts/engine_scorecard.py aggregate results/scorecard/<run-id> --baseline native_unfused
"""
from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import os
import platform
import random
import signal
import socket
import statistics
import subprocess
import sys
import threading
import time
from pathlib import Path

import httpx

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))

from inferstack.evaluation.metrics import StreamMeasurement  # noqa: E402

WORDS = ('cache latency kernel batch decode prefill tensor memory schedule token stream request engine model '
         'throughput queue warp block attention layer norm weight gradient shard replica router budget').split()
DEFAULT_WORKLOAD = {
    'version': 1,
    'cases': [
        {'name': 'interactive', 'prompt_words': 48, 'max_tokens': 128},
        {'name': 'moderate', 'prompt_words': 400, 'max_tokens': 256},
        {'name': 'prefill_heavy', 'prompt_words': 1500, 'max_tokens': 32},
        {'name': 'shared_prefix', 'prompt_words': 1200, 'max_tokens': 64, 'shared_prefix': True},
    ],
    'concurrency': [1, 4, 8, 16],
    'requests_per_cell': {'min': 8, 'per_concurrency': 4},
    'warmup_requests': 2,
    'temperature': 0.0,
    'seed': 20261109,
}


def free_port():
    with socket.socket() as s:
        s.bind(('127.0.0.1', 0))
        return s.getsockname()[1]


def sha(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def pct(xs, p):
    if not xs:
        return None
    ys = sorted(xs)
    k = (len(ys) - 1) * p
    lo = int(k)
    hi = min(lo + 1, len(ys) - 1)
    return ys[lo] * (1 - (k - lo)) + ys[hi] * (k - lo)


# --- workload --------------------------------------------------------------------------

def prompts(case, n, seed):
    """Deterministic prompts. `shared_prefix` cases share one long prefix and differ only in a suffix;
    other cases are unique per request (cold prefixes)."""
    rng = random.Random(f"{seed}:{case['name']}")
    words = case['prompt_words']

    def text(k, r):
        return ' '.join(r.choice(WORDS) for _ in range(k))
    if case.get('shared_prefix'):
        prefix = text(words, rng)
        return [f'{prefix}\nQuestion {i}: summarize the above in one sentence.' for i in range(n)]
    return [f'Request {i}. ' + text(words, random.Random(f"{seed}:{case['name']}:{i}")) for i in range(n)]


def cell_requests(workload, concurrency):
    spec = workload['requests_per_cell']
    return max(spec['min'], spec['per_concurrency'] * concurrency)


# --- process control -------------------------------------------------------------------

class GpuSampler:
    def __init__(self, interval=0.25):
        self.interval, self.samples, self._stop = interval, [], threading.Event()
        self.thread = None

    def __enter__(self):
        if shutil_which('nvidia-smi'):
            self.thread = threading.Thread(target=self._run, daemon=True)
            self.thread.start()
        return self

    def _run(self):
        while not self._stop.is_set():
            try:
                out = subprocess.check_output(['nvidia-smi', '--query-gpu=memory.used,utilization.gpu',
                                               '--format=csv,noheader,nounits'], text=True, timeout=5)
                mem, util = (float(x) for x in out.splitlines()[0].split(','))
                self.samples.append((mem, util))
            except Exception:  # noqa: BLE001 - sampling is best effort and recorded as missing
                pass
            self._stop.wait(self.interval)

    def __exit__(self, *exc):
        self._stop.set()
        if self.thread:
            self.thread.join(timeout=2)

    def summary(self):
        if not self.samples:
            return {'samples': 0, 'peak_memory_mb': None, 'mean_utilization_pct': None}
        return {'samples': len(self.samples), 'peak_memory_mb': max(m for m, _ in self.samples),
                'mean_utilization_pct': statistics.mean(u for _, u in self.samples)}


def shutil_which(name):
    import shutil
    return shutil.which(name)


def host_identity():
    info = {'python': sys.version.split()[0], 'platform': platform.platform(), 'gpus': []}
    if shutil_which('nvidia-smi'):
        try:
            out = subprocess.check_output(['nvidia-smi', '--query-gpu=name,memory.total,driver_version,uuid',
                                           '--format=csv,noheader'], text=True, timeout=10)
            info['gpus'] = [dict(zip(('name', 'memory_total', 'driver', 'uuid'), (x.strip() for x in line.split(','))))
                            for line in out.splitlines() if line.strip()]
        except Exception as exc:  # noqa: BLE001
            info['gpu_error'] = str(exc)[:200]
    try:
        info['commit'] = subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip()
        info['dirty'] = bool(subprocess.check_output(['git', '-C', str(ROOT), 'status', '--porcelain',
                                                      '--untracked-files=no'], text=True).strip())
    except Exception:  # noqa: BLE001
        info['commit'] = None
    return info


def launch(profile, port, log_path):
    argv = [a.replace('{port}', str(port)) for a in profile['argv']]
    env = {**os.environ, **{k: str(v) for k, v in (profile.get('env') or {}).items()}}
    if profile.get('path_prepend'):
        env['PATH'] = profile['path_prepend'] + os.pathsep + env.get('PATH', '')
    log = open(log_path, 'ab')
    proc = subprocess.Popen(argv, cwd=profile.get('cwd') or ROOT, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)
    return proc, log, argv


def stop(proc, grace=30):
    if proc.poll() is not None:
        return proc.returncode
    try:
        os.killpg(proc.pid, signal.SIGTERM)
    except ProcessLookupError:
        return proc.poll()
    try:
        proc.wait(grace)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        proc.wait(10)
    return proc.returncode


async def wait_ready(base, model, proc, timeout):
    deadline = time.monotonic() + timeout
    async with httpx.AsyncClient(timeout=5) as client:
        while time.monotonic() < deadline:
            if proc.poll() is not None:
                return False, f'process exited with {proc.returncode} before ready'
            try:
                served = [m.get('id') for m in (await client.get(base + '/v1/models')).json().get('data', [])]
                if model in served:
                    return True, served
            except Exception:  # noqa: BLE001 - not ready yet
                pass
            await asyncio.sleep(1)
    return False, 'readiness timeout'


async def identity(base):
    out = {}
    async with httpx.AsyncClient(timeout=10) as client:
        for path in ('/v1/models', '/version', '/get_server_info', '/v1/native/status'):
            try:
                r = await client.get(base + path)
                if r.status_code == 200:
                    out[path] = r.json()
            except Exception:  # noqa: BLE001 - optional identity endpoints
                pass
    return out


# --- measurement -------------------------------------------------------------------------

async def one(client, base, model, prompt, max_tokens, temperature, timeout):
    m = StreamMeasurement(time.perf_counter())
    error = None
    try:
        async with client.stream('POST', base + '/v1/completions', timeout=timeout, json={
                'model': model, 'prompt': prompt, 'max_tokens': max_tokens, 'temperature': temperature,
                'stream': True, 'stream_options': {'include_usage': True}}) as response:
            response.raise_for_status()
            async for line in response.aiter_lines():
                if line.startswith('data:'):
                    m.feed(line[5:].strip(), time.perf_counter())
                    if m.done:
                        break
    except Exception as exc:  # noqa: BLE001 - recorded per request
        error = f'{type(exc).__name__}: {exc}'[:300]
    r = m.result(time.perf_counter(), error)
    tokens, ttft = r['output_tokens'], r['ttft_ms']
    decode = (r['e2e_ms'] - ttft) / (tokens - 1) if r['success'] and tokens and tokens > 1 and ttft is not None \
        else None
    return {'success': r['success'], 'error': r['error'], 'ttft_ms': ttft, 'e2e_ms': r['e2e_ms'],
            'output_tokens': tokens, 'client_decode_ms': decode, 'finish_reason': r['finish_reason'],
            'content_chunks': r['content_chunks']}


SPEC_KEYS = ('spec_steps', 'spec_row_steps', 'spec_accepted', 'spec_step_ms', 'decode_steps',
             'packed_prefills', 'packed_rows', 'packed_prefix_hits', 'packed_prefix_groups', 'packed_tokens_saved',
             'packed_ms', 'gpu_thread_ms', 'gpu_thread_calls', 'spec_host_ms')


async def ttft_breakdown(base):
    """Native engine only: cumulative TTFT split + spec lane counters from /v1/native/status."""
    try:
        async with httpx.AsyncClient() as client:
            r = await client.get(base + '/v1/native/status', timeout=10)
        if r.status_code != 200:
            return None
        for diag in (r.json().get('models') or {}).values():
            if isinstance(diag, dict) and isinstance(diag.get('ttft_breakdown'), dict):
                sd = diag.get('static_decode') or {}
                return {**diag['ttft_breakdown'], **{'lane_' + k: sd.get(k, 0) or 0 for k in SPEC_KEYS}}
    except Exception:
        return None
    return None


def breakdown_delta(before, after):
    """Per-request means over one cell: queue/wait/prefill/total ms, decode steps and other
    jobs' prefill forwards that ran between a request's prefill prepare and its first token."""
    if not before or not after:
        return None
    d = {k: after[k] - before.get(k, 0) for k in after}
    n = d.pop('requests', 0)
    if n <= 0:
        return None
    out = {'requests': n, **{k + '_mean': round(v / n, 3) for k, v in d.items() if not k.startswith('lane_')}}
    steps, rows = d.get('lane_spec_steps', 0), d.get('lane_spec_row_steps', 0)
    out['lane'] = {'decode_steps': d.get('lane_decode_steps', 0), 'spec_steps': steps,
                   'spec_acceptance': round(d.get('lane_spec_accepted', 0) / rows, 4) if rows else None,
                   'tokens_per_row_step': round(1 + d.get('lane_spec_accepted', 0) / rows, 4) if rows else None,
                   'rows_per_step': round(rows / steps, 3) if steps else None,
                   'spec_step_ms': round(d.get('lane_spec_step_ms', 0) / steps, 3) if steps else None,
                   **{k: d.get('lane_' + k, 0) for k in ('packed_prefills', 'packed_rows', 'packed_prefix_hits',
                                                         'packed_prefix_groups', 'packed_tokens_saved')},
                   'packed_ms_per_prefill': round(d.get('lane_packed_ms', 0) / d['lane_packed_prefills'], 3)
                   if d.get('lane_packed_prefills') else None,
                   # P6: totals over the cell (ms); compare with the cell's wall_s
                   'packed_ms_total': round(d.get('lane_packed_ms', 0), 1),
                   'spec_step_ms_total': round(d.get('lane_spec_step_ms', 0), 1),
                   'spec_host_ms_total': round(d.get('lane_spec_host_ms', 0), 1),
                   'gpu_thread_ms_total': round(d.get('lane_gpu_thread_ms', 0), 1),
                   'gpu_thread_calls': d.get('lane_gpu_thread_calls', 0)}
    return out


async def run_cell(base, model, texts, max_tokens, concurrency, temperature, timeout):
    inflight = {'now': 0, 'max': 0, 'area': 0.0, 'last': time.perf_counter()}

    def tick(delta):
        now = time.perf_counter()
        inflight['area'] += inflight['now'] * (now - inflight['last'])
        inflight['last'] = now
        inflight['now'] += delta
        inflight['max'] = max(inflight['max'], inflight['now'])
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(limits=httpx.Limits(max_connections=concurrency * 2 + 4)) as client:
        async def task(i, text):
            async with sem:
                tick(+1)
                try:
                    return {'i': i, **await one(client, base, model, text, max_tokens, temperature, timeout)}
                finally:
                    tick(-1)
        start = time.perf_counter()
        rows = await asyncio.gather(*(task(i, t) for i, t in enumerate(texts)))
        wall = time.perf_counter() - start
    tick(0)
    return rows, {'wall_s': wall, 'attained_max_concurrency': inflight['max'],
                  'attained_mean_concurrency': inflight['area'] / wall if wall else None}


async def run_profile(profile, workload, round_dir, ready_timeout, request_timeout):
    round_dir.mkdir(parents=True, exist_ok=False)  # immutable: never reuse a run directory
    port = free_port()
    base = f'http://127.0.0.1:{port}'
    proc, log, argv = launch(profile, port, round_dir / 'server.log')
    status = {'profile': profile['name'], 'argv': argv, 'port': port, 'started_at': time.time()}
    try:
        t0 = time.monotonic()
        ready, detail = await wait_ready(base, profile['model'], proc, ready_timeout)
        status.update(ready=ready, ready_s=round(time.monotonic() - t0, 3))
        if not ready:
            status['error'] = str(detail)[:500]
            return status
        status['identity'] = await identity(base)
        cells = []
        with (round_dir / 'requests.jsonl').open('w') as out:
            for case in workload['cases']:
                warm = prompts({**case, 'name': case['name'] + ':warmup'}, workload['warmup_requests'],
                               workload['seed'])
                async with httpx.AsyncClient() as client:
                    for text in warm:  # unscored warmup
                        await one(client, base, profile['model'], text, case['max_tokens'],
                                  workload['temperature'], request_timeout)
                for concurrency in workload['concurrency']:
                    n = cell_requests(workload, concurrency)
                    texts = prompts(case, n, workload['seed'])
                    before = await ttft_breakdown(base)
                    with GpuSampler() as gpu:
                        rows, cell = await run_cell(base, profile['model'], texts, case['max_tokens'], concurrency,
                                                    workload['temperature'], request_timeout)
                    cell.update(case=case['name'], concurrency=concurrency, requests=n, gpu=gpu.summary())
                    split = breakdown_delta(before, await ttft_breakdown(base))
                    if split:
                        cell['ttft_breakdown'] = split
                    cells.append(cell)
                    for row in rows:
                        out.write(json.dumps({'profile': profile['name'], 'case': case['name'],
                                              'concurrency': concurrency, **row}) + '\n')
        status['cells'] = cells
        return status
    finally:
        status['exit_code'] = stop(proc)
        status['stopped_at'] = time.time()
        log.close()
        (round_dir / 'status.json').write_text(json.dumps(status, indent=2, default=str))


async def run(args):
    profiles = json.loads(Path(args.profiles).read_text())
    workload = json.loads(Path(args.workload).read_text()) if args.workload else DEFAULT_WORKLOAD
    selected = [p for p in profiles['profiles'] if not args.only or p['name'] in args.only]
    if not selected:
        raise SystemExit('no profiles selected')
    run_id = args.run_id or time.strftime('%Y%m%dT%H%M%SZ', time.gmtime())
    out = Path(args.out) / run_id
    out.mkdir(parents=True, exist_ok=False)
    manifest = {'run_id': run_id, 'created_at': time.time(), 'host': host_identity(), 'rounds': args.rounds,
                'workload': workload, 'workload_sha256': sha(workload), 'profiles': selected,
                'profiles_sha256': sha(selected), 'model': profiles.get('model'),
                'metric_definitions': __doc__.split('Metric definitions')[1].split('\n\n')[0]}
    (out / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str))
    for r in range(args.rounds):
        order = selected[r % len(selected):] + selected[:r % len(selected)]  # rotate: alternate execution order
        for profile in order:
            print(f'round {r} profile {profile["name"]}', flush=True)
            status = await run_profile(profile, workload, out / f'round-{r}' / profile['name'], args.ready_timeout,
                                       args.request_timeout)
            print(json.dumps({k: status.get(k) for k in ('profile', 'ready', 'ready_s', 'error')}), flush=True)
    print(f'run complete: {out}')
    return out


# --- aggregation ---------------------------------------------------------------------------

def _round_metrics(rows, cell):
    ok = [r for r in rows if r['success']]
    tokens = [r['output_tokens'] for r in ok]
    complete_usage = bool(ok) and all(isinstance(t, int) for t in tokens)
    return {
        'requests': len(rows), 'success_rate': len(ok) / len(rows) if rows else None,
        'ttft_p50_ms': pct([r['ttft_ms'] for r in ok if r['ttft_ms'] is not None], .5),
        'ttft_p95_ms': pct([r['ttft_ms'] for r in ok if r['ttft_ms'] is not None], .95),
        'e2e_p50_ms': pct([r['e2e_ms'] for r in ok], .5),
        'e2e_p95_ms': pct([r['e2e_ms'] for r in ok], .95),
        'client_decode_p50_ms': pct([r['client_decode_ms'] for r in ok if r['client_decode_ms'] is not None], .5),
        'output_tps': sum(tokens) / cell['wall_s'] if complete_usage and cell and cell.get('wall_s') else None,
        'mean_output_tokens': statistics.mean(tokens) if complete_usage else None,
        'attained_mean_concurrency': (cell or {}).get('attained_mean_concurrency'),
        'peak_memory_mb': ((cell or {}).get('gpu') or {}).get('peak_memory_mb'),
    }


METRICS_BETTER = {'ttft_p50_ms': 'min', 'ttft_p95_ms': 'min', 'e2e_p50_ms': 'min', 'e2e_p95_ms': 'min',
                  'client_decode_p50_ms': 'min', 'output_tps': 'max', 'success_rate': 'max',
                  'peak_memory_mb': 'min'}


def aggregate(run_dirs, baseline=None, min_rounds=3):
    per = {}       # (profile, case, conc) -> [round metrics]
    status = {}    # profile -> list of round statuses
    manifests = []
    for run_dir in map(Path, run_dirs):
        manifests.append(json.loads((run_dir / 'manifest.json').read_text()))
        for round_dir in sorted(run_dir.glob('round-*')):
            for prof_dir in sorted(p for p in round_dir.iterdir() if p.is_dir()):
                st = json.loads((prof_dir / 'status.json').read_text()) if (prof_dir / 'status.json').exists() \
                    else {'ready': False, 'error': 'no status (interrupted)'}
                status.setdefault(prof_dir.name, []).append({k: st.get(k) for k in ('ready', 'error', 'ready_s')})
                if not st.get('ready'):
                    continue
                cells = {(c['case'], c['concurrency']): c for c in st.get('cells') or []}
                rows = {}
                path = prof_dir / 'requests.jsonl'
                for line in (path.read_text().splitlines() if path.exists() else []):
                    row = json.loads(line)
                    rows.setdefault((row['case'], row['concurrency']), []).append(row)
                for key, cell_rows in rows.items():
                    per.setdefault((prof_dir.name, *key), []).append(_round_metrics(cell_rows, cells.get(key)))
    profiles = sorted(status)
    baseline = baseline if baseline in profiles else (profiles[0] if profiles else None)
    cells = []
    for (profile, case, conc), rounds in sorted(per.items(), key=lambda kv: (kv[0][1], kv[0][2], kv[0][0])):
        summary = {}
        for metric in list(METRICS_BETTER) + ['mean_output_tokens', 'attained_mean_concurrency']:
            values = [r[metric] for r in rounds if r.get(metric) is not None]
            summary[metric] = {'median': statistics.median(values), 'min': min(values), 'max': max(values),
                               'rounds': len(values)} if values else {'median': None, 'missing': True,
                                                                     'rounds': 0}
        cells.append({'profile': profile, 'case': case, 'concurrency': conc, 'rounds': len(rounds),
                      'metrics': summary})
    index = {(c['profile'], c['case'], c['concurrency']): c for c in cells}
    for cell in cells:
        base = index.get((baseline, cell['case'], cell['concurrency']))
        if base is None or cell['profile'] == baseline:
            continue
        comparison = {}
        for metric, better in METRICS_BETTER.items():
            a, b = cell['metrics'][metric], base['metrics'][metric]
            if a.get('median') is None or b.get('median') is None:
                comparison[metric] = {'relation': 'missing'}
                continue
            ratio = a['median'] / b['median'] if b['median'] else None
            if min(a['rounds'], b['rounds']) < min_rounds:
                relation = 'insufficient_rounds'
            elif a['min'] > b['max']:
                relation = 'behind' if better == 'min' else 'ahead'
            elif a['max'] < b['min']:
                relation = 'ahead' if better == 'min' else 'behind'
            else:
                relation = 'overlap'
            comparison[metric] = {'ratio_to_baseline': ratio, 'relation': relation}
        tok_a = cell['metrics']['mean_output_tokens'].get('median')
        tok_b = base['metrics']['mean_output_tokens'].get('median')
        comparison['equal_work'] = None if tok_a is None or tok_b is None else abs(tok_a - tok_b) <= 0.05 * tok_b
        cell['vs_baseline'] = comparison
    unsupported = {p: s for p, s in status.items() if not all(x['ready'] for x in s)}
    return {'version': 1, 'baseline': baseline, 'min_rounds_for_relation': min_rounds,
            'runs': [m['run_id'] for m in manifests], 'hosts': [m['host'] for m in manifests],
            'workload_sha256': sorted({m['workload_sha256'] for m in manifests}),
            'profiles': {p: {'rounds': len(s), 'ready_rounds': sum(1 for x in s if x['ready'])} for p, s in
                         status.items()},
            'unsupported_or_failed': unsupported, 'cells': cells,
            'claim': 'measurement only: relations need >= %d rounds with non-overlapping ranges; not a release '
                     'decision; quality is evaluated separately' % min_rounds}


def markdown(card):
    lines = [f"# Engine scorecard", '', f"Runs: {', '.join(card['runs'])} · baseline: `{card['baseline']}` · "
             f"{card['claim']}", '']
    if card['unsupported_or_failed']:
        lines += ['**Unsupported / failed profiles:** ' + ', '.join(
            f"`{p}` ({next((x['error'] for x in s if x.get('error')), 'not ready')})"
            for p, s in card['unsupported_or_failed'].items()), '']
    lines += ['| case | conc | profile | rounds | TTFT p50 ms | E2E p95 ms | decode ms/tok | out tok/s | '
              'vs baseline (tok/s) | equal work |', '|---|---|---|---|---|---|---|---|---|---|']

    def fmt(m):
        return '—' if m.get('median') is None else (f"{m['median']:.1f}" if m['rounds'] < 2 else
                                                       f"{m['median']:.1f} [{m['min']:.1f}–{m['max']:.1f}]")
    for c in card['cells']:
        m, vs = c['metrics'], c.get('vs_baseline') or {}
        rel = vs.get('output_tps', {})
        relation = '' if not vs else f"{rel.get('relation')}" + (
            f" ×{rel['ratio_to_baseline']:.2f}" if rel.get('ratio_to_baseline') else '')
        lines.append(f"| {c['case']} | {c['concurrency']} | {c['profile']} | {c['rounds']} | {fmt(m['ttft_p50_ms'])} "
                     f"| {fmt(m['e2e_p95_ms'])} | {fmt(m['client_decode_p50_ms'])} | {fmt(m['output_tps'])} | "
                     f"{relation} | {'' if not vs else vs.get('equal_work')} |")
    return '\n'.join(lines) + '\n'


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    r = sub.add_parser('run')
    r.add_argument('--profiles', required=True)
    r.add_argument('--workload')
    r.add_argument('--out', default='results/scorecard')
    r.add_argument('--rounds', type=int, default=3)
    r.add_argument('--run-id')
    r.add_argument('--only', nargs='*')
    r.add_argument('--ready-timeout', type=float, default=1200)
    r.add_argument('--request-timeout', type=float, default=300)
    a = sub.add_parser('aggregate')
    a.add_argument('run_dirs', nargs='+')
    a.add_argument('--baseline')
    a.add_argument('--min-rounds', type=int, default=3)
    a.add_argument('--out')
    args = parser.parse_args()
    if args.command == 'run':
        out = asyncio.run(run(args))
        card = aggregate([out], baseline=json.loads(Path(args.profiles).read_text()).get('baseline'), min_rounds=3)
        (out / 'scorecard.json').write_text(json.dumps(card, indent=2, default=str))
        (out / 'scorecard.md').write_text(markdown(card))
        print((out / 'scorecard.md').read_text())
        return
    card = aggregate(args.run_dirs, baseline=args.baseline, min_rounds=args.min_rounds)
    target = Path(args.out) if args.out else Path(args.run_dirs[0])
    target.mkdir(parents=True, exist_ok=True)
    (target / 'scorecard.json').write_text(json.dumps(card, indent=2, default=str))
    (target / 'scorecard.md').write_text(markdown(card))
    print(markdown(card))


if __name__ == '__main__':
    os.environ.setdefault('PYTHONUNBUFFERED', '1')
    main()
