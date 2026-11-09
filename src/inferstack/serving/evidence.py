"""Live serving evidence: per-request phase measurements, durable windows, Prometheus export.

Each request through the gateway produces one record with separate phases:
- `upstream_first_byte_ms`: first bytes from the engine (streams only);
- `first_content_ms`: first non-empty generated content (streams only; role, usage
  and finish events are metadata and never count);
- `e2e_ms`: full completion, streamed or not;
- `max_gap_ms`: longest interval between content events (a stall signal; content
  events are NOT tokens);
- `output_tokens` from engine usage when reported, and `itl_ms`: the mean observed
  client-side gap between tokens, only when every token arrived as its own content
  event (events == authoritative tokens). Grouped/coalesced events cannot show when
  tokens were generated, so ITL is unavailable for them (never 0). This is client
  timing, not engine token-emission timing; engine decode timing comes from engine
  telemetry (verifier `tpot_engine_ms`). The `itl` series is a per-request mean, so
  its percentiles are over requests, not over individual token gaps;
- `outcome`: ok | error | cancelled | timeout | rejected, with `rejection_source`
  for admission rejections (draining, unhealthy, capacity).

Records are written to the durable `serving_events` table (migration 6), so a
gateway restart or several gateway workers keep complete accounting: rollout windows
are queried from the store, never from one process's memory. Raw prompts and
outputs are never stored or used as metric labels. Percentiles are computed from
pooled raw samples, never averaged across replicas or workers.
"""
from __future__ import annotations

import json
import math
import socket
import threading
import time
import uuid
from dataclasses import asdict, dataclass, field

OUTCOMES = ('ok', 'error', 'cancelled', 'timeout', 'rejected')
# Smallest sample count at which a percentile is reported: 10 / (1 - p).
TAIL_MIN = {0.5: 1, 0.95: 200, 0.99: 1000}


def percentile(values, p):
    values = sorted(v for v in values if v is not None)
    if not values:
        return None
    k = (len(values) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(values) - 1)
    return values[lo] + (values[hi] - values[lo]) * (k - lo)


class LiveStream:
    """Incremental, non-raising SSE measurement for a proxied stream.

    Buffers partial events; times an event when its terminating blank line arrives.
    A malformed or error event marks the stream as errored without interrupting the
    client (the proxy forwards bytes unchanged)."""

    def __init__(self, started):
        self.started = started
        self.buffer = b''
        self.first_byte = None
        self.content_times = []
        self.tokens = None
        self.input_tokens = None
        self.done = False
        self.error = None

    def feed(self, chunk, now):
        if self.first_byte is None and chunk:
            self.first_byte = now
        self.buffer += chunk.replace(b'\r\n', b'\n')
        *events, self.buffer = self.buffer.split(b'\n\n')
        for event in events:
            data = [line[5:].strip() for line in event.split(b'\n') if line.startswith(b'data:')]
            if not data:
                continue
            payload = b'\n'.join(data)
            if payload == b'[DONE]':
                self.done = True
                continue
            try:
                item = json.loads(payload)
            except ValueError:
                self.error = self.error or 'malformed stream event'
                continue
            if not isinstance(item, dict):
                continue
            if 'error' in item:
                self.error = self.error or 'engine stream error'
                continue
            usage = item.get('usage') or {}
            count = usage.get('completion_tokens') if isinstance(usage, dict) else None
            if isinstance(count, int) and not isinstance(count, bool) and count >= 0:
                self.tokens = count
            prompt = usage.get('prompt_tokens') if isinstance(usage, dict) else None
            if isinstance(prompt, int) and not isinstance(prompt, bool) and prompt >= 0:
                self.input_tokens = prompt
            for choice in item.get('choices') or []:
                piece = choice.get('text')
                if piece is None:
                    piece = (choice.get('delta') or {}).get('content')
                if isinstance(piece, str) and piece:
                    self.content_times.append(now)

    def phases(self, ended):
        ms = lambda t: (t - self.started) * 1000 if t is not None else None
        times = self.content_times
        gaps = [(b - a) * 1000 for a, b in zip(times, times[1:])]
        itl = None
        if self.tokens is not None and self.tokens > 1 and len(times) == self.tokens:
            itl = (times[-1] - times[0]) * 1000 / (self.tokens - 1)  # one observed event per token
        return {'upstream_first_byte_ms': ms(self.first_byte), 'first_content_ms': ms(times[0]) if times else None,
                'e2e_ms': ms(ended), 'max_gap_ms': max(gaps) if gaps else None, 'content_events': len(times),
                'output_tokens': self.tokens, 'input_tokens': self.input_tokens, 'itl_ms': itl}


@dataclass
class RequestRecord:
    release_id: str
    deployment_id: str
    outcome: str
    stream: bool
    at: float = field(default_factory=time.time)
    upstream_first_byte_ms: float | None = None
    first_content_ms: float | None = None
    e2e_ms: float | None = None
    max_gap_ms: float | None = None
    content_events: int | None = None
    output_tokens: int | None = None
    itl_ms: float | None = None
    rejection_source: str | None = None
    status: int | None = None
    endpoint: str | None = None  # replica chosen by the endpoint picker, when the data plane reports it
    input_tokens: int | None = None
    shadow: bool = False  # mirrored copy: excluded from client-facing windows unless asked for

    def __post_init__(self):
        if self.outcome not in OUTCOMES:
            raise ValueError(f'unknown outcome {self.outcome!r}')


COLUMNS = ('id', 'at', 'worker', 'release_id', 'deployment_id', 'outcome', 'stream', 'status', 'rejection_source',
           'upstream_first_byte_ms', 'first_content_ms', 'e2e_ms', 'max_gap_ms', 'content_events', 'output_tokens',
           'itl_ms', 'endpoint', 'input_tokens', 'shadow')


class EvidenceStore:
    """Durable per-request serving evidence shared by every gateway worker."""

    def __init__(self, repository, *, worker=None):
        self.repository = repository
        self.worker = worker or f'{socket.gethostname()}:{uuid.uuid4().hex[:8]}'

    def record(self, record: RequestRecord):
        row = {**asdict(record), 'id': uuid.uuid4().hex, 'worker': self.worker, 'stream': int(record.stream),
               'shadow': int(record.shadow)}
        with self.repository.transaction() as db:
            db.execute(f'INSERT INTO serving_events({",".join(COLUMNS)}) VALUES({",".join("?" * len(COLUMNS))})',
                       tuple(row[c] for c in COLUMNS))

    def rows(self, *, deployment_id=None, release_id=None, since, until=None, shadow=False):
        """shadow=False: client traffic only; True: shadow copies only; None: both."""
        sql, params = 'SELECT * FROM serving_events WHERE at>=?', [since]
        if shadow is not None:
            sql, params = sql + ' AND COALESCE(shadow,0)=?', params + [int(bool(shadow))]
        if until is not None:
            sql, params = sql + ' AND at<?', params + [until]
        for column, value in (('deployment_id', deployment_id), ('release_id', release_id)):
            if value is not None:
                sql, params = sql + f' AND {column}=?', params + [value]
        with self.repository.connection() as db:
            return [{k: r[k] for k in r.keys()} for r in db.execute(sql + ' ORDER BY at', params)]

    def window(self, *, deployment_id=None, release_id=None, since, until=None, shadow=False):
        until = until if until is not None else time.time()
        rows = self.rows(deployment_id=deployment_id, release_id=release_id, since=since, until=until, shadow=shadow)
        return summarize(rows, since=since, until=until)

    def prune(self, before):
        with self.repository.transaction() as db:
            return db.execute('DELETE FROM serving_events WHERE at<?', (before,)).rowcount


def summarize(rows, *, since=None, until=None):
    """Window summary with explicit sample counts. Percentiles below their minimum
    sample count are None (never a guess); error and cancellation denominators are
    kept separate from rejections (requests that never reached the engine)."""
    served = [r for r in rows if r['outcome'] != 'rejected']
    ok = [r for r in served if r['outcome'] == 'ok']
    out = {'window': {'since': since, 'until': until}, 'requests': len(served),
           'errors': sum(1 for r in served if r['outcome'] in ('error', 'timeout')),
           'timeouts': sum(1 for r in served if r['outcome'] == 'timeout'),
           'cancelled': sum(1 for r in served if r['outcome'] == 'cancelled'),
           'rejected': sum(1 for r in rows if r['outcome'] == 'rejected'),
           'rejection_sources': {}, 'samples': {}}
    for r in rows:
        if r['outcome'] == 'rejected':
            out['rejection_sources'][r['rejection_source'] or 'unknown'] = \
                out['rejection_sources'].get(r['rejection_source'] or 'unknown', 0) + 1
    # Legacy field used by existing rollout stages: first byte for streams, full latency otherwise.
    ttfb = [r['upstream_first_byte_ms'] if r['stream'] else r['e2e_ms'] for r in ok]
    out['ttfb_p50_ms'] = percentile(ttfb, 0.5)
    out['latency_samples'] = sum(1 for v in ttfb if v is not None)
    metrics = {'ttft': [r['first_content_ms'] for r in ok if r['stream']],
               'e2e': [r['e2e_ms'] for r in ok], 'itl': [r['itl_ms'] for r in ok],
               'stall': [r['max_gap_ms'] for r in ok if r['stream']]}
    for name, values in metrics.items():
        values = [v for v in values if v is not None and math.isfinite(v)]
        out['samples'][name] = len(values)
        for p, label in ((0.5, 'p50'), (0.95, 'p95'), (0.99, 'p99')):
            out[f'{name}_{label}_ms'] = percentile(values, p) if len(values) >= TAIL_MIN[p] else None
    # Streams that produced no generated content are not "fast": they count separately.
    out['streams_without_content'] = sum(1 for r in ok if r['stream'] and r['first_content_ms'] is None)
    return out


class Metrics:
    """Process-local Prometheus exposition (text format 0.0.4), labelled by release and
    deployment only. Each gateway worker exports its own counters; Prometheus sums them."""
    BUCKETS = (5, 10, 25, 50, 100, 250, 500, 1000, 2500, 5000, 10000, 30000, 60000)
    HISTOGRAMS = {'first_content_ms': 'inferstack_gateway_ttft_ms', 'e2e_ms': 'inferstack_gateway_e2e_ms',
                  'itl_ms': 'inferstack_gateway_itl_ms', 'max_gap_ms': 'inferstack_gateway_stall_ms'}

    def __init__(self):
        self.lock = threading.Lock()
        self.counts, self.hist = {}, {}

    def observe(self, record: RequestRecord):
        labels = (record.release_id, record.deployment_id)
        with self.lock:
            key = (*labels, record.outcome)
            self.counts[key] = self.counts.get(key, 0) + 1
            for attr, name in self.HISTOGRAMS.items():
                value = getattr(record, attr)
                if value is None or record.outcome != 'ok':
                    continue
                h = self.hist.setdefault((name, *labels), {'buckets': [0] * len(self.BUCKETS), 'sum': 0.0, 'count': 0})
                for i, bound in enumerate(self.BUCKETS):
                    if value <= bound:
                        h['buckets'][i] += 1
                h['sum'] += value
                h['count'] += 1

    @staticmethod
    def _labels(release, deployment, **extra):
        esc = lambda v: str(v).replace('\\', '\\\\').replace('"', '\\"').replace('\n', '\\n')
        pairs = {'release': release, 'deployment': deployment, **extra}
        return '{' + ','.join(f'{k}="{esc(v)}"' for k, v in pairs.items()) + '}'

    def render(self, inflight=None):
        lines = ['# TYPE inferstack_gateway_requests_total counter']
        with self.lock:
            for (release, deployment, outcome), n in sorted(self.counts.items()):
                lines.append(f'inferstack_gateway_requests_total{self._labels(release, deployment, outcome=outcome)} {n}')
            for name in sorted(set(self.HISTOGRAMS.values())):
                lines.append(f'# TYPE {name} histogram')
                for (hname, release, deployment), h in sorted(self.hist.items()):
                    if hname != name:
                        continue
                    for bound, count in zip(self.BUCKETS, h['buckets']):
                        lines.append(f'{name}_bucket{self._labels(release, deployment, le=bound)} {count}')
                    lines.append(f'{name}_bucket{self._labels(release, deployment, le="+Inf")} {h["count"]}')
                    lines.append(f'{name}_sum{self._labels(release, deployment)} {h["sum"]}')
                    lines.append(f'{name}_count{self._labels(release, deployment)} {h["count"]}')
        if inflight:
            lines.append('# TYPE inferstack_gateway_inflight gauge')
            for deployment, n in sorted(inflight.items()):
                lines.append(f'inferstack_gateway_inflight{{deployment="{deployment}"}} {n}')
        return '\n'.join(lines) + '\n'
