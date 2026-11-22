"""G6: SLO watchdog. Serving below the SLO opens a settings campaign on its own, inside a spend cap.

A deployment's SLO is a load it must carry (`rate_rps`) with at least `min_good` of requests meeting every
per-request limit (TTFT and per-token, the goodput rule). Each `tick` takes one observation window, judges
it, and moves a small per-deployment state machine stored as a `watch` record:

  healthy/suspect -- `consecutive` breached windows --> open an incident, run the campaign (settings tune +
                     interleaved A/B confirm; no model calls, no code changes)
  campaign keep and the candidate's goodput@99 carries `rate_rps`
                  --> `apply` given: apply the patch, status `verifying`
                      no `apply`:    status `fix_pending` (promotion is a separate, gated step)
  verifying       -- `consecutive` healthy windows --> incident closed `resolved`
                  -- a breached window              --> incident closed `fix_failed`, status `escalated`
  campaign without a fix (no candidate, reject, inconclusive, keep but short of the rate)
                  --> incident closed `unresolved`, status `escalated`: settings are not enough, the next
                      step is a code change (G7) or a human; no new campaign until the deployment changes
  infra_failed    --> incident closed `infra_failed`, status `cooldown` (retried after `cooldown_s`)
  spend cap       --> a campaign that would pass `daily_cap_usd` is not started: incident `blocked_budget`

Windows with fewer than `min_requests` requests are `unmeasured` and never count as healthy or breached.
A deployment revision change (a promotion, a new model or GPU) clears `fix_pending`/`escalated` and starts
watching again. Observers: `probe_observer` (a short open-loop run at `rate_rps` against the live endpoint;
works for every engine) and `telemetry_observer` (normalized Prometheus p99 estimates; vLLM/SGLang).
"""
from __future__ import annotations

import asyncio
import inspect
import time
from dataclasses import asdict, dataclass

from inferstack.evaluation.goodput import GoodputCase

KEEP_WINDOWS = 12


@dataclass(frozen=True)
class ServingSLO:
    rate_rps: float
    cases: tuple            # GoodputCase limits, one per traffic class
    min_good: float = 0.99


def judge_window(window, slo, *, min_requests=20):
    """('breach' | 'ok' | 'unmeasured', reasons)."""
    if not window or (window.get('requests') or 0) < min_requests:
        return 'unmeasured', [f"{(window or {}).get('requests') or 0} requests < {min_requests}"]
    reasons = []
    if window.get('good_frac') is not None and window['good_frac'] < slo.min_good:
        reasons.append(f"good fraction {window['good_frac']:.2f} < {slo.min_good} at {slo.rate_rps} req/s")
    # engine p99s cover all traffic classes at once: judged against the loosest limit, so a breach is certain
    ttft, tpot = max(c.ttft_ms for c in slo.cases), max(c.tpot_ms for c in slo.cases)
    if window.get('ttft_p99_ms') is not None and window['ttft_p99_ms'] > ttft:
        reasons.append(f"p99 TTFT {window['ttft_p99_ms']:.0f} ms > {ttft:.0f} ms")
    if window.get('tpot_p99_ms') is not None and window['tpot_p99_ms'] > tpot:
        reasons.append(f"p99 per-token {window['tpot_p99_ms']:.1f} ms > {tpot:.1f} ms")
    if window.get('good_frac') is None and window.get('ttft_p99_ms') is None and window.get('tpot_p99_ms') is None:
        return 'unmeasured', ['no latency signal in window']
    return ('breach' if reasons else 'ok'), reasons


def probe_observer(slo, *, seconds=20.0, min_requests=20, warmup=1):
    """async (deployment body) -> window: open-loop requests at the SLO rate against the live endpoint."""
    from inferstack.engines.managed import DeploymentSpec, ManagedEngine
    from inferstack.evaluation.goodput import sweep

    async def observe(deployment):
        engine = ManagedEngine(DeploymentSpec(id=deployment['id'], engine='generic_openai',
                                              model=deployment['model'], base_url=deployment['base_url']))
        try:
            out = await sweep(engine.stream, list(slo.cases), rates=(slo.rate_rps,), seconds_per_rate=seconds,
                              min_requests=min_requests, warmup=warmup, stop_below=0.0)
        except Exception as exc:  # noqa: BLE001 - an unreachable endpoint is unmeasured, not healthy
            return {'source': 'probe', 'requests': 0, 'error': f'{type(exc).__name__}: {exc}'[:200]}
        finally:
            await engine.client.aclose()
        steps = [c['steps'][0] for c in out['cases'].values() if c['steps']]
        n, good = sum(s['requests'] for s in steps), sum(s['good'] for s in steps)
        misses = {}
        for s in steps:
            for k, v in s['misses'].items():
                misses[k] = misses.get(k, 0) + v
        return {'source': 'probe', 'rate_rps': slo.rate_rps, 'requests': n, 'good_frac': good / n if n else None,
                'misses': misses}
    return observe


def telemetry_observer(snapshot):
    """async (deployment body) -> window from a normalized telemetry snapshot (control.wiring.telemetry_source)."""
    async def observe(deployment):
        snap = await snapshot(deployment['id'])
        metrics = snap.get('metrics') or {}

        def p99(name):
            return (metrics.get(name) or {}).get('p99_estimate')
        return {'source': 'telemetry', 'requests': (metrics.get('engine_e2e_ms') or {}).get('count') or 0,
                'ttft_p99_ms': p99('engine_ttft_ms'), 'tpot_p99_ms': p99('engine_tpot_ms')}
    return observe


def fix_from(result, slo):
    """The campaign's patch when it was kept and its measured goodput@99 carries the SLO rate, else None."""
    if (result or {}).get('decision') != 'keep':
        return None
    metric = (((result.get('verdict') or {}).get('metrics') or {}).get('goodput99_rps') or {})
    carried = (metric.get('candidate') or {}).get('median')
    best = (result.get('tune') or {}).get('best') or {}
    if carried is None or carried < slo.rate_rps or not best.get('patch'):
        return None
    return {'patch': best['patch'], 'goodput99_rps': carried}


async def _call(fn, *args):
    if inspect.iscoroutinefunction(fn):
        return await fn(*args)
    return await asyncio.to_thread(fn, *args)


class Watchdog:
    """One SLO, many deployments. `campaign(deployment_id, incident) -> result` (TuneAction.run shape);
    `apply(deployment_id, patch) -> None` optional; `notify(event)` optional (escalations, budget blocks)."""

    def __init__(self, repository, observe, campaign, slo, *, apply=None, notify=None, consecutive=3,
                 cooldown_s=6 * 3600.0, daily_cap_usd=10.0, campaign_estimate_usd=1.0, min_requests=20,
                 clock=time.time):
        self.repository, self.observe, self.campaign, self.slo = repository, observe, campaign, slo
        self.apply, self.notify, self.consecutive, self.cooldown_s = apply, notify, consecutive, cooldown_s
        self.daily_cap_usd, self.campaign_estimate_usd = daily_cap_usd, campaign_estimate_usd
        self.min_requests, self.clock = min_requests, clock

    # --- records ----------------------------------------------------------------------

    def _read(self, kind, key):
        try:
            r = self.repository.read(kind, key)
            return r['body'], r['revision']
        except KeyError:
            return None, None

    def _write(self, kind, key, body, revision):
        return self.repository.put(kind, key, body, expected_revision=revision)

    def state(self, deployment_id):
        return self._read('watch', deployment_id)[0]

    def incident(self, incident_id):
        return self._read('incident', incident_id)[0]

    def _spent_today(self, watch):
        day = time.strftime('%Y-%m-%d', time.gmtime(self.clock()))
        spend = watch.get('spend') or {}
        return day, (spend.get('usd', 0.0) if spend.get('day') == day else 0.0)

    def _close(self, watch, outcome, **fields):
        inc, rev = self._read('incident', watch['incident'])
        inc.update(fields, outcome=outcome, closed_at=self.clock())
        self._write('incident', watch['incident'], inc, rev)
        watch['last_closed_at'] = self.clock()
        watch['last_incident'], watch['incident'] = watch['incident'], None
        if outcome not in ('resolved',) and self.notify:
            self.notify({'kind': 'watchdog_' + outcome, 'deployment_id': watch['deployment_id'],
                         'incident': watch['last_incident'], **fields})
        return inc

    # --- the loop step -------------------------------------------------------------------

    async def tick(self, deployment_id):
        dep = self.repository.read('deployment', deployment_id)
        watch, wrev = self._read('watch', deployment_id)
        watch = watch or {'deployment_id': deployment_id, 'status': 'healthy', 'streak': 0, 'ok_streak': 0,
                          'windows': [], 'incident': None, 'incidents': 0, 'deployment_revision': dep['revision']}
        if dep['revision'] != watch.get('deployment_revision'):
            # promoted fix, new model or new GPU: watch the new revision from scratch
            watch['deployment_revision'] = dep['revision']
            if watch['status'] in ('fix_pending', 'escalated'):
                watch['status'] = 'verifying' if watch['status'] == 'fix_pending' else 'healthy'
            watch['streak'] = watch['ok_streak'] = 0
        window = {**(await self.observe(dep['body'])), 'at': self.clock(), 'deployment_revision': dep['revision']}
        verdict, reasons = judge_window(window, self.slo, min_requests=self.min_requests)
        window.update(verdict=verdict, reasons=reasons)
        watch['windows'] = (watch['windows'] + [window])[-KEEP_WINDOWS:]
        status = watch['status']

        if status == 'cooldown' and self.clock() >= watch.get('last_closed_at', 0) + self.cooldown_s:
            status = watch['status'] = 'healthy'
            watch['streak'] = 0
        if status == 'verifying':
            if verdict == 'ok':
                watch['ok_streak'] += 1
                if watch['ok_streak'] >= self.consecutive:
                    self._close(watch, 'resolved', verified_windows=watch['ok_streak'])
                    watch['status'], watch['ok_streak'] = 'healthy', 0
            elif verdict == 'breach':
                self._close(watch, 'fix_failed', reasons=reasons)
                watch['status'] = 'escalated'
        elif status in ('healthy', 'suspect'):
            if verdict == 'breach':
                watch['streak'] += 1
                watch['status'] = 'suspect'
            elif verdict == 'ok':
                watch['streak'], watch['status'] = 0, 'healthy'
            if watch['streak'] >= self.consecutive:
                await self._open(watch, deployment_id, dep)
        self._write('watch', deployment_id, watch, wrev)
        return watch

    async def _open(self, watch, deployment_id, dep):
        watch['incidents'] += 1
        incident_id = f'{deployment_id}--inc{watch["incidents"]:04d}'
        breaches = [w for w in watch['windows'] if w['verdict'] == 'breach'][-self.consecutive:]
        incident = {'id': incident_id, 'deployment_id': deployment_id, 'deployment_revision': dep['revision'],
                    'opened_at': self.clock(), 'slo': {**asdict(self.slo), 'cases': [asdict(c) for c in self.slo.cases]},
                    'trigger': {'kind': 'slo_breach', 'windows': breaches}, 'outcome': None}
        self._write('incident', incident_id, incident, None)
        watch['incident'], watch['streak'] = incident_id, 0
        day, spent = self._spent_today(watch)
        if spent + self.campaign_estimate_usd > self.daily_cap_usd:
            self._close(watch, 'blocked_budget', spent_today_usd=spent, estimate_usd=self.campaign_estimate_usd,
                        cap_usd=self.daily_cap_usd)
            watch['status'] = 'cooldown'
            return
        watch['status'] = 'campaign'
        try:
            result = await _call(self.campaign, deployment_id, incident)
        except Exception as exc:  # noqa: BLE001 - a crashed campaign is an infra failure, retried after cooldown
            result = {'decision': 'infra_failed', 'reason': f'{type(exc).__name__}: {exc}'[:300], 'cost_usd': 0.0}
        cost = float(result.get('cost_usd') or 0.0)
        watch['spend'] = {'day': day, 'usd': round(spent + cost, 4)}
        summary = {'decision': result.get('decision'), 'reason': result.get('reason'), 'cost_usd': cost,
                   'best': (result.get('tune') or {}).get('best'),
                   'verdict_reasons': ((result.get('verdict') or {}).get('reasons') or [])[:5]}
        inc, rev = self._read('incident', incident_id)
        inc['campaign'] = summary
        self._write('incident', incident_id, inc, rev)
        fix = fix_from(result, self.slo)
        if fix is not None:
            inc, rev = self._read('incident', incident_id)
            inc['fix'] = fix
            self._write('incident', incident_id, inc, rev)
            if self.apply is None:
                watch['status'] = 'fix_pending'
                return
            await _call(self.apply, deployment_id, fix['patch'])
            watch['status'], watch['ok_streak'] = 'verifying', 0
            watch['deployment_revision'] = self.repository.read('deployment', deployment_id)['revision']
            return
        if result.get('decision') == 'infra_failed':
            self._close(watch, 'infra_failed', cost_usd=cost)
            watch['status'] = 'cooldown'
            return
        self._close(watch, 'unresolved', cost_usd=cost,
                    next='settings cannot carry the SLO rate; needs a code change (G7) or a human')
        watch['status'] = 'escalated'


__all__ = [
    'GoodputCase',
    'ServingSLO',
    'Watchdog',
    'fix_from',
    'judge_window',
    'probe_observer',
    'telemetry_observer',
]
