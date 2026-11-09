"""Serving cost attribution over a bounded interval (P5).

Gross infrastructure cost is every serving allocation's GPU-time inside the
interval (integrating its scale history: surge, idle, warm rollback and canary
capacity included) times its stamped price, plus operator-declared extra items
(storage, egress, CPU) with their own provenance. Each allocation is counted
exactly once, keyed by allocation ID, so shared capacity is never double counted.

Token-normalised figures use authoritative engine usage only (never SSE events):
- `usd_per_million_output_tokens`: gross / all output tokens served to clients;
- `usd_per_million_goodput_tokens`: gross / output tokens of requests that met the
  declared SLO (failed, cancelled, rejected and SLO-missing requests still cost
  money but produce no goodput).
A ratio is unavailable (None, with the reason) when any allocation has no known
price or any successful request lacks a token count. Shadow copies are cost
inside gross but never counted as client tokens.
"""
from __future__ import annotations

import json

from inferstack.infrastructure.allocations import SERVING_PREFIX


def _gpu_seconds(alloc, since, until):
    """GPU-seconds of one allocation inside [since, until), integrating its scale history."""
    start = alloc['active_at']
    if start is None:
        return 0.0
    end = alloc['released_at'] if alloc['released_at'] is not None else until
    history = (alloc['spec'] or {}).get('scale_history') or [[start, alloc['gpu_count']]]
    points = sorted([[float(t), int(g)] for t, g in history])
    total = 0.0
    for i, (t, gpus) in enumerate(points):
        seg_start = max(t, start, since)
        seg_end = min(points[i + 1][0] if i + 1 < len(points) else end, end, until)
        if seg_end > seg_start:
            total += (seg_end - seg_start) * gpus
    return total


def serving_cost(repository, *, since, until, slo=None, extra_costs=(), roles=None):
    """Cost report for serving between `since` and `until` (epoch seconds).

    slo: {'ttft_ms': x, 'e2e_ms': y} thresholds a request must meet to count as goodput.
    extra_costs: [{'item', 'usd', 'source', 'as_of'}] non-GPU costs for the same interval.
    roles: {deployment_id: 'stable'|'candidate'|'rollback'|...} for the capacity breakdown.
    """
    if not until > since:
        raise ValueError('until must be after since')
    slo, roles = slo or {}, roles or {}
    with repository.connection() as db:
        rows = db.execute('SELECT * FROM allocations WHERE worker_id LIKE ? AND active_at IS NOT NULL '
                          'AND active_at < ? AND (released_at IS NULL OR released_at > ?)',
                          (SERVING_PREFIX + '%', until, since)).fetchall()
        events = [{k: r[k] for k in r.keys()} for r in db.execute(
            'SELECT * FROM serving_events WHERE at>=? AND at<?', (since, until))]
    allocations, unpriced, gross_gpu, provenance = [], [], 0.0, {}
    by_role = {}
    for row in rows:
        alloc = {k: row[k] for k in row.keys()}
        alloc['spec'] = json.loads(alloc['spec']) if alloc['spec'] else {}
        seconds = _gpu_seconds(alloc, since, until)
        price = alloc['spec'].get('price')
        deployment = alloc['worker_id'][len(SERVING_PREFIX):]
        role = roles.get(deployment, 'unassigned')
        usd = None
        if seconds == 0:
            usd = 0.0
        elif price and price.get('usd_per_gpu_hour') is not None:
            usd = seconds / 3600 * price['usd_per_gpu_hour']
            provenance[f"{alloc['provider']}:{price.get('source')}"] = price.get('as_of')
        else:
            unpriced.append(alloc['id'])
        if usd is not None:
            gross_gpu += usd
        bucket = by_role.setdefault(role, {'gpu_seconds': 0.0, 'usd': 0.0, 'allocations': 0, 'unpriced': 0})
        bucket['gpu_seconds'] += seconds
        bucket['allocations'] += 1
        if usd is None:
            bucket['unpriced'] += 1
        else:
            bucket['usd'] += usd
        allocations.append({'id': alloc['id'], 'deployment': deployment, 'role': role, 'provider': alloc['provider'],
                            'gpu_seconds': seconds, 'usd': usd})
    extras = 0.0
    for item in extra_costs:
        if not item.get('source') or not item.get('as_of') or item.get('usd') is None or item['usd'] < 0:
            raise ValueError(f'extra cost {item.get("item")} needs a non-negative usd, source and as_of')
        extras += item['usd']
        provenance[f"{item['item']}:{item['source']}"] = item['as_of']
    client = [e for e in events if not e.get('shadow')]
    ok = [e for e in client if e['outcome'] == 'ok']
    missing_tokens = sum(1 for e in ok if e['output_tokens'] is None)
    output_tokens = sum(e['output_tokens'] or 0 for e in ok)
    input_tokens = sum(e['input_tokens'] or 0 for e in ok) if all(e.get('input_tokens') is not None for e in ok) \
        else None

    def meets(e):
        ttft = e['first_content_ms'] if e['stream'] else None
        if slo.get('ttft_ms') is not None and e['stream'] and (ttft is None or ttft > slo['ttft_ms']):
            return False
        if slo.get('e2e_ms') is not None and (e['e2e_ms'] is None or e['e2e_ms'] > slo['e2e_ms']):
            return False
        return True
    good = [e for e in ok if meets(e)]
    goodput_tokens = sum(e['output_tokens'] or 0 for e in good)
    gross = None if unpriced else gross_gpu + extras
    reasons = []
    if unpriced:
        reasons.append(f'{len(unpriced)} serving allocation(s) without a known price')
    if missing_tokens:
        reasons.append(f'{missing_tokens} successful request(s) without an authoritative token count')
    ratio = lambda tokens: None if reasons or gross is None or not tokens else gross / tokens * 1e6
    return {'scope': 'serving GPU allocations + declared extra costs', 'currency': 'USD',
            'interval': {'since': since, 'until': until, 'seconds': until - since},
            'gross_usd': gross, 'gpu_usd_known': gross_gpu, 'extra_usd': extras, 'unpriced_allocations': unpriced,
            'price_provenance': provenance, 'capacity_by_role': by_role, 'allocations': allocations,
            'requests': {'client': len(client), 'ok': len(ok), 'slo_met': len(good),
                         'failed': sum(1 for e in client if e['outcome'] in ('error', 'timeout')),
                         'cancelled': sum(1 for e in client if e['outcome'] == 'cancelled'),
                         'rejected': sum(1 for e in client if e['outcome'] == 'rejected'),
                         'shadow': len(events) - len(client)},
            'tokens': {'output': output_tokens if not missing_tokens else None, 'input': input_tokens,
                       'goodput_output': goodput_tokens if not missing_tokens else None},
            'slo': slo, 'usd_per_million_output_tokens': ratio(output_tokens),
            'usd_per_million_goodput_tokens': ratio(goodput_tokens), 'unavailable_reasons': reasons}


def reconcile_bill(report, billed_usd, *, tolerance=0.05):
    """Compare the attributed gross with the provider's bill for the same interval and scope."""
    if report['gross_usd'] is None:
        return {'reconciled': False, 'reason': 'attributed gross unavailable: ' + '; '.join(report['unavailable_reasons'])}
    if billed_usd <= 0:
        return {'reconciled': False, 'reason': 'billed amount must be positive'}
    delta = (report['gross_usd'] - billed_usd) / billed_usd
    return {'reconciled': abs(delta) <= tolerance, 'attributed_usd': report['gross_usd'], 'billed_usd': billed_usd,
            'relative_delta': delta, 'tolerance': tolerance}


def compare_capacity(fixed, scaled, *, request_tolerance=0.02):
    """Fixed vs autoscaled capacity on an equal workload and SLO; refuses unequal comparisons."""
    if fixed['slo'] != scaled['slo']:
        raise ValueError('comparison needs the same SLO')
    a, b = fixed['requests']['client'], scaled['requests']['client']
    if not a or abs(a - b) / a > request_tolerance:
        raise ValueError(f'workloads differ ({a} vs {b} client requests)')
    out = {}
    for key in ('gross_usd', 'usd_per_million_goodput_tokens'):
        x, y = fixed[key], scaled[key]
        out[key] = {'fixed': x, 'scaled': y, 'delta': None if x is None or y is None else y - x}
    out['slo_met_rate'] = {'fixed': fixed['requests']['slo_met'] / max(1, fixed['requests']['client']),
                           'scaled': scaled['requests']['slo_met'] / max(1, scaled['requests']['client'])}
    return out
