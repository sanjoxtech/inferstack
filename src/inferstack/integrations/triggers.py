"""Bounded automatic optimization triggers (periodic and SLO), feeding the same job pipeline.

A trigger is a stored spec: `{"deployment_id", "goal", "interval_s"}` (periodic) or
`{"deployment_id", "goal", "metric", "threshold", "direction": "above"|"below",
"window_s"}` (slo), plus mandatory bounds:
- `cooldown_s`: minimum time between firings;
- `budget`: per-job budget (required: triggered jobs never run unbudgeted);
- `max_jobs_per_day`: firing cap over a rolling 24 h;
- at most one unfinished job per trigger, and `max_active` unfinished triggered
  jobs across all triggers (TriggerService argument).
`tick()` is idempotent per evaluation window: the job request key includes the
trigger id and the window index, so concurrent tickers or retries create one job.
Admission is atomic: the cooldown/in-flight/global/daily checks, the job insert and
the firing-state update run in one writer transaction (SQLite BEGIN IMMEDIATE /
PostgreSQL writer advisory lock), re-reading the trigger row inside it, so tickers
straddling a window boundary cannot both pass admission.
SLO evaluation uses an observe(deployment_id, metric, window_s) callback that
returns a number or None; None (missing telemetry) never fires.
"""
from __future__ import annotations

import json
import math

TERMINAL = ('completed', 'failed', 'cancelled')


def validate_spec(kind, spec):
    if kind not in ('periodic', 'slo'):
        raise ValueError('trigger kind must be periodic or slo')
    for key in ('deployment_id', 'goal', 'cooldown_s', 'budget', 'max_jobs_per_day'):
        if key not in spec:
            raise ValueError(f'trigger spec requires {key}')
    def positive(key):
        value = spec.get(key)
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
            raise ValueError(f'{key} must be a positive number')
    positive('cooldown_s')
    if not isinstance(spec['max_jobs_per_day'], int) or not 1 <= spec['max_jobs_per_day'] <= 96:
        raise ValueError('max_jobs_per_day must be an integer in [1, 96]')
    if not isinstance(spec['budget'], dict) or not spec['budget']:
        raise ValueError('triggered jobs require a nonempty budget')
    if kind == 'periodic':
        positive('interval_s')
        if spec['interval_s'] < spec['cooldown_s']:
            raise ValueError('interval_s must be >= cooldown_s')
    else:
        positive('window_s')
        if spec.get('direction') not in ('above', 'below') or not isinstance(spec.get('metric'), str):
            raise ValueError('slo triggers need metric and direction above|below')
        if isinstance(spec.get('threshold'), bool) or not isinstance(spec.get('threshold'), (int, float)):
            raise ValueError('slo threshold must be a number')
    return spec


class TriggerService:
    def __init__(self, store, *, observe=None, max_active=2):
        self.store, self.observe, self.max_active = store, observe, max_active

    def create(self, trigger_id, kind, spec):
        validate_spec(kind, spec)
        store = self.store
        store.validate_budget(spec['budget'])
        now = store.clock()
        with store.transaction() as db:
            if db.execute('SELECT 1 FROM triggers WHERE id=?', (trigger_id,)).fetchone():
                raise ValueError(f'trigger {trigger_id} exists')
            db.execute('INSERT INTO triggers(id,kind,spec,state,fired_count,created_at,updated_at) '
                       'VALUES(?,?,?,?,?,?,?)', (trigger_id, kind, store.encode(spec), 'active', 0, now, now))
        return self.get(trigger_id)

    def get(self, trigger_id):
        with self.store.connection() as db:
            row = db.execute('SELECT * FROM triggers WHERE id=?', (trigger_id,)).fetchone()
        if row is None:
            raise KeyError(trigger_id)
        return {**{k: row[k] for k in row.keys()}, 'spec': json.loads(row['spec'])}

    def list(self):
        with self.store.connection() as db:
            ids = [r['id'] for r in db.execute('SELECT id FROM triggers ORDER BY id')]
        return [self.get(i) for i in ids]

    def set_state(self, trigger_id, state):
        if state not in ('active', 'paused'):
            raise ValueError('state must be active or paused')
        with self.store.transaction() as db:
            db.execute('UPDATE triggers SET state=?,updated_at=? WHERE id=?', (state, self.store.clock(), trigger_id))

    def _unfinished(self, db, trigger_id=None):
        sql = "SELECT COUNT(*) FROM jobs WHERE request_key LIKE ? AND state NOT IN ('completed','failed','cancelled')"
        return db.execute(sql, (f'trigger:{trigger_id}:%' if trigger_id else 'trigger:%',)).fetchone()[0]

    async def _breached(self, spec):
        if self.observe is None:
            return False, None
        value = await self.observe(spec['deployment_id'], spec['metric'], spec['window_s'])
        if value is None or not math.isfinite(value):
            return False, value
        return (value > spec['threshold'] if spec['direction'] == 'above' else value < spec['threshold']), value

    async def tick(self):
        now = self.store.clock()
        outcomes = []
        for trigger in self.list():
            spec, tid = trigger['spec'], trigger['id']
            if trigger['state'] != 'active':
                outcomes.append({'trigger': tid, 'action': 'paused'})
                continue
            if trigger['last_fired_at'] is not None and now - trigger['last_fired_at'] < spec['cooldown_s']:
                outcomes.append({'trigger': tid, 'action': 'cooldown'})
                continue
            if trigger['kind'] == 'periodic':
                due = trigger['last_fired_at'] is None or now - trigger['last_fired_at'] >= spec['interval_s']
                value, window = None, int(now // spec['interval_s'])
            else:
                due, value = await self._breached(spec)
                window = int(now // spec['cooldown_s'])
            if not due:
                outcomes.append({'trigger': tid, 'action': 'not_due', 'observed': value})
                continue
            payload = {'deployment_id': spec['deployment_id'], 'goal': spec['goal'], 'source': f'trigger:{trigger["kind"]}',
                       'trigger_id': tid, 'evidence_ids': [], **({'observed': {spec['metric']: value}}
                                                                  if trigger['kind'] == 'slo' else {})}
            with self.store.transaction() as db:
                current = db.execute('SELECT state,last_fired_at FROM triggers WHERE id=?', (tid,)).fetchone()
                if current is None or current['state'] != 'active':
                    outcomes.append({'trigger': tid, 'action': 'paused'})
                    continue
                if current['last_fired_at'] is not None and now - current['last_fired_at'] < spec['cooldown_s']:
                    outcomes.append({'trigger': tid, 'action': 'cooldown'})
                    continue
                fired_today = db.execute("SELECT COUNT(*) FROM jobs WHERE request_key LIKE ? AND created>?",
                                         (f'trigger:{tid}:%', now - 86400)).fetchone()[0]
                if self._unfinished(db, tid):
                    outcomes.append({'trigger': tid, 'action': 'job_in_flight'})
                    continue
                if self._unfinished(db) >= self.max_active:
                    outcomes.append({'trigger': tid, 'action': 'global_limit'})
                    continue
                if fired_today >= spec['max_jobs_per_day']:
                    outcomes.append({'trigger': tid, 'action': 'daily_limit'})
                    continue
                job_id = self.store.submit_in(db, f'trigger:{tid}:{window}', payload, budget=spec['budget'])
                db.execute('UPDATE triggers SET last_fired_at=?,last_job_id=?,fired_count=fired_count+1,updated_at=? '
                           'WHERE id=? AND (last_job_id IS NULL OR last_job_id<>?)', (now, job_id, now, tid, job_id))
            outcomes.append({'trigger': tid, 'action': 'fired', 'job_id': job_id, 'observed': value})
        return outcomes
