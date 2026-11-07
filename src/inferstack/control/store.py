"""Durable job store: leases, fencing, steps, budgets, deadlines, retries, workers.

All worker mutations require the live lease token of a running job (fencing).
A stale worker (expired or superseded lease) can never publish a result, step
outcome or usage charge. The store runs on SQLite or PostgreSQL with identical
semantics (see storage/db.py).
"""
from __future__ import annotations

import json
import math
import socket
import os
import time
import uuid
from contextlib import contextmanager

from inferstack.storage.db import open_database

BUDGET_KEYS = ('wall_seconds', 'gpu_seconds', 'provider_tokens', 'spend_usd', 'experiments')
TERMINAL = {'completed', 'failed', 'cancelled'}


class LeaseLost(RuntimeError):
    pass


class Store:
    """Job store. `path` is a SQLite path or a database URL (sqlite:///, postgresql://)."""

    def __init__(self, path, clock=time.time):
        self.database = open_database(path)
        self.path = self.database.url
        self.clock = clock
        from inferstack.storage.migrations import migrate
        self.schema_version = migrate(self.database, self.clock())

    @contextmanager
    def connection(self):
        with self.database.connection() as db:
            yield db

    @contextmanager
    def transaction(self):
        with self.database.transaction() as db:
            yield db

    @staticmethod
    def encode(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

    def event(self, db, job_id, kind, body):
        db.execute('INSERT INTO events(job_id,kind,body,at) VALUES(?,?,?,?)',
                   (job_id, kind, self.encode(body), self.clock()))

    # --- submission ---------------------------------------------------------

    @staticmethod
    def validate_budget(budget):
        if budget is None:
            return None
        if not isinstance(budget, dict) or not set(budget) <= set(BUDGET_KEYS):
            raise ValueError('budget keys must be a subset of ' + ','.join(BUDGET_KEYS))
        for key, value in budget.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'budget {key} must be positive and finite')
        return budget

    def submit(self, request_key, payload, *, budget=None, deadline_at=None, max_attempts=1):
        if not isinstance(request_key, str) or not request_key.strip():
            raise ValueError('nonempty request_key required')
        budget = self.validate_budget(budget)
        if deadline_at is not None and (not isinstance(deadline_at, (int, float)) or not math.isfinite(deadline_at)):
            raise ValueError('deadline_at must be an absolute UTC epoch timestamp')
        if isinstance(max_attempts, bool) or not isinstance(max_attempts, int) or not 1 <= max_attempts <= 10:
            raise ValueError('max_attempts must be in [1,10]')
        encoded = self.encode(payload)
        controls = self.encode({'budget': budget, 'deadline_at': deadline_at, 'max_attempts': max_attempts})
        with self.transaction() as db:
            existing = db.execute('SELECT * FROM jobs WHERE request_key=?', (request_key,)).fetchone()
            if existing:
                same = self.encode({'budget': json.loads(existing['budget']) if existing['budget'] else None,
                                    'deadline_at': existing['deadline_at'],
                                    'max_attempts': existing['max_attempts'] or 1}) == controls
                if existing['payload'] != encoded or not same:
                    raise ValueError('idempotency key reused with different payload')
                return existing['id']
            job_id = uuid.uuid4().hex
            now = self.clock()
            db.execute('INSERT INTO jobs(id,request_key,payload,state,created,updated,budget,usage,deadline_at,'
                       'max_attempts) VALUES(?,?,?,?,?,?,?,?,?,?)',
                       (job_id, request_key, encoded, 'pending', now, now,
                        self.encode(budget) if budget else None, self.encode({}), deadline_at, max_attempts))
            self.event(db, job_id, 'submitted', payload)
            return job_id

    # --- leases ---------------------------------------------------------------

    @staticmethod
    def duration(seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('lease duration must be positive and finite')

    def claim(self, lease_seconds=60, worker_id=None):
        self.duration(lease_seconds)
        with self.transaction() as db:
            now = self.clock()
            while True:
                row = db.execute("SELECT * FROM jobs WHERE (state='pending' AND (not_before IS NULL OR not_before<=?)) "
                                 "OR (state='running' AND lease_until<=?) ORDER BY created,id LIMIT 1",
                                 (now, now)).fetchone()
                if row is None:
                    return None
                if row['deadline_at'] is not None and row['deadline_at'] <= now:
                    error = {'type': 'DeadlineExceeded', 'retryable': False, 'message': 'deadline passed before claim'}
                    db.execute("UPDATE jobs SET state='failed',lease_token=NULL,lease_until=NULL,last_error=?,"
                               "result=?,updated=? WHERE id=?",
                               (self.encode(error), self.encode({'error': error}), now, row['id']))
                    self.event(db, row['id'], 'failed', {'error': error})
                    continue
                token = uuid.uuid4().hex
                db.execute("UPDATE jobs SET state='running',lease_token=?,lease_until=?,attempt=attempt+1,"
                           "worker_id=?,updated=? WHERE id=?",
                           (token, now + lease_seconds, worker_id, now, row['id']))
                self.event(db, row['id'], 'claimed', {'attempt': row['attempt'] + 1, 'worker_id': worker_id,
                                                      'recovered_from_expired_lease': row['state'] == 'running'})
                return {'id': row['id'], 'payload': json.loads(row['payload']), 'lease_token': token,
                        'attempt': row['attempt'] + 1, 'worker_id': worker_id,
                        'budget': json.loads(row['budget']) if row['budget'] else None,
                        'deadline_at': row['deadline_at']}

    def owned(self, db, job_id, token):
        row = db.execute("SELECT * FROM jobs WHERE id=? AND state='running' AND lease_token=? AND lease_until>?",
                         (job_id, token, self.clock())).fetchone()
        if row is None:
            raise LeaseLost(job_id)
        return row

    def heartbeat(self, job_id, token, lease_seconds=60):
        self.duration(lease_seconds)
        with self.transaction() as db:
            self.owned(db, job_id, token)
            db.execute('UPDATE jobs SET lease_until=?,updated=? WHERE id=?',
                       (self.clock() + lease_seconds, self.clock(), job_id))

    def finish(self, job_id, token, result, state='completed'):
        if state not in {'completed', 'failed'}:
            raise ValueError('invalid terminal state')
        encoded = self.encode(result)
        with self.transaction() as db:
            self.owned(db, job_id, token)
            error = self.encode(result.get('error')) if state == 'failed' and isinstance(result, dict) and result.get('error') else None
            db.execute('UPDATE jobs SET state=?,result=?,lease_token=NULL,lease_until=NULL,last_error=COALESCE(?,last_error),'
                       'updated=? WHERE id=?', (state, encoded, error, self.clock(), job_id))
            self.event(db, job_id, state, result)

    def fail(self, job_id, token, error, partial=None, backoff_seconds=5.0):
        """Terminal failure, or reschedule when the error is retryable and attempts remain."""
        with self.transaction() as db:
            row = self.owned(db, job_id, token)
            now = self.clock()
            retry = bool(error.get('retryable')) and row['attempt'] < (row['max_attempts'] or 1)
            if retry:
                db.execute("UPDATE jobs SET state='pending',lease_token=NULL,lease_until=NULL,worker_id=NULL,"
                           "last_error=?,not_before=?,updated=? WHERE id=?",
                           (self.encode(error), now + backoff_seconds * row['attempt'], now, job_id))
                self.event(db, job_id, 'retry_scheduled', {'error': error, 'attempt': row['attempt']})
                return 'pending'
            result = {'error': error, **({'partial': partial} if partial is not None else {})}
            db.execute("UPDATE jobs SET state='failed',result=?,lease_token=NULL,lease_until=NULL,last_error=?,"
                       "updated=? WHERE id=?", (self.encode(result), self.encode(error), now, job_id))
            self.event(db, job_id, 'failed', result)
            return 'failed'

    def cancel(self, job_id):
        # The lease token is retained (state no longer 'running', so it fences nothing)
        # so the cancelled worker can attach partial evidence exactly once.
        with self.transaction() as db:
            cursor = db.execute("UPDATE jobs SET state='cancelled',lease_until=NULL,updated=? "
                                "WHERE id=? AND state IN ('pending','running')", (self.clock(), job_id))
            if cursor.rowcount:
                self.event(db, job_id, 'cancelled', {})
            return bool(cursor.rowcount)

    def attach_partial(self, job_id, token, body):
        """A cancelled job's last lease holder may record partial evidence once."""
        with self.transaction() as db:
            row = db.execute("SELECT * FROM jobs WHERE id=? AND state='cancelled' AND lease_token=?",
                             (job_id, token)).fetchone()
            if row is None:
                raise LeaseLost(job_id)
            db.execute('UPDATE jobs SET lease_token=NULL,updated=? WHERE id=?', (self.clock(), job_id))
            self.event(db, job_id, 'partial_evidence', body)

    def get(self, job_id):
        with self.connection() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            result = {k: row[k] for k in row.keys()}
            result.pop('lease_token')
            for key in ('payload', 'result', 'budget', 'usage', 'last_error'):
                result[key] = json.loads(result[key]) if result.get(key) else (None if key != 'usage' else {})
            result['events'] = [{'kind': r['kind'], 'body': json.loads(r['body']), 'at': r['at']}
                                for r in db.execute('SELECT * FROM events WHERE job_id=? ORDER BY seq', (job_id,))]
            result['steps'] = self._steps(db, job_id)
            return result

    # --- steps ----------------------------------------------------------------

    def _steps(self, db, job_id):
        return [{'step_key': r['step_key'], 'state': r['state'], 'attempts': r['attempts'],
                 'idempotent': bool(r['idempotent']), 'worker_id': r['worker_id'],
                 'result': json.loads(r['result']) if r['result'] else None,
                 'error': json.loads(r['error']) if r['error'] else None}
                for r in db.execute('SELECT * FROM job_steps WHERE job_id=? ORDER BY started_at,step_key', (job_id,))]

    def step_begin(self, job_id, token, step_key, *, idempotent, worker_id=None):
        """Returns {'action': 'reuse'|'run'|'ambiguous', ...}.

        reuse: the step succeeded in an earlier attempt; its stored result applies.
        run: execute the step now (first attempt, or a safe re-run).
        ambiguous: a non-idempotent step was interrupted mid-flight; its side
        effects must be reconciled before it may run again.
        """
        with self.transaction() as db:
            self.owned(db, job_id, token)
            now = self.clock()
            row = db.execute('SELECT * FROM job_steps WHERE job_id=? AND step_key=?', (job_id, step_key)).fetchone()
            if row is not None and row['state'] == 'succeeded':
                return {'action': 'reuse', 'result': json.loads(row['result']) if row['result'] else None,
                        'attempts': row['attempts']}
            if row is not None and row['state'] == 'running' and not idempotent:
                return {'action': 'ambiguous', 'attempts': row['attempts'], 'worker_id': row['worker_id']}
            if row is None:
                db.execute('INSERT INTO job_steps(job_id,step_key,attempts,state,idempotent,worker_id,started_at) '
                           'VALUES(?,?,?,?,?,?,?)', (job_id, step_key, 1, 'running', int(idempotent), worker_id, now))
                attempts = 1
            else:
                attempts = row['attempts'] + 1
                db.execute("UPDATE job_steps SET attempts=?,state='running',worker_id=?,started_at=?,ended_at=NULL,"
                           "error=NULL WHERE job_id=? AND step_key=?", (attempts, worker_id, now, job_id, step_key))
            self.event(db, job_id, 'step_started', {'step': step_key, 'attempt': attempts})
            return {'action': 'run', 'attempts': attempts}

    def step_resolve_ambiguous(self, job_id, token, step_key, note):
        """Mark an interrupted non-idempotent step as safe to re-run after reconciliation."""
        with self.transaction() as db:
            self.owned(db, job_id, token)
            db.execute("UPDATE job_steps SET state='interrupted',ended_at=?,error=? WHERE job_id=? AND step_key=? "
                       "AND state='running'", (self.clock(), self.encode({'reconciled': note}), job_id, step_key))
            self.event(db, job_id, 'step_reconciled', {'step': step_key, 'note': note})

    def step_end(self, job_id, token, step_key, state, *, result=None, error=None):
        if state not in {'succeeded', 'failed', 'cancelled'}:
            raise ValueError('invalid step state')
        with self.transaction() as db:
            self.owned(db, job_id, token)
            db.execute('UPDATE job_steps SET state=?,ended_at=?,result=?,error=? WHERE job_id=? AND step_key=?',
                       (state, self.clock(), self.encode(result) if result is not None else None,
                        self.encode(error) if error is not None else None, job_id, step_key))
            self.event(db, job_id, 'step_' + state, {'step': step_key, **({'error': error} if error else {})})

    # --- budgets --------------------------------------------------------------

    def charge(self, job_id, token, amounts):
        """Add usage atomically under the lease. Returns the new usage totals."""
        for key, value in amounts.items():
            if key not in BUDGET_KEYS or key in ('experiments', 'wall_seconds', 'gpu_seconds') or isinstance(value, bool) or not isinstance(value, (int, float)) \
                    or not math.isfinite(value) or value < 0:
                raise ValueError(f'invalid usage charge {key}={value!r}')
        with self.transaction() as db:
            row = self.owned(db, job_id, token)
            usage = json.loads(row['usage']) if row['usage'] else {}
            for key, value in amounts.items():
                usage[key] = usage.get(key, 0) + value
            db.execute('UPDATE jobs SET usage=?,updated=? WHERE id=?', (self.encode(usage), self.clock(), job_id))
            self.event(db, job_id, 'usage_charged', amounts)
            return usage

    def exhausted(self, job_id):
        """Budget keys whose usage has reached the limit.

        Wall time counts from job creation. Experiments are derived from succeeded
        experiment steps, so the count cannot drift from the durable step record.
        """
        job = self.get(job_id)
        budget, usage = job['budget'] or {}, dict(job['usage'] or {})
        usage['wall_seconds'] = self.clock() - job['created']
        usage['experiments'] = sum(1 for s in job['steps']
                                   if s['step_key'].endswith('.experiment') and s['state'] == 'succeeded')
        now = self.clock()
        with self.connection() as db:
            rows = db.execute('SELECT active_at, released_at, gpu_count FROM allocations WHERE job_id=? '
                              'AND active_at IS NOT NULL', (job_id,)).fetchall()
        usage['gpu_seconds'] = usage.get('gpu_seconds', 0) + sum(
            max(0.0, (r['released_at'] or now) - r['active_at']) * r['gpu_count'] for r in rows)
        return sorted(k for k, limit in budget.items() if usage.get(k, 0) >= limit)

    # --- worker registry ------------------------------------------------------

    def register_worker(self, lease_seconds=60):
        self.duration(lease_seconds)
        worker_id = 'worker-' + uuid.uuid4().hex[:16]
        now = self.clock()
        with self.transaction() as db:
            db.execute('INSERT INTO workers VALUES(?,?,?,?,?,?,?)',
                       (worker_id, socket.gethostname(), os.getpid(), now, now, now + lease_seconds, 'active'))
        return worker_id

    def worker_heartbeat(self, worker_id, lease_seconds=60):
        now = self.clock()
        with self.transaction() as db:
            cursor = db.execute("UPDATE workers SET heartbeat_at=?,lease_until=? WHERE id=? AND state='active'",
                                (now, now + lease_seconds, worker_id))
            if not cursor.rowcount:
                raise LeaseLost(worker_id)

    def retire_worker(self, worker_id):
        with self.transaction() as db:
            db.execute("UPDATE workers SET state='retired',lease_until=? WHERE id=?", (self.clock(), worker_id))

    def workers(self):
        with self.connection() as db:
            return [{k: r[k] for k in r.keys()} for r in db.execute('SELECT * FROM workers ORDER BY started_at')]

    def worker_alive(self, db, worker_id):
        row = db.execute("SELECT 1 FROM workers WHERE id=? AND state='active' AND lease_until>?",
                         (worker_id, self.clock())).fetchone()
        return row is not None
