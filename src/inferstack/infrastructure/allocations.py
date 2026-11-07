"""Resource allocation ownership: intent before launch, identity, release, orphans.

Lifecycle: intent -> active -> released, with `orphaned` when cleanup could not be
proven. Intent is persisted (fenced by the job lease) *before* any external side
effect, so a crash at any point leaves a row the reconciler can resolve.

Idempotency: an allocation is keyed by (job, step, step-attempt, purpose). A
repeated request for the same key returns the existing allocation instead of
creating a second resource.
"""
from __future__ import annotations

import json
import uuid

LIVE_STATES = ('intent', 'active', 'releasing')


class AllocationManager:
    def __init__(self, store, worker_id, faults=None):
        self.store = store
        self.worker_id = worker_id
        self.faults = faults or (lambda point: None)

    @staticmethod
    def _row(row):
        out = {k: row[k] for k in row.keys()}
        out['spec'] = json.loads(out['spec'])
        out['external'] = json.loads(out['external']) if out['external'] else None
        return out

    def intent(self, job, step_key, *, purpose, provider, resource_type, spec, gpu_count=0):
        key = f"{job['id']}:{step_key}:{job.get('step_attempt', 1)}:{purpose}"
        with self.store.transaction() as db:
            self.store.owned(db, job['id'], job['lease_token'])
            existing = db.execute('SELECT * FROM allocations WHERE idempotency_key=?', (key,)).fetchone()
            if existing is not None:
                return self._row(existing)
            now = self.store.clock()
            alloc_id = 'alloc-' + uuid.uuid4().hex[:20]
            db.execute('INSERT INTO allocations(id,idempotency_key,job_id,step_key,worker_id,provider,resource_type,'
                       'spec,state,gpu_count,created_at,updated_at) VALUES(?,?,?,?,?,?,?,?,?,?,?,?)',
                       (alloc_id, key, job['id'], step_key, self.worker_id, provider, resource_type,
                        self.store.encode(spec), 'intent', gpu_count, now, now))
            self.store.event(db, job['id'], 'allocation_intent', {'allocation_id': alloc_id, 'purpose': purpose})
        self.faults('allocation_intent')
        return self.get(alloc_id)

    def activate(self, alloc_id, job, external):
        with self.store.transaction() as db:
            self.store.owned(db, job['id'], job['lease_token'])
            now = self.store.clock()
            cursor = db.execute("UPDATE allocations SET state='active',external=?,active_at=?,updated_at=? "
                                "WHERE id=? AND state='intent'", (self.store.encode(external), now, now, alloc_id))
            if not cursor.rowcount:
                raise RuntimeError(f'allocation {alloc_id} is not in intent state')
            self.store.event(db, job['id'], 'allocation_active', {'allocation_id': alloc_id})
        self.faults('allocation_active')

    def record_external(self, alloc_id, external):
        """Record identity of a launched resource even if activation cannot be fenced."""
        with self.store.transaction() as db:
            db.execute('UPDATE allocations SET external=COALESCE(external,?),updated_at=? WHERE id=?',
                       (self.store.encode(external), self.store.clock(), alloc_id))

    def release(self, alloc_id, note):
        """Mark released. Not lease-fenced: cleanup must always be recordable."""
        with self.store.transaction() as db:
            now = self.store.clock()
            row = db.execute('SELECT * FROM allocations WHERE id=?', (alloc_id,)).fetchone()
            if row is None:
                raise KeyError(alloc_id)
            if row['state'] == 'released':
                return self._row(row)
            db.execute("UPDATE allocations SET state='released',released_at=?,updated_at=?,note=? WHERE id=?",
                       (now, now, note, alloc_id))
            if row['job_id']:
                self.store.event(db, row['job_id'], 'allocation_released', {'allocation_id': alloc_id, 'note': note})
        return self.get(alloc_id)

    def mark_orphaned(self, alloc_id, note):
        with self.store.transaction() as db:
            now = self.store.clock()
            row = db.execute('SELECT job_id FROM allocations WHERE id=?', (alloc_id,)).fetchone()
            db.execute("UPDATE allocations SET state='orphaned',updated_at=?,note=? WHERE id=?", (now, note, alloc_id))
            if row and row['job_id']:
                self.store.event(db, row['job_id'], 'allocation_orphaned', {'allocation_id': alloc_id, 'note': note})

    def get(self, alloc_id):
        with self.store.connection() as db:
            row = db.execute('SELECT * FROM allocations WHERE id=?', (alloc_id,)).fetchone()
        if row is None:
            raise KeyError(alloc_id)
        return self._row(row)

    def list(self, *, state=None, job_id=None, step_key=None):
        sql, params = 'SELECT * FROM allocations WHERE 1=1', []
        for column, value in (('state', state), ('job_id', job_id), ('step_key', step_key)):
            if value is not None:
                sql += f' AND {column}=?'
                params.append(value)
        with self.store.connection() as db:
            return [self._row(r) for r in db.execute(sql + ' ORDER BY created_at,id', params)]

    def gpu_seconds(self, alloc):
        if not alloc['active_at'] or not alloc['released_at']:
            return 0.0
        return max(0.0, alloc['released_at'] - alloc['active_at']) * alloc['gpu_count']
