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
SERVING_PREFIX = 'serving:'  # owner of a kept candidate's runtime (not reconciled as a dead worker)


class ResourceConflict(RuntimeError):
    """The requested devices are held by another live allocation."""


class QuotaExceeded(ResourceConflict):
    """A provider quota (live allocations or GPUs) would be exceeded."""


def held_devices(db, provider):
    """Device identifiers held by live (or orphaned, i.e. unproven-free) allocations of a provider."""
    held = {}
    for row in db.execute('SELECT id, spec FROM allocations WHERE provider=? AND state IN (?,?,?,?)',
                          (provider, *LIVE_STATES, 'orphaned')).fetchall():
        for device in (json.loads(row['spec']) or {}).get('gpu_devices') or []:
            held[device] = row['id']
    return held


def place_devices(db, provider, spec, gpu_count):
    """Resolve and exclusively reserve identified devices inside the caller's writer transaction.

    `spec['gpu_devices']`: explicit devices, checked for exclusivity.
    `spec['gpu_pool']`: the host's identified devices; `gpu_count` free ones are chosen.
    Returns the spec with `gpu_devices` set. A GPU reservation without identified devices
    fails closed: a count alone is accounting, not an exclusive lease."""
    spec = dict(spec or {})
    held = held_devices(db, provider)
    explicit = list(spec.get('gpu_devices') or [])
    if explicit:
        clash = sorted(d for d in explicit if d in held)
        if clash:
            raise ResourceConflict(f'devices {clash} held by allocation {held[clash[0]]}')
        return spec
    if not gpu_count:
        return spec
    pool = list(spec.pop('gpu_pool', None) or [])
    if not pool:
        raise ResourceConflict(f'{gpu_count} GPU(s) requested without identified devices; configure the '
                               'host GPU inventory (device UUIDs) or an explicit gpu_devices reservation')
    free = [d for d in pool if d not in held]
    if len(free) < gpu_count:
        raise ResourceConflict(f'{gpu_count} GPU(s) requested, {len(free)} of {len(pool)} identified devices free')
    spec['gpu_devices'] = free[:gpu_count]
    return spec


class AllocationManager:
    def __init__(self, store, worker_id, faults=None, quotas=None):
        """quotas: {provider: {'max_live': int, 'max_gpus': int}} enforced at intent time."""
        self.store = store
        self.worker_id = worker_id
        self.faults = faults or (lambda point: None)
        self.quotas = quotas or {}

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
            quota = self.quotas.get(provider)
            if quota:
                live = db.execute('SELECT COUNT(*), COALESCE(SUM(gpu_count),0) FROM allocations WHERE provider=? '
                                  'AND state IN (?,?,?,?)', (provider, *LIVE_STATES, 'orphaned')).fetchone()
                if quota.get('max_live') is not None and live[0] + 1 > quota['max_live']:
                    raise QuotaExceeded(f'{provider}: {live[0]} live allocations, quota {quota["max_live"]}')
                if quota.get('max_gpus') is not None and live[1] + gpu_count > quota['max_gpus']:
                    raise QuotaExceeded(f'{provider}: {live[1]}+{gpu_count} GPUs exceeds quota {quota["max_gpus"]}')
            if (spec or {}).get('gpu_devices') or (spec or {}).get('gpu_pool'):
                # Exclusive device reservation: experiments never share a GPU with serving
                # or with another experiment on the same provider; placement is chosen
                # atomically under the writer lock.
                spec = place_devices(db, provider, spec, gpu_count)
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

    def transfer_to_serving(self, alloc_id, deployment_id, job):
        """Hand a kept candidate's live runtime from the experiment worker to serving ownership.

        The job lease fences the transfer; afterwards worker reconciliation no longer
        stops it. ReleaseRuntime owns its health, replacement and retirement.
        """
        owner = SERVING_PREFIX + deployment_id
        with self.store.transaction() as db:
            self.store.owned(db, job['id'], job['lease_token'])
            cursor = db.execute("UPDATE allocations SET worker_id=?,updated_at=? WHERE id=? AND state='active'",
                                (owner, self.store.clock(), alloc_id))
            if not cursor.rowcount:
                raise RuntimeError(f'allocation {alloc_id} is not active; cannot retain it for serving')
            self.store.event(db, job['id'], 'allocation_retained', {'allocation_id': alloc_id, 'owner': owner})
        return self.get(alloc_id)

    def rescale(self, alloc_id, gpu_count, *, max_gpus=None):
        """Record a change of GPUs held (scale up/down). GPU-seconds integrate the scale history."""
        if isinstance(gpu_count, bool) or not isinstance(gpu_count, int) or gpu_count < 0:
            raise ValueError('gpu_count must be a non-negative integer')
        with self.store.transaction() as db:
            row = db.execute('SELECT * FROM allocations WHERE id=?', (alloc_id,)).fetchone()
            if row is None or row['state'] != 'active':
                raise RuntimeError(f'allocation {alloc_id} is not active')
            quota = self.quotas.get(row['provider'])
            if quota and quota.get('max_gpus') is not None:
                others = db.execute('SELECT COALESCE(SUM(gpu_count),0) FROM allocations WHERE provider=? AND id<>? '
                                    'AND state IN (?,?,?,?)', (row['provider'], alloc_id, *LIVE_STATES,
                                                              'orphaned')).fetchone()[0]
                if others + gpu_count > quota['max_gpus']:
                    raise QuotaExceeded(f"{row['provider']}: rescale to {gpu_count} GPUs exceeds quota")
            spec = json.loads(row['spec'])
            history = spec.get('scale_history') or [[row['active_at'], row['gpu_count']]]
            now = self.store.clock()
            history.append([now, gpu_count])
            spec['scale_history'] = history
            db.execute('UPDATE allocations SET gpu_count=?,spec=?,updated_at=? WHERE id=?',
                       (gpu_count, self.store.encode(spec), now, alloc_id))
            if row['job_id']:
                self.store.event(db, row['job_id'], 'allocation_rescaled', {'allocation_id': alloc_id,
                                                                            'gpu_count': gpu_count})

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
