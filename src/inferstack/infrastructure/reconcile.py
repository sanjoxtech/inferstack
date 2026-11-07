"""Reconcile allocations left behind by dead or superseded workers.

Runs at worker start and on operator request. A live allocation is reconciled
when its owning worker's lease has expired. Outcomes:

- intent, no recorded identity: search for the allocation tag. Found -> stop it
  (the launch happened but was not recorded). Not found -> released, never launched.
- active (or identity recorded): stop the identified process if still alive.
- any stop that cannot be confirmed -> `orphaned`, visible to operators; `recover`
  retries, or an operator force-releases with a note after manual cleanup.

Allocations owned by other hosts are left for that host's reconciler.
"""
from __future__ import annotations

from .allocations import LIVE_STATES, AllocationManager


def _stop(provider, external):
    return provider.stop(external)


def reconcile_allocation(manager: AllocationManager, provider, alloc) -> str:
    if alloc['external'] is None:
        pids = provider.find_tagged(alloc['id'])
        if not pids:
            manager.release(alloc['id'], 'never_launched')
            return 'never_launched'
        ok = all(_stop(provider, provider.external_for(pid)) for pid in pids)
        if ok:
            manager.release(alloc['id'], 'unrecorded_launch_stopped')
            return 'unrecorded_launch_stopped'
        manager.mark_orphaned(alloc['id'], 'unrecorded_launch_stop_failed')
        return 'orphaned'
    if not provider.owns(alloc['external']):
        return 'foreign_host'
    if not provider.alive(alloc['external']):
        manager.release(alloc['id'], 'already_exited')
        return 'already_exited'
    if _stop(provider, alloc['external']):
        manager.release(alloc['id'], 'orphan_stopped')
        return 'orphan_stopped'
    manager.mark_orphaned(alloc['id'], 'stop_failed')
    return 'orphaned'


def reconcile(store, provider, *, reconciler_id='reconciler') -> list[dict]:
    """Resolve live allocations whose owning worker is no longer alive."""
    manager = AllocationManager(store, reconciler_id)
    actions = []
    with store.connection() as db:
        rows = db.execute('SELECT id, worker_id FROM allocations WHERE state IN (?,?,?) AND provider=?',
                          (*LIVE_STATES, provider.name)).fetchall()
        stale = [r['id'] for r in rows if not store.worker_alive(db, r['worker_id'])]
    for alloc_id in stale:
        alloc = manager.get(alloc_id)
        actions.append({'allocation_id': alloc_id, 'job_id': alloc['job_id'],
                        'outcome': reconcile_allocation(manager, provider, alloc)})
    return actions


def recover(store, provider, alloc_id, *, force_note=None) -> dict:
    """Operator recovery for an orphaned allocation: retry stop, or force-release."""
    manager = AllocationManager(store, 'operator')
    alloc = manager.get(alloc_id)
    if alloc['state'] != 'orphaned':
        raise ValueError(f'allocation {alloc_id} is {alloc["state"]}, not orphaned')
    if force_note:
        manager.release(alloc_id, 'operator_force_released: ' + force_note[:500])
        return manager.get(alloc_id)
    outcome = reconcile_allocation(manager, provider, alloc)
    return {**manager.get(alloc_id), 'outcome': outcome}
