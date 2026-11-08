"""Retention enforcement for the control-plane store.

Applies RetentionPolicy TTLs: events of terminal jobs older than
`event_ttl_days` are deleted (job rows, results and step records are kept, so
outcomes stay auditable); artifacts older than `artifact_ttl_days` that no
record, job result, event or allocation references are deleted. Running and
pending jobs are never touched. Returns counts so the run is auditable itself.
"""
from __future__ import annotations

import json
import os
import re

from inferstack.contracts.redaction import RetentionPolicy

_SHA = re.compile(r'[0-9a-f]{64}')


def enforce(repository, policy: RetentionPolicy, *, dry_run=False):
    now = repository.clock()
    report = {'events_deleted': 0, 'artifacts_deleted': 0, 'artifacts_kept_referenced': 0, 'dry_run': dry_run}
    if policy.event_ttl_days:
        cutoff = now - policy.event_ttl_days * 86400
        with repository.transaction() as db:
            where = ("FROM events WHERE at < ? AND job_id IN "
                     "(SELECT id FROM jobs WHERE state IN ('completed','failed','cancelled'))")
            report['events_deleted'] = db.execute('SELECT COUNT(*) ' + where, (cutoff,)).fetchone()[0]
            if not dry_run:
                db.execute('DELETE ' + where, (cutoff,))
    if policy.artifact_ttl_days and hasattr(repository, 'artifact_root'):
        cutoff = now - policy.artifact_ttl_days * 86400
        referenced = set()
        with repository.connection() as db:
            for sql in ('SELECT body FROM record_history', 'SELECT result FROM jobs WHERE result IS NOT NULL',
                        'SELECT body FROM events', 'SELECT spec FROM allocations'):
                for row in db.execute(sql):
                    referenced.update(_SHA.findall(row[0] or ''))
        for path in repository.artifact_root.iterdir():
            if not _SHA.fullmatch(path.name):
                continue
            if path.name in referenced:
                report['artifacts_kept_referenced'] += 1
            elif path.stat().st_mtime < cutoff:
                report['artifacts_deleted'] += 1
                if not dry_run:
                    os.unlink(path)
    return report


def policy_from(config: dict | None) -> RetentionPolicy:
    return RetentionPolicy(**json.loads(json.dumps(config or {})))
