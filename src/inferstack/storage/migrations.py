"""Ordered, recorded schema migrations for the control-plane database.

Rules:
- Migrations are append-only and numbered. A released migration is never edited.
- Each migration runs inside one writer-locked transaction together with its
  ledger row, so concurrent openers serialize and a crash leaves the previous
  version intact. The same DDL runs on SQLite and PostgreSQL via dialect tokens.
- A database stamped with a version newer than this build fails with
  IncompatibleSchema instead of being silently downgraded or partially read.
- Migrations preserve every existing row. v1.6 databases kept only the latest
  revision of each record; earlier revisions were overwritten before history
  existed and cannot be reconstructed. Backfilled history rows say so.
"""
from __future__ import annotations

import hashlib
import json

from inferstack.contracts.records import IncompatibleSchema


def _script(database, db, sql):
    # One statement per execute(): sqlite executescript() would COMMIT the migration.
    for statement in database.ddl(sql).split(';'):
        if statement.strip():
            db.execute(statement)


def _add_columns(database, db, table, columns):
    existing = database.columns(db, table)
    for name, ddl in columns:
        if name not in existing:
            db.execute(f'ALTER TABLE {table} ADD COLUMN {name} {database.ddl(ddl)}')


def _baseline(database, db, now):
    _script(database, db, '''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
            payload TEXT NOT NULL, state TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 0,
            lease_token TEXT, lease_until {FLOAT}, result TEXT,
            created {FLOAT} NOT NULL, updated {FLOAT} NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            seq {SERIAL_PK},
            job_id TEXT NOT NULL REFERENCES jobs(id),
            kind TEXT NOT NULL, body TEXT NOT NULL, at {FLOAT} NOT NULL
        );
        CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(state, lease_until, created);
        CREATE TABLE IF NOT EXISTS records (
            kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
            revision INTEGER NOT NULL, PRIMARY KEY(kind, id)
        )
    ''')


def classify(kind, body_text):
    """Return the schema version a stored body satisfies, or None for legacy/unvalidated."""
    from inferstack.contracts.records import SCHEMAS
    if kind not in SCHEMAS:
        return None
    body = json.loads(body_text)
    for version in sorted(SCHEMAS[kind], reverse=True):
        try:
            SCHEMAS[kind][version].model_validate(body)
            return version
        except (ValueError, TypeError):
            continue
    return None


def _history(database, db, now):
    _add_columns(database, db, 'records', [('schema_version', 'INTEGER'), ('created_at', '{FLOAT}'),
                                           ('updated_at', '{FLOAT}')])
    _script(database, db, '''CREATE TABLE IF NOT EXISTS record_history (
        kind TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
        schema_version INTEGER, body TEXT NOT NULL, body_sha256 TEXT NOT NULL,
        recorded_at {FLOAT} NOT NULL, origin TEXT NOT NULL,
        PRIMARY KEY(kind, id, revision))''')
    for row in db.execute('SELECT kind,id,body,revision FROM records').fetchall():
        version = classify(row[0], row[2])
        db.execute('UPDATE records SET schema_version=?, created_at=COALESCE(created_at,?), '
                   'updated_at=COALESCE(updated_at,?) WHERE kind=? AND id=?', (version, now, now, row[0], row[1]))
        db.execute('INSERT INTO record_history VALUES(?,?,?,?,?,?,?,?) ON CONFLICT DO NOTHING',
                   (row[0], row[1], row[3], version, row[2], hashlib.sha256(row[2].encode()).hexdigest(), now,
                    'migrated_from_v1_6:prior_revisions_unrecoverable' if row[3] > 1 else 'migrated_from_v1_6'))


def _references(database, db, now):
    _script(database, db, '''
        CREATE TABLE IF NOT EXISTS record_refs (
            kind TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
            ref_kind TEXT NOT NULL, ref_id TEXT NOT NULL, ref_revision INTEGER,
            PRIMARY KEY(kind, id, revision, ref_kind, ref_id)
        );
        CREATE INDEX IF NOT EXISTS record_refs_target ON record_refs(ref_kind, ref_id)
    ''')


def _execution(database, db, now):
    _add_columns(database, db, 'jobs', [
        ('worker_id', 'TEXT'), ('deadline_at', '{FLOAT}'), ('budget', 'TEXT'), ('usage', 'TEXT'),
        ('max_attempts', 'INTEGER'), ('not_before', '{FLOAT}'), ('last_error', 'TEXT')])
    _script(database, db, '''
        CREATE TABLE IF NOT EXISTS job_steps (
            job_id TEXT NOT NULL REFERENCES jobs(id), step_key TEXT NOT NULL,
            attempts INTEGER NOT NULL, state TEXT NOT NULL, idempotent INTEGER NOT NULL,
            worker_id TEXT, started_at {FLOAT}, ended_at {FLOAT}, result TEXT, error TEXT,
            PRIMARY KEY(job_id, step_key)
        );
        CREATE TABLE IF NOT EXISTS workers (
            id TEXT PRIMARY KEY, host TEXT NOT NULL, pid INTEGER NOT NULL,
            started_at {FLOAT} NOT NULL, heartbeat_at {FLOAT} NOT NULL,
            lease_until {FLOAT} NOT NULL, state TEXT NOT NULL
        );
        CREATE TABLE IF NOT EXISTS allocations (
            id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL,
            job_id TEXT, step_key TEXT, worker_id TEXT NOT NULL,
            provider TEXT NOT NULL, resource_type TEXT NOT NULL, spec TEXT NOT NULL,
            external TEXT, state TEXT NOT NULL, gpu_count INTEGER NOT NULL DEFAULT 0,
            created_at {FLOAT} NOT NULL, updated_at {FLOAT} NOT NULL,
            active_at {FLOAT}, released_at {FLOAT}, note TEXT
        );
        CREATE INDEX IF NOT EXISTS allocations_state ON allocations(state, worker_id)
    ''')


def _integrations(database, db, now):
    _script(database, db, '''
        CREATE TABLE IF NOT EXISTS outbox (
            id TEXT PRIMARY KEY, idempotency_key TEXT UNIQUE NOT NULL, topic TEXT NOT NULL,
            job_id TEXT, payload TEXT NOT NULL, state TEXT NOT NULL, attempts INTEGER NOT NULL DEFAULT 0,
            next_attempt_at {FLOAT} NOT NULL, last_error TEXT, created_at {FLOAT} NOT NULL,
            delivered_at {FLOAT}, response TEXT
        );
        CREATE INDEX IF NOT EXISTS outbox_due ON outbox(state, next_attempt_at);
        CREATE TABLE IF NOT EXISTS triggers (
            id TEXT PRIMARY KEY, kind TEXT NOT NULL, spec TEXT NOT NULL, state TEXT NOT NULL,
            last_fired_at {FLOAT}, last_job_id TEXT, fired_count INTEGER NOT NULL DEFAULT 0,
            created_at {FLOAT} NOT NULL, updated_at {FLOAT} NOT NULL
        )
    ''')


def _serving_evidence(database, db, now):
    _script(database, db, '''
        CREATE TABLE IF NOT EXISTS serving_events (
            id TEXT PRIMARY KEY, at {FLOAT} NOT NULL, worker TEXT NOT NULL, release_id TEXT NOT NULL,
            deployment_id TEXT NOT NULL, outcome TEXT NOT NULL, stream INTEGER NOT NULL, status INTEGER,
            rejection_source TEXT, upstream_first_byte_ms {FLOAT}, first_content_ms {FLOAT}, e2e_ms {FLOAT},
            max_gap_ms {FLOAT}, content_events INTEGER, output_tokens INTEGER, itl_ms {FLOAT}
        );
        CREATE INDEX IF NOT EXISTS serving_events_release ON serving_events(release_id, at);
        CREATE INDEX IF NOT EXISTS serving_events_deployment ON serving_events(deployment_id, at)
    ''')


def _routing_attribution(database, db, now):
    _add_columns(database, db, 'serving_events', [('endpoint', 'TEXT')])


MIGRATIONS = [
    (1, 'baseline jobs, events and records (v1.6 layout)', _baseline),
    (2, 'record schema versions and immutable revision history', _history),
    (3, 'record reference index', _references),
    (4, 'durable steps, budgets, deadlines, workers and resource allocations', _execution),
    (5, 'integration outbox and bounded optimization triggers', _integrations),
    (6, 'durable per-request serving evidence for rollout windows', _serving_evidence),
    (7, 'selected endpoint attribution for routed serving evidence', _routing_attribution),
]
LATEST = MIGRATIONS[-1][0]


def current_version(database, db) -> int:
    _script(database, db, 'CREATE TABLE IF NOT EXISTS schema_migrations '
                          '(version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at {FLOAT} NOT NULL)')
    row = db.execute('SELECT MAX(version) FROM schema_migrations').fetchone()
    return row[0] or 0


def migrate(database, now) -> int:
    """Apply pending migrations under the writer lock. Returns the final version."""
    with database.transaction() as db:
        version = current_version(database, db)
        if version > LATEST:
            raise IncompatibleSchema(f'database schema version {version} is newer than this build '
                                     f'supports ({LATEST}); upgrade InferStack before opening it')
        for number, name, apply in MIGRATIONS:
            if number > version:
                apply(database, db, now)
                db.execute('INSERT INTO schema_migrations VALUES(?,?,?)', (number, name, now))
    return LATEST
