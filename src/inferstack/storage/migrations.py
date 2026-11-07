"""Ordered, recorded schema migrations for the control-plane database.

Rules:
- Migrations are append-only and numbered. A released migration is never edited.
- Each migration runs inside one IMMEDIATE transaction together with its ledger row,
  so concurrent openers serialize and a crash leaves the previous version intact.
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


def _script(db, sql):
    # executescript() would COMMIT the enclosing migration transaction; run statements individually.
    for statement in sql.split(';'):
        if statement.strip():
            db.execute(statement)


def _baseline(db, now):
    _script(db, '''
        CREATE TABLE IF NOT EXISTS jobs (
            id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL,
            payload TEXT NOT NULL, state TEXT NOT NULL,
            attempt INTEGER NOT NULL DEFAULT 0,
            lease_token TEXT, lease_until REAL, result TEXT,
            created REAL NOT NULL, updated REAL NOT NULL
        );
        CREATE TABLE IF NOT EXISTS events (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            job_id TEXT NOT NULL REFERENCES jobs(id),
            kind TEXT NOT NULL, body TEXT NOT NULL, at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS jobs_claim ON jobs(state, lease_until, created);
        CREATE TABLE IF NOT EXISTS records (
            kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL,
            revision INTEGER NOT NULL, PRIMARY KEY(kind, id)
        );
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


def _history(db, now):
    columns = {row[1] for row in db.execute('PRAGMA table_info(records)')}
    for name, ddl in (('schema_version', 'INTEGER'), ('created_at', 'REAL'), ('updated_at', 'REAL')):
        if name not in columns:
            db.execute(f'ALTER TABLE records ADD COLUMN {name} {ddl}')
    db.execute('''CREATE TABLE IF NOT EXISTS record_history (
        kind TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
        schema_version INTEGER, body TEXT NOT NULL, body_sha256 TEXT NOT NULL,
        recorded_at REAL NOT NULL, origin TEXT NOT NULL,
        PRIMARY KEY(kind, id, revision))''')
    for row in db.execute('SELECT kind,id,body,revision FROM records').fetchall():
        version = classify(row[0], row[2])
        db.execute('UPDATE records SET schema_version=?, created_at=COALESCE(created_at,?), '
                   'updated_at=COALESCE(updated_at,?) WHERE kind=? AND id=?', (version, now, now, row[0], row[1]))
        db.execute('INSERT OR IGNORE INTO record_history VALUES(?,?,?,?,?,?,?,?)',
                   (row[0], row[1], row[3], version, row[2], hashlib.sha256(row[2].encode()).hexdigest(), now,
                    'migrated_from_v1_6:prior_revisions_unrecoverable' if row[3] > 1 else 'migrated_from_v1_6'))


def _references(db, now):
    _script(db, '''
        CREATE TABLE IF NOT EXISTS record_refs (
            kind TEXT NOT NULL, id TEXT NOT NULL, revision INTEGER NOT NULL,
            ref_kind TEXT NOT NULL, ref_id TEXT NOT NULL, ref_revision INTEGER,
            PRIMARY KEY(kind, id, revision, ref_kind, ref_id)
        );
        CREATE INDEX IF NOT EXISTS record_refs_target ON record_refs(ref_kind, ref_id);
    ''')


MIGRATIONS = [
    (1, 'baseline jobs, events and records (v1.6 layout)', _baseline),
    (2, 'record schema versions and immutable revision history', _history),
    (3, 'record reference index', _references),
]
LATEST = MIGRATIONS[-1][0]


def current_version(db) -> int:
    db.execute('CREATE TABLE IF NOT EXISTS schema_migrations '
               '(version INTEGER PRIMARY KEY, name TEXT NOT NULL, applied_at REAL NOT NULL)')
    row = db.execute('SELECT MAX(version) FROM schema_migrations').fetchone()
    return row[0] or 0


def migrate(db, now) -> int:
    """Apply pending migrations on an autocommit sqlite3 connection. Returns final version."""
    db.execute('BEGIN IMMEDIATE')
    try:
        version = current_version(db)
        if version > LATEST:
            raise IncompatibleSchema(f'database schema version {version} is newer than this build '
                                     f'supports ({LATEST}); upgrade InferStack before opening it')
        for number, name, apply in MIGRATIONS:
            if number > version:
                apply(db, now)
                db.execute('INSERT INTO schema_migrations VALUES(?,?,?)', (number, name, now))
        db.execute('COMMIT')
    except BaseException:
        db.execute('ROLLBACK')
        raise
    return LATEST
