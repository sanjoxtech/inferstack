from __future__ import annotations

import hashlib
import json
import math
import sqlite3
import time
import uuid
from contextlib import contextmanager
from pathlib import Path


class LeaseLost(RuntimeError):
    pass


class Store:
    """Single-host SQLite job store. All worker mutations require a live lease token."""

    def __init__(self, path, clock=time.time):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.clock = clock
        from inferstack.storage.migrations import migrate
        with self.connection() as db:
            self.schema_version = migrate(db, self.clock())

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None)
        db.row_factory = sqlite3.Row
        db.execute('PRAGMA foreign_keys=ON')
        db.execute('PRAGMA journal_mode=WAL')
        try:
            yield db
        finally:
            db.close()

    @contextmanager
    def transaction(self):
        with self.connection() as db:
            db.execute('BEGIN IMMEDIATE')
            try:
                yield db
                db.execute('COMMIT')
            except BaseException:
                db.execute('ROLLBACK')
                raise

    @staticmethod
    def encode(value):
        return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)

    def event(self, db, job_id, kind, body):
        db.execute('INSERT INTO events(job_id,kind,body,at) VALUES(?,?,?,?)',
                   (job_id, kind, self.encode(body), self.clock()))

    def submit(self, request_key, payload):
        if not isinstance(request_key, str) or not request_key.strip():
            raise ValueError('nonempty request_key required')
        encoded = self.encode(payload)
        with self.transaction() as db:
            existing = db.execute('SELECT * FROM jobs WHERE request_key=?', (request_key,)).fetchone()
            if existing:
                if existing['payload'] != encoded:
                    raise ValueError('idempotency key reused with different payload')
                return existing['id']
            job_id = uuid.uuid4().hex
            now = self.clock()
            db.execute('INSERT INTO jobs(id,request_key,payload,state,created,updated) VALUES(?,?,?,?,?,?)',
                       (job_id, request_key, encoded, 'pending', now, now))
            self.event(db, job_id, 'submitted', payload)
            return job_id

    @staticmethod
    def duration(seconds):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError('lease duration must be positive and finite')

    def claim(self, lease_seconds=60):
        self.duration(lease_seconds)
        with self.transaction() as db:
            now = self.clock()
            row = db.execute("SELECT * FROM jobs WHERE state='pending' OR "
                             "(state='running' AND lease_until<=?) ORDER BY created,id LIMIT 1", (now,)).fetchone()
            if row is None:
                return None
            token = uuid.uuid4().hex
            db.execute("UPDATE jobs SET state='running',lease_token=?,lease_until=?,attempt=attempt+1,updated=? WHERE id=?",
                       (token, now + lease_seconds, now, row['id']))
            self.event(db, row['id'], 'claimed', {'attempt': row['attempt'] + 1})
            return {'id': row['id'], 'payload': json.loads(row['payload']),
                    'lease_token': token, 'attempt': row['attempt'] + 1}

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
            db.execute('UPDATE jobs SET state=?,result=?,lease_token=NULL,lease_until=NULL,updated=? WHERE id=?',
                       (state, encoded, self.clock(), job_id))
            self.event(db, job_id, state, result)

    def cancel(self, job_id):
        with self.transaction() as db:
            cursor = db.execute("UPDATE jobs SET state='cancelled',lease_token=NULL,lease_until=NULL,updated=? "
                                "WHERE id=? AND state IN ('pending','running')", (self.clock(), job_id))
            if cursor.rowcount:
                self.event(db, job_id, 'cancelled', {})
            return bool(cursor.rowcount)

    def get(self, job_id):
        with self.connection() as db:
            row = db.execute('SELECT * FROM jobs WHERE id=?', (job_id,)).fetchone()
            if row is None:
                raise KeyError(job_id)
            result = dict(row)
            result.pop('lease_token')
            result['payload'] = json.loads(result['payload'])
            result['result'] = json.loads(result['result']) if result['result'] else None
            result['events'] = [{'kind': r['kind'], 'body': json.loads(r['body']), 'at': r['at']}
                                for r in db.execute('SELECT * FROM events WHERE job_id=? ORDER BY seq', (job_id,))]
            return result
