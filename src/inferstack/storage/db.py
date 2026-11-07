"""Database access for the control plane: SQLite (single host) and PostgreSQL (shared).

Both dialects expose the same small surface used by Store/Repository/migrations:
`connection()` and `transaction()` context managers yielding a connection whose
`execute(sql, params)` accepts `?` placeholders and returns rows addressable by
column name and by index.

Write serialization is identical in both dialects: a transaction holds the single
control-plane writer lock for its duration (SQLite `BEGIN IMMEDIATE`; PostgreSQL a
transaction-scoped advisory lock). This makes lease, fencing and compare-and-swap
behavior equivalent by construction. The control plane's write rate is low, so the
lost write concurrency is an intentional trade for semantic equivalence.
"""
from __future__ import annotations

import re
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from urllib.parse import urlparse

WRITER_LOCK_KEY = 0x1A7E5_7AC  # arbitrary constant advisory-lock key for the writer lock


class Row:
    """Row addressable by column name or position (sqlite3.Row-compatible subset)."""
    __slots__ = ('_values', '_index')

    def __init__(self, names, values):
        self._values = tuple(values)
        self._index = {n: i for i, n in enumerate(names)}

    def __getitem__(self, key):
        return self._values[self._index[key] if isinstance(key, str) else key]

    def keys(self):
        return list(self._index)

    def __iter__(self):
        return iter(self._values)

    def __len__(self):
        return len(self._values)


class Database:
    dialect: str

    def ddl(self, sql: str) -> str:
        """Expand dialect tokens in migration DDL."""
        raise NotImplementedError

    def columns(self, db, table: str) -> set[str]:
        raise NotImplementedError


class SQLiteDatabase(Database):
    dialect = 'sqlite'

    def __init__(self, path):
        self.path = str(Path(path).resolve())
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.url = 'sqlite:///' + self.path

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

    def ddl(self, sql):
        return sql.replace('{SERIAL_PK}', 'INTEGER PRIMARY KEY AUTOINCREMENT').replace('{FLOAT}', 'REAL')

    def columns(self, db, table):
        return {row[1] for row in db.execute(f'PRAGMA table_info({table})')}


class _PGConnection:
    _PLACEHOLDER = re.compile(r'\?')

    def __init__(self, raw):
        self.raw = raw

    def execute(self, sql, params=()):
        cursor = self.raw.cursor()
        cursor.execute(self._PLACEHOLDER.sub('%s', sql), tuple(params))
        return _PGCursor(cursor)


class _PGCursor:
    def __init__(self, cursor):
        self.cursor = cursor
        self.rowcount = cursor.rowcount
        self.names = [d.name for d in cursor.description] if cursor.description else []

    def fetchone(self):
        row = self.cursor.fetchone()
        return None if row is None else Row(self.names, row)

    def fetchall(self):
        return [Row(self.names, r) for r in self.cursor.fetchall()]

    def __iter__(self):
        return iter(self.fetchall())


class PostgresDatabase(Database):
    dialect = 'postgres'

    def __init__(self, url):
        try:
            import psycopg  # noqa: F401
        except ImportError as exc:
            raise RuntimeError('PostgreSQL support requires: pip install "inferstack[postgres]"') from exc
        self.url = url

    @contextmanager
    def connection(self):
        import psycopg
        raw = psycopg.connect(self.url, autocommit=True, connect_timeout=10)
        try:
            yield _PGConnection(raw)
        finally:
            raw.close()

    @contextmanager
    def transaction(self):
        import psycopg
        raw = psycopg.connect(self.url, autocommit=False, connect_timeout=10)
        conn = _PGConnection(raw)
        try:
            conn.execute('SELECT pg_advisory_xact_lock(?)', (WRITER_LOCK_KEY,))
            yield conn
            raw.commit()
        except BaseException:
            raw.rollback()
            raise
        finally:
            raw.close()

    def ddl(self, sql):
        return sql.replace('{SERIAL_PK}', 'BIGSERIAL PRIMARY KEY').replace('{FLOAT}', 'DOUBLE PRECISION')

    def columns(self, db, table):
        rows = db.execute('SELECT column_name FROM information_schema.columns WHERE table_name=? '
                          'AND table_schema=current_schema()', (table,)).fetchall()
        return {r[0] for r in rows}


def open_database(target) -> Database:
    """`target` is a filesystem path, `sqlite:///path`, or `postgresql://...`."""
    if isinstance(target, Database):
        return target
    text = str(target)
    scheme = urlparse(text).scheme
    if scheme in ('postgres', 'postgresql'):
        return PostgresDatabase(text)
    if scheme == 'sqlite':
        return SQLiteDatabase(text[len('sqlite:///'):] if text.startswith('sqlite:///') else urlparse(text).path)
    if scheme and len(scheme) > 1:
        raise ValueError(f'unsupported database URL scheme: {scheme}')
    return SQLiteDatabase(text)
