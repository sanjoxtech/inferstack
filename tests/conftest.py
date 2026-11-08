"""Shared fixtures. `database_url` runs a test on SQLite and on a real PostgreSQL.

PostgreSQL uses a throwaway local cluster from the installed server binaries
(initdb/pg_ctl), or INFERSTACK_TEST_POSTGRES_URL when provided (e.g. CI service).
When neither is available the PostgreSQL variant is skipped with the reason.
"""
import os
import shutil
import subprocess
import uuid
from pathlib import Path

import pytest

PG_BIN_CANDIDATES = [Path(p) for p in ('/usr/lib/postgresql/17/bin', '/usr/lib/postgresql/16/bin',
                                        '/usr/lib/postgresql/15/bin', '/opt/homebrew/bin', '/usr/local/bin')]


def _pg_bin():
    for candidate in PG_BIN_CANDIDATES:
        if (candidate / 'initdb').exists() and (candidate / 'pg_ctl').exists():
            return candidate
    found = shutil.which('initdb')
    return Path(found).parent if found else None


@pytest.fixture(scope='session')
def postgres_server(tmp_path_factory):
    url = os.environ.get('INFERSTACK_TEST_POSTGRES_URL')
    if url:
        yield {'base': url}
        return
    try:
        import psycopg  # noqa: F401
    except ImportError:
        pytest.skip('PostgreSQL variant requires psycopg (pip install "inferstack[postgres]")')
    bindir = _pg_bin()
    if bindir is None:
        pytest.skip('PostgreSQL variant requires server binaries (initdb/pg_ctl) or INFERSTACK_TEST_POSTGRES_URL')
    import tempfile
    root = Path(tempfile.mkdtemp(prefix='inferstack-pg-'))  # pytest's tmp root is not traversable by postgres
    run_as = []
    if os.geteuid() == 0:  # initdb refuses to run as root
        try:
            import pwd
            pwd.getpwnam('postgres')
        except KeyError:
            pytest.skip('running as root without a postgres user')
        shutil.chown(root, 'postgres')
        run_as = ['runuser', '-u', 'postgres', '--']
    port = 56000 + uuid.uuid4().int % 2000
    subprocess.run(run_as + [str(bindir / 'initdb'), '-D', str(root / 'data'), '-A', 'trust', '-U', 'infer'],
                   check=True, capture_output=True)
    subprocess.run(run_as + [str(bindir / 'pg_ctl'), '-D', str(root / 'data'), '-w', '-l', str(root / 'log'),
                             '-o', f"-k {root} -p {port} -c listen_addresses='' -c max_connections=200",
                             'start'], check=True, capture_output=True)
    try:
        yield {'base': f'postgresql://infer@/postgres?host={root}&port={port}', 'host': str(root), 'port': port}
    finally:
        subprocess.run(run_as + [str(bindir / 'pg_ctl'), '-D', str(root / 'data'), '-w', '-m', 'immediate', 'stop'],
                       capture_output=True)
        shutil.rmtree(root, ignore_errors=True)


@pytest.fixture(params=['sqlite', 'postgres'])
def database_url(request, tmp_path):
    if request.param == 'sqlite':
        yield str(tmp_path / 'control.db')
        return
    server = request.getfixturevalue('postgres_server')
    import psycopg
    name = 'test_' + uuid.uuid4().hex[:12]
    with psycopg.connect(server['base'], autocommit=True) as conn:
        conn.execute(f'CREATE DATABASE {name}')
    try:
        if 'host' in server:
            yield f"postgresql://infer@/{name}?host={server['host']}&port={server['port']}"
        else:
            from urllib.parse import urlparse, urlunparse
            parsed = urlparse(server['base'])
            yield urlunparse(parsed._replace(path='/' + name))
    finally:
        # External servers (CI service, hosted Postgres) keep nothing from the test run.
        with psycopg.connect(server['base'], autocommit=True) as conn:
            conn.execute(f'DROP DATABASE IF EXISTS {name} WITH (FORCE)')
