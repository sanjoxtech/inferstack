"""Package A acceptance: versioned contracts, migrations, references, capabilities."""
import json
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest
from fastapi.testclient import TestClient

from inferstack.contracts import IncompatibleSchema, redact, validate
from inferstack.contracts.redaction import RetentionPolicy, SecretInRecord
from inferstack.control.api import create_control_app
from inferstack.control.store import Store
from inferstack.engines.capabilities import Support, UnsupportedCapability, lookup, matrix, require
from inferstack.storage.migrations import LATEST
from inferstack.storage.repository import Repository

HF_REV = 'a' * 40
SHA = 'b' * 64


def repo(tmp_path):
    return Repository(tmp_path / 'db', tmp_path / 'artifacts')


def deployment(name, **extra):
    return {'id': name, 'engine': 'vllm', 'model': 'm', 'base_url': f'http://{name}', **extra}


def make_v16_database(path):
    """Exact table layout produced by the v1.6 Store + Repository, with data."""
    db = sqlite3.connect(path, isolation_level=None)
    db.executescript('''
        CREATE TABLE jobs (id TEXT PRIMARY KEY, request_key TEXT UNIQUE NOT NULL, payload TEXT NOT NULL,
            state TEXT NOT NULL, attempt INTEGER NOT NULL DEFAULT 0, lease_token TEXT, lease_until REAL,
            result TEXT, created REAL NOT NULL, updated REAL NOT NULL);
        CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL REFERENCES jobs(id),
            kind TEXT NOT NULL, body TEXT NOT NULL, at REAL NOT NULL);
        CREATE INDEX jobs_claim ON jobs(state, lease_until, created);
        CREATE TABLE records (kind TEXT NOT NULL,id TEXT NOT NULL,body TEXT NOT NULL,revision INTEGER NOT NULL,
            PRIMARY KEY(kind,id));
    ''')
    db.execute("INSERT INTO jobs VALUES('j1','key-1','{}','completed',1,NULL,NULL,'{\"decision\":\"keep\"}',1.0,2.0)")
    db.execute("INSERT INTO events(job_id,kind,body,at) VALUES('j1','submitted','{}',1.0)")
    db.execute("INSERT INTO records VALUES('deployment','base',?,3)", (json.dumps(deployment('base')),))
    db.execute("INSERT INTO records VALUES('deployment','loose','{\"engine\":\"native\"}',1)")
    db.execute("INSERT INTO records VALUES('evidence','note','{\"text\":\"free-form\"}',1)")
    db.close()


# --- migrations -------------------------------------------------------------

def test_v16_database_migrates_without_losing_data(tmp_path):
    path = tmp_path / 'db'
    make_v16_database(path)
    r = Repository(path, tmp_path / 'artifacts')
    assert r.schema_version == LATEST
    job = r.get('j1')
    assert job['result'] == {'decision': 'keep'} and [e['kind'] for e in job['events']] == ['submitted']
    base = r.read('deployment', 'base')
    assert base['revision'] == 3 and base['body'] == deployment('base') and base['schema_version'] == 1
    hist = r.history('deployment', 'base')
    assert len(hist) == 1 and hist[0]['revision'] == 3 and 'prior_revisions_unrecoverable' in hist[0]['origin']
    # Bodies that do not satisfy a schema are kept verbatim as legacy (schema_version NULL), not dropped.
    loose = r.read('deployment', 'loose')
    assert loose['body'] == {'engine': 'native'} and loose['schema_version'] is None
    with pytest.raises(IncompatibleSchema, match='legacy'):
        r.read_typed('deployment', 'loose')
    assert r.read_typed('deployment', 'base').base_url == 'http://base'
    # Old records keep working with CAS and gain history from here on.
    assert r.put_record('deployment', 'base', deployment('base', engine_version='0.11.0'), expected_revision=3) == 4
    assert [h['revision'] for h in r.history('deployment', 'base')] == [3, 4]
    assert r.read('deployment', 'base', revision=3)['body'] == deployment('base')


def test_reopening_is_idempotent_and_concurrent_openers_migrate_once(tmp_path):
    path = tmp_path / 'db'
    make_v16_database(path)
    with ThreadPoolExecutor(max_workers=6) as pool:
        versions = list(pool.map(lambda _: Store(path).schema_version, range(6)))
    assert versions == [LATEST] * 6
    db = sqlite3.connect(path)
    assert [row[0] for row in db.execute('SELECT version FROM schema_migrations ORDER BY version')] == list(range(1, LATEST + 1))
    assert db.execute("SELECT COUNT(*) FROM record_history WHERE kind='deployment' AND id='base'").fetchone()[0] == 1


def test_newer_database_schema_fails_clearly(tmp_path):
    path = tmp_path / 'db'
    Store(path)
    db = sqlite3.connect(path, isolation_level=None)
    db.execute('INSERT INTO schema_migrations VALUES(?,?,?)', (LATEST + 5, 'from the future', 1.0))
    db.close()
    with pytest.raises(IncompatibleSchema, match='newer than this build'):
        Store(path)


# --- typed records and references --------------------------------------------

def test_typed_records_validate_and_keep_history(tmp_path):
    r = repo(tmp_path)
    assert r.put_record('deployment', 'a', deployment('a')) == 1
    with pytest.raises(ValueError, match='expected_revision'):
        r.put_record('deployment', 'a', deployment('a'))
    with pytest.raises(ValueError):
        r.put_record('deployment', 'a', {**deployment('a'), 'engine': 'tensorrt'}, expected_revision=1)
    with pytest.raises(ValueError, match='must match'):
        r.put_record('deployment', 'b', deployment('a'))
    with pytest.raises(ValueError, match='unknown record kind'):
        r.put_record('spaceship', 'x', {})
    assert r.put_record('deployment', 'a', deployment('a', profile={'max_num_seqs': 64}), expected_revision=1) == 2
    assert [h['origin'] for h in r.history('deployment', 'a')] == ['typed', 'typed']


def test_invalid_cross_record_references_fail(tmp_path):
    r = repo(tmp_path)
    with pytest.raises(ValueError, match='dangling reference to model/qwen'):
        r.put_record('deployment', 'a', deployment('a', model_ref='qwen'))
    r.put_record('model', 'qwen', {'source': 'huggingface', 'repo_id': 'Qwen/Qwen3.5-2B-Base', 'revision': HF_REV})
    assert r.put_record('deployment', 'a', deployment('a', model_ref='qwen')) == 1
    assert r.referrers('model', 'qwen') == [{'kind': 'deployment', 'id': 'a', 'revision': 1, 'ref_revision': None}]
    with pytest.raises(ValueError, match='dangling reference to model/qwen@2'):
        r.put_record('deployment', 'b', deployment('b', model_ref='qwen@2'))
    # Plans must pin exact snapshots for reproducibility.
    plan = {'job_id': 'j', 'iteration': 0, 'hypothesis': 'h', 'baseline_ref': 'a', 'candidate_ref': 'a@1'}
    with pytest.raises(ValueError, match='baseline_ref must pin'):
        r.put_record('plan', 'p', plan)
    assert r.put_record('plan', 'p', {**plan, 'baseline_ref': 'a@1'}) == 1


def test_incompatible_record_schema_version_fails_clearly(tmp_path):
    r = repo(tmp_path)
    with pytest.raises(IncompatibleSchema, match='schema version 9 is not supported'):
        r.put_record('deployment', 'a', deployment('a'), schema_version=9)
    with pytest.raises(IncompatibleSchema):
        validate('objective', {}, version=0)


def test_immutable_identities_and_registered_metrics(tmp_path):
    with pytest.raises(ValueError, match='mutable'):
        validate('model', {'source': 'huggingface', 'repo_id': 'Qwen/Qwen3.5-2B-Base', 'revision': 'main'})
    with pytest.raises(ValueError, match='40-hex'):
        validate('model', {'source': 'huggingface', 'repo_id': 'x/y', 'revision': 'v1.0'})
    assert validate('model', {'source': 'foundry', 'repo_id': 'factorylm-2b', 'revision': 'candidate-2026-11-01'})
    with pytest.raises(ValueError, match='unregistered metric: ttft'):
        validate('objective', {'name': 'o', 'slos': [{'metric': 'ttft', 'direction': 'min'}]})
    assert validate('objective', {'name': 'o', 'slos': [{'metric': 'ttft_client_ms', 'direction': 'min',
                                                         'limit': 500, 'percentile': 95}]})
    with pytest.raises(ValueError, match='unregistered metric'):
        validate('trial', {'experiment_id': 'e', 'deployment_ref': 'a@1', 'role': 'baseline', 'repeat_index': 0,
                           'metrics': {'tps': 1.0}, 'raw_artifact_sha256': SHA, 'started_at': 1, 'ended_at': 2})
    # Missing metrics are explicit None, never coerced to zero.
    trial = validate('trial', {'experiment_id': 'e', 'deployment_ref': 'a@1', 'role': 'baseline', 'repeat_index': 0,
                               'metrics': {'tpot_engine_ms': None}, 'raw_artifact_sha256': SHA,
                               'started_at': 1, 'ended_at': 2})
    assert trial.metrics == {'tpot_engine_ms': None}
    with pytest.raises(ValueError, match='end at 100'):
        validate('rollout', {'model': 'm', 'release_ref': 'r', 'stages': [{'weight': 10, 'min_duration_s': 60}],
                             'updated_at': 1})


# --- redaction -----------------------------------------------------------------

def test_literal_credentials_rejected_and_env_references_allowed(tmp_path):
    r = repo(tmp_path)
    with pytest.raises(SecretInRecord, match=r'\$\.api_key'):
        r.put_record('deployment', 'a', {**deployment('a'), 'api_key': 'sk-live-0123456789abcdef'})
    with pytest.raises(SecretInRecord, match=r'\$\.profile\.headers\.authorization'):
        r.put_record('deployment', 'a', deployment('a', profile={'headers': {'authorization': 'Bearer x'}}))
    with pytest.raises(SecretInRecord):
        r.put_record('deployment', 'a', deployment('a', profile={'note': 'ghp_' + 'A' * 30}))
    assert r.put_record('deployment', 'a', deployment('a', api_key_env='VLLM_API_KEY')) == 1
    assert redact({'token': 'abc', 'api_key_env': 'X', 'nested': [{'password': 'p'}, 'sk-' + 'z' * 20]}) == \
        {'token': '[redacted]', 'api_key_env': 'X', 'nested': [{'password': '[redacted]'}, '[redacted]']}
    with pytest.raises(ValueError):
        RetentionPolicy(event_ttl_days=0)
    assert RetentionPolicy().capture_request_content is False


# --- capabilities ----------------------------------------------------------------

def test_capability_matrix_never_treats_unknown_as_supported():
    caps = matrix()
    assert {c.state for c in caps} == set(Support)
    assert lookup('inferstack_native', 'cuda_graph_replay').state is Support.UNKNOWN
    with pytest.raises(UnsupportedCapability, match='unknown'):
        require('inferstack_native', 'cuda_graph_replay')
    with pytest.raises(UnsupportedCapability, match='unsupported'):
        require('inferstack_native', 'paged_kv')
    assert require('vllm', 'paged_kv').basis == 'declared'
    assert all(c.basis != 'none' for c in caps if c.state is Support.SUPPORTED)
    with pytest.raises(ValueError):
        lookup('tensorrt', 'complete')


# --- releases keep working on the typed path -----------------------------------

def test_route_references_must_resolve(tmp_path):
    from inferstack.releases.manager import ReleaseManager
    r = repo(tmp_path)
    with pytest.raises(KeyError):
        ReleaseManager(r).route('m', 'missing')
    with pytest.raises(ValueError, match='dangling reference to release/missing'):
        r.put_record('route', 'Qwen/Qwen3.5-2B-Base', {'stable': 'missing', 'weight': 0})


# --- API -----------------------------------------------------------------------

def test_control_api_typed_records(tmp_path):
    r = repo(tmp_path)
    client = TestClient(create_control_app(r, 'test-token'))
    h = {'Authorization': 'Bearer test-token'}
    assert client.put('/control/records/deployment/d', headers=h,
                      json={'body': deployment('d', model_ref='nope')}).status_code == 409
    assert client.put('/control/records/deployment/d', headers=h,
                      json={'body': {**deployment('d'), 'engine': 'bad'}}).status_code == 422
    assert client.put('/control/records/deployment/d', headers=h,
                      json={'body': {**deployment('d'), 'api_key': 'sk-0123456789abcdefgh'}}).status_code == 422
    assert client.put('/control/records/deployment/d', headers=h,
                      json={'body': deployment('d'), 'schema_version': 7}).status_code == 422
    assert client.put('/control/records/release/x', headers=h, json={'body': {}}).status_code == 422
    assert client.put('/control/records/deployment/d', headers=h, json={'body': deployment('d')}).json() == {'revision': 1}
    assert client.put('/control/records/deployment/d', headers=h,
                      json={'body': deployment('d', engine_version='0.11'), 'expected_revision': 1}).status_code == 200
    assert [x['revision'] for x in client.get('/control/records/deployment/d/history', headers=h).json()['history']] == [1, 2]
    assert client.get('/control/records/deployment/d?revision=1', headers=h).json()['body']['engine_version'] is None
    caps = client.get('/control/capabilities?engine=generic_openai', headers=h).json()['capabilities']
    assert caps and {c['engine'] for c in caps} == {'generic_openai'}
    schemas = client.get('/control/schemas', headers=h).json()
    assert schemas['record_schemas']['deployment'] == 1 and 'tpot_engine_ms' in schemas['metrics']
    assert schemas['database_schema_version'] == LATEST
