from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from inferstack.control.store import Store
from inferstack.contracts.records import IncompatibleSchema, schema_for, valid_id
from inferstack.contracts.redaction import reject_secrets


class Repository(Store):
    """Revision-checked records, immutable revision history and content-addressed artifacts.

    `put` is the untyped write path kept for internal/legacy callers: it records the
    body with schema_version NULL. `put_record` is the typed path: it validates the
    body against the kind's versioned schema, rejects literal credentials, checks
    that every reference resolves, and records the references. Both paths write an
    immutable history row per revision inside the same transaction.
    """

    def __init__(self, path, artifact_root):
        super().__init__(path)
        self.artifact_root=Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True,exist_ok=True)

    # --- writes ---------------------------------------------------------------

    def _write(self, db, kind, identifier, encoded, expected_revision, schema_version, origin):
        old=db.execute('SELECT revision FROM records WHERE kind=? AND id=?',(kind,identifier)).fetchone()
        revision=old['revision'] if old else 0
        if expected_revision is None and old:raise ValueError('expected_revision required for replacement')
        if expected_revision is not None and expected_revision!=revision:raise ValueError('revision conflict')
        now=self.clock()
        db.execute('INSERT INTO records(kind,id,body,revision,schema_version,created_at,updated_at) VALUES(?,?,?,?,?,?,?) '
                   'ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body,revision=excluded.revision,'
                   'schema_version=excluded.schema_version,updated_at=excluded.updated_at',
                   (kind,identifier,encoded,revision+1,schema_version,now,now))
        db.execute('INSERT INTO record_history VALUES(?,?,?,?,?,?,?,?)',
                   (kind,identifier,revision+1,schema_version,encoded,hashlib.sha256(encoded.encode()).hexdigest(),now,origin))
        return revision+1

    def put(self, kind, identifier, body, expected_revision=None):
        if not kind or not identifier:raise ValueError('record identity required')
        encoded=self.encode(body)
        with self.transaction() as db:
            return self._write(db,kind,identifier,encoded,expected_revision,None,'untyped')

    def put_record(self, kind, identifier, body, expected_revision=None, schema_version=None):
        """Typed write. Raises ValueError (invalid/conflict/dangling) or IncompatibleSchema."""
        if not valid_id(identifier):raise ValueError(f'invalid record id: {identifier!r}')
        if not isinstance(body,dict):raise ValueError('record body must be an object')
        reject_secrets(body)
        cls=schema_for(kind,schema_version)
        model=cls.model_validate(body)
        if 'id' in cls.model_fields and getattr(model,'id')!=identifier:
            raise ValueError('body id must match record identifier')
        refs=model.references()
        encoded=self.encode(model.model_dump(mode='json'))
        with self.transaction() as db:
            for ref_kind,ref_id,ref_rev in refs:
                if ref_kind==kind and ref_id==identifier:raise ValueError('record cannot reference itself')
                if ref_rev is None:
                    found=db.execute('SELECT 1 FROM records WHERE kind=? AND id=?',(ref_kind,ref_id)).fetchone()
                else:
                    found=db.execute('SELECT 1 FROM record_history WHERE kind=? AND id=? AND revision=?',
                                     (ref_kind,ref_id,ref_rev)).fetchone()
                if found is None:
                    raise ValueError(f'dangling reference to {ref_kind}/{ref_id}'+(f'@{ref_rev}' if ref_rev else ''))
            revision=self._write(db,kind,identifier,encoded,expected_revision,cls.VERSION,'typed')
            for ref_kind,ref_id,ref_rev in refs:
                db.execute('INSERT OR IGNORE INTO record_refs VALUES(?,?,?,?,?,?)',
                           (kind,identifier,revision,ref_kind,ref_id,ref_rev))
            return revision

    # --- reads ----------------------------------------------------------------

    def read(self,kind,identifier,revision=None):
        with self.connection() as db:
            if revision is None:
                row=db.execute('SELECT * FROM records WHERE kind=? AND id=?',(kind,identifier)).fetchone()
            else:
                row=db.execute('SELECT * FROM record_history WHERE kind=? AND id=? AND revision=?',
                               (kind,identifier,revision)).fetchone()
            if row is None:raise KeyError((kind,identifier) if revision is None else (kind,identifier,revision))
            return {'id':identifier,'kind':kind,'revision':row['revision'],'schema_version':row['schema_version'],
                    'body':json.loads(row['body'])}

    def read_typed(self,kind,identifier,revision=None):
        record=self.read(kind,identifier,revision)
        if record['schema_version'] is None:
            raise IncompatibleSchema(f'{kind}/{identifier}@{record["revision"]} is a legacy untyped record; '
                                     're-write it with put_record to validate it')
        return schema_for(kind,record['schema_version']).model_validate(record['body'])

    def history(self,kind,identifier):
        with self.connection() as db:
            rows=db.execute('SELECT * FROM record_history WHERE kind=? AND id=? ORDER BY revision',(kind,identifier)).fetchall()
        if not rows:raise KeyError((kind,identifier))
        return [{'revision':r['revision'],'schema_version':r['schema_version'],'body_sha256':r['body_sha256'],
                 'recorded_at':r['recorded_at'],'origin':r['origin'],'body':json.loads(r['body'])} for r in rows]

    def referrers(self,kind,identifier):
        with self.connection() as db:
            return [dict(r) for r in db.execute('SELECT kind,id,revision,ref_revision FROM record_refs '
                                                'WHERE ref_kind=? AND ref_id=? ORDER BY kind,id,revision',(kind,identifier))]

    def list_records(self,kind,limit=100):
        if not 1<=limit<=1000:raise ValueError('limit out of bounds')
        with self.connection() as db:
            return [{'id':r['id'],'revision':r['revision'],'schema_version':r['schema_version'],'body':json.loads(r['body'])}
                    for r in db.execute('SELECT * FROM records WHERE kind=? ORDER BY id LIMIT ?',(kind,limit))]

    # --- artifacts ------------------------------------------------------------

    def artifact(self,content:bytes):
        digest=hashlib.sha256(content).hexdigest()
        target=self.artifact_root/digest
        fd,temp=tempfile.mkstemp(dir=self.artifact_root)
        try:
            with os.fdopen(fd,'wb') as f:
                f.write(content);f.flush();os.fsync(f.fileno())
            os.replace(temp,target)
        finally:
            if os.path.exists(temp):os.unlink(temp)
        return {'sha256':digest,'bytes':len(content)}

    def read_artifact(self,digest):
        if not isinstance(digest,str) or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('invalid digest')
        data=(self.artifact_root/digest).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('artifact corruption')
        return data
