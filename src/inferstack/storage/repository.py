from __future__ import annotations
import hashlib
import json
import os
import tempfile
from pathlib import Path
from inferstack.control.store import Store


class Repository(Store):
    def __init__(self, path, artifact_root):
        super().__init__(path)
        self.artifact_root=Path(artifact_root).resolve()
        self.artifact_root.mkdir(parents=True,exist_ok=True)
        with self.connection() as db:
            db.execute('CREATE TABLE IF NOT EXISTS records (kind TEXT NOT NULL,id TEXT NOT NULL,body TEXT NOT NULL,revision INTEGER NOT NULL,PRIMARY KEY(kind,id))')

    def put(self, kind, identifier, body, expected_revision=None):
        if not kind or not identifier:raise ValueError('record identity required')
        encoded=self.encode(body)
        with self.transaction() as db:
            old=db.execute('SELECT revision FROM records WHERE kind=? AND id=?',(kind,identifier)).fetchone()
            revision=old['revision'] if old else 0
            if expected_revision is None and old:raise ValueError('expected_revision required for replacement')
            if expected_revision is not None and expected_revision!=revision:raise ValueError('revision conflict')
            db.execute('INSERT INTO records VALUES(?,?,?,?) ON CONFLICT(kind,id) DO UPDATE SET body=excluded.body,revision=excluded.revision', (kind,identifier,encoded,revision+1))
            return revision+1

    def read(self,kind,identifier):
        with self.connection() as db:
            row=db.execute('SELECT * FROM records WHERE kind=? AND id=?',(kind,identifier)).fetchone()
            if row is None:raise KeyError((kind,identifier))
            return {'id':identifier,'kind':kind,'revision':row['revision'],'body':json.loads(row['body'])}

    def list_records(self,kind,limit=100):
        if not 1<=limit<=1000:raise ValueError('limit out of bounds')
        with self.connection() as db:
            return [{'id':r['id'],'revision':r['revision'],'body':json.loads(r['body'])} for r in db.execute('SELECT * FROM records WHERE kind=? ORDER BY id LIMIT ?',(kind,limit))]

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
        if len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):raise ValueError('invalid digest')
        data=(self.artifact_root/digest).read_bytes()
        if hashlib.sha256(data).hexdigest()!=digest:raise ValueError('artifact corruption')
        return data
