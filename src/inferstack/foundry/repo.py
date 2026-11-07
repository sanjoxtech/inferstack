from __future__ import annotations
import json
from pathlib import Path
from urllib.parse import urlparse

class FileSystemFoundryRepository:
    def __init__(self, root:str|Path): self.root=Path(root)
    def resolve(self, ref:str)->dict:
        if not ref.startswith('foundry://'):
            raise ValueError(f'expected foundry:// reference, got {ref}')
        body=ref[len('foundry://'):]
        model, _, release=body.partition('@')
        release=release or 'candidate'
        if release=='candidate':
            # Prefer explicit candidate directory, then first manifest tagged candidate.
            direct=self.root/model/'candidate'/'manifest.json'
            if direct.exists(): return json.loads(direct.read_text())
            for p in sorted((self.root/model).glob('*/manifest.json')):
                d=json.loads(p.read_text())
                if d.get('stage')=='candidate': return d
            raise FileNotFoundError(f'no candidate manifest for {model}')
        p=self.root/model/release/'manifest.json'
        if not p.exists(): raise FileNotFoundError(p)
        return json.loads(p.read_text())
