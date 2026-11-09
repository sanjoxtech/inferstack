"""Immutable model artifact cache: verified publish, read-only serving, bounded eviction.

Layout under an operator-configured root (a PVC or node-local volume, never an
arbitrary hostPath by default)::

    <root>/entries/<key>/            published, read-only entry
    <root>/entries/<key>/.inferstack-manifest.json   pinned identity + file digests
    <root>/staging/<key>.<nonce>/    in-progress downloads (never read by serving)

`key` is derived from the pinned identity (source, repo, weights/tokenizer/config
revisions and the pinned file checksums), so two revisions can never share an
entry. A downloader (the only component holding artifact credentials) fetches
into staging, verifies every pinned checksum, writes the manifest last and then
publishes with a single atomic rename. Serving containers mount the cache
read-only and call `verify` before reporting ready: a missing, incomplete or
corrupt entry fails readiness instead of loading wrong weights. Cache entries are
evicted least-recently-used within a byte budget, never while leased by a live
runtime.
"""
from __future__ import annotations

import errno
import hashlib
import json
import os
import shutil
import time
import uuid
from pathlib import Path

MANIFEST = '.inferstack-manifest.json'


class CacheError(RuntimeError):
    """Cache entry unavailable, incomplete, corrupt, or the cache cannot accept it."""


def cache_key(model):
    """Content identity of a pinned model artifact (record body)."""
    if not model.get('checksums'):
        raise CacheError('model record has no pinned file checksums; the cache only stores verifiable artifacts')
    identity = {k: model.get(k) for k in ('source', 'repo_id', 'revision', 'tokenizer_revision', 'config_revision')}
    identity['checksums'] = dict(sorted(model['checksums'].items()))
    digest = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    safe = ''.join(c if c.isalnum() or c in '-_.' else '-' for c in model['repo_id'])[:80]
    return f'{safe}-{digest[:24]}', identity


def file_sha256(path, chunk=1 << 20):
    digest = hashlib.sha256()
    with open(path, 'rb') as handle:
        while block := handle.read(chunk):
            digest.update(block)
    return digest.hexdigest()


class ModelCache:
    def __init__(self, root, *, max_bytes=None, clock=time.time):
        self.root = Path(root)
        self.entries, self.staging, self.leases = self.root / 'entries', self.root / 'staging', self.root / 'leases'
        self.max_bytes, self.clock = max_bytes, clock

    def _writable(self):
        """Writer-side layout; serving mounts the cache read-only and never calls this."""
        for d in (self.entries, self.staging, self.leases):
            d.mkdir(parents=True, exist_ok=True)

    def path(self, model):
        return self.entries / cache_key(model)[0]

    # --- publish (downloader side: holds credentials, writes the cache) -----------

    def publish(self, model, fetch):
        """fetch(dest_dir, files) downloads the pinned files into dest_dir. Returns the entry path.

        Idempotent: an already published, verified entry is returned unchanged."""
        key, identity = cache_key(model)
        self._writable()
        final = self.entries / key
        if final.exists():
            self.verify(model)
            return final
        stage = self.staging / f'{key}.{uuid.uuid4().hex[:8]}'
        stage.mkdir()
        try:
            fetch(stage, sorted(model['checksums']))
            sizes = {}
            for rel, expected in model['checksums'].items():
                target = (stage / rel).resolve()
                if stage.resolve() not in target.parents or not target.is_file():
                    raise CacheError(f'pinned file {rel} missing after fetch')
                actual = file_sha256(target)
                if actual != expected:
                    raise CacheError(f'checksum mismatch for {rel}: {actual} != pinned {expected}')
                sizes[rel] = target.stat().st_size
            manifest = {'key': key, 'identity': identity, 'bytes': sum(sizes.values()), 'sizes': sizes,
                        'published_at': self.clock()}
            if self.max_bytes is not None:
                self.evict(self.max_bytes - manifest['bytes'], protect={key})
            (stage / MANIFEST).write_text(json.dumps(manifest, sort_keys=True))  # written last
            for p in stage.rglob('*'):
                if p.is_file():
                    p.chmod(0o444)
            try:
                os.rename(stage, final)  # atomic publish; readers never see a partial entry
            except OSError as exc:
                if final.exists():  # a concurrent downloader published the same identity first
                    shutil.rmtree(stage, ignore_errors=True)
                    self.verify(model)
                    return final
                raise CacheError(f'cannot publish cache entry: {exc}') from exc
            return final
        except OSError as exc:
            shutil.rmtree(stage, ignore_errors=True)
            if exc.errno == errno.ENOSPC:
                raise CacheError('cache volume full while staging the artifact') from exc
            raise CacheError(f'artifact fetch failed: {type(exc).__name__}: {exc}') from exc
        except BaseException:
            shutil.rmtree(stage, ignore_errors=True)
            raise

    # --- read (serving side: read-only) ------------------------------------------

    def verify(self, model, *, deep=True):
        """Raise CacheError unless the published entry matches the pinned identity exactly.

        deep=True re-hashes every file (startup check); deep=False checks the manifest
        and sizes only (cheap periodic check)."""
        key, identity = cache_key(model)
        entry = self.entries / key
        manifest_path = entry / MANIFEST
        if not entry.is_dir():
            raise CacheError(f'cache miss: {key} not published')
        if not manifest_path.is_file():
            raise CacheError(f'incomplete cache entry {key}: no manifest')
        try:
            manifest = json.loads(manifest_path.read_text())
        except ValueError as exc:
            raise CacheError(f'corrupt manifest for {key}') from exc
        if manifest.get('identity') != identity:
            raise CacheError(f'cache entry {key} identity does not match the pinned model')
        for rel, expected in model['checksums'].items():
            target = entry / rel
            if not target.is_file():
                raise CacheError(f'cache entry {key} missing {rel}')
            if target.stat().st_size != manifest['sizes'].get(rel):
                raise CacheError(f'cache entry {key} size mismatch for {rel}')
            if deep and file_sha256(target) != expected:
                raise CacheError(f'cache entry {key} corrupt: {rel} digest changed')
        return {'key': key, 'path': str(entry), 'bytes': manifest['bytes']}

    # --- leases and eviction ----------------------------------------------------

    def lease(self, model, holder):
        key, _ = cache_key(model)
        self._writable()
        (self.leases / f'{key}.{holder}').write_text(str(self.clock()))
        (self.entries / key).exists() and os.utime(self.entries / key)
        return key

    def unlease(self, model, holder):
        key, _ = cache_key(model)
        (self.leases / f'{key}.{holder}').unlink(missing_ok=True)

    def leased(self):
        return {p.name.rsplit('.', 1)[0] for p in self.leases.iterdir()} if self.leases.is_dir() else set()

    def usage(self):
        total = 0
        for entry in (self.entries.iterdir() if self.entries.is_dir() else ()):
            try:
                total += json.loads((entry / MANIFEST).read_text())['bytes']
            except (OSError, ValueError, KeyError):
                continue
        return total

    def evict(self, budget, *, protect=()):
        """Remove least-recently-used unleased entries until usage <= budget. Returns removed keys.

        Raises CacheError when leased/protected entries alone exceed the budget."""
        held = self.leased() | set(protect)
        entries = sorted((e for e in self.entries.iterdir() if e.is_dir()), key=lambda e: e.stat().st_mtime)
        removed, usage = [], self.usage()
        for entry in entries:
            if usage <= budget:
                break
            if entry.name in held:
                continue
            try:
                size = json.loads((entry / MANIFEST).read_text())['bytes']
            except (OSError, ValueError, KeyError):
                size = 0
            for p in entry.rglob('*'):
                p.chmod(0o644) if p.is_file() else None
            shutil.rmtree(entry, ignore_errors=True)
            removed.append(entry.name)
            usage -= size
        if usage > budget:
            raise CacheError(f'cache budget exceeded by leased entries ({usage} > {budget} bytes)')
        return removed


def local_fetcher(source_dir):
    """fetch() copying pinned files from a local directory (tests, pre-staged mirrors)."""
    source_dir = Path(source_dir)

    def fetch(dest, files):
        for rel in files:
            target = dest / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(source_dir / rel, target)
    return fetch


def huggingface_fetcher(repo_id, revision):
    """fetch() for the downloader Job: pinned files from the Hugging Face Hub at an exact commit.

    Credentials come only from the Job's environment (HF_TOKEN via its Secret)."""
    def fetch(dest, files):
        try:
            from huggingface_hub import hf_hub_download
        except ImportError as exc:
            raise CacheError('huggingface_hub is required in the downloader image') from exc
        for rel in files:
            hf_hub_download(repo_id, rel, revision=revision, local_dir=str(dest))
    return fetch


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description='Publish or verify a pinned model artifact cache entry.')
    parser.add_argument('action', choices=('publish', 'verify'))
    parser.add_argument('--root', required=True)
    parser.add_argument('--model-json', required=True, help='pinned model record body (with checksums)')
    parser.add_argument('--source', default='huggingface', help="'huggingface' or a local directory path")
    parser.add_argument('--max-bytes', type=int)
    args = parser.parse_args(argv)
    model = json.loads(args.model_json)
    try:
        if args.action == 'verify':
            result = ModelCache(args.root).verify(model)
        else:
            fetch = huggingface_fetcher(model['repo_id'], model['revision']) if args.source == 'huggingface' \
                else local_fetcher(args.source)
            result = {'path': str(ModelCache(args.root, max_bytes=args.max_bytes).publish(model, fetch))}
    except CacheError as exc:
        print(json.dumps({'ok': False, 'error': str(exc)}))
        raise SystemExit(2)
    print(json.dumps({'ok': True, **result}))


if __name__ == '__main__':
    main()
