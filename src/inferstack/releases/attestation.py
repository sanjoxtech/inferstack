"""Release attestation: the running runtime must match the frozen release manifest.

A release pins model/tokenizer revisions, engine version, container image or
build digest and plugins. Attestation compares each pinned field with what the
runtime (or the provider that launched it) actually reports. Every check is
True (matches), False (mismatch) or None (cannot be verified). Status:
- `mismatch` if any check is False;
- `unverified` if any required check is None, including identity the manifest
  failed to pin (a release without a pinned model revision is not attestable);
- `attested` only when every required check is True.

Runtime observations come from the engine adapter (served models, version, and
any identity the engine reports such as `model_revision`) plus an optional
`observe_launch(release)` callback returning what the provider launched
(image digest from `docker inspect`/Pod `imageID`, argv revisions, build digest).
"""
from __future__ import annotations

import httpx

from inferstack.engines.adapters import adapter_for


def _expected(repository, deployment):
    expected = {'model': deployment['model'], 'engine_version': deployment.get('engine_version'),
                'image_digest': deployment.get('image_digest'), 'plugin_refs': sorted(deployment.get('plugin_refs') or []),
                'build_sha256': (deployment.get('profile') or {}).get('build_sha256'),
                'model_revision': None, 'tokenizer_revision': None}
    if deployment.get('model_ref'):
        from inferstack.contracts.records import parse_ref
        identifier, revision = parse_ref(deployment['model_ref'])
        model = repository.read('model', identifier, revision=revision) if revision else repository.read('model', identifier)
        expected['model_revision'] = model['body']['revision']
        expected['tokenizer_revision'] = model['body'].get('tokenizer_revision') or model['body']['revision']
    return expected


def compare(expected, identity, launched):
    """Pure comparison; returns (status, checks)."""
    launched = launched or {}

    def observed(key):
        return identity.get(key) if identity.get(key) is not None else launched.get(key)

    checks = {'model_served': expected['model'] in (identity.get('served_models') or []),
              'model_revision': None, 'tokenizer_revision': None}
    for key in ('model_revision', 'tokenizer_revision'):
        if expected[key] is not None and observed(key) is not None:
            checks[key] = observed(key) == expected[key]
    version = identity.get('version') or launched.get('engine_version')
    checks['engine_version'] = None if expected['engine_version'] is None or version is None \
        else version == expected['engine_version']
    # Artifact identity: a container image digest or a build digest must be pinned and observed.
    artifact = None
    if expected['image_digest']:
        artifact = None if observed('image_digest') is None else observed('image_digest') == expected['image_digest']
    elif expected['build_sha256']:
        artifact = None if observed('build_sha256') is None else observed('build_sha256') == expected['build_sha256']
    checks['artifact_digest'] = artifact
    if expected['plugin_refs']:
        seen = observed('plugin_refs')
        checks['plugins'] = None if seen is None else sorted(seen) == expected['plugin_refs']
    if False in checks.values():
        return 'mismatch', checks
    return ('unverified' if None in checks.values() else 'attested'), checks


def release_attestor(repository, *, observe_launch=None, transport=None, timeout=10.0):
    """attest(release_record) for RolloutController, reading the release's runtime endpoint."""
    async def attest(release):
        deployment = release['body']['deployment']['body']
        expected = _expected(repository, deployment)
        async with httpx.AsyncClient(base_url=deployment['base_url'].rstrip('/') + '/', timeout=timeout,
                                     transport=transport) as client:
            identity = await adapter_for(deployment['engine'], client).runtime_identity()
        launched = await observe_launch(release) if observe_launch else None
        status, checks = compare(expected, identity, launched)
        return {'status': status, 'checks': checks, 'expected': expected,
                'identity': {k: v for k, v in identity.items() if k != 'flags'}, 'launched': launched}
    return attest
