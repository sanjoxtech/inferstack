"""New-model onboarding: immutable identity, architecture support, execution profile.

Steps: resolve the artifact to immutable revisions (a mutable ref such as `main`
is resolved to a commit when a resolver is available, otherwise refused); read
its config at that commit; classify task and architecture; match against the
engine support table; emit a ModelArtifact record and an execution profile for
the reference path, or an explicit capability gap. Onboarding never creates a
deployment and never claims support for an architecture it cannot vouch for.

Support table semantics:
- inferstack_native supports exactly the architectures its runtime implements.
- vllm/sglang support many architectures, but this repo has no verified list;
  they are reported as `needs_engine_probe` until a launch on that engine
  version succeeds (recorded as observed capability evidence).
"""
from __future__ import annotations

import json
import re

from inferstack.contracts.records import MUTABLE_REVISIONS

NATIVE_ARCHITECTURES = {
    'qwen3_5': {'architectures': {'Qwen3_5ForCausalLM', 'Qwen3_5ForConditionalGeneration'},
                'runtime': 'QwenHybridContinuousEngine', 'notes': 'hybrid linear-attention (GDN) + attention'},
}
TEXT_GENERATION_SUFFIXES = ('ForCausalLM', 'ForConditionalGeneration', 'LMHeadModel')
THIRD_PARTY = ('vllm', 'sglang')


class OnboardingError(ValueError):
    pass


def hf_resolver(repo_id, revision):
    """Resolve a Hugging Face revision to a commit and read config.json at that commit."""
    try:
        from huggingface_hub import HfApi, hf_hub_download
    except ImportError as exc:
        raise OnboardingError('huggingface_hub is required to resolve Hugging Face revisions '
                              '(pip install "inferstack[hf]") or pass an immutable commit and config') from exc
    commit = HfApi().model_info(repo_id, revision=revision).sha
    with open(hf_hub_download(repo_id, 'config.json', revision=commit)) as f:
        return commit, json.load(f)


def classify(config: dict) -> dict:
    text = config.get('text_config') if isinstance(config.get('text_config'), dict) else {}
    model_type = config.get('model_type')
    architectures = list(config.get('architectures') or [])
    task = 'text-generation' if any(a.endswith(TEXT_GENERATION_SUFFIXES) for a in architectures) else 'other'
    return {'model_type': model_type, 'text_model_type': text.get('model_type'), 'architectures': architectures,
            'task': task, 'dtype': config.get('torch_dtype') or config.get('dtype') or text.get('torch_dtype'),
            'max_position_embeddings': config.get('max_position_embeddings') or text.get('max_position_embeddings'),
            'requires_remote_code': bool(config.get('auto_map'))}


def onboard(repository, *, model_id, repo_id, revision, source='huggingface', tokenizer_revision=None,
            config=None, resolver=hf_resolver, allow_remote_code=False, record=True):
    """Return an onboarding outcome. Writes the `model` record only for a usable artifact."""
    if not re.fullmatch(r'[0-9a-f]{40}', revision or '') or config is None:
        if source != 'huggingface':
            if revision in MUTABLE_REVISIONS:
                raise OnboardingError(f'{source} revision {revision!r} is mutable; supply an immutable release id')
            if config is None:
                raise OnboardingError('config is required for non-Hugging Face artifacts')
        else:
            requested = revision
            revision, config = resolver(repo_id, revision)
            if not re.fullmatch(r'[0-9a-f]{40}', revision or ''):
                raise OnboardingError(f'resolver did not return an immutable commit for {requested!r}')
    tokenizer_revision = tokenizer_revision or revision
    if tokenizer_revision in MUTABLE_REVISIONS:
        raise OnboardingError('tokenizer revision must be immutable')
    facts = classify(config)
    gaps, engines = [], {}
    if facts['task'] != 'text-generation':
        gaps.append(f"task is not text generation (architectures={facts['architectures']})")
    if facts['requires_remote_code'] and not allow_remote_code:
        gaps.append('model requires trust_remote_code; refused unless explicitly allowed')
    native = NATIVE_ARCHITECTURES.get(facts['model_type']) or NATIVE_ARCHITECTURES.get(facts['text_model_type'])
    if native and set(facts['architectures']) & native['architectures']:
        engines['inferstack_native'] = {'status': 'supported', 'runtime': native['runtime'], 'notes': native['notes']}
    else:
        engines['inferstack_native'] = {'status': 'unsupported',
                                        'reason': f"native runtime implements {sorted(NATIVE_ARCHITECTURES)} only"}
    for engine in THIRD_PARTY:
        engines[engine] = {'status': 'needs_engine_probe',
                           'reason': 'no verified architecture list; prove support with a launch on the pinned version'}
    if gaps:
        status, reference = 'unsupported', None
    elif engines['inferstack_native']['status'] == 'supported':
        status, reference = 'supported', 'inferstack_native'
    else:
        status, reference = 'needs_engine_probe', None
    outcome = {'model_id': model_id, 'status': status, 'repo_id': repo_id, 'revision': revision,
               'tokenizer_revision': tokenizer_revision, 'facts': facts, 'engines': engines, 'gaps': gaps,
               'reference_engine': reference, 'deployment_created': False}
    if reference:
        outcome['execution_profile'] = {'engine': reference, 'dtype': 'bfloat16' if facts['dtype'] in (None, 'bfloat16')
                                        else facts['dtype'], 'max_model_len': facts['max_position_embeddings'],
                                        'reference_path': True, 'optimizations': 'none (reference baseline first)'}
    if record and status != 'unsupported':
        body = {'source': source, 'repo_id': repo_id, 'revision': revision, 'tokenizer_revision': tokenizer_revision,
                'config_revision': revision, 'task': 'text-generation',
                'architecture': (facts['architectures'] or [None])[0], 'dtype': facts['dtype']}
        try:
            repository.put_record('model', model_id, body)
        except ValueError as exc:
            if 'expected_revision' not in str(exc):
                raise
            from inferstack.contracts.records import validate
            current = repository.read('model', model_id)
            if current['body'] != validate('model', body).model_dump(mode='json'):
                repository.put_record('model', model_id, body, expected_revision=current['revision'])
        outcome['model_record'] = f"{model_id}@{repository.read('model', model_id)['revision']}"
    return outcome
