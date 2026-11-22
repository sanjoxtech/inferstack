"""G7: model-backed coding agent for the implementer handoff (OpenRouter / DeepSeek / any OpenAI-compatible).

The agent sees the handoff (hypothesis, description, allowed files at the pinned revision, the previous
attempt's rejection reason) and answers with whole new file contents plus optional candidate tests. The
diff is computed here, from the pinned revision, not trusted from the model, and goes through the desk's
normal validation (patch confinement, compile, the controller's independent tests in the sandbox). The
agent never sees or touches tests, the evaluator or benchmark code, and never runs anything itself.
Every call is one journaled attempt; tokens are charged to the attempt.
"""
from __future__ import annotations

import json
import subprocess
import tempfile
from pathlib import Path

MAX_FILE_CHARS = 60_000

SYSTEM = """You change the InferStack inference engine's source to test one performance hypothesis.
Rules (a change that breaks one is rejected by the controller, which you cannot influence):
- Edit only the files listed under allowed_files. Never touch tests, the evaluator or benchmark scripts.
- Greedy outputs must stay token-identical to the current code: no skipped work, no changed sampling,
  no shortcuts that only hold for benchmark traffic, no detecting or special-casing benchmark prompts.
- Keep the change small and explain it in `summary`.
- Optionally add pytest files that check your change (names test_<something>.py).
Answer with one JSON object:
{"files": {"<allowed path>": "<complete new file content>"}, "tests": {"test_x.py": "<pytest file>"},
 "summary": "<what changed and why it should be faster>"}"""


def file_at(source_root, revision, path):
    out = subprocess.run(['git', '-C', str(source_root), 'show', f'{revision}:{path}'], capture_output=True)
    return out.stdout.decode(errors='replace') if out.returncode == 0 else None


def unified_diff(old_files, new_files):
    """git-format diff from {path: old text or None} to {path: new text}, computed in a scratch repo."""
    with tempfile.TemporaryDirectory(prefix='coder-') as tmp:
        root = Path(tmp)
        git = ['git', '-C', tmp, '-c', 'core.hooksPath=/dev/null', '-c', 'user.name=coder', '-c',
               'user.email=coder@localhost']
        subprocess.run(['git', 'init', '-q', tmp], check=True, capture_output=True)
        for path, text in old_files.items():
            if text is not None:
                (root / path).parent.mkdir(parents=True, exist_ok=True)
                (root / path).write_text(text)
        subprocess.run([*git, 'add', '-A'], check=True, capture_output=True)
        subprocess.run([*git, 'commit', '-q', '--allow-empty', '-m', 'pinned'], check=True, capture_output=True)
        for path, text in new_files.items():
            (root / path).parent.mkdir(parents=True, exist_ok=True)
            (root / path).write_text(text if text.endswith('\n') else text + '\n')
        subprocess.run([*git, 'add', '-A'], check=True, capture_output=True)
        return subprocess.run([*git, 'diff', '--cached', '--no-ext-diff', '--no-textconv'], capture_output=True,
                              text=True, check=True).stdout


class ModelImplementer:
    def __init__(self, provider, *, source_root, name=None):
        self.provider, self.source_root = provider, Path(source_root)
        self.name = name or f"{getattr(provider, 'name', 'model')}:{getattr(provider, 'model', '?')}"

    def messages(self, handoff, feedback):
        files = {p: (file_at(self.source_root, handoff['source_revision'], p) or '')[:MAX_FILE_CHARS]
                 for p in handoff['allowed_files']}
        task = {'hypothesis': handoff['hypothesis'], 'description': handoff['description'],
                'allowed_files': handoff['allowed_files'], 'current_files': files,
                'limits': {'max_patch_bytes': handoff['limits']['max_patch_bytes']},
                'previous_attempt_rejected_because': feedback}
        return [{'role': 'system', 'content': SYSTEM}, {'role': 'user', 'content': json.dumps(task)}]

    async def run(self, desk, impl_id, feedback=None):
        """One journaled attempt. Returns the implementation record after validation."""
        handoff = desk.read(impl_id)['body']['handoff']
        number = desk.begin_attempt(impl_id, {'kind': 'agent', 'name': self.name})
        cost = {}
        try:
            response = await self.provider.chat(self.messages(handoff, feedback), json_mode=True)
            if response.usage.get('total_tokens'):
                cost['provider_tokens'] = response.usage['total_tokens']
            answer = json.loads(response.content or '')
            files, tests = answer.get('files'), answer.get('tests') or {}
            if not isinstance(files, dict) or not files or not all(isinstance(v, str) for v in files.values()):
                raise ValueError('answer has no files')
            if not isinstance(tests, dict):
                raise ValueError('tests must be an object')
            old = {p: file_at(self.source_root, handoff['source_revision'], p) for p in files}
            diff = unified_diff(old, files)
        except Exception as exc:  # noqa: BLE001 - provider error, malformed answer: a failed attempt
            return desk.finish_attempt(impl_id, number, outcome='failed', logs=f'{type(exc).__name__}: {exc}'[:2000],
                                       cost=cost)
        return desk.finish_attempt(impl_id, number, outcome='submitted', diff=diff, candidate_tests=tests,
                                   logs=str(answer.get('summary') or '')[:4000], cost=cost)


__all__ = ['ModelImplementer', 'unified_diff']
