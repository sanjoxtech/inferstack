"""Read-only review reproductions, 2026-11-09; not release acceptance tests.

Run with the repository's Python environment. Uses temporary SQLite/artifacts,
no GPU, provider, cluster, or production database. The JSON describes observed
gaps; a successful process exit does not mean the product is qualified.
"""
from __future__ import annotations

import asyncio
import json
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from inferstack.execution.executor import BuildFailed, IsolatedExecutor, with_flags
from inferstack.control.worker import OptimizationWorker
from inferstack.optimization.dag import PlanRejected, validate_plan
from inferstack.optimization.experiment import EndpointExecutor
from inferstack.research.techniques import assess
from inferstack.serving.evidence import LiveStream
from inferstack.storage.repository import Repository


def plan(params, action_type='config_change'):
    return {
        'hypothesis': 'Review probe for a bounded candidate treatment.',
        'baseline_ref': 'base@1', 'evidence_ids': [],
        'actions': [
            {'id': 'change', 'type': action_type, 'depends_on': [], 'params': params},
            {'id': 'measure', 'type': 'measure', 'depends_on': ['change'], 'params': {}},
        ],
        'expected_measurements': ['e2e_ms'],
        'gates': ['correctness', 'quality', 'health', 'comparable', 'complete'],
        'estimate': {'gpu_seconds': 1, 'wall_seconds': 1, 'provider_tokens': 0},
        'stop_rule': {'max_iterations': 1, 'stop_on': ['keep']},
    }


def validate(repo, raw):
    return validate_plan(raw, repository=repo, baseline_id='base', candidate_ids=[],
                         allowed_evidence=[], history_signatures=set())


async def cleanup_lease_probe(repo):
    class Planner:
        async def plan(self, request, history):
            return {'hypothesis': 'Probe slow cleanup and lease renewal'}

    class SlowProvider:
        def stop(self, external):
            time.sleep(0.65)  # deterministic stand-in for a synchronous process-stop wait
            return True

    class Allocations:
        def release(self, identifier, note):
            pass  # no real process/allocation exists in this probe

    class Executor(EndpointExecutor):
        def __init__(self):
            self.active, self.owned = {}, {}
            self.provider, self.allocations = SlowProvider(), Allocations()

        async def execute(self, plan, namespace, context=None):
            self.owned[namespace] = ('probe-allocation', {})
            return {'id': namespace}

    class Verifier:
        async def verify(self, plan, candidate):
            return {'decision': 'reject', 'reasons': ['probe only']}

    repo.submit('cleanup-probe', {'deployment_id': 'probe'})
    result = await OptimizationWorker(repo, Planner(), Executor(), Verifier(),
                                      lease_seconds=0.3, max_iterations=1).once()
    return {'job_state_after_worker_returns': result['state'],
            'terminal_result': result['result'],
            'events': [e['kind'] for e in result['events']],
            'gap_observed': result['state'] == 'running' and result['result'] is None}


def main():
    observed = {}
    with TemporaryDirectory(prefix='inferstack-harness-review-') as directory:
        root = Path(directory)
        repo = Repository(root / 'db', root / 'artifacts')
        repo.put_record('deployment', 'base', {
            'id': 'base', 'engine': 'vllm', 'model': 'same-name',
            'base_url': 'http://baseline.invalid', 'engine_version': '0.11.0',
            'profile': {'dtype': 'bfloat16'},
        })
        repo.put_record('deployment', 'old', {
            'id': 'old', 'engine': 'vllm', 'model': 'same-name',
            'base_url': 'http://old.invalid', 'engine_version': '0.10.0',
            'profile': {'dtype': 'float16'},
        })
        raw = repo.artifact(b'{"review": "failed trial"}')
        repo.put_record('trial', 'failed', {
            'experiment_id': 'review-experiment', 'deployment_ref': 'old@1',
            'role': 'candidate', 'repeat_index': 0, 'metrics': {'e2e_ms': 999.0},
            'raw_artifact_sha256': raw['sha256'], 'errors': ['quality gate failed'],
            'started_at': 1, 'ended_at': 2,
        })
        repo.put_record('technique', 'tech', {
            'name': 'Review technique', 'summary': 'Probe only',
            'engines': [{'engine': 'vllm'}], 'proven_by': ['failed@1'],
        })
        proof = assess(repo, 'tech', 'base')
        observed['failed_trial_reused_across_version_and_dtype'] = {
            'verdict': proof['verdict'], 'proven_by': proof['proven_by'],
            'gap_observed': proof['verdict'] == 'compatible_proven',
        }

        source = plan({'technique_id': 'tech', 'files': ['src/inferstack/kernels/review.py'],
                       'description': 'Implement a new technique'}, 'source_patch')
        _, normalized = validate(repo, source)
        error = None
        try:
            IsolatedExecutor.__new__(IsolatedExecutor).build_source(
                root, normalized['actions'][0], {}, [])
        except BuildFailed as exc:
            error = str(exc)
        observed['description_only_source_action'] = {
            'validator_accepted': True, 'executor_error': error,
            'gap_observed': error is not None and 'patch_sha256' in error,
        }

        argv = with_flags(['vllm', 'serve', 'model'], 'vllm', {'load_format': 'runai_streamer'})
        error = None
        try:
            validate(repo, plan({'profile_patch': {'load_format': 'runai_streamer'}}))
        except PlanRejected as exc:
            error = str(exc)
        observed['loader_profile_contract_mismatch'] = {
            'executor_argv': argv, 'planner_error': error,
            'gap_observed': error is not None and '--load-format' in argv,
        }

    with TemporaryDirectory(prefix='inferstack-native-review-') as directory:
        root = Path(directory)
        repo = Repository(root / 'db', root / 'artifacts')
        repo.put_record('deployment', 'base', {
            'id': 'base', 'engine': 'inferstack_native', 'model': 'qwen',
            'base_url': 'http://baseline.invalid',
        })
        patch = {'decode_priority': True}
        validate(repo, plan({'profile_patch': patch}))
        error = None
        try:
            with_flags(['python', '-m', 'inferstack'], 'inferstack_native', patch)
        except BuildFailed as exc:
            error = str(exc)
        observed['native_profile_contract_mismatch'] = {
            'validator_accepted': True, 'executor_error': error,
            'gap_observed': error is not None,
        }

    stream = LiveStream(0)
    event = {'choices': [{'delta': {'content': 'two tokens'}}],
             'usage': {'completion_tokens': 2}}
    stream.feed(('data: ' + json.dumps(event) + '\n\n').encode(), 1)
    phases = stream.phases(1.2)
    observed['coalesced_content_reports_zero_itl'] = {
        'content_events': phases['content_events'], 'output_tokens': phases['output_tokens'],
        'itl_ms': phases['itl_ms'],
        'gap_observed': phases['content_events'] == 1 and phases['itl_ms'] == 0,
    }
    with TemporaryDirectory(prefix='inferstack-cleanup-review-') as directory:
        root = Path(directory)
        repo = Repository(root / 'db', root / 'artifacts')
        observed['synchronous_cleanup_starves_lease_renewal'] = asyncio.run(cleanup_lease_probe(repo))
    print(json.dumps({'purpose': 'gap reproductions, not qualification', 'probes': observed}, indent=2))


if __name__ == '__main__':
    main()
