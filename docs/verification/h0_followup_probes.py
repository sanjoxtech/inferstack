"""Independent H0 boundary probes against commit 52b9f8d, 2026-11-09.

Uses temporary SQLite and the implementation's positive proof fixture. No GPU,
external API, actual processes or production state. Reports remaining gaps;
a successful script exit does not qualify H0.
"""
import asyncio
import json
import runpy
import sys
import time
from pathlib import Path
from tempfile import TemporaryDirectory

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))

from inferstack.control.worker import OptimizationWorker
from inferstack.research.techniques import assess
from inferstack.serving.evidence import LiveStream
from inferstack.storage.repository import Repository


def proof_probe(kind):
    fixture = runpy.run_path(str(ROOT / 'tests/test_research.py'))['proof_fixture']
    with TemporaryDirectory(prefix='inferstack-h0-proof-') as directory:
        root = Path(directory)
        repo = Repository(root / 'db', root / 'artifacts')
        positive = fixture(repo)
        assert positive['verdict'] == 'compatible_proven'
        target = repo.read('deployment', 'target')['body']
        if kind == 'tokenizer':
            model = repo.read('model', 'mdl')['body']
            model['tokenizer_revision'] = 'c' * 40
            repo.put_record('model', 'new-tokenizer', model)
            target['model_ref'] = 'new-tokenizer@1'
        elif kind == 'image':
            target['image_digest'] = 'sha256:' + 'd' * 64
        elif kind == 'hardware_runtime':
            hardware = repo.read('hardware', 'a100')['body']
            hardware.update(driver_version='580.65.06', cuda_version='13.0', topology='PCIe')
            repo.put_record('hardware', 'different-runtime', hardware)
            target['hardware_ref'] = 'different-runtime@1'
        else:
            raise ValueError(kind)
        repo.put_record('deployment', 'target', target, expected_revision=1)
        result = assess(repo, 'tech', 'target')
        return {'changed': kind, 'verdict': result['verdict'], 'proven_by': result['proven_by'],
                'gap_observed': result['verdict'] == 'compatible_proven'}


async def reconcile_probe():
    with TemporaryDirectory(prefix='inferstack-h0-reconcile-') as directory:
        root = Path(directory)
        repo = Repository(root / 'db', root / 'artifacts')
        identifier = repo.submit('restart-probe', {'deployment_id': 'probe'})
        old = repo.claim(0.3)
        repo.step_begin(identifier, old['lease_token'], 'iter0.experiment', idempotent=False)
        with repo.transaction() as db:
            db.execute('UPDATE jobs SET lease_until=? WHERE id=?', (repo.clock() - 1, identifier))

        class Allocations:
            released = False

            def list(self, **filters):
                return [] if self.released else [{'id': 'probe-allocation', 'state': 'active',
                                                  'external': {'allocation_id': 'probe-allocation'}}]

            def release(self, identifier, note):
                self.released = True

        class Provider:
            def owns(self, external):
                return True

            def alive(self, external):
                return True

            def stop(self, external):
                time.sleep(0.65)
                return True

        class Planner:
            async def plan(self, request, history):
                return {'hypothesis': 'Restart must renew its lease during reconciliation'}

        executed = []

        class Executor:
            async def execute(self, plan, namespace):
                executed.append(namespace)
                return {'id': namespace}

            async def cleanup(self, candidate):
                pass

        class Verifier:
            async def verify(self, plan, candidate):
                return {'decision': 'reject'}

        allocations = Allocations()
        result = await OptimizationWorker(repo, Planner(), Executor(), Verifier(), lease_seconds=0.3,
                                          max_iterations=1, allocations=allocations,
                                          provider=Provider()).once()
        return {'job_state': result['state'], 'terminal_result': result['result'],
                'reconciled_resource': allocations.released, 'executed_new_candidate': bool(executed),
                'gap_observed': result['state'] == 'running' and result['result'] is None}


def main():
    observed = {f'proof_scope_{kind}': proof_probe(kind)
                for kind in ('tokenizer', 'image', 'hardware_runtime')}
    stream = LiveStream(0)
    events = [
        {'choices': [{'delta': {'content': 'hello'}}]},
        {'choices': [{'delta': {'content': ' world'}}]},
        {'choices': [], 'usage': {'completion_tokens': 2}},
    ]
    # Separate SSE events can arrive in one HTTP chunk, all stamped with `now`.
    stream.feed(''.join('data: ' + json.dumps(event) + '\n\n' for event in events).encode(), 1)
    phases = stream.phases(1.2)
    observed['transport_coalescing_itl'] = {
        'content_events': phases['content_events'], 'tokens': phases['output_tokens'],
        'itl_ms': phases['itl_ms'], 'gap_observed': phases['itl_ms'] == 0,
    }
    observed['slow_restart_reconciliation'] = asyncio.run(reconcile_probe())
    print(json.dumps({'purpose': 'H0 independent gap reproductions', 'probes': observed}, indent=2))


if __name__ == '__main__':
    main()
