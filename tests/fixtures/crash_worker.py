"""Worker process for crash/restart tests: real store, executor, verifier and processes.

Only the planner is a deterministic test double (no provider call). `--crash-at`
makes the process die with os._exit(137) at a named transition, skipping every
finally block, exactly like a SIGKILL at that point.
"""
import argparse
import asyncio
import json
import os
from pathlib import Path

from inferstack.control.worker import OptimizationWorker
from inferstack.infrastructure import AllocationManager, LocalProcessProvider, reconcile
from inferstack.optimization.experiment import EndpointExecutor, EndpointVerifier
from inferstack.storage.repository import Repository


class FixedPlanner:
    def __init__(self, repo):
        self.repo = repo

    async def plan(self, request, history):
        base = self.repo.read('deployment', request['deployment_id'])
        cand = self.repo.read('deployment', 'candidate')
        return {'candidate_id': 'candidate', 'hypothesis': 'registered alternate', 'evidence_ids': [],
                'baseline_id': base['id'], 'baseline_revision': base['revision'],
                'candidate_revision': cand['revision']}


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--db', required=True)
    parser.add_argument('--state', required=True)
    parser.add_argument('--crash-at')
    parser.add_argument('--hang-at')
    parser.add_argument('--hang-seconds', type=float, default=0)
    parser.add_argument('--lease', type=float, default=1.0)
    parser.add_argument('--iterations', type=int, default=1)
    args = parser.parse_args()

    def faults(point):
        if point == args.crash_at:
            os._exit(137)
        if point == args.hang_at:
            import time
            Path(args.state, 'hanging').write_text(point)
            time.sleep(args.hang_seconds)  # wedged: blocks the event loop, so heartbeats stop too
            Path(args.state, 'woke').write_text(point)

    repo = Repository(args.db, Path(args.state) / 'artifacts')
    worker_id = repo.register_worker(args.lease)
    provider = LocalProcessProvider()
    actions = reconcile(repo, provider)
    allocations = AllocationManager(repo, worker_id, faults=faults)
    executor = EndpointExecutor(repo, Path(args.state) / 'logs', allocations=allocations, provider=provider)
    verifier = EndpointVerifier(repo, executor, 'baseline', [{'prompt': '2+2', 'expected': 'four', 'max_tokens': 4}],
                                [{'metric': 'e2e_ms', 'direction': 'min', 'epsilon': 0.5}], repeats=3)
    worker = OptimizationWorker(repo, FixedPlanner(repo), executor, verifier, lease_seconds=args.lease,
                                max_iterations=args.iterations, worker_id=worker_id, allocations=allocations,
                                provider=provider, faults=faults)
    result = await worker.once()
    repo.retire_worker(worker_id)
    print(json.dumps({'reconciled': actions, 'state': result and result['state']}))


if __name__ == '__main__':
    asyncio.run(main())
