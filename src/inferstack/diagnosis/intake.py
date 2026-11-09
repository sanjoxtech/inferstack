"""One intake desk for FactoryOS issues, operator reports and submitted research (harness H2).

Every input becomes a typed `problem` record with lineage to its source (feedback or
evidence record) and a diagnosis (diagnose.py). What happens next depends only on the
diagnosis, never on what the submitted text asks for:

- owner `inference` with an evidenced symptom -> `queued`: one optimization job whose
  goal is composed from the category, numeric observations and the structured
  diagnosis (no free text from the submitter);
- owner `model` -> `routed_elsewhere` (Model Foundry); owner `application` ->
  `routed_elsewhere` (FactoryOS); no inference job is created;
- not enough evidence -> `needs_measurement` with explicit measurement requests; new
  observations can be attached later and re-diagnosed;
- submitted research text -> stored as untrusted evidence, `claim_recorded`: a claim to
  investigate, never execution authority and never a dependency install.
"""
from __future__ import annotations

import hashlib
import json

from inferstack.diagnosis.diagnose import diagnose, numeric_observations

ACTIONABLE = {'latency', 'throughput', 'capacity', 'cost', 'error'}
ROUTE = {'model': 'model_foundry', 'application': 'factoryos'}


def _problem_id(*parts):
    return 'prb-' + hashlib.sha256('\x1f'.join(parts).encode()).hexdigest()[:20]


class ProblemDesk:
    def __init__(self, repository, *, live_window_s=3600):
        self.repository, self.live_window_s = repository, live_window_s

    # --- inputs -----------------------------------------------------------------

    def _live(self, deployment_id):
        if not deployment_id:
            return None
        try:
            from inferstack.serving.evidence import EvidenceStore
            now = self.repository.clock()
            window = EvidenceStore(self.repository).window(deployment_id=deployment_id,
                                                           since=now - self.live_window_s, until=now)
            return window if window.get('requests') or window.get('rejected') else None
        except Exception:  # noqa: BLE001 - absence of live evidence is itself recorded by the diagnosis
            return None

    def from_feedback(self, issue_id, *, callback=None):
        """Diagnose a stored FactoryOS/operator/telemetry issue and route it."""
        from inferstack.contracts.records import parse_ref
        record = self.repository.read('feedback', issue_id)
        issue = record['body']
        deployment_id = parse_ref(issue['deployment_ref'])[0] if issue.get('deployment_ref') else None
        problem_id = _problem_id('feedback', issue_id)
        body = {'source': issue['source'], 'kind': 'symptom', 'deployment_ref': issue.get('deployment_ref'),
                'feedback_ref': f"{issue_id}@{record['revision']}", 'category': issue['category'],
                'summary': f"{issue['category']} issue {issue_id} from {issue['source']}",
                'created_at': self.repository.clock()}
        return self._diagnose_and_route(problem_id, body, issue.get('observations') or {}, deployment_id,
                                        idempotency=issue['idempotency_key'], callback=callback)

    def from_research(self, text, *, source_key, title=None, uri=None, deployment_id=None, technique_id=None):
        """Store submitted research as untrusted evidence and record it as a claim to investigate."""
        from inferstack.research import EvidenceStore
        evidence_id = EvidenceStore(self.repository).ingest_text(
            text, source_key=source_key, provenance='user_supplied', title=(title or '')[:300] or None, uri=uri,
            retrieved_at=self.repository.clock())
        evidence = self.repository.read('evidence', evidence_id)
        assessment = None
        if technique_id and deployment_id:
            from inferstack.research.techniques import assess
            try:
                result = assess(self.repository, technique_id, deployment_id)
                assessment = {k: result[k] for k in ('technique_id', 'verdict', 'blockers', 'unknowns',
                                                     'proven_by', 'executable')}
            except (KeyError, ValueError) as exc:
                assessment = {'technique_id': technique_id, 'error': str(exc)[:200]}
        problem_id = _problem_id('research', evidence_id, deployment_id or '')
        deployment_ref = None
        if deployment_id:
            deployment_ref = f"{deployment_id}@{self.repository.read('deployment', deployment_id)['revision']}"
        body = {'source': 'research', 'kind': 'technique_claim', 'deployment_ref': deployment_ref,
                'evidence_ref': f"{evidence_id}@{evidence['revision']}", 'category': 'unknown',
                'summary': f"research claim: {(title or 'untitled')[:200]}",
                'diagnosis': {'version': 1, 'claim': True, 'assessment': assessment,
                              'next': 'investigate as a hypothesis: measure the target deployment and, if the '
                                      'technique is registered and compatible, plan a bounded experiment',
                              'untrusted_content': True},
                'state': 'claim_recorded', 'created_at': self.repository.clock()}
        return self._store(problem_id, body)

    def add_observations(self, problem_id, observations):
        """Attach new measurements to a problem and re-diagnose it (a new revision)."""
        from inferstack.contracts.records import parse_ref
        record = self.repository.read('problem', problem_id)
        body = dict(record['body'])
        if body['kind'] != 'symptom':
            raise ValueError('observations attach to symptoms; research claims are investigated by campaigns')
        if body['state'] in ('queued', 'routed_elsewhere'):
            raise ValueError(f"problem already {body['state']}")
        merged = {**(body['diagnosis'].get('observations') or {}), **numeric_observations(observations)}
        deployment_id = parse_ref(body['deployment_ref'])[0] if body.get('deployment_ref') else None
        idempotency = f"{problem_id}:{record['revision'] + 1}"
        return self._diagnose_and_route(problem_id, body, merged, deployment_id, idempotency=idempotency,
                                        expected_revision=record['revision'])

    # --- routing ----------------------------------------------------------------

    def _diagnose_and_route(self, problem_id, body, observations, deployment_id, *, idempotency, callback=None,
                            expected_revision=0):
        live = self._live(deployment_id)
        diagnosis = diagnose(body['category'], observations, live)
        diagnosis['observations'] = numeric_observations(observations)
        diagnosis['live_window'] = {k: live.get(k) for k in ('requests', 'errors', 'rejected', 'ttft_p95_ms',
                                                            'e2e_p95_ms', 'stall_p95_ms')} if live else None
        body = {**body, 'diagnosis': diagnosis}
        owner = diagnosis['owner']
        if owner in ROUTE:
            body.update(state='routed_elsewhere')
            body['diagnosis']['route_to'] = ROUTE[owner]
        elif owner == 'inference' and body['category'] == 'quality':
            # A quality regression attributed to an inference release is not a speed-optimization
            # job: compare against the previous release and consider a rollback first.
            body['state'] = 'diagnosed'
            body['diagnosis']['next'] = ('compare outputs of the current and previous release on held-out inputs; '
                                         'roll back if the regression is confirmed')
        elif owner == 'inference' and diagnosis['symptom_evidenced'] and body['category'] in ACTIONABLE \
                and deployment_id:
            body['job_id'] = self._submit(problem_id, body, deployment_id, idempotency, callback)
            body['state'] = 'queued'
        else:
            body['state'] = 'needs_measurement'
        return self._store(problem_id, body, expected_revision=expected_revision)

    def _submit(self, problem_id, body, deployment_id, idempotency, callback):
        diagnosis = body['diagnosis']
        # Composed only from structured fields: category, numbers, bottleneck, requests.
        goal = (f"Address {body['summary']} on {deployment_id}. Diagnosis: owner={diagnosis['owner']}, "
                f"bottleneck={diagnosis['bottleneck']} ({diagnosis['confidence']}). Numeric observations: "
                f"{json.dumps(diagnosis['observations'], sort_keys=True)[:1000]}.")
        if diagnosis['bottleneck'] == 'uncertain':
            goal += ' The bottleneck is not yet attributed: prefer experiments that also measure it.'
        payload = {'deployment_id': deployment_id, 'goal': goal, 'source': body['source'],
                   'problem_ref': problem_id, 'diagnosis': {k: diagnosis[k] for k in ('owner', 'bottleneck',
                                                                                       'confidence')},
                   'evidence_ids': []}
        if body.get('feedback_ref'):
            payload['feedback_ref'] = body['feedback_ref']
        if callback:
            payload['outcome_callback'] = callback
        return self.repository.submit('feedback:' + idempotency, payload)

    def _store(self, problem_id, body, *, expected_revision=0):
        try:
            current = self.repository.read('problem', problem_id)
        except KeyError:
            current = None
        if current is not None and expected_revision == 0:
            return current  # idempotent intake: the same source maps to the same problem
        self.repository.put_record('problem', problem_id, body, expected_revision=expected_revision)
        return self.repository.read('problem', problem_id)
