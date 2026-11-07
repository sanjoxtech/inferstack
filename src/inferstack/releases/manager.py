from __future__ import annotations
import hashlib
import json


class ReleaseManager:
    def __init__(self, repository):self.repository=repository

    def create(self,job_id):
        job=self.repository.get(job_id)
        if job['state']!='completed' or job['result'].get('decision')!='keep':
            raise ValueError('release requires completed, verified experiment')
        experiment=job['result']['experiments'][-1]
        candidate=experiment['candidate']
        evidence_ref=experiment['verdict']['evidence']
        evidence=json.loads(self.repository.read_artifact(evidence_ref['sha256']))
        if evidence['verdict']['decision']!='keep' or evidence['candidate']!=candidate:
            raise ValueError('evidence does not verify candidate')
        record=self.repository.read('deployment',candidate['deployment_id'])
        digest=hashlib.sha256(self.repository.encode(record['body']).encode()).hexdigest()
        if record['revision']!=candidate['revision'] or digest!=candidate['spec_sha256']:
            raise ValueError('deployment changed after verification')
        manifest={'deployment':record,'evidence':evidence_ref,'job_id':job_id,
                  'verification_scope':evidence['scope']}
        artifact=self.repository.artifact(self.repository.encode(manifest).encode())
        identifier=artifact['sha256']
        # Idempotent by immutable manifest digest.
        try:self.repository.put('release',identifier,manifest)
        except ValueError:
            if self.repository.read('release',identifier)['body']!=manifest:raise
        return identifier

    def route(self,model,stable_release,*,candidate_release=None,weight=0,expected_revision=0):
        if isinstance(weight,bool) or not isinstance(weight,int) or not 0<=weight<=100:
            raise ValueError('canary weight must be integer percentage')
        if weight and not candidate_release:raise ValueError('candidate required')
        for identifier in filter(None,(stable_release,candidate_release)):
            release=self.repository.read('release',identifier)
            if release['body']['deployment']['body']['model']!=model:
                raise ValueError('release model mismatch')
        body={'stable':stable_release,'candidate':candidate_release,'weight':weight}
        return self.repository.put('route',model,body,expected_revision)

    def select(self,model,request_id):
        route=self.repository.read('route',model)['body']
        bucket=int(hashlib.sha256(request_id.encode()).hexdigest()[:8],16)%100
        release_id=route['candidate'] if route['candidate'] and bucket<route['weight'] else route['stable']
        return self.repository.read('release',release_id)

    def promote(self,model,expected_revision):
        record=self.repository.read('route',model)
        if record['revision']!=expected_revision:raise ValueError('revision conflict')
        body=record['body']
        if not body['candidate']:raise ValueError('no canary to promote')
        next_body={'stable':body['candidate'],'candidate':None,'weight':0,'previous':body['stable']}
        return self.repository.put('route',model,next_body,expected_revision)

    def rollback(self,model,expected_revision):
        record=self.repository.read('route',model)
        if record['revision']!=expected_revision:raise ValueError('revision conflict')
        body=record['body']
        target=body['stable'] if body['candidate'] else body.get('previous')
        if target is None:raise ValueError('no rollback target')
        return self.repository.put('route',model,{'stable':target,'candidate':None,'weight':0},expected_revision)
