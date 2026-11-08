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
        # The release must be covered by the bound verification policy: every
        # required gate recorded True in the evidence itself.
        policy=evidence.get('policy') or {}
        required=policy.get('required') or []
        if not required or any(evidence.get('gates',{}).get(g) is not True for g in required):
            raise ValueError('evidence does not cover its verification policy')
        record=self.repository.read('deployment',candidate['deployment_id'])
        digest=hashlib.sha256(self.repository.encode(record['body']).encode()).hexdigest()
        if record['revision']!=candidate['revision'] or digest!=candidate['spec_sha256']:
            raise ValueError('deployment changed after verification')
        manifest={'deployment':record,'evidence':evidence_ref,'job_id':job_id,
                  'verification_scope':evidence['scope'],'verification_policy_sha256':policy.get('sha256')}
        serving=experiment.get('serving') or {}
        if serving.get('runnable'):manifest['runnable']=serving['runnable']
        if serving.get('allocation_id'):manifest['serving_allocation_id']=serving['allocation_id']
        artifact=self.repository.artifact(self.repository.encode(manifest).encode())
        identifier=artifact['sha256']
        # Idempotent by immutable manifest digest.
        try:self.repository.put_record('release',identifier,manifest)
        except ValueError:
            from inferstack.contracts.records import validate
            stored=self.repository.read('release',identifier)['body']
            if stored not in (manifest,validate('release',manifest).model_dump(mode='json')):raise
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
        return self.repository.put_record('route',model,body,expected_revision)

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
        return self.repository.put_record('route',model,next_body,expected_revision)

    def rollback(self,model,expected_revision):
        record=self.repository.read('route',model)
        if record['revision']!=expected_revision:raise ValueError('revision conflict')
        body=record['body']
        target=body['stable'] if body['candidate'] else body.get('previous')
        if target is None:raise ValueError('no rollback target')
        return self.repository.put_record('route',model,{'stable':target,'candidate':None,'weight':0,'previous':body.get('previous') if body['candidate'] else None},expected_revision)
