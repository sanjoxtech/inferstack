import pytest
from inferstack.storage.repository import Repository
from inferstack.releases.manager import ReleaseManager


def test_route_cas_promotion_and_rollback(tmp_path):
    repo=Repository(tmp_path/'db',tmp_path/'artifacts')
    for name in ('old','new'):
        repo.put('release',name,{'deployment':{'body':{'model':'m'}}})
    manager=ReleaseManager(repo)
    assert manager.route('m','old',candidate_release='new',weight=100)==1
    assert manager.select('m','request')['id']=='new'
    with pytest.raises(ValueError,match='revision conflict'):manager.promote('m',0)
    assert manager.promote('m',1)==2
    assert manager.rollback('m',2)==3
    assert manager.select('m','request')['id']=='old'
    with pytest.raises(ValueError):manager.route('other','new')


def test_unverified_job_cannot_release(tmp_path):
    repo=Repository(tmp_path/'db',tmp_path/'artifacts')
    job=repo.submit('a',{})
    with pytest.raises(ValueError,match='verified'):ReleaseManager(repo).create(job)

def test_release_pins_evidence_and_rejects_mutation(tmp_path):
    import hashlib
    repo=Repository(tmp_path/'db',tmp_path/'artifacts')
    spec={'id':'d','model':'m','engine':'generic_openai','base_url':'http://candidate'}
    repo.put('deployment','d',spec)
    candidate={'id':'attempt','deployment_id':'d','revision':1,
               'spec_sha256':hashlib.sha256(repo.encode(spec).encode()).hexdigest()}
    evidence={'verdict':{'decision':'keep'},'candidate':candidate,'scope':'test exact-match suite'}
    artifact=repo.artifact(repo.encode(evidence).encode())
    job=repo.submit('a',{})
    lease=repo.claim()
    repo.finish(job,lease['lease_token'],{'decision':'keep','experiments':[
        {'candidate':candidate,'verdict':{'decision':'keep','evidence':artifact}}]})
    manager=ReleaseManager(repo)
    release=manager.create(job)
    assert manager.create(job)==release
    repo.put('deployment','d',{**spec,'base_url':'http://changed'},1)
    with pytest.raises(ValueError,match='changed'):manager.create(job)
    assert repo.read('release',release)['body']['deployment']['body']==spec
