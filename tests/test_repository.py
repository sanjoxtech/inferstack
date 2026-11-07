import pytest
from inferstack.storage.repository import Repository


def test_records_require_compare_and_swap(tmp_path):
    r=Repository(tmp_path/'db',tmp_path/'artifacts')
    assert r.put('deployment','a',{'engine':'native'})==1
    with pytest.raises(ValueError):r.put('deployment','a',{})
    assert r.put('deployment','a',{'engine':'vllm'},expected_revision=1)==2
    with pytest.raises(ValueError):r.put('deployment','a',{},expected_revision=1)
    assert r.read('deployment','a')['body']['engine']=='vllm'


def test_artifact_integrity_and_path_validation(tmp_path):
    r=Repository(tmp_path/'db',tmp_path/'artifacts')
    a=r.artifact(b'profile')
    assert r.read_artifact(a['sha256'])==b'profile'
    with pytest.raises(ValueError):r.read_artifact('../secret')
    (tmp_path/'artifacts'/a['sha256']).write_bytes(b'changed')
    with pytest.raises(ValueError):r.read_artifact(a['sha256'])
