import pytest
from inferstack.evaluation.metrics import StreamMeasurement
from inferstack.evaluation.judge import Objective, evaluate


def test_metadata_and_grouped_content_not_tokens():
    s=StreamMeasurement(0)
    s.feed('{"choices":[]}',1)
    s.feed('{"choices":[{"text":"hello world"}]}',2)
    s.feed('{"choices":[],"usage":{"completion_tokens":2}}',3)
    s.feed('[DONE]',4)
    r=s.result(4)
    assert r['output_tokens']==2 and r['content_chunks']==1
    assert r['ttft_ms']==2000 and r['tpot_ms'] is None


def test_missing_usage_and_truncated_stream():
    s=StreamMeasurement(0)
    s.feed('{"choices":[{"text":"hi"}]}',1)
    r=s.result(2)
    assert not r['success'] and r['output_tokens'] is None


def test_malformed_event_rejected():
    with pytest.raises(ValueError): StreamMeasurement(0).feed('oops',1)


def test_judge_requires_measurements_and_gates():
    gates=dict.fromkeys(['correctness','quality','health','comparable','complete'],True)
    obj=[Objective('latency','min',100)]
    assert evaluate([{}]*3,[{}]*3,obj,gates=gates)['decision']=='inconclusive'
    assert evaluate([{'latency':50}]*3,[{'latency':40}]*3,obj,gates={})['decision']=='inconclusive'
    assert evaluate([{'latency':50}]*3,[{'latency':40}]*3,obj,gates=gates)['decision']=='keep'
    assert evaluate([{'latency':50}]*3,[{'latency':60}]*3,obj,gates=gates)['decision']=='reject'
    gates['correctness']=False
    assert evaluate([{'latency':50}]*3,[{'latency':20}]*3,obj,gates=gates)['decision']=='reject'


def test_noise_requires_retest():
    gates=dict.fromkeys(['correctness','quality','health','comparable','complete'],True)
    r=evaluate([{'x':50}]*3,[{'x':v} for v in (1,30,80)],[Objective('x','min')],gates=gates)
    assert r['decision']=='retest'
