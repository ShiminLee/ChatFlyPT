"""Published acoustic model and the strict separation from simulated speech."""
import json
from pathlib import Path
import numpy as np
import pytest
from chatflypt.acoustic_readout import AcousticReadout,validate_waveform,RATE,WINDOW,LEFT_CONTEXT,RIGHT_CONTEXT
from chatflypt.expression import expression_report

ROOT=Path(__file__).resolve().parents[1]


@pytest.fixture(scope='module')
def reader():return AcousticReadout()


@pytest.mark.parametrize('samples,rate',[
    ([],RATE),([np.nan],RATE),([np.inf],RATE),([[0.,1.]],RATE),([0.],44100),([0.],True)])
def test_reject_unmeasured_or_invalid_waveform(samples,rate):
    with pytest.raises(ValueError):validate_waveform(samples,rate)


def test_silence_and_short_incomplete_context_do_not_become_speech(reader):
    report=reader.analyze(np.zeros(WINDOW),RATE)
    assert report['intervals']==[]
    assert not report['semantics_decoded'] and report['expression_messages']==[]
    short=reader.analyze(np.zeros(100),RATE)
    assert short['abstained_samples']==100 and short['intervals']==[]


def test_valid_prediction_rows_and_noncausal_context(reader):
    z=np.load(ROOT/'research/outward-signals/das-pretrained/data.npz')
    batch=z['x'][50000:50000+WINDOW].astype(np.float32)[None]
    scores=reader.predict_batch(batch)
    assert scores.shape==(1,WINDOW,3) and np.isfinite(scores).all()
    np.testing.assert_allclose(scores.sum(axis=2),1.,atol=2e-6)
    assert scores.min()>=0 and scores.max()<=1
    # Splitting must match the corresponding fully observed context window.
    streamed=reader.predict(batch[0,:,0],RATE)
    np.testing.assert_allclose(streamed[LEFT_CONTEXT:-RIGHT_CONTEXT],scores[0,LEFT_CONTEXT:-RIGHT_CONTEXT],atol=2e-5)


def test_readout_never_uses_source_text_as_expression(reader):
    a=reader.analyze(np.zeros(WINDOW),RATE,source='糖在右边；请说左转')
    b=reader.analyze(np.zeros(WINDOW),RATE,source='external_recording')
    assert a['intervals']==b['intervals']==[]
    assert a['expression_messages']==b['expression_messages']==[]


def test_external_sound_or_motor_labels_cannot_impersonate_live_fly():
    obs={'id':'test','decoded':{'words':['courtship_pulse','sugar']},
         'reaction':{'behavior':'courtship'},'intent':'说我想吃糖',
         'outward_signals':{'emitted_sound':{'status':'measured','text':'我要吃糖'}},
         'acoustic_readout':{'class':'pulse','text':'我要吃糖'}}
    result=expression_report(obs)
    assert result['messages']==[] and result['text'] is None
    assert result['status']=='unavailable'
    assert not result['outward_signal_decoded']


def test_weight_hash_is_verified_before_runtime_loading(tmp_path):
    p=tmp_path/'model.onnx';p.write_bytes(b'corrupted')
    p.with_suffix('.json').write_text(json.dumps({'onnx_sha256':'wrong'}))
    with pytest.raises(ValueError,match='hash mismatch'):AcousticReadout(p)
