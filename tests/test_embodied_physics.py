"""Real full-loop acceptance, including deterministic replay of its input tape."""
import json
import numpy as np

from chatflypt.neural_session import replay_events
from chatflypt.embodied import run_embodied
from chatflypt.environment import validate_objects
from chatflypt.malecns import MaleCNS
from chatflypt.paths import ROOT
from chatflypt.sensory_catalogue import build_catalogue


def test_measured_contacts_return_to_same_neural_and_physical_state():
    brain=MaleCNS();catalogue=build_catalogue(save=False)
    environment={'schema':'chatflypt.environment.v1','revision':1,
        'objects':validate_objects([{'id':'contact_sugar','type':'sugar','position_mm':[0,0,.1],'size_mm':[5,4,.2]}])}
    folder=ROOT/'reports/scene-pathway/embodied-contact'
    result=run_embodied(brain,brain.rows,catalogue,'MaleCNS v1.0',[],environment,91505,'ground',
                        duration_ms=300.,output_dir=folder)
    feedback=result['feedback'];samples=feedback['samples'];frames=result['body']['trajectory']
    assert [s['time_ms'] for s in samples]==[0,100,200]
    assert feedback['neural_reset_count']==feedback['body_reset_count']==1
    assert result['body']['controls']['neural_feedback']
    assert result['body']['summary']['finite_state'] and not result['body']['summary']['physics_warnings']
    for sample in samples:
        at=next(f for f in frames if abs(f['time_ms']-sample['time_ms'])<1e-7)
        np.testing.assert_array_equal(at['position_mm'],sample['sensor_pose']['position_mm'])
        assert len(sample['vision']['ommatidia'])==2
        for item in sample['items']:
            assert catalogue['channels'][item['channel']]['dataset']=='MaleCNS v1.0'
    assert any(i['channel']=='leg_sweet_local' for s in samples for i in s['items'])
    assert any(s['evidence'] for s in samples)
    # Final tape is a measurement-derived record. Rerunning it uses no body,
    # target, user intent, environment labels, or decoder.
    replay=replay_events(brain,result['events'])
    for name in ('counts','spike_ids','spike_steps','state_record'):
        np.testing.assert_array_equal(getattr(result['trial'],name),getattr(replay,name))
    summary={'duration_ms':300,'feedback_samples':len(samples),'spikes':int(replay.counts.sum()),
        'body_summary':result['body']['summary'],'replay_bitwise_identical':True,
        'body_continuity_verified':True,'neural_resets':1,'body_resets':1,
        'input_cells':len(result['events']['input_neuron_ids']),'input_events':len(result['events']['event_steps']),
        'feedback_positions_mm':[s['sensor_pose']['position_mm'] for s in samples]}
    (folder/'acceptance.json').write_text(json.dumps(summary,indent=2),encoding='utf8')
