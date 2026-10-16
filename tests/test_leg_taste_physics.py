"""Verify the leg bridge with real MuJoCo contacts and real MaleCNS events."""
import json
from pathlib import Path

import numpy as np

from chatflypt.environment import probe_environment, sample_environment, validate_objects
from chatflypt.sensory_catalogue import build_catalogue
from chatflypt.sensory_events import encode_inputs
from chatflypt.social import to_engine_tape
from chatflypt.brain import recorded_trial
from chatflypt.malecns import MaleCNS


def test_real_sugar_platform_contacts_enter_exact_leg_candidate_cells():
    catalogue=build_catalogue(save=False)
    environment={'revision':1,'objects':validate_objects([{'id':'contact_sugar','type':'sugar',
        'position_mm':[0,0,.1],'size_mm':[5,4,.2]}])}
    probe=probe_environment(environment,seed=91501)
    sample=sample_environment(environment,probe,catalogue)
    selected=[i for i in sample['items'] if i['channel'] in ('leg_sweet_local','leg_appetitive_ascending')]
    assert selected
    contact_evidence=[e for e in sample['evidence'] if e['mode']=='mujoco_leg_taste_candidate']
    for row in contact_evidence:
        assert row['contacts'] and row['side'] in ('left','right')
        spec=catalogue['channels'][row['channel']]
        assert row['neuron_ids']==spec['body_part_ids'][row['body_part']][row['side']]
        assert not set(row['neuron_ids']) & set(catalogue['channels']['sugar_specific']['neuron_ids'])
    events=encode_inputs(catalogue,selected,91501)['MaleCNS v1.0']
    expected={int(i) for row in contact_evidence for i in row['neuron_ids']}
    assert set(events['input_neuron_ids'])==expected and len(events['event_steps'])>0
    brain=MaleCNS();tape=to_engine_tape(brain,events)
    trial=recorded_trial(brain,tape,91501,events['duration_ms'])
    internal=brain.nonsensory.copy();internal[tape[0]]=False
    assert trial.counts[internal].sum()>0 or np.any(trial.state_record[2:,internal]>1e-6)
    root=Path(__file__).resolve().parents[1];folder=root/'reports/scene-pathway';folder.mkdir(parents=True,exist_ok=True)
    raw=folder/'leg-contact-neural.npz'
    np.savez_compressed(raw,counts=trial.counts,spike_ids=trial.spike_ids,spike_steps=trial.spike_steps,
        state_record=trial.state_record,neuron_ids=brain.ids,driven_indices=tape[0],
        input_neuron_ids=events['input_neuron_ids'],event_neuron_ids=events['event_neuron_ids'],event_steps=events['event_steps'])
    report={'environment':environment,'contacts':probe['environment_contacts'],'candidate_inputs':selected,
        'evidence':contact_evidence,'input_cells':len(expected),'input_events':len(events['event_steps']),
        'internal_spikes':int(trial.counts[internal].sum()),
        'membrane_internal_cells':int(np.any(trial.state_record[2:,internal]>1e-6,axis=0).sum()),
        'raw_record':str(raw.relative_to(root)),'neural_simulation_run':True,
        'sugar_label_decoded':False,'mapping_status':'population_projection_match_candidate'}
    (folder/'leg-contact.json').write_text(json.dumps(report,ensure_ascii=False,indent=2),encoding='utf8')
