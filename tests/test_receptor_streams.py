"""Receptor noise stays independent, while window boundaries remain replayable."""
import numpy as np
from chatflypt.embodied import receptor_seed, merge_measured_tapes


def test_equal_receptor_rates_do_not_reuse_the_same_noise_sequence():
    sources=('external','vision','proprioception')
    sequences=[]
    for source in sources:
        sequences.append(np.concatenate([
            np.random.default_rng(receptor_seed(42,step,source)).random(1000)<.05
            for step in range(0,10000,100)]))
    assert all(not np.array_equal(a,b) for i,a in enumerate(sequences) for b in sequences[i+1:])
    # Restarting the RNG for each physical source with one shared seed gives
    # correlation 1; independent streams are near zero at this sample size.
    correlation=np.corrcoef(sequences)
    assert np.max(np.abs(correlation[np.triu_indices(3,1)]))<.03
    one=[receptor_seed(42,t,'vision') for t in range(100)]
    two=[receptor_seed(42,t,'vision') for part in (range(30),range(30,100)) for t in part]
    assert one==two


def test_independent_sources_can_still_have_coincident_events():
    a={'dataset':'MaleCNS v1.0','id_namespace':'malecns_body_id','dt_ms':.1,
       'duration_ms':10.,'event_steps':np.array([1,4]),'event_neuron_ids':np.array([10,11]),
       'input_neuron_ids':np.array([10,11])}
    b=dict(a,event_steps=np.array([1,7]),event_neuron_ids=np.array([10,12]),input_neuron_ids=np.array([10,12]))
    combined=merge_measured_tapes(a,b)
    assert combined['event_steps'].tolist()==[1,1,4,7]
    assert combined['event_neuron_ids'].tolist()==[10,10,11,12]
    assert combined['input_neuron_ids'].tolist()==[10,11,12]
