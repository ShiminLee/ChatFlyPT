"""Summaries depend on measured model activity, never on requested directions."""
import numpy as np
from .brain import DT_MS


def summarize_reaction(brain,trial,driven,rows):
    classes=rows.super_class.astype(str)
    sensory=classes.str.contains('sensory').to_numpy(copy=True)
    sensory[np.asarray(driven,dtype=int)]=True
    masks={
        'sensory':sensory,
        'central_brain':classes.isin(['cb_intrinsic','central']).to_numpy(),
        'vnc_intrinsic':classes.eq('vnc_intrinsic').to_numpy(),
        'ascending':classes.isin(['ascending_neuron','ascending']).to_numpy(),
        'DN':classes.isin(['descending_neuron','descending']).to_numpy(),
        'vnc_motor':classes.eq('vnc_motor').to_numpy(),
        'cb_motor':classes.isin(['cb_motor','motor']).to_numpy(),
    }
    def summary(mask):
        times=trial.spike_steps[mask[trial.spike_ids]] if len(trial.spike_ids) else np.empty(0)
        return {'spikes':int(trial.counts[mask].sum()),
                'active_cells':int(np.count_nonzero(trial.counts[mask])),
                'first_spike_ms':float(times.min()*DT_MS) if len(times) else None}
    compartments={k:summary(v) for k,v in masks.items()}
    motor=np.flatnonzero(rows.cell_type.isin(['DNa01','DNa02','MDN','MN9']).to_numpy())
    evidence=[{'neuron_id':str(brain.ids[i]),'type':str(rows.iloc[i].cell_type),
        'side':str(rows.iloc[i].side),'spikes':int(trial.counts[i])} for i in motor]
    result={'non_sensory_spikes':int(trial.counts[~sensory].sum()),
        'active_non_sensory_neurons':int(np.count_nonzero(trial.counts[~sensory])),
        'duration_ms':trial.duration_ms,'compartments':compartments,
        **{k:compartments[k] for k in ['DN','vnc_motor','cb_motor']},'motor_evidence':evidence,
        'behavior_inference':'Not assigned: output cell spikes are not an observed movement or subjective intention.'}
    if trial.state_record is not None:
        area=trial.state_record[:2,~sensory]
        peaks=trial.state_record[2:,~sensory]
        result['membrane_response']={
            'internal_cells':int(np.count_nonzero(np.any(peaks>1e-6,axis=0))),
            'depolarized_cells':int(np.count_nonzero(peaks[0]>1e-6)),
            'hyperpolarized_cells':int(np.count_nonzero(peaks[1]>1e-6)),
            'positive_area_mv_ms':float(area[0].sum()),
            'negative_area_mv_ms':float(area[1].sum()),
            'max_depolarization_mv':float(peaks[0].max(initial=0)),
            'max_hyperpolarization_mv':float(peaks[1].max(initial=0)),
            'scope':'LIF voltage relative to -52 mV; includes subthreshold responses, excludes sensory and driven cells.'}
    return result
