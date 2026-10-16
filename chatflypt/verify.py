"""Compare our engine against Brian2 on an identical deterministic input tape."""
import numpy as np
from scipy.sparse import csr_matrix
import brian2 as b2
from .brain import simulate, DT_MS
from .paths import dump, REPORTS


def brian_reference(indptr, indices, weights, tape):
    b2.start_scope()
    b2.prefs.codegen.target = "numpy"
    b2.defaultclock.dt = DT_MS * b2.ms
    inputs, ptr, events, steps = tape
    n = len(indptr)-1
    group = b2.NeuronGroup(n,
        'dv/dt=(-52*mV-v+g)/(20*ms):volt (unless refractory)\n'
        'dg/dt=-g/(5*ms):volt (unless refractory)\nrfc:second',
        threshold='v > -45*mV', reset='v=-52*mV;g=0*mV',
        refractory='rfc', method='linear')
    group.v = -52*b2.mV
    group.rfc = 2.2*b2.ms
    group.rfc[inputs] = 0*b2.ms
    syn = b2.Synapses(group, group, 'w:volt', on_pre='g+=w', delay=1.8*b2.ms)
    sources = np.repeat(np.arange(n), np.diff(indptr))
    syn.connect(i=sources, j=indices)
    syn.w = weights*b2.mV
    # Same stateupdate template as PoissonInput; randomness is replaced by tape.
    kicks = np.zeros((steps, len(inputs)))
    event_steps = np.repeat(np.arange(steps), np.diff(ptr))
    inverse = {int(v): i for i,v in enumerate(inputs)}
    for t, i in zip(event_steps, events):
        kicks[t, inverse[int(i)]] += 68.75
    objects = []
    for k, i in enumerate(inputs):
        stim = b2.TimedArray(kicks[:,k]*b2.mV, dt=DT_MS*b2.ms)
        runner = group[int(i):int(i)+1].run_regularly(f'v += drive_{k}(t)', when='synapses')
        group.namespace[f'drive_{k}'] = stim
        objects.append(runner)
    monitor = b2.SpikeMonitor(group)
    net = b2.Network(group,syn,monitor,*objects)
    net.run(steps*DT_MS*b2.ms, namespace={})
    return np.asarray(monitor.i), np.rint(monitor.t/b2.ms/DT_MS).astype(int), np.asarray(group.v/b2.mV), np.asarray(group.g/b2.mV)


def verify_small():
    rng = np.random.default_rng(3301)
    n, steps = 48, 1500
    dense = rng.uniform(0, 60, (n,n)) * (rng.random((n,n)) < .13)
    dense[2::3] *= -1
    np.fill_diagonal(dense,0)
    graph=csr_matrix(dense)
    inputs=np.array([0,1,2,3],np.int32)
    times, columns=np.where(rng.random((steps,len(inputs))) < .045)
    ptr=np.r_[0,np.cumsum(np.bincount(times,minlength=steps))].astype(np.int64)
    tape=(inputs,ptr,inputs[columns],steps)
    ours=simulate(graph.indptr,graph.indices,graph.data,*tape,True)
    reference=brian_reference(graph.indptr,graph.indices,graph.data,tape)
    # Within-timestep monitor ordering differs; compare event multisets.
    a=sorted(zip(ours[2].tolist(),ours[1].tolist()))
    b=sorted(zip(reference[1].tolist(),reference[0].tolist()))
    result={'case':'48-cell excitatory/inhibitory recurrent graph, fixed input tape',
            'spikes_custom':len(a),'spikes_brian2':len(b),'spikes_identical':a==b,
            'max_voltage_error_mv':float(np.max(np.abs(ours[3]-reference[2]))),
            'max_conductance_error_mv':float(np.max(np.abs(ours[4]-reference[3])))}
    result['passed']=result['spikes_identical'] and result['max_voltage_error_mv']<1e-8 and result['max_conductance_error_mv']<1e-8
    dump(REPORTS/'engine-verification-small.json',result)
    print(result,flush=True)
    if not result['passed']:
        raise AssertionError('Simulator does not match Brian2')
    return result


if __name__=='__main__':
    verify_small()
