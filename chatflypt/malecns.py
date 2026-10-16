"""MaleCNS v1.0 brain + VNC simulation, with explicit model assumptions.

This ports the existing Shiu-style LIF equations to a separate connectome.
It is a computational model, not a physiologically calibrated MaleCNS model.
"""
import argparse
import json
import time
import numpy as np
import pandas as pd
import pyarrow as pa
from scipy.sparse import csr_matrix
from .brain import Brain, Trial, simulate, DT_MS
from .paths import ROOT, dump, sha256
from .social import MALE, to_engine_tape

OUT = ROOT / 'data/derived/malecns'
REPORT = ROOT / 'reports/full-pathway'
EDGES = MALE.parent / 'connectome-weights-male-cns-v1.0-minconf-0.5.feather'
TRANSMITTERS = MALE.parent / 'body-neurotransmitters-male-cns-v1.0.feather'
SIGN = {'acetylcholine': 1, 'gaba': -1, 'glutamate': -1, 'histamine': -1}


def build_graph():
    start=time.perf_counter()
    rows=pd.read_feather(MALE).fillna('')
    rows=rows[rows.superclass.ne('')].sort_values('bodyId').reset_index(drop=True)
    ids=rows.bodyId.to_numpy(np.int64)
    if len(np.unique(ids))!=len(ids): raise ValueError('Duplicate annotated body IDs')
    nt=pd.read_feather(TRANSMITTERS).set_index('body').reindex(ids).fillna('')
    signs=nt.consensus_nt.map(SIGN).fillna(0).to_numpy(np.int8)
    index=pd.Index(ids)
    sources=[];targets=[];weights=[]
    raw_rows=0;annotated_edges=0;unmodeled_edges=0;total_synapses=0
    with pa.memory_map(str(EDGES),'r') as handle:
        reader=pa.ipc.open_file(handle)
        for b in range(reader.num_record_batches):
            batch=reader.get_batch(b)
            pre=index.get_indexer(batch.column('body_pre').to_numpy())
            post=index.get_indexer(batch.column('body_post').to_numpy())
            weight=batch.column('weight').to_numpy()
            raw_rows+=len(pre)
            valid=(pre>=0)&(post>=0)&(weight>0)
            pre,post,weight=pre[valid],post[valid],weight[valid]
            annotated_edges+=len(pre);total_synapses+=int(weight.sum())
            known=signs[pre]!=0
            unmodeled_edges+=int((~known).sum())
            sources.append(pre[known].astype(np.int32))
            targets.append(post[known].astype(np.int32))
            weights.append(weight[known].astype(np.float64)*signs[pre[known]]*.275)
            if b%200==0: print(f'MaleCNS graph {b+1}/{reader.num_record_batches} batches',flush=True)
    matrix=csr_matrix((np.concatenate(weights),(np.concatenate(sources),np.concatenate(targets))),
                      shape=(len(ids),len(ids)))
    matrix.sum_duplicates();matrix.sort_indices()
    OUT.mkdir(parents=True,exist_ok=True)
    np.savez(OUT/'graph.npz',ids=ids,indptr=matrix.indptr,indices=matrix.indices,weights=matrix.data)
    provenance={'dataset':'MaleCNS v1.0','id_namespace':'malecns_body_id','neurons':len(ids),
        'selection':f'All {len(ids)} bodies with a nonempty superclass annotation; unclassified fragments excluded',
        'raw_edge_rows':raw_rows,'annotated_edge_rows':annotated_edges,'annotated_synapse_count':total_synapses,
        'modeled_edges':matrix.nnz,'edges_without_fast_transmitter_model':unmodeled_edges,
        'transmitter_signs':SIGN,'transmitter_counts':nt.consensus_nt.value_counts().to_dict(),
        'other_transmitters':'Unknown and monoaminergic outgoing effects are zero; no invented fast synaptic sign.',
        'equations':'Existing Shiu-style LIF engine; 0.1 ms timestep, 1.8 ms delay, 0.275 mV/synapse gain.',
        'calibration':'Parameters transferred from a brain model; no MaleCNS/VNC physiological calibration.',
        'source_sha256':{p.name:sha256(p) for p in [MALE,EDGES,TRANSMITTERS]},
        'graph_sha256':sha256(OUT/'graph.npz'),'build_seconds':time.perf_counter()-start,
        'sources':['https://male-cns.janelia.org/download/','https://www.nature.com/articles/s41586-024-07763-9']}
    dump(OUT/'graph.json',provenance);dump(REPORT/'graph.json',provenance)
    print(json.dumps({k:provenance[k] for k in ['neurons','modeled_edges','build_seconds']}),flush=True)


class MaleCNS(Brain):
    dataset='MaleCNS v1.0'
    id_namespace='malecns_body_id'
    def __init__(self):
        if not (OUT/'graph.npz').exists(): build_graph()
        with np.load(OUT/'graph.npz',allow_pickle=False) as f:
            self.ids=f['ids'];self.indptr=f['indptr'];self.indices=f['indices'];self.weights=f['weights']
        self.version='malecns-v1.0'
        self.lookup={str(i):j for j,i in enumerate(self.ids)}
        self.rows=pd.read_feather(MALE).set_index('bodyId').reindex(self.ids).fillna('')
        sensory=self.rows.superclass.str.contains('sensory').to_numpy()
        self.nonsensory=~sensory
        self.motor=self.rows.superclass.isin(['descending_neuron','vnc_motor','cb_motor']).to_numpy()
        self.source_connectivity=EDGES

    def run_events(self,events,record_spikes=True):
        duration=float(events['duration_ms'])
        if (not np.isfinite(duration) or not 0 < duration <= 10000 or
                abs(duration / DT_MS - round(duration / DT_MS)) > 1e-8):
            raise ValueError('Duration must be a multiple of 0.1 ms in (0, 10000]')
        tape=to_engine_tape(self,events)
        result=simulate(self.indptr,self.indices,self.weights,*tape,record_spikes)
        return Trial(result[0],result[1],result[2],int(events['seed']),duration)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['build']);parser.parse_args()
    build_graph()
