"""Infer candidate visual columns from actual same-side R1-R6 -> L1/L2 edges.

This produces a review artifact, not a camera-to-connectome registration. Both
L1 and L2 must independently agree, with at least 75% of their combined weight
in that column. Unassigned sides and conflicts are retained as unresolved.
"""
import json
import sys
from collections import defaultdict,Counter
from pathlib import Path
import numpy as np
import pandas as pd

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from chatflypt.malecns import MaleCNS
from chatflypt.social import MALE
from chatflypt.paths import dump,sha256


def main():
    brain=MaleCNS()
    rows=pd.read_feather(MALE).set_index('bodyId').reindex(brain.ids)
    kinds=rows['type'].fillna('').to_numpy();side=rows['rootSide'].fillna('').to_numpy()
    target_side=rows['somaSide'].fillna('').to_numpy()
    coords=rows[['assignedOlHex1','assignedOlHex2']].to_numpy(float)
    phot=np.flatnonzero(kinds=='R1-R6');result=[]
    for i in phot:
        votes={'L1':defaultdict(float),'L2':defaultdict(float)};targets=[]
        lo,hi=brain.indptr[i:i+2]
        for j,w in zip(brain.indices[lo:hi],brain.weights[lo:hi]):
            if kinds[j] not in votes or side[i] not in ('L','R') or target_side[j]!=side[i]:continue
            if not np.isfinite(coords[j]).all() or not np.equal(coords[j],np.floor(coords[j])).all():continue
            column=tuple(map(int,coords[j]));votes[kinds[j]][column]+=abs(float(w))
            targets.append({'body_id':str(brain.ids[j]),'type':str(kinds[j]),'column':column,
                            'model_weight':float(w)})
        best={k:max(v,key=v.get) if v else None for k,v in votes.items()}
        agreed=best['L1'] is not None and best['L1']==best['L2']
        total=sum(sum(v.values()) for v in votes.values())
        fraction=sum(v.get(best['L1'],0) for v in votes.values())/total if agreed and total else 0.
        accepted=agreed and fraction>=.75
        reason='candidate' if accepted else 'L1_L2_disagree_or_missing' if not agreed else 'insufficient_column_dominance'
        result.append({'body_id':str(brain.ids[i]),'entry_side':str(side[i]),'status':reason,
            'candidate_column':best['L1'] if accepted else None,'weight_fraction':fraction,
            'l1_best':best['L1'],'l2_best':best['L2'],'targets':targets})
    assert len({r['body_id'] for r in result})==len(phot)
    accepted=[r for r in result if r['status']=='candidate']
    folder=ROOT/'research/retinotopy'
    pd.DataFrame([{'body_id':r['body_id'],'entry_side':r['entry_side'],
        'column_hex1':r['candidate_column'][0],'column_hex2':r['candidate_column'][1],
        'weight_fraction':r['weight_fraction']} for r in accepted]).to_csv(folder/'malecns-r1-r6-column-candidates.csv',index=False)
    report={'dataset':'MaleCNS v1.0','source_annotations_sha256':sha256(MALE),
        'graph':'data/derived/malecns/graph.npz; signed fast-transmitter model edges',
        'selection':'Exact R1-R6 inputs; exact same-side L1/L2 targets; independently agreeing annotated columns; 75% combined weight',
        'threshold_is_engineering':True,'photoreceptors':len(result),'statuses':dict(Counter(r['status'] for r in result)),
        'candidate_counts_by_side':dict(Counter(r['entry_side'] for r in accepted)),
        'unique_candidate_columns':len({(r['entry_side'],*r['candidate_column']) for r in accepted}),
        'executable_camera_registration':False,'used_in_live_simulation':False,
        'limitations':['Candidates for downstream cartridge/column assignment, not measured photoreceptor optical axes.',
            'The model graph omits edges whose fast-transmitter sign is unknown; this is not an exhaustive anatomical reconstruction.',
            'Camera alignment, absolute orientation, receptor physiology and independent spatial-stimulus validation remain missing.'],
        'cells':result}
    dump(folder/'column-candidate-audit.json',report)
    print(json.dumps({k:report[k] for k in ('photoreceptors','statuses','candidate_counts_by_side','unique_candidate_columns','executable_camera_registration')},indent=2))


if __name__=='__main__':main()
