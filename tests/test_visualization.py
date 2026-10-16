import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from types import SimpleNamespace

from chatflypt.brain import Trial
from chatflypt.visualization import NeuralView


def test_scene_preserves_recorded_spikes_and_real_edges_without_inventing_motion():
    graph=csr_matrix(np.array([[0.,1.,0.],[0.,0.,-1.],[0.,0.,0.]]))
    view=NeuralView.__new__(NeuralView)
    view.brain=SimpleNamespace(ids=np.array([720575940628857210,720575940628857211,720575940628857212]),
        indptr=graph.indptr,indices=graph.indices,weights=graph.data)
    view.rows=pd.DataFrame({'cell_type':['input','central','DNa02'],'cell_class':['sensory','central','descending'],
                           'side':['right','','right']})
    view.valid=np.ones(3,bool);view.base=np.arange(3);view.positions=np.zeros((3,3))
    view.classes=np.array([1,0,2]);view.turn_cells=np.array([2])
    trial=Trial(np.array([2,1,1]),np.array([0,1,0,2]),np.array([11,50,125,199]),17,20.)
    scene=view.activity(trial,np.array([0]))
    assert scene['totals']==[2,2]
    assert scene['frames']==[[0,1,1,1],[0,1,2,1]]
    assert all(isinstance(n[0],str) for n in scene['nodes'])
    assert scene['nodes'][0][4]==3
    assert set(tuple(x) for x in scene['edges'])=={(0,1,1),(1,2,-1)}
    assert scene['motor_evidence'][0]['spikes']==1
    assert scene['body']['trajectory']==[]
    assert scene['body']['status']=='not_connected'
