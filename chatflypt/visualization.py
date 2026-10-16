"""Small browser payloads derived from annotated positions and recorded spikes."""
import numpy as np
import pandas as pd
from .brain import DT_MS
from .stimuli import ANNOTATIONS


class NeuralView:
    def __init__(self, brain):
        self.brain = brain
        table = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False, dtype={'root_id': str}).fillna('')
        self.rows = table.set_index(table.root_id.astype(str)).reindex(brain.ids.astype(str)).fillna('')
        # Reindex can add unannotated cells; retain exact IDs without float coercion.
        self.rows['root_id'] = self.rows.index
        raw = self.rows[['pos_x', 'pos_y', 'pos_z']].apply(pd.to_numeric, errors='coerce').to_numpy(float)
        self.valid = np.isfinite(raw).all(axis=1)
        lo, hi = np.nanpercentile(raw[self.valid], [1, 99], axis=0)
        center = (lo + hi) / 2
        scale = max(hi - lo) / 2
        self.positions = np.nan_to_num((raw - center) / scale).clip(-1.5, 1.5)
        self.positions[:, 1] *= -1
        valid = np.flatnonzero(self.valid)
        self.base = np.sort(np.random.default_rng(17).choice(valid, min(2100, len(valid)), replace=False))
        self.classes = np.zeros(len(brain.ids), np.int8)
        self.classes[self.rows.super_class.isin(['sensory', 'sensory_ascending']).to_numpy()] = 1
        self.classes[self.rows.super_class.isin(['descending', 'motor']).to_numpy()] = 2
        self.turn_cells = np.flatnonzero(self.rows.cell_type.isin(['DNa01', 'DNa02']).to_numpy())

    def nodes(self, indices, driven=()):
        driven = set(map(int, driven))
        schematic=getattr(self,'schematic',None)
        types=self.rows.cell_type.to_numpy()
        classes=self.rows.cell_class.to_numpy()
        # Root IDs remain strings: JavaScript cannot represent these int64 IDs exactly.
        return [[str(self.brain.ids[i]), *np.round(self.positions[i], 4).tolist(),
                 3 if int(i) in driven else int(self.classes[i]),
                 str(types[i] or classes[i] or '未分类'),
                 'schematic' if schematic is not None and schematic[i] else 'anatomical']
                for i in indices]

    def resting(self):
        return {'nodes': self.nodes(self.base), 'edges': [], 'frames': [], 'duration_ms': 0,
                'bin_ms': 10, 'total_neurons': len(self.brain.ids), 'sampled_neurons': len(self.base),
                'source': getattr(self,'position_source','FlyWire v783 annotated representative positions'),
                'dataset':getattr(self.brain,'dataset','FlyWire v783'),
                'geometry_note': getattr(self,'geometry_note','神经元位置采样；非完整神经突形态。')}

    def activity(self, trial, driven):
        active = np.flatnonzero((trial.counts > 0) & self.valid)
        hot = active[np.argsort(trial.counts[active], kind='stable')[-700:]]
        state_hot=np.empty(0,dtype=int)
        if trial.state_record is not None:
            power=trial.state_record[2:].max(axis=0)
            responding=np.flatnonzero((power>1e-6)&self.valid)
            state_hot=responding[np.argsort(power[responding],kind='stable')[-500:]]
        indices = np.unique(np.concatenate((self.base, hot, state_hot, driven, self.turn_cells)))
        indices = indices[self.valid[indices]]
        reverse = np.full(len(self.brain.ids), -1, np.int32)
        reverse[indices] = np.arange(len(indices))
        bin_ms = 10
        bins = int(np.ceil(trial.duration_ms / bin_ms))
        counts = np.zeros((bins, len(indices)), np.int32)
        if len(trial.spike_ids):
            local = reverse[trial.spike_ids]
            timebins = np.minimum((trial.spike_steps * DT_MS / bin_ms).astype(int), bins - 1)
            keep = local >= 0
            np.add.at(counts, (timebins[keep], local[keep]), 1)
            totals = np.bincount(timebins, minlength=bins)
        else:
            totals = np.zeros(bins, dtype=int)
        frames = []
        for row in counts:
            nz = np.flatnonzero(row)
            frames.append(np.column_stack((nz, row[nz])).ravel().tolist())
        edges = []
        for src in hot[::-1][:160]:
            start, end = self.brain.indptr[src:src + 2]
            targets = self.brain.indices[start:end]
            weights = self.brain.weights[start:end]
            candidates = np.flatnonzero((reverse[targets] >= 0) & (trial.counts[targets] > 0))
            best = candidates[np.argsort(np.abs(weights[candidates]))[-4:]]
            edges.extend([[int(reverse[src]), int(reverse[targets[k]]), 1 if weights[k] > 0 else -1]
                          for k in best])
        result={'nodes': self.nodes(indices, driven), 'edges': edges, 'frames': frames,
                'totals': totals.tolist(), 'duration_ms': trial.duration_ms, 'bin_ms': bin_ms,
                'total_neurons': len(self.brain.ids), 'sampled_neurons': len(indices),
                'source': 'Current simulation spike recording; sampled nodes, real connectivity',
                'dataset':getattr(self.brain,'dataset','FlyWire v783'),
                'geometry_note': getattr(self,'geometry_note','神经元位置采样；非完整神经突形态。'),
                'motor_evidence': [{'neuron_id': str(self.brain.ids[i]),
                    'id_namespace':getattr(self.brain,'id_namespace','flywire_root_id'),
                    'type': str(self.rows.iloc[i].cell_type), 'side': str(self.rows.iloc[i].side),
                    'spikes': int(trial.counts[i])} for i in self.turn_cells],
                'body': {'status': 'not_connected', 'trajectory': [],
                         'note': '身体动力学尚未接入，未测量到动作轨迹。'}}
        if trial.state_record is not None:
            peak=trial.state_record[2:,indices]
            nonzero=np.flatnonzero(peak.max(axis=0)>1e-6)
            result['voltage_peaks']=[[int(i),float(peak[0,i]),float(peak[1,i])] for i in nonzero]
            result['voltage_note']='膜电位着色为整次试验的正/负偏移峰值，不是脉冲，也不是该时间点的瞬时电位。'
        return result


class MaleNeuralView(NeuralView):
    """Actual soma locations; missing peripheral positions explicitly schematic."""
    def __init__(self,brain):
        self.brain=brain
        rows=brain.rows.copy()
        self.rows=pd.DataFrame({'cell_type':rows['type'],'cell_class':rows['class'],
            'super_class':rows.superclass,'side':rows.somaSide.replace({'L':'left','R':'right'})})
        raw=np.full((len(rows),3),np.nan)
        for i,(soma,near) in enumerate(zip(rows.somaLocation,rows.tosomaLocation)):
            point=soma if isinstance(soma,(np.ndarray,list)) and len(soma)==3 else near
            if isinstance(point,(np.ndarray,list)) and len(point)==3:raw[i]=point
        available=np.isfinite(raw).all(axis=1)
        lo,hi=np.percentile(raw[available],[1,99],axis=0)
        xyz=np.nan_to_num((raw-(lo+hi)/2)/(max(hi-lo)/2)).clip(-1.5,1.5)
        # Place neuraxis along the vertical screen dimension (dataset z).
        self.positions=xyz[:,[0,2,1]];self.positions[:,1]*=-1
        self.schematic=~available
        rng=np.random.default_rng(3701)
        missing=np.flatnonzero(~available)
        # Avoid constructing a whole heterogeneous pandas Series for each
        # missing coordinate. This path can involve >100,000 peripheral cells.
        root=rows.rootSide.to_numpy();soma=rows.somaSide.to_numpy()
        sides=np.where(root!='',root,soma)[missing]
        vnc=rows.superclass.str.contains('vnc').to_numpy()[missing]
        noise=rng.normal(size=(len(missing),3))*np.array([.04,.12,.08])
        self.positions[missing,0]=np.where(sides=='R',.8,np.where(sides=='L',-.8,0))+noise[:,0]
        self.positions[missing,1]=np.where(vnc,-.55,.45)+noise[:,1]
        self.positions[missing,2]=noise[:,2]
        self.valid=np.ones(len(rows),bool)
        # Include both brain and VNC in the resting sample.
        self.base=np.sort(rng.choice(len(rows),min(2400,len(rows)),replace=False))
        self.classes=np.zeros(len(rows),np.int8)
        self.classes[~brain.nonsensory]=1
        self.classes[brain.motor]=2
        self.turn_cells=np.flatnonzero(rows['type'].isin(['DNa01','DNa02','MDN']).to_numpy())
        self.position_source='MaleCNS annotated soma/tosoma locations; missing locations use labeled schematic peripheral positions'
        self.geometry_note='脑与腹神经索的位置采样；缺少坐标的细胞使用标明的示意位置，非完整神经突形态。'
