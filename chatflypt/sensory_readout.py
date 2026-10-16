"""Count-only, lateralized probes for the general sensory event protocol.

These classify simulated interventions, not thoughts. Input text, requested
side, receptor IDs, and event metadata never enter the inference interface.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
import time

import numpy as np
import pandas as pd
from scipy import sparse
from threadpoolctl import threadpool_limits

from .brain import Brain, simulate
from .experiment import Decoder
from .paths import ROOT, dump, sha256
from .social import to_engine_tape
from .stimuli import ANNOTATIONS

OUT = ROOT / 'data/derived/general-sensory'
REPORT = ROOT / 'reports/general-sensory'
DATASETS = {'flywire': 'FlyWire v783', 'malecns': 'MaleCNS v1.0'}


def dataset_key(dataset):
    return next(k for k, v in DATASETS.items() if v == dataset)


class SensoryDecoder(Decoder):
    def __init__(self, dataset, path=None):
        path = path or OUT / f'{dataset_key(dataset)}-decoder.npz'
        super().__init__(path)
        with np.load(path, allow_pickle=False) as data:
            self.labels = data['labels'].tolist()
            self.names = dict(zip(self.labels, data['names'].tolist()))
            self.catalogue_fingerprint = str(data['catalogue_fingerprint'])
            self.qualified = data['qualified'] if 'qualified' in data else np.ones(len(self.labels),dtype=bool)

    def decode(self, counts):
        scores = self.scores(counts)
        if scores.ndim != 1:
            raise ValueError('One neural count vector required')
        words = [w for w, s, t, q in zip(self.labels, scores, self.thresholds,self.qualified) if s >= t and q]
        return {'words': words, 'scores': dict(zip(self.labels, map(float, scores))),
                'descriptions': {w: self.names[w] for w in words},
                'unqualified_words':[w for w,s,t,q in zip(self.labels,scores,self.thresholds,self.qualified) if s>=t and not q],
                'status': 'decoded', 'score_type': 'linear intervention scores, not probabilities',
                'scope': 'Aggregate internal spike counts; temporal order and natural perception are not decoded.'}


class UncalibratedDecoder:
    """Advanced annotation groups can run before any label readout is validated."""
    indices = np.empty(0, dtype=np.int32)

    def decode(self, counts):
        return {'words': [], 'scores': {}, 'descriptions': {}, 'status': 'uncalibrated',
                'scope': 'No validated general sensory probe for this input configuration.'}


def sensory_response_text(decoded, reaction):
    """Render only measured internal activity and the count-only probe output."""
    membrane=reaction.get('membrane_response',{}).get('internal_cells',0)
    if reaction.get('non_sensory_spikes', 0) == 0 and membrane == 0:
        return '本次没有记录到可读出的内部神经活动。'
    if decoded.get('status') == 'uncalibrated':
        return '记录到了内部神经活动；这组输入尚无校准过的感觉读出，暂不赋予感觉含义。'
    labels = [decoded.get('descriptions', {}).get(w, w) for w in decoded.get('words', [])]
    if not labels:
        return '记录到了内部神经活动，但当前读出器未识别出稳定的感觉通道。'
    text = '内部活动读出为：' + '、'.join(labels) + '。'
    if reaction.get('non_sensory_spikes',0)==0 and membrane:
        text+='依据为阈下膜电位变化；本次未记录到内部神经脉冲。'
    if decoded.get('unqualified_words'):
        text+='另有未通过单通道验证的候选标签，未作感觉判断。'
    if any(reaction.get(k, {}).get('spikes', 0) > 0 for k in ['DN', 'vnc_motor', 'cb_motor']):
        body=reaction.get('body_simulation',{})
        coverage=body.get('motor_coverage',{})
        if body.get('physics_run') and body.get('status')=='ok' and body.get('motor_activity_in_window',True):
            text+='运动神经活动已驱动身体物理仿真；轨迹来自工程控制器，尚未验证真实果蝇的动作规律。'
        elif body.get('status')=='no_motor_activity' or body.get('motor_activity_in_window') is False:
            text+='记录到运动相关神经活动，但未形成当前控制器可用的动作命令。'
            if body.get('physics_run'):
                text+='本轮身体记录仅含姿态维持和环境物理作用。'
        elif body.get('status') in ('error','timeout','unavailable'):
            text+='记录到运动相关神经活动；本次身体仿真未完成。'
        else:
            text += '同时记录到运动相关输出活动；动作结果尚未验证。'
        if coverage.get('unmapped_active_neurons'):
            text+=f"累计记录中有 {coverage['unmapped_active_neurons']} 个活跃运动相关细胞未接入身体执行器；详见运动连接覆盖。"
    return text


def canonical_specs(catalogue, dataset):
    return [s for s in catalogue['channels'].values()
            if s['dataset'] == dataset and s['status'] != 'unmapped'
            and not s.get('advanced', False) and s.get('neuron_ids')]


def fingerprint(catalogue):
    return hashlib.sha256(json.dumps(catalogue, sort_keys=True).encode()).hexdigest()


def protocol(catalogue, dataset):
    specs = canonical_specs(catalogue, dataset)
    labels, names = [], []
    configs = []
    for spec in specs:
        sides = [s for s in ['left', 'right'] if spec['side_ids'].get(s)]
        for side in sides:
            label=spec.get('readout_group',spec['id']) + ':' + side
            if label not in labels:
                labels.append(label)
                group=catalogue['channels'].get(spec.get('readout_group'),spec)
                names.append(('左侧' if side == 'left' else '右侧') + group['label'])
        for side in [*sides, *(['bilateral'] if len(sides) == 2 else [])]:
            configs.append({'channel': spec['id'], 'side': side})
    rng = np.random.default_rng(2026091917)
    cases = []

    def add(split, items, scale=1., duration=300, kind='single'):
        cases.append({'split': split, 'seed': 700000 + len(cases), 'kind': kind,
                      'items': [dict(item, intensity=float(scale), onset_ms=50,
                                     duration_ms=duration) for item in items]})

    for split, repeats in [('train', 3), ('validation', 1), ('test', 2), ('ood', 1)]:
        for config in configs:
            for j in range(repeats):
                scale = float(rng.uniform(.75, 1.25)) if split != 'ood' else .5
                add(split, [config], scale, 300 if split != 'ood' else 180)
        add(split, [], kind='blank')
    seen = set()
    for split, total in [('train', 64), ('validation', 24), ('test', 24)]:
        for i in range(total):
            while True:
                n = int(rng.integers(2, min(5, len(specs)) + 1))
                picked = rng.choice(len(specs), n, replace=False)
                items = [{'channel': specs[k]['id'],
                          'side': str(rng.choice([s for s in ['left', 'right', 'bilateral']
                                                if specs[k]['side_ids'].get(s)]))}
                         for k in picked]
                key = tuple(sorted((c['channel'], c['side']) for c in items))
                if key not in seen:
                    seen.add(key)
                    break
            add(split, items, float(rng.uniform(.8, 1.2)), kind='mixture')
    return {'dataset': dataset, 'labels': labels, 'names': names, 'cases': cases,
            'catalogue_fingerprint': fingerprint(catalogue), 'seed_overlap': False,
            'mixture_configurations_disjoint': True,
            'features': 'sqrt counts; exclude all sensory and catalogue input cells',
            'fit': 'Ridge alpha=10; per-label F1 thresholds on validation only',
            'interpretation': 'Artificial intervention identity and input side; not language comprehension.'}


def targets(case, catalogue, labels):
    active = set()
    for item in case['items']:
        sides = ['left', 'right'] if item['side'] == 'bilateral' else [item['side']]
        for side in sides:
            if catalogue['channels'][item['channel']]['side_ids'].get(side):
                spec=catalogue['channels'][item['channel']]
                active.add(spec.get('readout_group',item['channel']) + ':' + side)
    return [int(label in active) for label in labels]


def train(key, workers=3):
    from sklearn.linear_model import Ridge
    from sklearn.metrics import f1_score
    from .sensory_catalogue import build_catalogue
    from .sensory_events import encode_inputs
    catalogue = build_catalogue()
    dataset = DATASETS[key]
    plan = protocol(catalogue, dataset)
    OUT.mkdir(parents=True, exist_ok=True)
    dump(REPORT / f'{key}-protocol.json', plan)
    if key == 'flywire':
        brain = Brain()
        table = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False).fillna('')
        rows = table.set_index(table.root_id.astype(str)).reindex(brain.ids.astype(str)).fillna('')
        allowed = ~rows.super_class.str.contains('sensory').to_numpy(copy=True)
    else:
        from .malecns import MaleCNS
        brain = MaleCNS()
        allowed = brain.nonsensory.copy()
    direct = {brain.lookup[str(i)] for s in catalogue['channels'].values()
              if s['dataset'] == dataset for i in s['neuron_ids']}
    allowed[list(direct)] = False
    brain.run([], 1, 0)
    digest = hashlib.sha256((json.dumps(plan, sort_keys=True) +
                            sha256(ROOT/'chatflypt/sensory_events.py') +
                            sha256(ROOT/'chatflypt/brain.py')).encode()).hexdigest()[:16]
    cache = OUT / 'trials' / key / digest
    cache.mkdir(parents=True, exist_ok=True)
    cases = plan['cases']
    start = time.perf_counter()

    def run(case):
        path = cache / f"{case['seed']}.npz"
        if not path.exists():
            if case['items']:
                events = encode_inputs(catalogue, case['items'], case['seed'])[dataset]
                tape = to_engine_tape(brain, events)
                counts = simulate(brain.indptr, brain.indices, brain.weights, *tape, False)[0]
            else:
                counts = brain.run([], 500, case['seed']).counts
            pending=path.with_name(path.stem+'.pending.npz')
            np.savez_compressed(pending, counts=counts)
            pending.replace(path)
        return path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(run, case) for case in cases]
        for i, future in enumerate(as_completed(futures), 1):
            future.result()
            if i % 20 == 0 or i == len(cases):
                print(f'{key} sensory corpus {i}/{len(cases)} {time.perf_counter()-start:.1f}s', flush=True)
    vectors = []
    for case in cases:
        with np.load(cache / f"{case['seed']}.npz", allow_pickle=False) as data:
            vectors.append(sparse.csr_matrix(data['counts'][None, :]))
    x = sparse.vstack(vectors, format='csr')
    y = np.array([targets(c, catalogue, plan['labels']) for c in cases])
    split = np.array([c['split'] for c in cases])
    tr, va = split == 'train', split == 'validation'
    indices = np.flatnonzero(allowed & (np.asarray(x[tr].getnnz(axis=0)).ravel() > 0))
    features = x[:, indices].astype(float)
    features.data = np.sqrt(features.data)
    with threadpool_limits(limits=3):
        model = Ridge(alpha=10, solver='lsqr', tol=1e-6).fit(features[tr], y[tr])
        scores = model.predict(features)
    thresholds = []
    for k in range(len(plan['labels'])):
        best = max(np.linspace(.2, .8, 61),
                   key=lambda t: (f1_score(y[va, k], scores[va, k] >= t, zero_division=0), -abs(t-.5)))
        thresholds.append(float(best))
    # A response label must first recover its own isolated validation inputs.
    # This is a universal necessary check, not a channel-name exclusion or a
    # filter derived from test/acceptance labels. Failed scores stay auditable.
    singles=np.array([c['kind']=='single' for c in cases])
    internal=np.asarray(x[:,allowed].sum(axis=1)).ravel()
    qualified=[]
    for k in range(len(plan['labels'])):
        own=va & singles & y[:,k].astype(bool)
        qualified.append(bool(own.any() and np.all(internal[own]>0)
                              and np.all(scores[own,k]>=thresholds[k])))
    qualified=np.array(qualified,dtype=bool)
    path = OUT / f'{key}-decoder.npz'
    temporary = OUT / f'{key}-decoder.tmp.npz'
    np.savez_compressed(temporary, indices=indices, weights=model.coef_, intercept=model.intercept_,
                        thresholds=thresholds, labels=plan['labels'], names=plan['names'],
                        catalogue_fingerprint=plan['catalogue_fingerprint'],qualified=qualified)
    temporary.replace(path)
    sparse.save_npz(OUT / f'{key}-counts.npz', x)

    def metrics(mask, values=None):
        values = scores[mask] if values is None else values
        predicted = (values >= thresholds) & qualified
        return {'trials': int(mask.sum()),
                'exact_set_accuracy': float(np.mean(np.all(predicted == y[mask], axis=1))),
                'micro_f1': float(f1_score(y[mask], predicted, average='micro', zero_division=0)),
                'per_label_f1': dict(zip(plan['labels'], f1_score(y[mask], predicted, average=None, zero_division=0).tolist()))}
    test = split == 'test'
    report = {'dataset': dataset, 'trials': len(cases), 'features': len(indices),
              'model_sha256': sha256(path), 'catalogue_fingerprint': plan['catalogue_fingerprint'],
              'all_sensory_and_direct_cells_excluded': bool(np.all(allowed[indices])),
              'qualification_rule':'Every isolated validation trial targeting this label must have internal spikes and a positive readout. No test labels are used.',
              'qualified_labels':dict(zip(plan['labels'],qualified.tolist())),
              'test': metrics(test), 'ood': metrics(split == 'ood'),
              'single': metrics(test & (np.array([c['kind'] for c in cases]) == 'single')),
              'mixture': metrics(test & (np.array([c['kind'] for c in cases]) == 'mixture')),
              'controls': {'shuffled_observations': metrics(test, scores[test][np.random.default_rng(90).permutation(test.sum())]),
                           'zero_observations': metrics(test, np.tile(model.intercept_, (test.sum(), 1)))},
              'test_predictions': [dict(c, predicted=[w for w, s, t, q in zip(plan['labels'], scores[i], thresholds,qualified) if s >= t and q],
                                        internal_spikes=int(x[i, allowed].sum()))
                                   for i, c in enumerate(cases) if c['split'] in ['test', 'ood']],
              'limitations': ['Artificial envelopes, uncalibrated physiology.',
                              'OOD changes duration and amplitude; no natural sensory or movement validation.',
                              'Advanced annotation channels lack a fitted meaning readout.']}
    dump(REPORT / f'{key}-readout.json', report)
    print(json.dumps({k: v for k, v in report.items() if k in ['dataset', 'trials', 'test', 'ood']}, ensure_ascii=False), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('--workers', type=int, default=3)
    args = parser.parse_args()
    train(args.dataset, args.workers)
