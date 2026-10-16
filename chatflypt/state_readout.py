"""Blind sensory probe using spikes and observed subthreshold membrane voltage.

No text, stimulus IDs, sides, or directly driven/sensory cells are features.
This changes observation of the LIF model, not its dynamics or synaptic signs.
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
from .brain import Brain, recorded_trial
from .paths import ROOT, dump, sha256
from .sensory_readout import DATASETS, dataset_key, canonical_specs, protocol, targets
from .social import to_engine_tape
from .stimuli import ANNOTATIONS

OUT = ROOT / 'data/derived/state-sensory'
REPORT = ROOT / 'reports/gap-closure'
FEATURES = 'sqrt spike count; log1p positive and negative voltage area in mV ms'
MIN_VOLTAGE_MV = 1e-6


def signature(catalogue, dataset):
    keys = ['id', 'readout_group', 'neuron_ids', 'side_ids', 'rate_hz',
            'profile', 'profile_parameters']
    specs = [{k: s.get(k) for k in keys} for s in canonical_specs(catalogue, dataset)]
    return hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()


def features(counts, state):
    if state is None or state.shape != (4, len(counts)):
        raise ValueError('Recorded neural membrane state is required')
    return np.concatenate((np.sqrt(counts), np.log1p(state[0]), np.log1p(state[1])))


def fit_probe(x, y):
    """Same ridge objective, solving the small sample Gram matrix directly.

    There are hundreds of trials but tens of thousands of recorded features.
    Dense centered Cholesky avoids separate long LSQR iterations for each label.
    The alpha, intercept and feature transformations are unchanged.
    """
    from sklearn.linear_model import Ridge
    with threadpool_limits(limits=3):
        return Ridge(alpha=10, solver='cholesky').fit(x.toarray(), y)


def choose_thresholds(scores, targets, isolated, signals):
    """Use validation rows only; enforce the reporting criteria before F1.

    A higher-F1 cutoff must not displace another cutoff that actually satisfies
    the isolated-positive and precision requirements. Keep the original grid.
    """
    grid = np.linspace(.2, .8, 61)
    chosen = []
    for k in range(targets.shape[1]):
        truth = targets[:, k].astype(bool)
        positive = (scores[:, k, None] >= grid) & np.asarray(signals)[:, None]
        tp = np.sum(positive & truth[:, None], axis=0)
        fp = np.sum(positive & ~truth[:, None], axis=0)
        fn = np.sum(~positive & truth[:, None], axis=0)
        own = np.asarray(isolated) & truth
        qualified = (own.any() & np.all(positive[own], axis=0) &
                     (tp / np.maximum(1, tp + fp) >= .9))
        f1 = 2 * tp / np.maximum(1, 2 * tp + fp + fn)
        best = max(range(len(grid)), key=lambda i: (bool(qualified[i]), f1[i], -abs(grid[i] - .5)))
        chosen.append(float(grid[best]))
    return np.array(chosen)


class StateDecoder:
    def __init__(self, dataset, path=None):
        path = path or OUT / f'{dataset_key(dataset)}-decoder.npz'
        with np.load(path, allow_pickle=False) as f:
            for name in ['feature_indices', 'indices', 'weights', 'intercept', 'thresholds', 'qualified']:
                setattr(self, name, f[name])
            self.labels = f['labels'].tolist()
            self.names = dict(zip(self.labels, f['names'].tolist()))
            self.catalogue_signature = str(f['catalogue_signature'])

    def decode_trial(self, trial):
        return self.decode(trial.counts, trial.state_record)

    def decode(self, counts, state_record=None):
        values = features(counts, state_record)[self.feature_indices]
        scores = self.weights @ values + self.intercept
        positive = scores >= self.thresholds
        # An intercept alone must never become a sensory report.
        if not (np.any(counts[self.indices]) or
                np.any(state_record[2:,self.indices]>MIN_VOLTAGE_MV)):
            positive[:] = False
        words = [w for w, yes, q in zip(self.labels, positive, self.qualified) if yes and q]
        return {'words': words, 'scores': dict(zip(self.labels, map(float, scores))),
                'descriptions': {w: self.names[w] for w in words},
                'unqualified_words': [w for w, yes, q in zip(self.labels, positive, self.qualified) if yes and not q],
                'status': 'decoded', 'score_type': 'linear neural-state intervention scores, not probabilities',
                'scope': 'Spikes and subthreshold LIF voltage; artificial input identity, not natural perception.'}


def setup(dataset):
    if dataset == DATASETS['flywire']:
        brain = Brain()
        table = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False).fillna('')
        rows = table.set_index(table.root_id.astype(str)).reindex(brain.ids.astype(str)).fillna('')
        allowed = ~rows.super_class.str.contains('sensory').to_numpy(copy=True)
    else:
        from .malecns import MaleCNS
        brain = MaleCNS()
        allowed = brain.nonsensory.copy()
    return brain, allowed


def run_case(brain, catalogue, dataset, case):
    from .sensory_events import encode_inputs
    if case['items']:
        events = encode_inputs(catalogue, case['items'], case['seed'])[dataset]
        tape = to_engine_tape(brain, events)
        duration = events['duration_ms']
    else:
        duration = 500.
        tape = brain.input_tape([], duration, case['seed'])
    return recorded_trial(brain, tape, case['seed'], duration, False)


def train(key, workers=4):
    from sklearn.metrics import f1_score
    catalogue = json.loads((ROOT/'data/derived/sensory/catalogue.json').read_text(encoding='utf8'))
    dataset = DATASETS[key]
    plan = protocol(catalogue, dataset)
    plan.update(features=FEATURES, model='passive voltage observation; unchanged LIF dynamics',
                catalogue_signature=signature(catalogue, dataset))
    # Alias/evidence-only catalogue edits do not change any input tape. Reuse
    # an existing identical intervention protocol only when its functional
    # mapping and current engine/encoder digest still locate the recorded cache.
    previous=REPORT/f'{key}-state-protocol.json'
    if previous.exists():
        old=json.loads(previous.read_text(encoding='utf8'))
        if all(old.get(k)==plan.get(k) for k in ['cases','labels','names','catalogue_signature','features','model']):
            old_digest=hashlib.sha256((json.dumps(old,sort_keys=True)+sha256(ROOT/'chatflypt/brain.py')+
                                      sha256(ROOT/'chatflypt/sensory_events.py')).encode()).hexdigest()[:16]
            if (OUT/'trials'/key/old_digest).is_dir():plan=old
    dump(REPORT / f'{key}-state-protocol.json', plan)
    brain, allowed = setup(dataset)
    direct = {brain.lookup[str(i)] for s in catalogue['channels'].values()
              if s['dataset'] == dataset for i in s['neuron_ids']}
    allowed[list(direct)] = False
    digest = hashlib.sha256((json.dumps(plan, sort_keys=True) + sha256(ROOT/'chatflypt/brain.py') +
                             sha256(ROOT/'chatflypt/sensory_events.py')).encode()).hexdigest()[:16]
    cache = OUT / 'trials' / key / digest
    cache.mkdir(parents=True, exist_ok=True)
    cases = plan['cases']; start = time.perf_counter()
    # Compile once before worker threads start.
    run_case(brain, catalogue, dataset, {'items': [], 'seed': 0})

    def run(case):
        path = cache / f"{case['seed']}.npz"
        if not path.exists():
            trial = run_case(brain, catalogue, dataset, case)
            temp = path.with_suffix('.tmp.npz')
            np.savez_compressed(temp, counts=trial.counts, state=trial.state_record,
                                seed=case['seed'], duration_ms=trial.duration_ms)
            temp.replace(path)
        return path

    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = [pool.submit(run, c) for c in cases]
        for i, job in enumerate(as_completed(jobs), 1):
            job.result()
            if i % 25 == 0 or i == len(cases):
                print(f'{key} state trials {i}/{len(cases)} {time.perf_counter()-start:.1f}s', flush=True)
    print(f'{key} assembling recorded features', flush=True)
    vectors = []; internal = []
    for case in cases:
        with np.load(cache/f"{case['seed']}.npz", allow_pickle=False) as f:
            vectors.append(sparse.csr_matrix(features(f['counts'], f['state'])[None, :]))
            internal.append(bool(np.any(f['counts'][allowed]) or np.any(f['state'][2:, allowed] > MIN_VOLTAGE_MV)))
    x = sparse.vstack(vectors, format='csr')
    y = np.array([targets(c, catalogue, plan['labels']) for c in cases])
    split = np.array([c['split'] for c in cases]); tr = split == 'train'; va = split == 'validation'
    permitted = np.tile(allowed, 3)
    selected = np.flatnonzero(permitted & (np.asarray(x[tr].getnnz(axis=0)).ravel() > 0))
    x = x[:, selected]
    print(f'{key} fitting ridge: {tr.sum()} trials, {len(selected)} features', flush=True)
    with threadpool_limits(limits=3):
        model = fit_probe(x[tr], y[tr])
        scores = model.predict(x)
    single = np.array([c['kind']=='single' for c in cases])
    thresholds = choose_thresholds(scores[va], y[va], single[va], np.array(internal)[va])
    positive = (scores >= thresholds) & np.array(internal)[:, None]
    qualified = []; validation = {}
    for k, label in enumerate(plan['labels']):
        own = va & single & y[:, k].astype(bool)
        tp = int(np.sum(positive[va, k] & y[va, k].astype(bool)))
        fp = int(np.sum(positive[va, k] & ~y[va, k].astype(bool)))
        precision = tp / max(1, tp + fp)
        q = bool(own.any() and np.all(positive[own, k]) and precision >= .9)
        qualified.append(q)
        validation[label] = {'true_positives': tp, 'false_positives': fp,
                             'precision': precision, 'qualified': q}
    qualified = np.array(qualified, dtype=bool)
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT/f'{key}-decoder.npz'; temp = path.with_suffix('.tmp.npz')
    np.savez_compressed(temp, feature_indices=selected, indices=np.unique(selected % len(brain.ids)),
        weights=model.coef_, intercept=model.intercept_, thresholds=thresholds, qualified=qualified,
        labels=plan['labels'], names=plan['names'], catalogue_signature=plan['catalogue_signature'])
    temp.replace(path)
    def metric(mask, values=None):
        pred = ((scores[mask] if values is None else values) >= thresholds) & qualified
        if values is None: pred &= np.array(internal)[mask, None]
        return {'trials': int(mask.sum()), 'exact_set_accuracy': float(np.mean(np.all(pred == y[mask], axis=1))),
                'micro_f1': float(f1_score(y[mask], pred, average='micro', zero_division=0)),
                'per_label_f1': dict(zip(plan['labels'], f1_score(y[mask], pred, average=None, zero_division=0).tolist()))}
    test = split == 'test'
    report = {'dataset': dataset, 'trials': len(cases), 'features': len(selected), 'feature_definition': FEATURES,
        'estimator': {'algorithm': 'ridge', 'alpha': 10, 'solver': 'cholesky', 'fit_intercept': True},
        'threshold_selection': 'Validation only; prioritize reporting qualification, then F1; original 0.20..0.80 grid in 0.01 increments.',
        'catalogue_signature': plan['catalogue_signature'], 'model_sha256': sha256(path), 'cache': str(cache.relative_to(ROOT)),
        'all_sensory_and_direct_cells_excluded': bool(np.all(permitted[selected])),
        'qualified_labels': dict(zip(plan['labels'], qualified.tolist())),
        'qualification_validation': validation,
        'qualification_rule': 'All isolated positive validation trials detected; precision >= 0.90 across ALL validation trials, including negative/wrong-side/mixed inputs; counts>0 or voltage peak>1e-6 mV.',
        'test': metric(test),
        'single': metric(test & single), 'mixture': metric(test & np.array([c['kind']=='mixture' for c in cases])), 'ood': metric(split=='ood'),
        'controls': {'shuffled': metric(test, scores[test][np.random.default_rng(98).permutation(test.sum())]),
                     'zero': metric(test, np.zeros_like(scores[test]))},
        'test_predictions': [dict(c, predicted=[w for w,p,q in zip(plan['labels'],positive[i],qualified) if p and q])
                             for i,c in enumerate(cases) if c['split'] in ['test','ood']],
        'limitations': ['Fixed synthetic intervention benchmark; not natural sensory accuracy.',
                       'Electrical response is under unchanged LIF equations, not calibrated graded visual physiology.']}
    dump(REPORT/f'{key}-state-readout.json', report)
    print(json.dumps({k:report[k] for k in ['dataset','trials','test','ood']}, ensure_ascii=False), flush=True)


def requalify(key, select_validation_thresholds=False):
    """Qualify completed probes using positives AND negatives from validation.

    Called before new independent neural tests. Weights stay fixed; an explicit
    selection pass can apply the current validation-only threshold rule.
    No test or language outcome chooses which labels can be verbalized.
    """
    from sklearn.metrics import f1_score
    path=OUT/f'{key}-decoder.npz'
    with np.load(path,allow_pickle=False) as f:artifact={k:f[k] for k in f.files}
    report=json.loads((REPORT/f'{key}-state-readout.json').read_text(encoding='utf8'))
    plan=json.loads((REPORT/f'{key}-state-protocol.json').read_text(encoding='utf8'))
    catalogue=json.loads((ROOT/'data/derived/sensory/catalogue.json').read_text(encoding='utf8'))
    if signature(catalogue,plan['dataset'])!=str(artifact['catalogue_signature']):raise ValueError('Stale sensory map')
    values=[];signals=[]
    for case in plan['cases']:
        with np.load(ROOT/report['cache']/f"{case['seed']}.npz",allow_pickle=False) as f:
            values.append(features(f['counts'],f['state'])[artifact['feature_indices']])
            indices=artifact['indices']
            signals.append(bool(np.any(f['counts'][indices]) or np.any(f['state'][2:,indices]>MIN_VOLTAGE_MV)))
    with threadpool_limits(limits=3):
        scores=np.array(values)@artifact['weights'].T+artifact['intercept']
    y=np.array([targets(c,catalogue,plan['labels']) for c in plan['cases']],dtype=bool)
    split=np.array([c['split'] for c in plan['cases']]);va=split=='validation'
    singles=np.array([c['kind']=='single' for c in plan['cases']]);signals=np.array(signals)
    if select_validation_thresholds:
        artifact['thresholds']=choose_thresholds(scores[va],y[va],singles[va],signals[va])
        report['threshold_selection']='Validation only; prioritize reporting qualification, then F1; original 0.20..0.80 grid in 0.01 increments.'
    positive=(scores>=artifact['thresholds'])&signals[:,None]
    qualified=[];validation={}
    for k,label in enumerate(plan['labels']):
        own=va&singles&y[:,k]
        tp=int(np.sum(positive[va,k]&y[va,k]));fp=int(np.sum(positive[va,k]&~y[va,k]))
        precision=tp/max(1,tp+fp)
        q=bool(own.any() and np.all(positive[own,k]) and precision>=.9)
        qualified.append(q)
        validation[label]={'true_positives':tp,'false_positives':fp,'precision':precision,'qualified':q}
    artifact['qualified']=np.array(qualified,dtype=bool)
    temp=path.with_suffix('.tmp.npz');np.savez_compressed(temp,**artifact);temp.replace(path)
    predicted=positive&artifact['qualified']
    def metric(mask):
        return {'trials':int(mask.sum()),'exact_set_accuracy':float(np.mean(np.all(predicted[mask]==y[mask],axis=1))),
                'micro_f1':float(f1_score(y[mask],predicted[mask],average='micro',zero_division=0)),
                'per_label_f1':dict(zip(plan['labels'],f1_score(y[mask],predicted[mask],average=None,zero_division=0).tolist()))}
    test=split=='test'
    report.update(qualified_labels=dict(zip(plan['labels'],qualified)),qualification_validation=validation,
        qualification_rule='All isolated positive validation trials detected; precision >= 0.90 across ALL validation trials, including negative/wrong-side/mixed inputs; counts>0 or voltage peak>1e-6 mV.',
        model_sha256=sha256(path),test=metric(test),single=metric(test&singles),
        mixture=metric(test&np.array([c['kind']=='mixture' for c in plan['cases']])),ood=metric(split=='ood'))
    shuffled=predicted[test][np.random.default_rng(98).permutation(test.sum())]
    report['controls']={'shuffled':{'micro_f1':float(f1_score(y[test],shuffled,average='micro',zero_division=0))},
                        'zero':{'micro_f1':0.,'predicted_labels':[]}}
    report['test_predictions']=[dict(c,predicted=[w for w,p in zip(plan['labels'],predicted[i]) if p])
        for i,c in enumerate(plan['cases']) if c['split'] in ['test','ood']]
    dump(REPORT/f'{key}-state-readout.json',report)
    print(key,'qualified',sum(qualified),'/',len(qualified),'test F1',report['test']['micro_f1'],flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', choices=DATASETS)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    train(args.dataset, args.workers)
