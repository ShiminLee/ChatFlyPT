"""Short full-graph run of the unmodified Shiu model; NOT a paper replication."""
from __future__ import annotations
import argparse
import ast
import hashlib
import importlib.util
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import brian2 as b2

ROOT = Path(__file__).resolve().parents[1]
VENDOR = ROOT / 'vendor/shiu-brain'


def notebook_literal(name):
    notebook = json.loads((VENDOR / 'figures.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        try:
            tree = ast.parse(''.join(cell.get('source', [])))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == name for t in node.targets):
                try:
                    return ast.literal_eval(node.value)
                except (ValueError, TypeError):
                    continue
    raise ValueError(f'No literal for {name}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--duration-ms', type=float, default=100)
    parser.add_argument('--version', choices=['630', '783'], default='630')
    parser.add_argument('--stimulus-set', choices=['shiu-paper', 'eon-783'], default='shiu-paper')
    args = parser.parse_args()
    if not 0 < args.duration_ms <= 10000:
        parser.error('duration must be in (0, 10000] ms')
    comp, con = (('2023_03_23_completeness_630_final.csv','2023_03_23_connectivity_630_final.parquet')
                 if args.version == '630' else ('Completeness_783.csv','Connectivity_783.parquet'))
    spec = importlib.util.spec_from_file_location('shiu_original', VENDOR / 'model.py')
    model = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(model)
    b2.prefs.codegen.target = 'numpy'  # avoids requiring a system C++ compiler on Windows
    b2.defaultclock.dt = 0.1 * b2.ms
    neurons = pd.read_csv(VENDOR / comp, index_col=0)
    mapping = {int(fid): i for i, fid in enumerate(neurons.index)}
    sugar = notebook_literal('neu_sugar')
    bitter = notebook_literal('neu_bitter')
    mn9 = notebook_literal('id_mn9')
    conditions = [('no_input', 0, 0), ('sugar', 150, 0), ('sugar_bitter', 150, 150)]
    stimulus_source = 'upstream Shiu figures.ipynb literal assignments (v630)'
    if args.stimulus_set == 'eon-783':
        if args.version != '783':
            parser.error('eon-783 stimuli require --version 783')
        tree = ast.parse((ROOT / 'vendor/eon-fly-brain/code/benchmark.py').read_text(encoding='utf-8'))
        definitions = next(ast.literal_eval(n.value) for n in tree.body
                           if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == 'EXPERIMENTS' for t in n.targets))
        sugar = definitions['sugar']['neu_exc']
        bitter = []  # no validated bitter list in this experiment set
        conditions = [('no_input', 0, 0), ('sugar', definitions['sugar']['stim_rate'], 0)]
        stimulus_source = 'pinned Eon code/benchmark.py EXPERIMENTS sugar (v783); no inferred ID mapping'
    missing = [str(x) for x in sugar + bitter + [mn9] if x not in mapping]
    if missing:
        failure = {'dataset_version': args.version, 'stimulus_set': args.stimulus_set,
                   'missing_ids': missing, 'status': 'rejected_before_simulation',
                   'reason': 'Version-specific neuron IDs must not be silently replaced or dropped'}
        (ROOT / f'reports/stimulus-check-{args.version}-{args.stimulus_set}.json').write_text(json.dumps(failure,indent=2)+'\n')
        raise ValueError(f'Notebook IDs missing from v{args.version}: {missing}; do not silently remap')
    trials = []
    output = ROOT / 'reports/simulation-output'
    output.mkdir(parents=True, exist_ok=True)
    for label, sweet_hz, bitter_hz in conditions:
        b2.start_scope()
        b2.seed(42)
        params = dict(model.default_params)
        params.update(t_run=args.duration_ms*b2.ms, n_run=1, r_poi=sweet_hz*b2.Hz, r_poi2=bitter_hz*b2.Hz)
        start = time.perf_counter()
        print(f'Running {label}, FlyWire v{args.version}, {len(neurons):,} neurons', flush=True)
        spikes = model.run_trial([mapping[x] for x in sugar], [mapping[x] for x in bitter], [], VENDOR/comp, VENDOR/con, params)
        mn9_count = len(spikes.get(mapping[mn9], []))
        total = sum(len(s) for s in spikes.values())
        downstream = sum(len(s) for n,s in spikes.items() if n not in {mapping[x] for x in sugar+bitter})
        frame = model.construct_dataframe([spikes], label, {i:int(fid) for i,fid in enumerate(neurons.index)})
        frame.to_parquet(output / f'{args.version}-{label}.parquet', index=False)
        trials.append({'condition': label, 'seed': 42, 'duration_ms': args.duration_ms,
                       'sugar_hz': sweet_hz, 'bitter_hz': bitter_hz, 'total_spikes': total,
                       'active_neurons': len(spikes), 'non_input_spikes': downstream,
                       'mn9_spikes': mn9_count, 'mn9_rate_hz': mn9_count/(args.duration_ms/1000),
                       'wall_seconds': time.perf_counter()-start})
        print(json.dumps(trials[-1]), flush=True)
    assert trials[0]['total_spikes'] == 0, 'Unexpected spontaneous firing in this model'
    assert trials[1]['non_input_spikes'] > 0, 'Stimulus did not propagate beyond input neurons'
    with (VENDOR/'model.py').open('rb') as stream:
        model_sha = hashlib.file_digest(stream,'sha256').hexdigest()
    result = {'purpose': 'installation/data/execution smoke test; not statistical or biological validation',
              'dataset': f'FlyWire v{args.version}', 'neurons': len(neurons),
              'backend': 'Brian2 numpy', 'brian2_version': b2.__version__, 'dt_ms': 0.1,
              'upstream_model_sha256': model_sha, 'upstream_code_modified': False,
              'stimulus_ids_source': stimulus_source, 'stimulus_set': args.stimulus_set,
              'mn9_id': str(mn9), 'trials': trials, 'checks_passed': True}
    (ROOT/f'reports/shiu-smoke-{args.version}.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf-8')


if __name__ == '__main__':
    main()
