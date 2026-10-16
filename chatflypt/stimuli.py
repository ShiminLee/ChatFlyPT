"""Explicit engineering stimulus dictionary; not the author's missing 16 words."""
import ast
import json
import numpy as np
import pandas as pd
from .paths import ROOT, ARTIFACTS, REPORTS, dump, sha256

WORDS = ['sugar','bitter','water','salt','vinegar','yeast','geosmin','pheromone',
         'warm','cold','humid','dry','bright','touch','sound','wind']
FIRST_SENTENCE = ['vinegar','yeast','sugar','warm','bright']
ANNOTATIONS = ROOT/'vendor/flywire-annotations/supplemental_files/Supplemental_file1_neuron_annotations.tsv'


def notebook_literal(name):
    notebook=json.loads((ROOT/'vendor/shiu-brain/figures.ipynb').read_text(encoding='utf-8'))
    for cell in notebook['cells']:
        if cell['cell_type']!='code':
            continue
        try:
            tree=ast.parse(''.join(cell.get('source',[])))
        except SyntaxError:
            continue
        for node in tree.body:
            if isinstance(node,ast.Assign) and any(isinstance(t,ast.Name) and t.id==name for t in node.targets):
                try:
                    return ast.literal_eval(node.value)
                except (TypeError, ValueError):
                    pass
    raise KeyError(name)


def build_manifest(brain):
    annotation=pd.read_csv(ANNOTATIONS,sep='\t',low_memory=False).fillna('')
    rows=annotation.set_index(annotation.root_id.astype(str)).reindex([str(x) for x in brain.ids]).fillna('')
    def select(**filters):
        mask=np.ones(len(rows),bool)
        for key,value in filters.items():
            mask &= rows[key].eq(value).to_numpy()
        return [str(x) for x in brain.ids[mask]]
    tree=ast.parse((ROOT/'vendor/eon-fly-brain/code/benchmark.py').read_text(encoding='utf-8'))
    experiments=next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='EXPERIMENTS' for t in n.targets))
    specs={
        'sugar':([str(x) for x in experiments['sugar']['neu_exc']],150,'Eon v783 sugar GRNs, published IDs'),
        'bitter':(select(super_class='sensory',cell_class='gustatory',cell_sub_class='bitter'),150,'FlyWire v783 bitter annotation; differs from paper v630 subset'),
        'water':([str(x) for x in notebook_literal('neu_water')],150,'Shiu paper water GRNs; every ID verified present in v783'),
        'salt':([str(x) for x in notebook_literal('neu_ir94e')],150,'Shiu Ir94e GRNs; low-salt proxy'),
        'vinegar':(select(cell_type='ORN_DM1')+select(cell_type='ORN_VA2'),100,'Partial vinegar proxy DM1+VA2; not a complete measured odor response'),
        'yeast':(select(cell_type='ORN_DM2'),100,'Synthetic yeast-associated odor channel DM2; not specific to yeast'),
        'geosmin':(select(cell_type='ORN_DA2'),100,'Synthetic geosmin-associated ORN_DA2 channel'),
        'pheromone':(select(cell_type='ORN_DA1'),100,'Synthetic cVA-associated ORN_DA1 channel'),
        'warm':(select(cell_type='TRN_VP2'),150,'Annotated heating cells; tonic activation proxy, no calibrated Celsius'),
        'cold':(select(cell_type='TRN_VP3a')+select(cell_type='TRN_VP3b'),150,'Annotated cold cells; tonic activation proxy'),
        'humid':(select(cell_type='HRN_VP5',cell_sub_class='moist'),150,'Ir68a-associated VP5 moist cells; tonic proxy, no calibrated relative humidity; corrected from cooling HRN_VP1l'),
        'dry':(select(cell_type='HRN_VP4'),150,'Annotated dry sensory cells'),
        'bright':(select(cell_type='R1-6'),80,'128 evenly sampled R1-6 cells; engineering light proxy, LIF approximation'),
        'touch':(select(cell_type='BM_Taste'),150,'Taste-bristle mechanosensory stimulation proxy'),
        'sound':(select(super_class='sensory',cell_sub_class='auditory'),100,'Annotated auditory cells; Poisson activation, no acoustic waveform'),
        'wind':(select(super_class='sensory',cell_sub_class='wind_gravity'),100,'Annotated wind/gravity cells; simplified engineering proxy'),
    }
    spec=specs['bright']
    ordered=sorted(spec[0])
    specs['bright']=([ordered[i] for i in np.linspace(0,len(ordered)-1,128,dtype=int)],spec[1],spec[2])
    manifest={'dataset':'FlyWire v783','mapping_revision':'2026-09-19-v2-moist-VP5',
              'original_author_vocabulary_available':False,
              'scope':'Independent 16-channel proxy dictionary. Labels describe modeled interventions, not validated subjective states.',
              'annotation_sha256':sha256(ANNOTATIONS),'words':{},
              'sources':[
                  'https://github.com/flyconnectome/flywire_annotations',
                  'https://github.com/philshiu/Drosophila_brain_model',
                  'https://github.com/eonsystemspbc/fly-brain',
                  'https://www.nature.com/articles/nature07983',
                  'https://doi.org/10.1016/j.cub.2020.06.028',
                  'https://elifesciences.org/articles/26654',
                  'https://pmc.ncbi.nlm.nih.gov/articles/PMC11766258/']}
    seen=set()
    for word in WORDS:
        ids,hz,note=specs[word]
        if not ids or any(x not in brain.lookup for x in ids):
            raise ValueError(f'Empty or incompatible stimulus: {word}')
        overlap=seen.intersection(ids)
        seen.update(ids)
        manifest['words'][word]={'neuron_ids':ids,'rate_hz':hz,'interpretation':note,
                                 'overlap_with_prior_channels':sorted(overlap)}
    manifest['overlap_policy']='Sum requested Poisson rates on shared cells. This is an explicit engineering mixture rule.'
    sensory=rows.super_class.isin(['sensory','sensory_ascending']).to_numpy(copy=True)
    explicitly_driven=np.array([brain.lookup[x] for x in seen])
    sensory[explicitly_driven]=True
    manifest['readout_indices']=np.flatnonzero(~sensory).tolist()
    manifest['motor_indices']=np.flatnonzero(rows.super_class.isin(['descending','motor']).to_numpy() & ~sensory).tolist()
    manifest['auditory_commands']={
        direction:select(super_class='sensory',cell_sub_class='auditory',side=direction)
        for direction in ['left','right']}
    manifest['auditory_command_note']='Left/right sound channels are an engineered code, not English comprehension. Stop is silence.'
    dump(ARTIFACTS/'stimuli.json',manifest)
    dump(REPORTS/'stimuli-summary.json',{k:v for k,v in manifest.items() if k not in ['readout_indices','motor_indices']})
    return manifest


def groups_for(manifest, words, scale=1.0):
    if len(set(words))!=len(words):
        raise ValueError('Duplicate stimulus')
    rates={}
    for w in words:
        spec=manifest['words'][w]
        for root_id in spec['neuron_ids']:
            rates[root_id]=rates.get(root_id,0.0)+spec['rate_hz']*scale
    by_rate={}
    for root_id,hz in sorted(rates.items()):
        by_rate.setdefault(hz,[]).append(root_id)
    return [(ids,hz) for hz,ids in by_rate.items()]
