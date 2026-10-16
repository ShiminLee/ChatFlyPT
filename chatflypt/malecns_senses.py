"""Same-connectome sensory counterparts with explicit anatomical provenance.

Only population identity is transferred from literature/FlyWire channel names.
No FlyWire ID, decoder weight, behavior or calibrated receptor curve is copied.
"""
from functools import lru_cache
from copy import deepcopy
import json

import numpy as np

from .paths import ROOT

DATASET = 'MaleCNS v1.0'
GRAPH = ROOT / 'data/derived/malecns/graph.npz'
CHANNEL_MAP = {**{name: f'mc_{name}' for name in (
    'bright', 'vinegar', 'yeast', 'geosmin', 'cva', 'co2',
    'warm', 'cold', 'moist', 'dry', 'evaporative_cooling')},
    # Conspecific song: the FlyWire registry exposes it under both a modality
    # name and a stimulus name; both resolve to the one MaleCNS auditory entry.
    'auditory': 'mc_auditory', 'courtship_pulse': 'mc_auditory'}

# Exact official annotation selections. Missing or unknown identities do not
# gain a function through fuzzy matches or proximity to a known population.
# A selector may be a class name plus exact types, or a class/subclass mapping.
POPULATIONS = (
    ('bright', 'visual', ('R1-R6',), 'vision', '外层视感受器 R1–R6', 'vision', 10., 'eye'),
    ('vinegar', 'olfactory', ('ORN_DM1', 'ORN_VA2'), 'olfaction', '醋相关 DM1/VA2 候选', 'vinegar', 100., 'antenna'),
    ('yeast', 'olfactory', ('ORN_DM2',), 'olfaction', '酵母相关 DM2 代理', 'malecns', 100., 'antenna'),
    ('geosmin', 'olfactory', ('ORN_DA2',), 'olfaction', '土臭素相关 DA2', 'geosmin', 100., 'antenna'),
    ('cva', 'olfactory', ('ORN_DA1',), 'olfaction', 'cVA 相关 DA1', 'cva', 100., 'antenna'),
    ('co2', 'olfactory', ('ORN_V',), 'olfaction', '外部二氧化碳 V 通道', 'co2', 100., 'antenna'),
    ('warm', 'thermosensory', ('TRN_VP2',), 'temperature', '升温相关 VP2', 'temperature', 100., 'antenna'),
    ('cold', 'thermosensory', ('TRN_VP3a', 'TRN_VP3b'), 'temperature', '冷却相关 VP3a/b', 'temperature', 100., 'antenna'),
    ('moist', 'hygrosensory', ('HRN_VP5',), 'humidity', '潮湿相关 VP5', 'humidity', 100., 'antenna'),
    ('dry', 'hygrosensory', ('HRN_VP4',), 'humidity', '干燥相关 VP4', 'humidity', 100., 'antenna'),
    ('evaporative_cooling', 'hygrosensory', ('HRN_VP1d',), 'temperature', '蒸发冷却候选 VP1d', 'temperature', 100., 'antenna'),
    ('auditory', {'class': 'mechanosensory', 'subclass': 'auditory'}, None, 'audition',
     '约翰斯顿氏器听觉候选', 'song', 150., 'antenna', 'pulse_envelope'),
)


@lru_cache(maxsize=2)
def _graph_membership(path, modified_ns, size):
    # Reading only IDs and row offsets avoids a second 25-million-edge graph.
    with np.load(path, allow_pickle=False) as graph:
        ids, indptr = graph['ids'], graph['indptr']
    return {str(i): int(n) for i, n in zip(ids, np.diff(indptr))}


def graph_membership():
    if not GRAPH.exists():
        return {}, {'status': 'unavailable', 'path': str(GRAPH.relative_to(ROOT))}
    stat = GRAPH.stat()
    counts = _graph_membership(str(GRAPH), stat.st_mtime_ns, stat.st_size)
    provenance = GRAPH.with_suffix('.json')
    metadata = json.loads(provenance.read_text(encoding='utf-8')) if provenance.exists() else {}
    return counts, {'status': 'checked', 'path': str(GRAPH.relative_to(ROOT)),
                    'graph_sha256': metadata.get('graph_sha256'),
                    'criterion': 'Present body ID and at least one outgoing edge in the current signed model.'}


def add_malecns_senses(rows, add, channels):
    """Called by the catalogue builder before the residual anatomy inventory."""
    counts, graph = graph_membership()
    for source, cls, types, modality, label, evidence, rate, part, *rest in POPULATIONS:
        profile = rest[0] if rest else 'tonic'
        cid = CHANNEL_MAP[source]
        if isinstance(cls, dict):
            selected = rows[rows['class'].eq(cls['class']) & rows['subclass'].eq(cls['subclass'])]
            selection = {'class': cls['class'], 'subclass': cls['subclass'], 'side': 'rootSide',
                         'outgoing': 'current signed graph, positive edge count'}
        else:
            selected = rows[rows['class'].eq(cls) & rows.type.isin(types)]
            selection = {'class': cls, 'type': list(types), 'side': 'rootSide',
                         'outgoing': 'current signed graph, positive edge count'}
        notes = [
            'New MaleCNS population: no inherited FlyWire or pre-existing sensory decoder label/weights.',
            'Population identity is supported by type/glomerulus correspondence; natural stimulus tuning in these individual EM cells is not measured.',
            'Unknown peripheral sides and cells without modeled outgoing edges are excluded from executable side selections and retained in the audit.',
        ]
        if source == 'bright':
            notes += ['Normal embodied input uses a local ommatidium-to-column candidate bridge inferred from actual R1-6 to lamina connections; cross-specimen retinal geometry is not calibrated and coverage is incomplete. Independent population interventions do not use that spatial bridge. No spectral tuning or object recognition is validated.',
                      'Photoreceptors are graded/histaminergic; event approximation and absence of a calibrated tonic lamina baseline can yield only inhibition. No artificial excitatory sign or baseline is added.',
                      'The reconstructed R1–R6 population is asymmetric and incomplete; missing cells are not fabricated.']
        elif source == 'yeast':
            notes += ['DM2 is a retained engineering odor proxy, not a unique or complete biological yeast code.']
        elif source == 'vinegar':
            notes += ['DM1/VA2 are only part of the concentration-dependent apple-cider-vinegar response; no complete odor mixture or behavioral valence is assigned.']
        elif source in ('warm', 'cold', 'moist', 'dry', 'evaporative_cooling'):
            notes += ['Glomerular modality correspondence does not calibrate temperature, humidity, dynamic adaptation, or response polarity to a natural physical waveform.']
            if source == 'evaporative_cooling':
                notes += ['VP1d evaporative-cooling assignment remains putative; no wet-bulb model or default environment presentation.']
        elif source == 'auditory':
            notes += ['Selection is the official auditory subclass in the antennal nerve; JO-FV/EV/ED wind-and-gravity units and the grooming subclass are excluded and are never treated as auditory.',
                      'MaleCNS leg chordotonal organs are proprioceptive (MetaLN/MesoLN/ProLN) and are never used as a Johnston\'s-organ proxy.',
                      'The 36 ms interpulse interval and 5 ms gate follow the synthetic-song example of the cited study; sound pressure, carrier waveform, phase locking and adaptation are not modelled.']
        add(cid, 'MaleCNS · ' + label, modality, selected, DATASET,
            source=evidence, status='candidate', advanced=True, body_parts=[part],
            rate=rate, profile=profile, limitations=notes,
            selection=selection)
        spec = channels[cid]
        missing = [i for i in spec['neuron_ids'] if i not in counts]
        blocked = [i for i in spec['neuron_ids'] if i in counts and counts[i] == 0]
        usable = {i for i in spec['neuron_ids'] if counts.get(i, 0) > 0}
        anatomical_sides = {side: list(ids) for side, ids in spec['side_ids'].items()}
        spec['side_ids'] = {side: [i for i in ids if i in usable]
                            for side, ids in spec['side_ids'].items()}
        spec['body_part_ids'] = {part: {side: [i for i in ids if i in usable]
                                      for side, ids in mapping.items()}
                                 for part, mapping in spec['body_part_ids'].items()}
        if not spec['side_ids']['bilateral']:
            spec['status'] = 'unmapped'
        spec.update({
            'source_channel': source, 'readout_group': cid,
            'readout_status': 'untrained_new_population',
            'anatomical_side_ids': anatomical_sides,
            'graph_audit': {**graph, 'missing_from_graph_ids': missing,
                'without_modeled_outgoing_ids': blocked,
                'executable_cell_count': len(spec['side_ids']['bilateral']),
                'modeled_outgoing_edge_count': sum(counts.get(i, 0) for i in spec['side_ids']['bilateral'])},
            'functional_mapping': {'status': 'cross_dataset_anatomical_candidate',
                'annotation_types': list(types) if types else None,
                'annotation_selector': selection,
                'peripheral_side': 'rootSide',
                'individual_receptor_measurements': False,
                'physical_transduction_calibrated': False},
        })


def target_channel(catalogue, channel, dataset=DATASET):
    """Resolve a measured modality to one dataset; never fall back across IDs."""
    original = catalogue['channels'].get(channel)
    if original is None:
        raise ValueError(f'Unknown sensory channel: {channel}')
    target = channel if original['dataset'] == dataset else (
        CHANNEL_MAP.get(channel) if dataset == DATASET else None)
    spec = catalogue['channels'].get(target)
    if spec is None or spec['dataset'] != dataset or spec['status'] == 'unmapped':
        raise ValueError(f'No executable {dataset} counterpart for {channel}')
    return target


def remap_environment_inputs(catalogue, items, dataset=DATASET):
    """Return mapped measured inputs and explicit exclusions, without mutations.

    This function only resolves population identity. Call it after actual
    environment sampling; it does not present objects or generate neural events.
    """
    mapped, exclusions = [], []
    for item in items:
        try:
            cid = target_channel(catalogue, item['channel'], dataset)
            spec = catalogue['channels'][cid]
            side_map = spec['side_ids']
            if item.get('body_part') is not None:
                side_map = spec.get('body_part_ids', {}).get(item['body_part'], {})
            if not side_map.get(item.get('side', 'bilateral')):
                raise ValueError('Requested measured side/body part has no verified input cells')
            mapped.append({**item, 'channel': cid})
        except ValueError as exc:
            exclusions.append({'channel': item['channel'], 'dataset': dataset, 'reason': str(exc)})
    return mapped, exclusions


def remap_environment_sample(catalogue, sample, dataset=DATASET):
    """Copy a complete measured sample with consistent population evidence."""
    result = deepcopy(sample)
    result['items'], exclusions = remap_environment_inputs(catalogue, sample.get('items', []), dataset)
    result['unmapped'] = [*result.get('unmapped', []), *exclusions]
    projections = []
    for old in sample.get('items', []):
        # Separate resolution preserves source metadata even when a prior item
        # was excluded; zipping accepted items would shift channel attribution.
        try:
            target = target_channel(catalogue, old['channel'], dataset)
        except ValueError:
            continue
        spec = catalogue['channels'][target]
        projections.append({'source_channel': old['channel'], 'channel': target,
            'dataset': dataset, 'id_namespace': spec['id_namespace'],
            'side': old.get('side', 'bilateral'), 'body_part': old.get('body_part'),
            'readout_status': spec.get('readout_status'),
            'physical_transduction_calibrated': False})
    for evidence in result.get('evidence', []):
        if evidence.get('channel') is None:
            continue
        source = evidence['channel']
        try:
            target = target_channel(catalogue, source, dataset)
        except ValueError:
            evidence['neural_projection_status'] = 'unmapped_for_selected_dataset'
            continue
        evidence.update(source_channel=source, channel=target, dataset=dataset)
        if 'neuron_ids' in evidence:
            spec = catalogue['channels'][target]
            sides = spec.get('body_part_ids', {}).get(evidence.get('body_part'), spec['side_ids'])
            evidence['neuron_ids'] = list(sides.get(evidence.get('side', 'bilateral'), []))
    result['neural_dataset'] = dataset
    result['sensory_projections'] = projections
    return result
