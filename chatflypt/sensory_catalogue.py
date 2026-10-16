"""An annotation-backed registry and coverage audit of modeled sensory inputs.

This inventories available neurons rather than claiming complete biological
transduction. Candidate channels expose that distinction, and unassigned entry
sides or functions are never invented. The two connectomes remain separate.
"""
import argparse
import hashlib
import json
import re
from collections import Counter

import pandas as pd

from .paths import ROOT, dump, sha256
from .social import MALE
from .stimuli import ANNOTATIONS

OUT = ROOT / 'data/derived/sensory'
REPORT = ROOT / 'reports/full-pathway'
FW = 'FlyWire v783'
MC = 'MaleCNS v1.0'
NAMESPACES = {FW: 'flywire_root_id', MC: 'malecns_body_id'}
SOURCES = {
    'flywire': 'https://www.nature.com/articles/s41586-024-07686-5',
    'malecns': 'https://male-cns.janelia.org/download/',
    'vision': 'https://www.nature.com/articles/s41598-020-74742-1',
    'polarization': 'https://pubmed.ncbi.nlm.nih.gov/27170135/',
    'vinegar': 'https://pubmed.ncbi.nlm.nih.gov/19396157/',
    'geosmin': 'https://pubmed.ncbi.nlm.nih.gov/23217715/',
    'cva': 'https://www.nature.com/articles/nature06808',
    'co2': 'https://www.nature.com/articles/nature05466',
    'temperature': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC7443704/',
    'humidity': 'https://elifesciences.org/articles/26654',
    'johnston': 'https://www.nature.com/articles/nature07810',
    'song': 'https://www.nature.com/articles/s41467-017-02453-9',
    'touch': 'https://www.nature.com/articles/nature14024',
    'proprioception': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC12154767/',
    'leg_proprioception': 'https://pubmed.ncbi.nlm.nih.gov/8788252/',
    'high_salt': 'https://elifesciences.org/articles/37167',
    'acid': 'https://www.nature.com/articles/s41467-021-23490-5',
    'fatty_acids': 'https://elifesciences.org/articles/30115',
    'amino_acids': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC5258133/',
    'gut_nutrients': 'https://pmc.ncbi.nlm.nih.gov/articles/PMC8611812/',
    'gustatory_connectome': 'https://doi.org/10.1016/j.cell.2026.08.016',
    'water_receptor': 'https://www.nature.com/articles/nature09011',
    'ir56d_taste': 'https://www.nature.com/articles/s41467-018-06453-1',
}
COMMON_LIMITATIONS = [
    'Rate and time profiles are engineering inputs, not calibrated physical stimulus-to-spike transformations.',
    'Side denotes annotated peripheral nerve entry, not requested movement direction or a fabricated split.',
]
NERVE_PARTS = {'ProLN': 'foreleg', 'MesoLN': 'midleg', 'MetaLN': 'hindleg',
               'ADMN': 'wing', 'DMetaN': 'haltere', 'PDMN': 'notum',
               'AN': 'antenna', 'MxLbN': 'labellum', 'PhN': 'pharynx', 'aPhN': 'pharynx'}


def _ids(rows):
    return sorted(rows.neuron_id.astype(str).unique().tolist(), key=int)


def _sides(rows):
    left = _ids(rows[rows.entry_side.eq('left')])
    right = _ids(rows[rows.entry_side.eq('right')])
    central = _ids(rows[rows.entry_side.eq('central')])
    # "bilateral" explicitly requests the available known peripheral field.
    # Centrally placed ocellar inputs are included only here, never in one side.
    return {'left': left, 'right': right,
            'bilateral': sorted(set(left + right + central), key=int)}


def _load_tables():
    fly = pd.read_csv(ANNOTATIONS, sep='\t', low_memory=False).fillna('')
    valid_fly = set(pd.read_csv(ROOT / 'vendor/shiu-brain/Completeness_783.csv', index_col=0).index.astype(str))
    raw_fly_sensory = fly[fly.super_class.str.contains('sensory')]
    absent = _ids(raw_fly_sensory.assign(neuron_id=raw_fly_sensory.root_id.astype(str))[
        ~raw_fly_sensory.root_id.astype(str).isin(valid_fly)])
    fly = raw_fly_sensory[raw_fly_sensory.root_id.astype(str).isin(valid_fly)].copy()
    fly['neuron_id'] = fly.root_id.astype(str)
    fly['entry_side'] = fly.side.map({'left': 'left', 'right': 'right', 'center': 'central'}).fillna('unknown')
    fly['body_part'] = fly.nerve.map(NERVE_PARTS).fillna('unknown')
    fly.loc[fly.cell_class.eq('visual'), 'body_part'] = 'eye'
    fly.loc[fly.cell_sub_class.eq('eye bristle'), 'body_part'] = 'eye'
    fly.loc[fly.cell_sub_class.eq('head bristle'), 'body_part'] = 'head'

    male = pd.read_feather(MALE).fillna('')
    male = male[male.superclass.str.contains('sensory')].copy()
    male['neuron_id'] = male.bodyId.astype(str)
    male['entry_side'] = male.rootSide.map({'L': 'left', 'R': 'right'}).fillna('unknown')
    male['body_part'] = male.entryNerve.map(NERVE_PARTS).fillna('unknown')
    male.loc[male['class'].eq('visual'), 'body_part'] = 'eye'
    male.loc[male.subclass.eq('abdomen'), 'body_part'] = 'abdomen'
    # Entry nerve is used where it identifies the peripheral origin. We do not
    # guess a leg identity for ProCN or other accessory nerves.
    male.loc[male.subclass.eq('neck'), 'body_part'] = 'neck'
    return {FW: fly, MC: male}, absent


def build_catalogue(save=True):
    tables, absent = _load_tables()
    fly, male = tables[FW], tables[MC]
    channels = {}

    def add(cid, label, modality, rows, dataset=FW, *, status='candidate', source='flywire',
            aliases=(), qualities=(), body_parts=(), rate=100., profile='tonic', limitations=(),
            advanced=False, selection=None):
        if cid in channels:
            raise ValueError(f'Duplicate sensory channel {cid}')
        side_ids = _sides(rows)
        known = set(side_ids['bilateral'])
        ids = _ids(rows)
        parts = {part: _sides(part_rows) for part, part_rows in rows.groupby('body_part', sort=True)
                 if part != 'unknown'}
        for part in body_parts:
            # Explicit aggregate names (e.g. head/body) refer to this complete
            # channel. Specific component maps retain their exact subsets.
            if part not in parts:
                parts[part] = side_ids
        if not ids or not known:
            status = 'unmapped'
        record = {'id': cid, 'label': label, 'modality': modality, 'dataset': dataset,
                  'id_namespace': NAMESPACES[dataset], 'neuron_ids': ids, 'side_ids': side_ids,
                  'unassigned_side_ids': sorted(set(ids) - known, key=int),
                  'status': status, 'source': SOURCES.get(source, source),
                  'sources': list(dict.fromkeys([SOURCES.get(source, source),
                                                SOURCES['flywire' if dataset == FW else 'malecns']])),
                  'limitations': [*COMMON_LIMITATIONS, *limitations],
                  'rate_hz': float(rate), 'profile': profile,
                  'profile_parameters': ({'decay_ms': 15.} if profile == 'touch_transient' else
                      {'interval_ms': 36., 'width_ms': 5.} if profile == 'pulse_envelope' else {}),
                  'aliases': list(dict.fromkeys([cid, *aliases])), 'quality_aliases': list(qualities),
                  'body_parts': list(body_parts), 'body_part_ids': parts,
                  'advanced': advanced,
                  'readout_group': 'auditory' if cid == 'courtship_pulse' else cid,
                  'selection': selection or 'Exact annotated class/type selection; see source code',
                  'annotated_types': sorted(set(rows.cell_type if dataset == FW else rows.type))}
        if record['unassigned_side_ids']:
            record['limitations'].append('Cells with unassigned entry side are listed but excluded from all executable side selections.')
        channels[cid] = record

    visual = fly.cell_class.eq('visual')
    for cid, typ, label, aliases in [
        ('bright', 'R1-6', '光照：外层视感受器 R1–R6', ['光照', '亮光', '明亮', '光', 'bright', 'light']),
        ('vision_r7', 'R7', 'R7 紫外敏感视感受器', ['R7', '紫外光', '紫外线', 'ultraviolet']),
        ('vision_r8', 'R8', 'R8 蓝／绿敏感视感受器', ['R8', '蓝绿光']),
    ]:
        add(cid, label, 'vision', fly[visual & fly.cell_type.eq(typ) & ~fly.cell_sub_class.eq('DRA')],
            source='vision', aliases=aliases, body_parts=['eye'], rate=10.,
            limitations=['Graded photoreceptor signals are approximated by LIF input events; no image, wavelength transfer curve, adaptation, or motion field is implemented.'])
    add('vision_dra', '背缘区偏振视觉候选', 'vision', fly[visual & fly.cell_sub_class.eq('DRA')],
        source='polarization', aliases=['偏振光', '偏振', 'polarized light'], body_parts=['eye'], rate=10.,
        limitations=['DRA population only; no e-vector angle or spatial polarization field encoder.'])
    add('vision_ocelli', '单眼视觉候选', 'vision', fly[visual & fly.cell_sub_class.eq('ocellar')],
        aliases=['单眼光照', 'ocelli'], body_parts=['eye'], rate=10.,
        limitations=['Central ocellar entries are selectable only through the bilateral/default field.'])

    olfactory = fly.cell_class.eq('olfactory')
    for cid, types, label, aliases, source, status, notes in [
        ('vinegar', ['ORN_DM1', 'ORN_VA2'], '醋相关气味候选', ['醋', '醋味', 'vinegar'], 'vinegar', 'candidate',
         ['DM1/VA2 are a partial vinegar proxy; natural vinegar activates a broader, concentration-dependent combination.']),
        ('yeast', ['ORN_DM2'], '酵母相关气味代理', ['酵母', '酵母味', 'yeast'], 'flywire', 'candidate',
         ['Legacy DM2 engineering proxy. No claim that DM2 uniquely or fully encodes yeast odor; the annotation establishes anatomy only.']),
        ('geosmin', ['ORN_DA2'], '土臭素通道', ['土臭素', '土腥味', 'geosmin'], 'geosmin', 'ready', []),
        ('cva', ['ORN_DA1'], 'cVA 信息素通道', ['cVA', 'cva', 'cVA信息素', '信息素'], 'cva', 'ready',
         ['The generic interface alias denotes cVA only; other pheromones are not equivalent.']),
        ('co2', ['ORN_V'], '二氧化碳通道', ['二氧化碳', 'CO2', 'co2', 'carbon dioxide'], 'co2', 'ready', []),
    ]:
        add(cid, label, 'olfaction', fly[olfactory & fly.cell_type.isin(types)], aliases=aliases,
            source=source, status=status, body_parts=['antenna'], limitations=notes)

    taste = fly.cell_class.eq('gustatory')
    for cid, sub, label, aliases, notes in [
        ('sweet_water', 'sugar/water', '糖／水感觉候选群', ['糖水', '糖水联合群', 'sugar-water population'],
         ['This historical FlyWire LB3/LB2d grouping is annotated sugar/water and cannot separate those functions. Separate MaleCNS subtype candidates are available as sugar_specific/water_specific. The older left-only IDs are never copied to construct a right-side population.']),
        ('bitter', 'bitter', '苦味感觉群', ['苦', '苦味', 'bitter'], []),
        ('low_salt', 'low-salt', '低盐感觉群', ['低盐', '盐味', '咸味', '盐', 'salt'],
         ['Low-salt annotation does not establish high-salt or arbitrary concentration coding.']),
    ]:
        add(cid, label, 'gustation', fly[taste & fly.cell_sub_class.eq(sub)],
            aliases=aliases, body_parts=['labellum'], limitations=notes)

    # The newer MaleCNS subtype-to-receptor assignments do not retroactively
    # relabel the coarser FlyWire LB3 population. These are distinct anatomical
    # candidates, inferred from population projection-field matching in Figure
    # 2D/E of Tastekin et al. (Cell 2026), not receptor measurements of each EM
    # cell. In particular, LB3b also participates in low-salt responses.
    for cid, types, label, aliases, receptor, figure, notes in [
        ('water_specific', ['LB3a'], '水感觉候选（LB3a）',
         ['水', 'water', '水特异', '水特异输入', 'water-specific'], 'ppk28', '2D/E',
         ['ppk28 is an external hypo-osmolarity/water sensor, not an internal thirst state.']),
        ('sugar_specific', ['LB3b', 'LB3c'], '糖感觉候选（LB3b/c）',
         ['糖', '甜味', 'sugar', 'sweet', '糖特异', '糖特异输入', 'sugar-specific'], 'Gr64f', '2D/E',
         ['LB3b also matches Ir56b low-salt neurons. The channel name identifies a sugar-related population distinct from LB3a, not exclusive molecular selectivity for sugar.']),
        ('high_salt', ['LB3d'], '高盐相关候选（LB3d）',
         ['高盐', '高浓度盐'], 'Ir7c/ppk23 (putative glutamatergic subset)', '2D/E',
         ['LB3d is one putative high-salt pathway, not the complete combinatorial high-salt code; bitter and other GRNs also contribute.',
          'Official MaleCNS consensus_nt assigns all 26 LB3d cells acetylcholine, while individual predicted_nt is 11 acetylcholine, 7 glutamate and 8 unclear, without experimental ground truth. This differs from the putative glutamatergic match in Cell 2026. The consensus-sign model can excite appetitive circuits; high-salt motor valence is unresolved, not validated avoidance.',
          'Its putative glutamatergic output is not relabeled as excitatory to force downstream spikes. Salt deprivation and concentration-dependent tuning are not modeled.']),
        ('fatty_acid_taste', ['claw_tpGRN'], '己酸相关味觉候选（IR56d）',
         ['脂肪酸', '脂肪味', '己酸', 'hexanoic acid'], 'Ir56d/Gr64e taste-peg population', '2I/J',
         ['The broad fatty-acid aliases select a hexanoic-acid-related IR56d taste-peg proxy, not every fatty acid or every IR56d bristle/leg neuron.',
          'This population also responds to carbonation. The same input pattern cannot distinguish carbonation from hexanoic acid; it is not the antennal CO2 ORN channel.',
          'IR56d-dependent hexanoic-acid responses are physiologically documented (Nature Communications 2018); the bridge to claw_tpGRN is the proposed Cell 2026 population match.']),
    ]:
        rows = male[male['class'].eq('gustatory') & male.type.isin(types)]
        add(cid, label, 'gustation', rows, MC, source='gustatory_connectome',
            aliases=aliases, body_parts=['labellum'], rate=100.,
            selection={'class': 'gustatory', 'type': types, 'side': 'rootSide'},
            limitations=[
                f'The receptor assignment is a proposed population morphology/connectivity match (Cell 2026 Figure {figure}), not single-cell receptor expression or physiological validation.',
                'Body IDs are selected from the local official MaleCNS v1.0 type annotations; the article Table S1 body-ID list has not been independently cross-checked.',
                *notes])
        spec = channels[cid]
        spec['evidence'] = {'receptor_match': receptor,
            'method': 'Population projection-field and connectivity matching; putative functional assignment.',
            'figure': figure, 'doi': '10.1016/j.cell.2026.08.016',
            'annotation_types': types, 'individual_receptor_measurements': False}
        missing_nerve = _ids(rows[rows.entryNerve.eq('')])
        spec['missing_entry_nerve_ids'] = missing_nerve
        if missing_nerve:
            spec['limitations'].append('Some typed labellar cells lack entryNerve annotation; their explicitly annotated rootSide is retained. These IDs are listed in missing_entry_nerve_ids.')
        if cid in ['water_specific', 'high_salt']:
            spec['sources'].append(SOURCES['water_receptor' if cid == 'water_specific' else 'high_salt'])
        if cid == 'fatty_acid_taste':
            spec['sources'].append(SOURCES['ir56d_taste'])
            spec['proxy_tastant'] = 'hexanoic acid'
            spec['not_distinguished_from'] = ['carbonation at the same IR56d taste-peg population']

    # Leg taste is not labellar taste. Figure 4J maps Gr64f/Ir56b projections
    # to LgLG4; Figure 3J maps Gr61a projections to LgAG2 (putative appetitive).
    # Retain independent leg/side IDs and keep these new, untrained candidates
    # outside the old decoder's canonical vocabulary/signature.
    for cid,typ,label,receptor,figure in [
        ('leg_sweet_local','LgLG4','腿部甜味相关局部候选群','Gr64f / Ir56b','4J'),
        ('leg_appetitive_ascending','LgAG2','腿部适口味觉上行候选群','Gr61a','3J'),
    ]:
        rows=male[male['class'].eq('gustatory') & male.type.eq(typ) & male.entryNerve.isin(['ProLN','MesoLN','MetaLN'])]
        add(cid,label,'gustation',rows,MC,source='gustatory_connectome',advanced=True,rate=50.,
            body_parts=sorted(set(rows.body_part)),selection={'class':'gustatory','type':[typ],'entryNerve':['ProLN','MesoLN','MetaLN'],'side':'rootSide'},
            limitations=['腿部接触味觉候选；受体身份来自群体投射匹配，不是每个 EM 细胞的直接受体测量。',
                '糖物品接触只作为候选工程刺激；该群不具有已经验证的糖专一性，也不表示完整盐味/苦味等腿部味觉。',
                '只刺激实际接触的腿及侧别；未定位单根感器，输入频率和接触力尚未经生理标定。',
                '该新增感觉群没有经过独立盲读出训练，不沿用唇部糖味标签。'])
        channels[cid]['functional_mapping']={'status':'projection_match_candidate','putative_receptor':receptor,'figure':figure}

    for cid, types, label, aliases, modality, status, notes in [
        ('warm', ['TRN_VP2'], '升温通道', ['温暖', '暖', '升温', '变热', 'warm'], 'temperature', 'ready', []),
        ('cold', ['TRN_VP3a', 'TRN_VP3b'], '低温通道', ['寒冷', '冷', '降温', 'cold'], 'temperature', 'ready', []),
        ('moist', ['HRN_VP5'], '潮湿通道', ['潮湿', '湿润', '变湿', 'humid', 'moist'], 'humidity', 'ready', []),
        ('dry', ['HRN_VP4'], '干燥通道', ['干燥', '变干', 'dry'], 'humidity', 'ready', []),
        ('evaporative_cooling', ['HRN_VP1d'], '蒸发冷却候选通道', ['蒸发冷却'], 'temperature', 'candidate',
         ['VP1d evaporative-cooling assignment is putative; no wet-bulb physics model.']),
    ]:
        add(cid, label, modality, fly[fly.cell_type.isin(types)], aliases=aliases,
            source='temperature', status=status, body_parts=['antenna'], limitations=notes)

    mechanical = fly.cell_class.eq('mechanosensory')
    audible = mechanical & fly.cell_sub_class.eq('auditory') & fly.cell_type.str.match(r'^JO-[AB]')
    add('auditory', 'JO-A/B 听觉输入', 'audition', fly[audible], source='johnston',
        aliases=['声音', '声响', '声', '听觉', 'sound'], body_parts=['antenna'], rate=100.,
        limitations=['Broad auditory population; no calibrated acoustic frequency, waveform, particle velocity, or phase locking.'])
    add('courtship_pulse', '求偶脉冲声包络', 'audition', fly[audible], source='song',
        aliases=['求偶鸣声', '求偶脉冲声', '求偶歌声'], body_parts=['antenna'], rate=150., profile='pulse_envelope',
        limitations=['36 ms inter-pulse interval is literature-inspired; 5 ms gate and 150 Hz neural input rate are assumptions, not acoustic carrier frequency.'])
    add('wind_gravity', 'JO 风／重力感觉群', 'mechanosensation',
        fly[mechanical & fly.cell_sub_class.eq('wind_gravity')], source='johnston',
        aliases=['风', '吹风', '气流', '重力', 'wind', 'gravity'], body_parts=['antenna'],
        limitations=['Wind and gravity are not independently identified by this broad population input; no force/direction transfer function.'])
    for cid, sub, label, part, aliases in [
        ('head_touch', 'head bristle', '头部触毛接触', 'head', ['头部触碰', '头部接触']),
        ('eye_touch', 'eye bristle', '眼间触毛接触', 'eye', ['眼部触碰', '眼部接触']),
    ]:
        add(cid, label, 'touch', fly[mechanical & fly.cell_sub_class.eq(sub)], aliases=aliases,
            qualities=['触碰', '接触', '碰', 'touch'], body_parts=[part], rate=150., profile='touch_transient',
            limitations=['Group stimulation does not select one individual bristle or reproduce measured contact force.'])

    tactile = male['class'].eq('mechanosensory_tactile')
    bristle = male.subclass.eq('mechanosensory bristle')
    for cid, nerve, part, label in [
        ('foreleg_touch', 'ProLN', 'foreleg', '前腿触毛接触'),
        ('midleg_touch', 'MesoLN', 'midleg', '中腿触毛接触'),
        ('hindleg_touch', 'MetaLN', 'hindleg', '后腿触毛接触'),
        ('wing_touch', 'ADMN', 'wing', '翅触毛接触'),
    ]:
        selected = male[tactile & bristle & male.entryNerve.eq(nerve)]
        add(cid, label, 'touch', selected, MC, source='touch' if part != 'wing' else 'proprioception',
            aliases=[label], qualities=['触碰', '接触', '碰', 'touch'], body_parts=[part],
            rate=150., profile='touch_transient',
            limitations=['Anatomical candidate population; neither exact distal sensillum identity nor an ensuing movement is assigned.'])
    add('body_touch', '背胸体表接触候选', 'touch', male[tactile & male.entryNerve.eq('PDMN')], MC,
        aliases=['身体触碰', '背部触碰', '体表触碰'], qualities=['触碰', '接触', '碰', 'touch'],
        body_parts=['body', 'notum'], rate=150., profile='touch_transient',
        limitations=['Mapped body surface is the posterior dorsal mesothoracic/notum field, not the entire abdomen or every body patch.'])

    proprio = male['class'].eq('mechanosensory_proprioceptive')
    for cid, sub, label, aliases in [
        ('chordotonal_proprioception', 'chordotonal organ', '弦音器本体输入', ['弦音器', 'chordotonal']),
        ('hair_plate_proprioception', 'hair plate', '毛板本体输入', ['毛板', 'hair plate']),
        ('campaniform_proprioception', 'campaniform sensilla', '钟状感器本体输入', ['钟状感器', 'campaniform']),
        ('haltere_proprioception', 'haltere', '平衡棒本体输入', ['平衡棒', 'haltere']),
    ]:
        selected = male[proprio & male.subclass.eq(sub)]
        parts = sorted(set(selected.body_part) - {'unknown'})
        add(cid, label, 'proprioception', selected, MC, source='proprioception', aliases=aliases,
            body_parts=parts, rate=50.,
            limitations=['No calibrated joint angle, load, rotation, or direction-specific receptor tuning; anatomical population activation only.'])
    leg_nerves = ['ProLN', 'MesoLN', 'MetaLN']
    add('leg_proprioception', '腿部本体感觉总群', 'proprioception', male[proprio & male.entryNerve.isin(leg_nerves)], MC,
        source='leg_proprioception', aliases=['腿部本体感觉', '关节位置', '关节运动', '肢体位置'],
        body_parts=['leg', 'foreleg', 'midleg', 'hindleg'], rate=50.,
        limitations=['Aggregates heterogeneous leg proprioceptors; flexion/extension and joint angle are not inferred.'])

    # Same MaleCNS individual can receive measured visual/odor/temperature input.
    # New populations are advanced candidates, isolated from old decoder labels.
    from .malecns_senses import add_malecns_senses
    add_malecns_senses(male, add, channels)

    # Preserve every remaining sensory annotation in the audit. An advanced
    # anatomical type is not assigned a fabricated odor, pain, taste or action.
    for dataset, rows in tables.items():
        used = {i for c in channels.values() if c['dataset'] == dataset for i in c['neuron_ids']}
        remaining = rows[~rows.neuron_id.isin(used)]
        cols = ['cell_class', 'cell_sub_class', 'cell_type'] if dataset == FW else ['class', 'subclass', 'type']
        for key, group in remaining.groupby(cols, sort=True, dropna=False):
            cell_class, subclass, typ = key
            slug = re.sub('[^a-z0-9]+', '_', (typ or subclass or cell_class or 'unclassified').lower()).strip('_')
            suffix = hashlib.sha256('\0'.join(key).encode()).hexdigest()[:8]
            cid = f"advanced_{'fw' if dataset == FW else 'mc'}_{slug}_{suffix}"
            function_unknown = cell_class in ['', 'unknown_sensory', 'mechanosensory_tbc'] or not typ
            modality = {'visual': 'vision', 'olfactory': 'olfaction', 'gustatory': 'gustation',
                        'thermosensory': 'temperature', 'hygrosensory': 'humidity',
                        'mechanosensory': 'mechanosensation', 'mechanosensory_tactile': 'touch',
                        'mechanosensory_proprioceptive': 'proprioception', 'chemosensory': 'chemosensation'}.get(cell_class, 'unknown')
            if modality == 'unknown':
                function_unknown = True
            add(cid, f'注释群 {typ or subclass or cell_class or "未命名"}', modality, group, dataset,
                source='flywire' if dataset == FW else 'malecns', advanced=True,
                status='unmapped' if function_unknown else 'candidate', rate=50.,
                body_parts=sorted(set(group.body_part) - {'unknown'}),
                limitations=['Advanced anatomical grouping only. Exact stimulus selectivity is unresolved; no everyday-language alias is assigned.'],
                selection=dict(zip(cols, key)))

    for cid, label, modality, aliases, reason in [
        ('hunger', '内部饥饿状态', 'interoception', ['饥饿', '饿', 'hunger'], 'No validated metabolic-state-to-neuron model or exact receptor mapping in this registry.'),
        ('thirst', '内部口渴状态', 'interoception', ['口渴', '渴', 'thirst'], 'Internal osmolarity and drinking state are not equivalent to external water or humidity input.'),
        ('nociception', '伤害感受', 'nociception', ['疼痛', '痛', '伤害', 'pain'], 'No validated adult nociceptor body-ID set or damaging-stimulus transfer function; no substitution with touch or heat.'),
    ]:
        add(cid, label, modality, fly.iloc[:0], status='unmapped', aliases=aliases, limitations=[reason], rate=0.)

    for cid, label, modality, aliases, source, reason in [
        ('acid_taste', '酸味／pH 味觉', 'gustation', ['酸味', '酸性', 'pH'], 'acid',
         'OtopLa-dependent acid taste is documented, but the corresponding exact bilateral body IDs and pH response model are not resolved here; vinegar odor is not acid taste.'),
        ('amino_acid_taste', '氨基酸味觉', 'gustation', ['氨基酸', '氨基酸味'], 'amino_acids',
         'Amino-acid taste depends on molecular and cellular context; no validated exact connectome population mapping is available in this registry.'),
        ('oxygen_state', '内部氧／缺氧状态', 'interoception', ['氧气', '缺氧', '氧水平'], 'malecns',
         'No validated adult neural ID mapping or oxygen physiology model; neither external CO2 ORNs nor arbitrary central cells stand in for internal oxygen sensing.'),
        ('internal_gas_state', '内部气体／酸碱状态', 'interoception', ['内部二氧化碳', '内部气体', '体内酸碱'], 'malecns',
         'Internal gas and acid-base regulation are not represented by the external antennal CO2 channel; exact sensors and transfer functions remain unresolved.'),
        ('gut_stretch', '消化道／嗉囊机械扩张', 'interoception', ['消化道扩张', '胃扩张', '嗉囊扩张'], 'gut_nutrients',
         'Piezo-mediated crop stretch has experimental support, but its functional receptor population is not reliably assigned to the available body IDs.'),
        ('internal_nutrients', '内部营养／代谢状态', 'interoception', ['营养状态', '血糖', '体内葡萄糖', '代谢状态'], 'gut_nutrients',
         'Internal glucose/nutrient signals involve hormonal and neural state; no validated state-variable or exact-ID implementation is present.'),
    ]:
        add(cid, label, modality, fly.iloc[:0], status='unmapped', aliases=aliases,
            source=source, limitations=[reason], rate=0.)

    coverage = {}
    for dataset, rows in tables.items():
        related = [c for c in channels.values() if c['dataset'] == dataset]
        all_ids = set(rows.neuron_id)
        accounted = {i for c in related for i in c['neuron_ids']}
        executable = {i for c in related if c['status'] in ['ready', 'candidate'] for i in c['side_ids']['bilateral']}
        named = {i for c in related if not c['advanced'] for i in c['neuron_ids']}
        unassigned = _ids(rows[rows.entry_side.eq('unknown')])
        coverage[dataset] = {'sensory_cells_in_engine': len(all_ids), 'accounted_cells': len(accounted),
            'named_channel_cells': len(named), 'executable_cells': len(executable),
            'not_executable_cells': len(all_ids - executable), 'not_executable_ids': sorted(all_ids - executable, key=int),
            'unassigned_entry_side_cells': len(unassigned), 'unassigned_entry_side_ids': unassigned,
            'unaccounted_ids': sorted(all_ids - accounted, key=int),
            'channel_status_counts': dict(Counter(c['status'] for c in related)),
            'advanced_channel_count': sum(c['advanced'] for c in related)}
        if all_ids != accounted:
            raise AssertionError(f'Incomplete sensory accounting for {dataset}')
    result = {'schema': 1, 'channels': channels, 'coverage': coverage, 'dt_ms': .1,
              'scope': 'Annotation inventory plus explicit simulation candidates; not all biological sensory transduction is solved.',
              'status_meaning': {'ready': 'Evidence-backed anatomical input available; physical encoding still uncalibrated.',
                                 'candidate': 'Anatomical target present; selectivity, body localization or physical encoding incomplete.',
                                 'unmapped': 'Cannot safely assign an executable functional channel or known entry side.'},
              'side_definition': {'FlyWire v783': 'Sensory side in annotation paper is the peripheral nerve entry side.',
                                  'MaleCNS v1.0': 'rootSide; never somaSide or arbitrary array splitting.',
                                  'bilateral': 'Union of known left/right/central peripheral entries; unknown entries excluded.'},
              'flywire_annotation_ids_absent_from_engine': absent,
              'source_sha256': {'flywire': sha256(ANNOTATIONS), 'malecns': sha256(MALE)},
              'sources': SOURCES,
              'encoding_assumptions': {'intensity': 'Dimensionless multiplier in [0,2], not physical concentration/force/lux.',
                  'overlap': 'Overlapping input rates sum per neuron and time bin; no duplicate spikes per bin.',
                  'visual': '10 Hz LIF event proxy for graded receptor activity; no pixel/image encoder.',
                  'timing': '0.1 ms bins; default onset 50 ms, duration 300 ms; inputs 50..2000 ms and onset 0..4000 ms; window extends 100 ms beyond final input, at most 5000 ms; profiles uncalibrated.'}}
    if save:
        dump(OUT / 'catalogue.json', result)
        dump(REPORT / 'sensory-coverage.json', {k: v for k, v in result.items() if k != 'channels'})
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=['build'])
    parser.parse_args()
    catalogue = build_catalogue()
    print(json.dumps({'channels': len(catalogue['channels']), 'coverage': {
        name: {k: v for k, v in row.items() if not k.endswith('_ids')}
        for name, row in catalogue['coverage'].items()}}, ensure_ascii=False, indent=2))
