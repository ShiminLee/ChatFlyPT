"""Auditable body measurements and MaleCNS proprioceptor candidate routing.

Anatomical eligibility, measured physical variables, published population
activation and an explicitly uncalibrated event-rate adapter are separate
facts. A controller's local feedback alone is not evidence that the connectome
has received proprioceptive input. Unmapped receptor families abstain.
"""
from functools import lru_cache
import hashlib
import json
import math
import re

from .paths import ROOT

SCHEMA = 'chatflypt.proprioception.v1'
LEGS = ('lf', 'lm', 'lh', 'rf', 'rm', 'rh')
LEG_NERVES = {'ProLN': 'f', 'MesoLN': 'm', 'MetaLN': 'h'}
ANNOTATIONS = ROOT / 'data/raw/malecns-v1.0/body-annotations-male-cns-v1.0-minconf-0.5.feather'
GRAPH = ROOT / 'data/derived/malecns/graph.npz'
SOURCES = [
    'https://male-cns.janelia.org/download/',
    'https://doi.org/10.1016/j.neuron.2018.09.009',
    'https://doi.org/10.1016/j.neuron.2023.07.009',
    'https://github.com/sagrawal/Lee_2024',
    'https://doi.org/10.1038/s41586-025-09554-2',
]
ACTIVATION_SOURCE = ('https://github.com/chrisjdallmann/feco-inhibition/blob/'
    'e1233f4a987c532c9f1ab42273af21a0a6a50393/code/utils/imaging_predict_gcamp.m')
# Ported from the author's MIT-licensed activation model; original and license
# are preserved in data/raw/proprioception-dallmann2025. These coefficients fit
# normalized CALCIUM, not spikes/s. No GCaMP imaging filter is added to neurons.
CLAW_COEFFICIENTS = (-4.28350195279596e-09, -3.08701145496934e-07,
    0.000123726627502515, 0.000474757347677466, -0.0357758464550021)
ENGINEERING_PEAK_RATE_HZ = 50.  # Existing proprioceptor experiment scale.
CLUB_VELOCITY_THRESHOLD_DEG_S = 5.  # Passive-movement model in the paper.
# These are type-congruence CANDIDATES inferred from directly labelled cells in
# the same annotation release. They do not establish flexion/extension tuning,
# and a missing synonym on another cell is never relabelled as direct evidence.
FECO_TYPES = {
    'SNpp39': 'hook', 'SNpp41': 'hook',
    'SNpp40': 'club', 'SNpp47': 'club', 'SNpp56': 'club',
    'SNpp57': 'club', 'SNpp60': 'club',
    'SNpp50': 'claw', 'SNpp51': 'claw',
}
FAMILY_PARAMETERS = {
    'claw': ['flexion_or_extension_identity', 'per_cell_angle_tuning',
             'model_to_anatomical_angle_registration', 'angle_history_hysteresis',
             'measured_firing_rate_curve'],
    'hook': ['flexion_or_extension_identity', 'model_to_anatomical_direction_registration',
             'velocity_and_adaptation_tuning', 'measured_firing_rate_curve'],
    'club': ['cell_specific_vibration_tuning', 'high_frequency_displacement_samples',
             'movement_response_and_adaptation', 'measured_firing_rate_curve'],
    'hair_plate': ['sensillum_location_and_deflection', 'joint_association',
                   'measured_firing_rate_curve'],
    'campaniform': ['sensillum_location_orientation', 'cuticle_strain_transfer',
                    'force_unit_calibration', 'measured_firing_rate_curve'],
    'unresolved': ['receptor_identity', 'joint_or_load_association',
                   'measured_firing_rate_curve'],
}


def _text(value):
    return value if isinstance(value, str) else ''


def build_proprioceptor_inventory(rows, outgoing_ids=None):
    """Build exact namespace/side/nerve candidates from annotation records.

    ``outgoing_ids`` must denote cells with at least one edge in the actual
    active neural graph, not merely an annotation or raw anatomical synapse.
    None means connectivity has not been checked, not that every cell works.
    """
    outgoing = None if outgoing_ids is None else {str(int(i)) for i in outgoing_ids}
    groups = {}
    excluded = []
    seen = set()
    total = 0
    for row in rows:
        if (row.get('class') != 'mechanosensory_proprioceptive'
                or 'sensory' not in _text(row.get('superclass'))):
            continue
        total += 1
        neuron_id = str(int(row['bodyId']))
        if neuron_id in seen:
            raise ValueError('Duplicate MaleCNS proprioceptor ID')
        seen.add(neuron_id)
        nerve, side = _text(row.get('entryNerve')), _text(row.get('rootSide'))
        typ, subclass = _text(row.get('type')), _text(row.get('subclass'))
        synonyms = _text(row.get('synonyms'))
        evidence = {'neuron_id': neuron_id, 'type': typ, 'subclass': subclass,
                    'synonyms': synonyms, 'entry_nerve': nerve, 'root_side': side,
                    'modeled_outgoing': None if outgoing is None else neuron_id in outgoing}
        if nerve not in LEG_NERVES or side not in ('L', 'R'):
            excluded.append(dict(evidence, reason='No unambiguous main-leg-nerve and side mapping'))
            continue
        leg = side.lower() + LEG_NERVES[nerve]
        family = next((f for f in ('claw', 'hook', 'club')
                       if re.search(r'\bFeCO\s+' + f + r'\b', synonyms)), None)
        evidence_level = 'direct_annotation_synonym' if family else 'annotated_subclass'
        if family is None and subclass == 'chordotonal organ' and typ in FECO_TYPES:
            family = FECO_TYPES[typ]
            evidence_level = 'type_congruence_candidate'
        if family is None:
            family = {'hair plate': 'hair_plate', 'campaniform sensilla': 'campaniform'}.get(subclass, 'unresolved')
        key = f'{leg}:{family}'
        group = groups.setdefault(key, {'key': key, 'leg': leg, 'family': family,
            'cells': [], 'candidate_neuron_ids': [], 'executable_neuron_ids': [],
            'missing_parameters': list(FAMILY_PARAMETERS[family]),
            'transduction_status': 'missing_empirical_calibration',
            'direction_tuning': 'unknown', 'biologically_calibrated': False})
        group['candidate_neuron_ids'].append(neuron_id)
        if evidence['modeled_outgoing'] is True:
            group['executable_neuron_ids'].append(neuron_id)
        group['cells'].append(dict(evidence, identity_evidence=evidence_level))
    ordered = []
    for key in sorted(groups):
        group = groups[key]
        for field in ('candidate_neuron_ids', 'executable_neuron_ids'):
            group[field].sort(key=int)
        group['cells'].sort(key=lambda r: int(r['neuron_id']))
        ordered.append(group)
    return {'schema': SCHEMA, 'dataset': 'MaleCNS v1.0', 'id_namespace': 'malecns_body_id',
            'annotation_cells': total, 'leg_candidate_cells': sum(len(g['cells']) for g in ordered),
            'groups': ordered, 'unmapped_cells': excluded,
            'connectivity_checked': outgoing is not None, 'sources': SOURCES,
            'biologically_calibrated': False}


@lru_cache(maxsize=1)
def load_proprioceptor_inventory():
    import numpy as np
    import pandas as pd
    rows = pd.read_feather(ANNOTATIONS).fillna('').to_dict('records')
    outgoing = None
    if GRAPH.exists():
        with np.load(GRAPH, allow_pickle=False) as graph:
            outgoing = graph['ids'][np.diff(graph['indptr']) > 0]
    result = build_proprioceptor_inventory(rows, outgoing)
    with ANNOTATIONS.open('rb') as stream:
        result['annotation_sha256'] = hashlib.file_digest(stream, 'sha256').hexdigest()
    result['mapping_sha256'] = hashlib.sha256(json.dumps(result['groups'], sort_keys=True).encode()).hexdigest()
    return result


def _finite(value, label):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f'{label} must be finite')
    return float(value)


def _joint_map(frame, field):
    values = frame.get(field, {})
    if not isinstance(values, dict):
        raise ValueError(f'{field} must map actual joint names to measurements')
    return {str(k).removeprefix('fly/'): _finite(v, field) for k, v in values.items()}


def _leg_for_geom(name):
    match = re.match(r'^fly/(lf|lm|lh|rf|rm|rh)_', name or '')
    return match.group(1) if match else None


def _included_angles(frame):
    names, positions = frame.get('body_names', []), frame.get('body_positions_mm', [])
    if len(names) != len(positions):
        raise ValueError('Body position/name counts differ')
    if len(set(names)) != len(names):
        raise ValueError('Duplicate body segment names')
    xyz = dict(zip(names, positions))
    result = {}
    for leg in LEGS:
        keys = [f'{leg}_trochanterfemur', f'{leg}_tibia', f'{leg}_tarsus1']
        if not all(key in xyz for key in keys):
            continue
        points = [[_finite(x, 'body coordinate') for x in xyz[k]] for k in keys]
        if any(len(p) != 3 for p in points):
            raise ValueError('Body coordinates must be 3-vectors')
        u = [a - b for a, b in zip(points[0], points[1])]
        v = [a - b for a, b in zip(points[2], points[1])]
        denom = math.sqrt(sum(x*x for x in u) * sum(x*x for x in v))
        if denom > 0:
            result[leg] = math.acos(max(-1., min(1., sum(a*b for a, b in zip(u, v)) / denom)))
    return result


def measure_proprioception(frame, inventory=None):
    """Route measured body state to anatomical candidates, never to fake rates.

    The frame accepts named ``joint_positions_rad`` and ``joint_velocities_rad_s``
    from MuJoCo qpos/qvel, ``body_contacts`` including ground contacts, and
    ``time_ms``. Optional ``body_names``/``body_positions_mm`` provide geometric
    femur-tibia included angles. Raw model angles are not relabelled as animal
    angles; summed contact force magnitudes are not cuticle strain or net force.

    A missing measurement is None/unknown, not a measured zero. The empty neural
    input list is an explicit abstention pending independently sourced tuning,
    not proof that a real fly's proprioceptors are silent.
    """
    inventory = load_proprioceptor_inventory() if inventory is None else inventory
    if inventory.get('dataset') != 'MaleCNS v1.0' or inventory.get('id_namespace') != 'malecns_body_id':
        raise ValueError('Proprioceptor ID namespaces must not be mixed')
    time_ms = _finite(frame.get('time_ms'), 'time_ms')
    positions = _joint_map(frame, 'joint_positions_rad')
    velocities = _joint_map(frame, 'joint_velocities_rad_s')
    angles = _included_angles(frame)
    contacts = frame.get('body_contacts')
    if contacts is not None and not isinstance(contacts, list):
        raise ValueError('body_contacts must be a list of actual contacts')
    loads = {leg: [] for leg in LEGS}
    rejected = 0
    for contact in contacts or []:
        leg = _leg_for_geom(contact.get('body_geom'))
        if (leg is None or contact.get('active_constraint') is not True
                or _finite(contact.get('distance_mm'), 'contact distance') > 0):
            rejected += 1
            continue
        normal = _finite(contact.get('normal_force_model_units'), 'normal force')
        tangent = _finite(contact.get('tangential_force_model_units'), 'tangential force')
        if normal < 0 or tangent < 0:
            raise ValueError('Contact force magnitudes must be nonnegative')
        loads[leg].append({'body_geom': contact['body_geom'], 'other_geom': contact.get('other_geom'),
                          'normal_force_model_units': normal,
                          'tangential_force_model_units': tangent})
    legs = []
    for leg in LEGS:
        joint = f'{leg}_trochanterfemur-{leg}_tibia-pitch'
        groups = [g for g in inventory['groups'] if g['leg'] == leg]
        leg_joints = {k: v for k, v in positions.items() if re.search(r'(^|-)'+leg+r'_', k)}
        leg_velocities = {k: v for k, v in velocities.items() if re.search(r'(^|-)'+leg+r'_', k)}
        legs.append({'leg': leg, 'joint_positions_rad': leg_joints,
            'joint_velocities_rad_s': leg_velocities,
            'tibia_model_angle_rad': positions.get(joint),
            'tibia_model_velocity_rad_s': velocities.get(joint),
            'femur_tibia_included_angle_rad': angles.get(leg),
            'contact_sampling_available': contacts is not None,
            'surface_contacts': loads[leg],
            'normal_contact_magnitude_sum_model_units': (sum(c['normal_force_model_units'] for c in loads[leg])
                                                       if contacts is not None else None),
            'force_calibrated': False, 'cuticle_strain': None,
            'receptors': [{k: g[k] for k in ('key', 'family', 'candidate_neuron_ids',
                'executable_neuron_ids', 'missing_parameters', 'direction_tuning', 'transduction_status')}
                for g in groups]})
    return {'schema': SCHEMA, 'dataset': 'MaleCNS v1.0', 'id_namespace': 'malecns_body_id',
        'time_ms': time_ms, 'source': 'measured_physics', 'legs': legs,
        'mapping_sha256': inventory.get('mapping_sha256'),
        'annotation_sha256': inventory.get('annotation_sha256'),
        'rejected_contact_records': rejected, 'neural_input_items': [],
        'biologically_calibrated': False, 'neural_feedback_applied': False,
        'status': 'measured_candidate_routing_only',
        'unknowns': ['No per-cell tuning or firing-rate calibration is installed.',
            'Model qpos/qvel coordinates are not anatomical tuning coordinates.',
            'Contact force magnitudes do not provide campaniform cuticle strain.',
            'Discrete body sampling has not been validated for FeCO high-frequency vibration; '
            'neither 100 ms nor 10 ms samples establish adequate vibration encoding.'],
        'sources': SOURCES}


def population_activation(angle_deg, velocity_deg_s, family):
    """Published dimensionless FeCO population activation, prior to GCaMP.

    Geometric included angles use the 90-degree centre of the source passive
    dataset, as described in Dallmann 2025 Methods. Their separate behaviour
    dataset used 80 degrees to account for a tracking offset; the source code's
    unconditional 80 is therefore NOT transplanted to model qpos. No exact
    animal-to-body registration or individual sensory-neuron tuning is claimed.
    """
    if family == 'claw':
        angle_deg = _finite(angle_deg, 'included angle in degrees')
        if not 0 <= angle_deg <= 180:
            raise ValueError('Included angle must be between 0 and 180 degrees')
        x = angle_deg - 90.
        value = 0.
        for coefficient in CLAW_COEFFICIENTS:
            value = value*x + coefficient
        # The fitted normalized calcium curve can be slightly negative; a
        # nonnegative event rate requires rectification. This is an adapter.
        return max(0., min(1., value)), value
    if family == 'club':
        velocity_deg_s = _finite(velocity_deg_s, 'angular velocity in degrees/s')
        value = float(abs(velocity_deg_s) > CLUB_VELOCITY_THRESHOLD_DEG_S)
        return value, value
    raise ValueError('Only published pooled claw/club activation is supported')


def encode_proprioceptive_feedback(frame, seed, duration_ms=100., *, inventory=None,
                                  peak_rate_hz=ENGINEERING_PEAK_RATE_HZ):
    """Physical knee geometry/velocity -> limited candidate FeCO event tape.

    Return ``(tape, reception_report)``. The tape has the existing sensory-event
    schema and only MaleCNS IDs with actual model out-edges. The population
    activation model is published; the activation-to-rate gain and homogeneous
    within-population assignment are explicitly engineering approximations.
    Hook direction, force transduction, vibration and untyped organs abstain.
    No user message, intended action, readout or motor command is consumed.
    """
    import numbers
    import numpy as np
    from .brain import DT_MS
    duration_ms = _finite(duration_ms, 'duration_ms')
    peak_rate_hz = _finite(peak_rate_hz, 'peak_rate_hz')
    if not 0 < duration_ms <= 100. or abs(duration_ms / DT_MS - round(duration_ms / DT_MS)) > 1e-8:
        raise ValueError('Feedback duration must align to the neural clock and be at most 100 ms')
    if not 0 <= peak_rate_hz <= 150.:
        raise ValueError('Engineering peak rate must be within 0..150 Hz')
    if isinstance(seed, bool) or not isinstance(seed, numbers.Integral) or not 0 <= seed < 2**32:
        raise ValueError('Seed must be an unsigned 32-bit integer')
    inventory = load_proprioceptor_inventory() if inventory is None else inventory
    report = measure_proprioception(frame, inventory)
    if not inventory.get('connectivity_checked'):
        raise ValueError('Active graph connectivity must be checked before FeCO encoding')
    rates = {}
    activations = []
    for leg in report['legs']:
        for receptor in leg['receptors']:
            family = receptor['family']
            angle = leg['femur_tibia_included_angle_rad']
            velocity = leg['tibia_model_velocity_rad_s']
            if family not in ('claw', 'club'):
                continue
            if (family == 'claw' and angle is None) or (family == 'club' and velocity is None):
                continue
            value, raw = population_activation(
                math.degrees(angle) if angle is not None else None,
                math.degrees(velocity) if velocity is not None else None, family)
            rate = value * peak_rate_hz
            ids = receptor['executable_neuron_ids']
            activations.append({'receptor_group': receptor['key'], 'family': family,
                'angle_deg': math.degrees(angle) if angle is not None else None,
                'angular_speed_deg_s': abs(math.degrees(velocity)) if velocity is not None else None,
                'published_activation_raw': raw, 'activation_rectified': value,
                'population_rate_hz': rate, 'neuron_ids': ids,
                'identity_scope': 'annotated_or_same_type_candidate', 'biologically_calibrated': False})
            if rate > 0:
                for neuron_id in ids:
                    if neuron_id in rates:
                        raise ValueError('A proprioceptor cannot receive two pooled family assignments')
                    rates[neuron_id] = rate
    steps = round(duration_ms / DT_MS)
    time_parts, origin_parts = [], []
    for neuron_id, rate in sorted(rates.items(), key=lambda item: int(item[0])):
        digest = hashlib.sha256(f'{seed}:proprioception:malecns_body_id:{neuron_id}'.encode()).digest()
        rng = np.random.default_rng(int.from_bytes(digest[:8], 'little'))
        times = np.flatnonzero(rng.random(steps) < rate * DT_MS / 1000.).astype(np.int32)
        time_parts.append(times)
        origin_parts.append(np.full(len(times), int(neuron_id), dtype=np.int64))
    times = np.concatenate(time_parts) if time_parts else np.empty(0, np.int32)
    origins = np.concatenate(origin_parts) if origin_parts else np.empty(0, np.int64)
    order = np.argsort(times, kind='stable')
    tape = {'dataset': 'MaleCNS v1.0', 'id_namespace': 'malecns_body_id', 'seed': int(seed),
        'duration_ms': duration_ms, 'dt_ms': DT_MS,
        'input_neuron_ids': np.array(sorted(map(int, rates)), np.int64),
        'event_steps': times[order], 'event_neuron_ids': origins[order]}
    report.update(status='limited_population_feedback_encoded',
        neural_feedback_applied=False,  # The caller must record successful neural delivery.
        neural_feedback_encoded=True, encoded_event_count=len(times), activations=activations,
        encoded_neuron_ids=sorted(rates, key=int), input_source='measured_body_proprioception',
        engineering_peak_rate_hz=peak_rate_hz,
        activation_source=ACTIVATION_SOURCE,
        activation_units='dimensionless fitted population-calcium activation, not measured firing rate',
        activation_angle_centre_deg=90., club_velocity_threshold_deg_s=CLUB_VELOCITY_THRESHOLD_DEG_S,
        gcamp_filter_applied=False,
        encoding_limits=['Group mean activation assigned homogeneously to candidate cells.',
            f'{peak_rate_hz:g} Hz peak is an explicit engineering event scale, not a fitted firing rate.',
            'Hook direction and force-to-campaniform transduction remain unresolved.',
            'Movement-only club model does not encode high-frequency substrate vibration.',
            'Presynaptic state-dependent modulation is left to the current connectome dynamics; it is not separately calibrated.'])
    # Exact-ID physical feedback uses an event tape, not catalogue input items.
    report.pop('neural_input_items', None)
    return tape, report
