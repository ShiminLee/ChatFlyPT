"""Physical measurement, anatomical exclusion and candidate FeCO feedback."""
import copy
import math

import numpy as np
import pytest

from chatflypt.proprioception import (
    build_proprioceptor_inventory, encode_proprioceptive_feedback,
    load_proprioceptor_inventory, measure_proprioception, population_activation,
)


def row(neuron_id, typ='SNpp50', synonym='', side='R', nerve='MetaLN', subclass='chordotonal organ'):
    return dict(bodyId=neuron_id, type=typ, synonyms=synonym, rootSide=side,
                entryNerve=nerve, subclass=subclass,
                superclass='vnc_sensory', **{'class': 'mechanosensory_proprioceptive'})


@pytest.fixture
def inventory():
    return build_proprioceptor_inventory([
        row(1, synonym='FeCO claw'), row(2), row(3, typ='SNpp40', synonym='FeCO club'),
        row(4, typ='SNpp39', synonym='FeCO hook'),
        row(5, typ='SNpp50', side='L', nerve='ProLN'),
        row(6, typ='SNpp05', nerve='ADMN', subclass='campaniform sensilla'),
        row(7, side='unknown'), row(8, typ='SNppxx', subclass='campaniform sensilla')],
        outgoing_ids=[1, 3, 4, 5, 6, 7, 8])


def frame(angle=90., velocity=0.):
    a = math.radians(angle)
    joint = 'fly/rh_trochanterfemur-rh_tibia-pitch'
    return {'time_ms': 100., 'joint_positions_rad': {joint: 1.25},
        'joint_velocities_rad_s': {joint: math.radians(velocity)},
        'body_names': ['rh_trochanterfemur', 'rh_tibia', 'rh_tarsus1'],
        'body_positions_mm': [[1., 0., 0.], [0., 0., 0.], [math.cos(a), math.sin(a), 0.]],
        'body_contacts': []}


def test_identity_side_nerve_and_graph_are_separate_evidence(inventory):
    assert inventory['annotation_cells'] == 8
    assert inventory['leg_candidate_cells'] == 6
    claw = next(g for g in inventory['groups'] if g['key'] == 'rh:claw')
    assert claw['candidate_neuron_ids'] == ['1', '2']
    assert claw['executable_neuron_ids'] == ['1']
    assert [c['identity_evidence'] for c in claw['cells']] == [
        'direct_annotation_synonym', 'type_congruence_candidate']
    assert claw['direction_tuning'] == 'unknown'
    assert {r['neuron_id'] for r in inventory['unmapped_cells']} == {'6', '7'}


def test_compound_type_is_not_silently_cast_to_named_feco_family():
    value = build_proprioceptor_inventory([row(1, typ='SNpp50,SNpp51')], [1])
    assert value['groups'][0]['family'] == 'unresolved'


def test_measured_geometry_not_target_or_raw_qpos(inventory):
    f = frame(angle=35., velocity=-60.)
    f.update(user_intent='left turn', commands={'forward': 1.}, joint_motor_offsets_rad={'rh': 3.})
    measured = measure_proprioception(f, inventory)
    leg = next(r for r in measured['legs'] if r['leg'] == 'rh')
    assert leg['femur_tibia_included_angle_rad'] == pytest.approx(math.radians(35.))
    assert leg['tibia_model_angle_rad'] == 1.25
    assert leg['tibia_model_velocity_rad_s'] == pytest.approx(math.radians(-60.))
    assert measured['neural_input_items'] == []
    assert not measured['neural_feedback_applied']


def test_missing_measurement_is_unknown_not_zero(inventory):
    measured = measure_proprioception({'time_ms': 0.}, inventory)
    for leg in measured['legs']:
        assert leg['tibia_model_velocity_rad_s'] is None
        assert leg['normal_contact_magnitude_sum_model_units'] is None
        assert not leg['contact_sampling_available']


def test_ground_load_is_measured_without_claiming_cuticle_strain(inventory):
    f = frame()
    contact = {'body_geom': 'fly/rh_tarsus5', 'other_geom': 'ground',
        'distance_mm': -.001, 'active_constraint': True,
        'normal_force_model_units': 4., 'tangential_force_model_units': 2.}
    f['body_contacts'] = [contact, dict(contact, distance_mm=.001),
                         dict(contact, active_constraint=False)]
    measured = measure_proprioception(f, inventory)
    leg = next(r for r in measured['legs'] if r['leg'] == 'rh')
    assert leg['normal_contact_magnitude_sum_model_units'] == 4.
    assert len(leg['surface_contacts']) == 1
    assert leg['cuticle_strain'] is None
    assert not leg['force_calibrated']
    assert measured['rejected_contact_records'] == 2


def test_published_polynomial_is_rectified_without_fictional_resting_rate():
    activation, raw = population_activation(90., 0., 'claw')
    assert raw == -.0357758464550021
    assert activation == 0.
    assert population_activation(0., 0., 'claw')[0] > .8
    assert 0 < population_activation(180., 0., 'claw')[0] < population_activation(0., 0., 'claw')[0]


@pytest.mark.parametrize('velocity,expected', [(-5.0001, 1.), (-5., 0.), (0., 0.), (5., 0.), (5.0001, 1.)])
def test_published_club_threshold_degrees_per_second(velocity, expected):
    assert population_activation(None, velocity, 'club') == (expected, expected)


def test_real_measured_change_changes_only_eligible_input_ids(inventory):
    still, _ = encode_proprioceptive_feedback(frame(90., 0.), 42, inventory=inventory)
    moved, report = encode_proprioceptive_feedback(frame(20., -100.), 42, inventory=inventory)
    assert len(still['event_steps']) == 0
    assert set(moved['input_neuron_ids']) == {1, 3}
    assert len(moved['event_steps']) > 0
    assert set(moved['event_neuron_ids']) <= {1, 3}
    assert report['engineering_peak_rate_hz'] == 50.
    assert not report['biologically_calibrated']
    assert not report['neural_feedback_applied']
    assert report['neural_feedback_encoded']
    assert not report['gcamp_filter_applied']


def test_same_seed_and_measurement_reproduce_tape_without_intent(inventory):
    a = frame(30., 70.)
    b = dict(a, message='run towards sugar', expected_behavior='escape', commands={'forward': 1})
    t1, r1 = encode_proprioceptive_feedback(a, 123, inventory=inventory)
    t2, r2 = encode_proprioceptive_feedback(b, 123, inventory=inventory)
    for key in ('input_neuron_ids', 'event_steps', 'event_neuron_ids'):
        np.testing.assert_array_equal(t1[key], t2[key])
    assert r1 == r2


def test_unknown_input_or_connectivity_abstains_or_fails(inventory):
    tape, report = encode_proprioceptive_feedback({'time_ms': 0.}, 1, inventory=inventory)
    assert len(tape['event_steps']) == 0
    assert report['activations'] == []
    with pytest.raises(ValueError, match='connectivity'):
        encode_proprioceptive_feedback(frame(), 1,
            inventory=build_proprioceptor_inventory([row(1)]))
    foreign = copy.deepcopy(inventory)
    foreign['dataset'] = 'FlyWire'
    with pytest.raises(ValueError, match='namespace'):
        encode_proprioceptive_feedback(frame(), 1, inventory=foreign)


def test_actual_malecns_inventory_has_no_cross_leg_or_unmodeled_ids():
    inventory = load_proprioceptor_inventory()
    assert inventory['annotation_cells'] == 1454
    assert inventory['leg_candidate_cells'] == 645
    assert inventory['connectivity_checked']
    assert len(inventory['annotation_sha256']) == 64
    for group in inventory['groups']:
        assert set(group['executable_neuron_ids']) <= set(group['candidate_neuron_ids'])
        for cell in group['cells']:
            assert cell['root_side'].lower() == group['leg'][0]
            assert {'ProLN': 'f', 'MesoLN': 'm', 'MetaLN': 'h'}[cell['entry_nerve']] == group['leg'][1]


@pytest.mark.parametrize('field,value', [('time_ms', float('nan')),
    ('joint_positions_rad', {'rh_tibia': float('inf')}), ('joint_velocities_rad_s', [1, 2])])
def test_invalid_physical_measurements_fail(field, value, inventory):
    f = frame()
    f[field] = value
    with pytest.raises(ValueError):
        encode_proprioceptive_feedback(f, 1, inventory=inventory)
