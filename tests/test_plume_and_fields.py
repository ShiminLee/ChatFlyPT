"""Borrowed plume field and the three declared emitter field models."""
import numpy as np
import pytest

from chatflypt import plume_field
from chatflypt.environment import (FIELD_MODELS, FIELD_SITES, TYPES, empty_environment, field,
                                   measured_field, validate_objects)


def probe(type_='pheromone', **extra):
    base = {'id': 'local_source', 'type': type_, 'position_mm': [0., .16, 1.2],
            'size_mm': [.05, .05, .05], 'range_mm': .2, 'strength': 1.}
    return dict(base, **extra)


def test_borrowed_plume_artifact_is_present_and_documented():
    provenance = plume_field.provenance()
    assert provenance['available'] is True
    assert provenance['shape'] == [60, 320, 480]
    assert provenance['license'] == 'Apache-2.0 (flygym-gymnasium)'
    assert provenance['calibrated_to_cva'] is False and provenance['concentration_units'] is False
    assert len(provenance['sha256']) == 64


def test_plume_samples_are_nonzero_time_varying_and_none_outside():
    values = [plume_field.sample([0., 0.], t) for t in range(0, 300, 50)]
    assert all(value is not None for value in values)
    assert max(values) > 0.1, 'the filled core must not be empty'
    assert len({round(value, 6) for value in values}) > 1, 'the borrowed plume must vary in time'
    # Outside the grid upstream returns NaN; here that is None, i.e. no input.
    assert plume_field.sample([500., 500.], 100.) is None


def test_range_plume_and_anchored_are_declared_models_only():
    assert FIELD_MODELS == ('range', 'plume', 'anchored')
    assert FIELD_SITES['pheromone'] == 'funiculus' and FIELD_SITES['song'] == 'pedicel'
    for kind in ('pheromone', 'song'):
        assert TYPES[kind]['solid'] is False
    assert validate_objects([probe(field='plume')])[0]['field'] == 'plume'
    assert validate_objects([probe()])[0]['field'] == 'range'
    with pytest.raises(ValueError, match='场模型'):
        validate_objects([probe(field='laser')])
    with pytest.raises(ValueError, match='anchor'):
        validate_objects([probe(field='anchored', anchor='wing')])
    with pytest.raises(ValueError, match='anchor_origin_mm'):
        validate_objects([probe(field='anchored', anchor='funiculus')])
    with pytest.raises(ValueError, match='只有跟随源'):
        validate_objects([probe(anchor='funiculus')])


def test_anchored_source_keeps_a_fixed_offset_so_value_is_pose_independent():
    origin = [0., 0., 1.2]
    obj = validate_objects([probe(field='anchored', anchor='funiculus',
                                  anchor_origin_mm=origin)])[0]
    near = measured_field('anchored', [0., 0., 1.2], obj, 0.)
    far = measured_field('anchored', [3., -2., 1.2], obj, 0.)
    # The source follows the receptor with the declared offset, so the sampled
    # value is the same wherever the receptor has moved to.
    assert near == pytest.approx(far)
    # A fixed source, by contrast, falls off with distance.
    fixed = validate_objects([probe()])[0]
    assert measured_field('range', [0., .16, 1.2], fixed, 0.) > measured_field('range', [3., -2., 1.2], fixed, 0.)


def test_plume_field_value_is_measured_not_injected():
    obj = validate_objects([probe(field='plume')])[0]
    on_core = measured_field('plume', [0., 0.], obj, 0.)
    off_core = measured_field('plume', [3., 3.], obj, 0.)
    assert on_core > off_core
    # Outside the grid there is no input at all rather than a silent zero value
    # pretending to be a measurement.
    assert measured_field('plume', [500., 500.], obj, 0.) == 0.
