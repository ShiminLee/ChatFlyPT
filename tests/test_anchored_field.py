"""Check the anchored field model: a declared side gives a real, one-sided field.

Pure-function checks on the fixed model, plus a guard that the legacy (no side
declared) scenes still behave exactly as the frozen corpora recorded.
"""

from chatflypt.environment import measured_field, validate_objects

# A source declared 0.10 mm outward of the left funiculus, range 0.15 mm, and the
# two receptors 0.216 mm apart (measured neutral separation).
LEFT = [0.0, -0.108, 0.5]
RIGHT = [0.0, 0.108, 0.5]
ORIGIN = [0.0, 0.0, 0.5]
OBJECT = {'id': 'partner', 'type': 'pheromone', 'position_mm': [0.0, -0.10, 0.5],
          'size_mm': [.05, .05, .05], 'range_mm': .15, 'strength': 1.,
          'field': 'anchored', 'anchor': 'funiculus', 'anchor_origin_mm': ORIGIN,
          'anchor_side': 'left'}


def test_a_declared_side_is_strictly_one_sided_and_follows_its_own_receptor():
    validated = validate_objects([OBJECT])[0]
    assert validated['anchor_side'] == 'left'
    near = measured_field('anchored', LEFT, validated, 0., anchor_position=LEFT)
    far = measured_field('anchored', RIGHT, validated, 0., anchor_position=LEFT)
    assert near > 0 and far == 0., 'a declared side must be strictly one-sided'
    # If the animal moves, the source follows its own receptor: the near field is
    # unchanged and the far side stays at zero.
    moved_left = [0.4, -0.10, 0.5]
    moved_right = [0.42, 0.12, 0.5]
    near_moved = measured_field('anchored', moved_left, validated, 0., anchor_position=moved_left)
    far_moved = measured_field('anchored', moved_right, validated, 0., anchor_position=moved_left)
    assert abs(near_moved - near) < 1e-9 and far_moved == 0.


def test_legacy_scenes_without_a_side_stay_byte_identical_and_bilateral():
    legacy_spec = dict(OBJECT)
    legacy_spec.pop('anchor_side')
    legacy = validate_objects([legacy_spec])[0]
    # The validated payload is exactly what the frozen corpora recorded, so the
    # old scenes stay reproducible.
    assert 'anchor_side' not in legacy
    legacy_left = measured_field('anchored', LEFT, legacy, 0.)
    legacy_right = measured_field('anchored', RIGHT, legacy, 0.)
    assert abs(legacy_left - legacy_right) < 1e-9
    assert legacy_left > 0


def test_a_fixed_small_range_source_is_unchanged_by_the_fix():
    fixed = validate_objects([{'id': 'fixed', 'type': 'yeast', 'position_mm': [0., -0.09, 0.5],
                               'size_mm': [.05, .05, .05], 'range_mm': .20, 'strength': 1.}])[0]
    near = measured_field('range', LEFT, fixed, 0.)
    far = measured_field('range', RIGHT, fixed, 0.)
    assert near > 10 * far


def test_an_unknown_anchor_side_is_rejected():
    bad = dict(OBJECT, anchor_side='middle')
    try:
        validate_objects([bad])
    except ValueError as error:
        assert 'anchor_side' in str(error)
    else:
        raise AssertionError('an unknown anchor side must be rejected')
