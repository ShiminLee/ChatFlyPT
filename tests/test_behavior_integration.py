"""Integration boundaries for exact FlyWire IDs and honest movement reporting."""
from types import SimpleNamespace

import numpy as np
import pytest

from chatflypt import visualization
from chatflypt.behavior_readout import build_behavior_drive
from chatflypt.sensory_readout import sensory_response_text
from chatflypt.server import validate_body_context


def test_missing_annotation_cannot_round_long_ids_or_prevent_behavior_readout(tmp_path, monkeypatch):
    ids = np.array([720575940600000001, 720575940600000003, 720575940600000005], dtype=np.int64)
    path = tmp_path / 'annotations.tsv'
    path.write_text('root_id\tpos_x\tpos_y\tpos_z\tsuper_class\tcell_type\tcell_class\tside\n'
                    f'{ids[0]}\t0\t0\t0\tdescending\tDNg13\tdescending\tleft\n'
                    f'{ids[1]}\t1\t1\t1\tdescending\tDNa02\tdescending\tright\n', encoding='utf-8')
    monkeypatch.setattr(visualization, 'ANNOTATIONS', path)
    view = visualization.NeuralView(SimpleNamespace(ids=ids))
    assert view.rows.root_id.tolist() == ids.astype(str).tolist()
    trial = SimpleNamespace(counts=np.array([1, 0, 0]), spike_ids=np.array([0]),
                            spike_steps=np.array([100]), duration_ms=40., seed=1)
    drive = build_behavior_drive(trial, ids, view.rows, dataset='FlyWire v783')
    assert drive['motor_spikes'] == 1
    assert drive['has_command_in_window']


@pytest.mark.parametrize('context', ['ground', 'tethered_flight'])
def test_explicit_supported_body_contexts(context):
    assert validate_body_context(context) == context


@pytest.mark.parametrize('context', ['free_flight', None, {}, 1])
def test_unsupported_body_contexts_are_rejected(context):
    with pytest.raises(ValueError):
        validate_body_context(context)


@pytest.mark.parametrize('status, expected', [
    ('no_motor_activity', '未形成当前控制器可用的动作命令'),
    ('error', '本次身体仿真未完成'),
    ('timeout', '本次身体仿真未完成'),
])
def test_neural_motor_spikes_do_not_imply_successful_movement(status, expected):
    text = sensory_response_text({'words': ['sugar'], 'descriptions': {'sugar': '糖'}},
        {'non_sensory_spikes': 1, 'DN': {'spikes': 1},
         'body_simulation': {'status': status, 'physics_run': False}})
    assert expected in text
    assert '已驱动身体物理仿真' not in text
