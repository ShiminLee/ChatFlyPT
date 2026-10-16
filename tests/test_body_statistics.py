"""Controller eligibility, causal held values, and independent observation windows."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from chatflypt.behavior_body import BodyControlStatistics, COMMAND_KEYS, command_execution
from chatflypt.body_feedback import BodySession
from chatflypt.joint_motor import LEGS, held_rows_in_window, joint_summary


def command(**changes):
    return dict(dict.fromkeys(COMMAND_KEYS, 0.), **changes)


@pytest.mark.parametrize('key', ['escape', 'landing', 'antenna', 'rub', 'anterior'])
def test_gesture_threshold_ignores_tails_but_preserves_raw_command(key):
    for value, active in [(0., False), (.007, False), (.01, False), (.010001, True)]:
        original = command(**{key: value})
        frozen = deepcopy(original)
        status = command_execution(original)
        assert status['requested_command_nonzero'] == bool(value)
        assert status['neural_command_active'] is active
        assert original == frozen


@pytest.mark.parametrize('key', ['left', 'right', 'feeding', 'song', 'flight'])
def test_source_programs_without_dead_zone_keep_small_signals(key):
    status = command_execution(command(**{key: .0001}))
    assert status['requested_command_nonzero'] and status['neural_command_active']
    assert command_execution(command(left=-.0001))['neural_command_active']


def test_joint_targets_and_passive_override_are_reported_without_editing_values():
    joints = dict.fromkeys(LEGS, 0.); joints['rf'] = -.0001
    assert command_execution(command(), joints)['effective_programs']['joint_targets']
    passive = command_execution(command(escape=.3), joints, passive=True)
    assert passive['requested_command_nonzero'] and not passive['neural_command_active']
    assert joints['rf'] == -.0001


def test_window_force_and_extrema_keep_real_zero_command_posture_force():
    stats = BodyControlStatistics()
    stats.record(command(), {}, 40.)
    stats.record(command(escape=.007), {}, 2.)
    stats.record(command(escape=.004), {}, 7.)
    result = stats.report(); c = result['controls']
    assert result['requested_command_in_window'] and not result['motor_activity_in_window']
    assert c['integration_steps'] == 3 and c['powered_steps'] == 0
    assert c['requested_command_steps'] == 2 and c['zero_command_steps'] == 1
    assert c['max_force_during_zero_commands'] == 40.
    assert c['max_force_without_effective_commands'] == 40.
    # Real zero was used in this window, so this minimum really is zero.
    assert result['command_summary']['escape'] == {'min': 0., 'max': .007}
    later = BodyControlStatistics(100.)
    later.record(command(escape=.004), {}, 3.)
    later.record(command(escape=.002), {}, 5.)
    report = later.report()
    assert report['controls']['max_force_during_zero_commands'] is None
    assert report['controls']['max_actuator_force_model_units'] == 5.
    assert report['command_summary']['escape'] == {'min': .002, 'max': .004}


def test_held_window_values_include_prior_bin_exclude_endpoint_and_no_fake_zero():
    times = [20., 40., 60.]
    rows = [{'x': .2}, {'x': .4}, {'x': .9}]
    assert held_rows_in_window(times, rows, 30., 60., {'x': 0.}) == rows[:2]
    assert held_rows_in_window(times, rows, 40., 60., {'x': 0.}) == rows[1:2]
    assert held_rows_in_window(times, rows, 0., 20., {'x': 0.}) == [{'x': 0.}]
    assert held_rows_in_window(times, rows, 60., 60., {'x': 0.}) == []
    drive = {'duration_ms': 60., 'available_at_ms': times,
             'joint_motor': {'offsets_rad': [dict.fromkeys(LEGS, -r['x']) for r in rows]}}
    assert joint_summary(drive, 30., 60.)['lf'] == {'min': -.4, 'max': -.2}
    assert joint_summary(drive, 0., 20.)['lf'] == {'min': 0., 'max': 0.}


def test_persistent_advance_carries_command_across_message_and_resets_only_stats(monkeypatch):
    # Exercise the actual worker scheduler without neural simulation or MuJoCo.
    # The fake body records every unchanged control value it receives.
    monkeypatch.setattr('chatflypt.behavior_readout._validate_behavior_drive', lambda d: None)
    session = BodySession.__new__(BodySession)
    data = SimpleNamespace(actuator_force=np.array([40.]))
    calls = []
    body = SimpleNamespace(elapsed=0., context='ground', joint_motor_offsets=dict.fromkeys(LEGS, 0.))
    def advance(dt, values):
        calls.append((round(body.elapsed*1000, 7), deepcopy(values)))
        body.elapsed += dt/1000
    body.advance = advance
    session.body=body; session.data=data; session.sim=SimpleNamespace(timestep=.001)
    session.np=np; session.statistics=BodyControlStatistics()
    session.touch=SimpleNamespace(update=lambda t: None)
    session.trajectory=[]; session.check=lambda: None
    session.frame=lambda: {'time_ms': body.elapsed*1000}
    session.probe=lambda refresh: {'time_ms': body.elapsed*1000}
    first={'duration_ms':30., 'context':'ground','available_at_ms':[20.,40.,60.],
           'commands':[command(escape=.007),command(escape=.004),command(escape=.8)]}
    frozen=deepcopy(first)
    session.advance(first, False)
    assert session.statistics.report()['controls']['max_force_during_zero_commands']==40.
    original_body=session.body
    session.begin_observation()
    assert session.body is original_body and body.elapsed == pytest.approx(.03)
    data.actuator_force[:]=5.
    session.advance(dict(first, duration_ms=60.), False)
    report=session.statistics.report(); c=report['controls']
    assert c['integration_steps']==30 and c['requested_command_steps']==30
    assert c['powered_steps']==0 and c['zero_command_steps']==0
    assert c['max_actuator_force_model_units']==5.
    assert c['max_force_during_zero_commands'] is None
    assert report['command_summary']['escape']=={'min':.004,'max':.007}
    assert calls[30][1]['escape']==.007 and calls[40][1]['escape']==.004
    assert calls[-1][1]['escape']==.004  # The 60 ms command cannot act before 60.
    assert first==frozen
