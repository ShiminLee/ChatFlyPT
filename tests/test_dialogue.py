"""Blind Chinese narration from decoded labels, without any live model/API request."""
import json

import pytest

from chatflypt.dialogue import sensory_report, sensation_text


def decoded(*words, **extra):
    return {"words": list(words), **extra}


def reaction(spikes=12, cells=0, **extra):
    return {"non_sensory_spikes": spikes, "membrane_response": {"internal_cells": cells}, **extra}


def test_sided_touch_and_body_part_come_only_from_decoded_labels():
    report = sensory_report(decoded("foreleg_touch:right"), reaction())
    assert report["text"] == "我感到右前腿被碰到了。"
    assert report["feelings"][0]["body_part"] == "foreleg"
    assert report["feelings"][0]["side"] == "right"
    assert sensation_text(decoded("left_foreleg_touch"), reaction()) == "我感到左前腿被碰到了。"
    assert "两侧翅膀" in sensation_text(decoded("wing_touch:bilateral"), reaction())


@pytest.mark.parametrize("label,phrase", [
    ("sugar", "甜味"), ("bitter", "苦味"), ("water", "水"), ("salt", "盐味"),
    ("vinegar", "醋相关气味"), ("yeast", "酵母相关气味"), ("geosmin", "土臭素"),
    ("pheromone", "cVA"), ("warm", "温暖"), ("cold", "寒冷"), ("humid", "潮湿"),
    ("dry", "干燥"), ("bright", "光"), ("touch", "触碰"), ("sound", "声音"), ("wind", "风"),
])
def test_all_legacy_sensations_are_short_chinese_first_person(label, phrase):
    text = sensation_text(decoded(label), reaction())
    assert text.startswith("我") and phrase in text
    assert "spike" not in text and "神经" not in text


def test_general_channels_do_not_gain_unidentified_specificity():
    assert "求偶" not in sensation_text(decoded("auditory:left"), reaction())
    assert "求偶脉冲声" in sensation_text(decoded("courtship_pulse"), reaction())
    assert "分不清" in sensation_text(decoded("sweet_water:right"), reaction())
    assert "蓝／绿" in sensation_text(decoded("vision_r8:left"), reaction())
    assert "风或重力" in sensation_text(decoded("wind_gravity:right"), reaction())
    assert "己酸" in sensation_text(decoded("fatty_acid_taste:left"), reaction())
    assert "平衡棒" in sensation_text(decoded("haltere_proprioception:right"), reaction())
    assert "已" not in sensation_text(decoded("leg_proprioception:left"), reaction())


def test_zero_activity_overrides_even_selected_words_or_motor_claims():
    report = sensory_report(decoded("sugar", "right_foreleg_touch"),
                            reaction(0, body_simulation={"status": "ok", "physics_run": True}))
    assert report["status"] == "no_signal"
    assert report["text"] == "我这轮没有收到新的感觉信号。"
    assert report["labels"] == report["feelings"] == []


def test_subthreshold_response_can_support_state_decoder_labels_without_spikes():
    report = sensory_report(decoded("vision_r7:right", status="decoded"), reaction(0, cells=7))
    assert report["status"] == "decoded" and "紫外光" in report["text"]
    assert any("阈下" in note for note in report["limitations"])
    assert "阈下" not in report["text"]


def test_uncalibrated_status_and_unqualified_labels_are_never_sensations():
    report = sensory_report(decoded("cold:left", status="uncalibrated"), reaction())
    assert report["status"] == "uncalibrated" and not report["labels"]
    assert "寒冷" not in report["text"]
    assert sensory_report(decoded("cold:left", unqualified_words=["cold:left"]), reaction())["status"] == "unidentified"
    # Legacy 'uncalibrated score' means no probability calibration, not no probe.
    legacy = decoded("cold", score_type="uncalibrated linear scores; not probabilities")
    assert "寒冷" in sensation_text(legacy, reaction())


def test_unknown_labels_and_injected_descriptions_are_not_forwarded():
    report = sensory_report(decoded("novel_sensation", descriptions={"novel_sensation": "秘密原文"}), reaction())
    assert report["status"] == "unidentified"
    assert "秘密原文" not in json.dumps(report, ensure_ascii=False)
    empty = sensory_report({}, {})
    assert empty["status"] == "unavailable"


@pytest.mark.parametrize("action,phrase", [("left", "向左"), ("right", "向右"), ("forward", "向前"), ("stop", "停止")])
def test_engineering_feedback_is_reception_not_completed_movement(action, phrase):
    report = sensory_report(decoded("command:" + action), reaction())
    assert report["status"] == "command"
    assert report["text"] == f"我收到了“{phrase}”的回传信号。"
    silence = sensory_report(decoded(status="silence"), reaction(0))
    assert silence["text"] == "我这轮没有收到新的感觉信号。"
