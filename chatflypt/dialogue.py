"""Chinese sensory narration from blind readouts.

This module has no API client, input parser, stimulus catalogue access, or model
execution. User text, intended stimuli, descriptions, targets and history are
deliberately never forwarded to any external model.
"""
from collections.abc import Mapping
import math
import re


# These labels are decoder outputs, never input-text aliases. Descriptions are
# intentionally local: arbitrary decoded.descriptions cannot inject facts.
_LABELS = {
    "sugar": ("taste", "甜味"), "sugar_specific": ("taste", "甜味"),
    "bitter": ("taste", "苦味"), "water": ("taste", "水"),
    "water_specific": ("taste", "水"), "salt": ("taste", "盐味"),
    "low_salt": ("taste", "盐味"), "high_salt": ("taste", "高盐相关味道"),
    "fatty_acid_taste": ("taste", "己酸相关味道"),
    "sweet_water": ("ambiguous_taste", "糖或水相关的味觉"),
    "leg_sugar": ("signal", "腿部甜味相关信号"),
    "leg_sweet_local": ("signal", "腿部甜味相关信号"),
    "leg_appetitive_ascending": ("signal", "腿部适口味觉上行信号"),
    "vinegar": ("odor", "醋相关气味"), "yeast": ("odor", "酵母相关气味"),
    "geosmin": ("odor", "土臭素相关气味"),
    "cva": ("odor", "cVA 信息素相关气味"), "pheromone": ("odor", "cVA 信息素相关气味"),
    "co2": ("signal", "二氧化碳相关信号"),
    "warm": ("condition", "温暖"), "cold": ("condition", "寒冷"),
    "humid": ("condition", "潮湿"), "moist": ("condition", "潮湿"),
    "dry": ("condition", "干燥"), "evaporative_cooling": ("signal", "蒸发冷却相关信号"),
    "bright": ("light", "光"), "vision_r7": ("light", "紫外光相关信号"),
    "vision_r8": ("light", "蓝／绿光相关信号"),
    "vision_dra": ("light", "偏振光相关信号"), "vision_ocelli": ("light", "单眼光照信号"),
    "sound": ("sound", "声音"), "auditory": ("sound", "声音"),
    "courtship_pulse": ("sound", "求偶脉冲声相关信号"),
    "wind": ("signal", "风相关信号"), "wind_gravity": ("signal", "风或重力相关信号"),
    "touch": ("touch", ""), "foreleg_touch": ("touch", "前腿"),
    "midleg_touch": ("touch", "中腿"), "hindleg_touch": ("touch", "后腿"),
    "head_touch": ("touch", "头部"), "eye_touch": ("touch", "眼睛附近"),
    "wing_touch": ("touch", "翅膀"), "body_touch": ("touch", "身体"),
    "chordotonal_proprioception": ("proprioception", "关节"),
    "hair_plate_proprioception": ("proprioception", "关节位置"),
    "campaniform_proprioception": ("proprioception", "身体受力"),
    "haltere_proprioception": ("proprioception", "平衡棒"),
    "leg_proprioception": ("proprioception", "腿部"),
}
_SIDES = {None: "", "left": "左侧", "right": "右侧", "bilateral": "两侧"}
_COMMAND_TEXT = {"left": "向左", "right": "向右", "forward": "向前", "stop": "停止"}
_BODY_PARTS = {"foreleg_touch": "foreleg", "midleg_touch": "midleg", "hindleg_touch": "hindleg",
               "head_touch": "head", "eye_touch": "eye", "wing_touch": "wing", "body_touch": "body",
               "leg_proprioception": "leg", "haltere_proprioception": "haltere"}


def _count(value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    return int(value) if math.isfinite(value) and 0 <= value <= 10**15 and value == int(value) else None


def _label(value):
    if not isinstance(value, str) or len(value) > 100:
        return None
    if value in ("left_foreleg_touch", "right_foreleg_touch"):
        return "foreleg_touch", value.split("_", 1)[0]
    parts = value.split(":")
    if len(parts) == 2 and parts[0] == "command" and parts[1] in _COMMAND_TEXT:
        return "command", parts[1]
    if len(parts) == 1 and value in _LABELS:
        return value, None
    if len(parts) == 2 and parts[0] in _LABELS and parts[1] in ("left", "right", "bilateral"):
        return parts[0], parts[1]
    return None


def _feeling(channel, side):
    if channel == "command":
        return {"channel": "command", "side": None, "body_part": None,
                "command": side, "text": f"我收到了“{_COMMAND_TEXT[side]}”的回传信号。"}
    category, noun = _LABELS[channel]
    where = _SIDES[side]
    prefix = "我" + ("在" + where if where else "")
    if category == "touch":
        if noun in ("前腿", "中腿", "后腿"):
            body = {None: "", "left": "左", "right": "右", "bilateral": "两侧"}[side] + noun
        else:
            body = where + noun
        text = "我感到" + (body + "被碰到了。" if body else "了触碰。")
    elif category == "odor":
        text = prefix + "闻到了" + noun + "。"
    elif category == "taste":
        text = prefix + "尝到了" + noun + "。"
    elif category == "sound":
        text = prefix + "听到了" + noun + "。"
    elif category == "condition":
        text = prefix + "感到" + noun + "。"
    elif category == "ambiguous_taste":
        text = prefix + "感到糖或水相关的味觉，暂时分不清是哪一种。"
    elif category == "proprioception":
        text = "我察觉到了" + where + noun + "的本体感觉信号。"
    else:
        text = prefix + "察觉到了" + noun + "。"
    return {"channel": channel, "side": side, "body_part": _BODY_PARTS.get(channel), "text": text}


def sensory_report(decoded, reaction):
    """Render only recognized blind labels with measured internal signal.

    Old decoders omit status; their selected words remain supported. A linear
    score's 'uncalibrated' score_type means not a probability, and is different
    from status='uncalibrated', which forbids any sensory assertion.
    """
    if not isinstance(decoded, Mapping) or not isinstance(reaction, Mapping):
        raise TypeError("decoded and reaction must be mappings")
    spikes = _count(reaction.get("non_sensory_spikes"))
    membrane = reaction.get("membrane_response", {})
    cells = _count(membrane.get("internal_cells")) if isinstance(membrane, Mapping) else None
    activity = {"non_sensory_spikes": spikes, "membrane_internal_cells": cells}
    result = {"schema": "chatflypt.sensory-dialogue.v1", "text": "", "status": "unavailable",
              "labels": [], "feelings": [], "activity": activity,
              "limitations": ["第一人称是模拟感觉读出的表达，不是主观体验的证据。",
                              "侧别是读出的感觉入口侧，不等于物体位置或运动方向。"]}
    if spikes is None and cells is None:
        result["text"] = "我还没有可用的感觉读出。"
        return result
    if not (spikes or cells):
        result.update(status="no_signal", text="我这轮没有收到新的感觉信号。")
        return result
    if decoded.get("status") == "uncalibrated":
        result.update(status="uncalibrated", text="我有了内部反应，但还不能确定是什么感觉。")
        return result
    if decoded.get("status") not in (None, "decoded", "command", "ok"):
        result.update(status="unavailable", text="我有了内部反应，但这次感觉读出不可用。")
        return result
    words = decoded.get("words", [])
    if not isinstance(words, (list, tuple, set, frozenset)):
        raise TypeError("decoded.words must be a collection of predicted labels")
    unqualified = decoded.get("unqualified_words", [])
    unqualified = {w for w in unqualified if isinstance(w, str)} if isinstance(unqualified, (list, tuple, set, frozenset)) else set()
    seen = set()
    for word in sorted(w for w in words if isinstance(w, str)):
        label = _label(word)
        if label is None or word in unqualified or label in seen:
            continue
        seen.add(label)
        result["labels"].append(word)
        result["feelings"].append(_feeling(*label))
    if not result["feelings"]:
        result.update(status="unidentified", text="我有了内部反应，但还分辨不出具体感觉。")
    else:
        result["status"] = "command" if all(f["channel"] == "command" for f in result["feelings"]) else "decoded"
        # Avoid repeating identical sentences from synonymous legacy labels.
        result["text"] = "".join(dict.fromkeys(f["text"] for f in result["feelings"]))
    if not spikes and cells:
        result["limitations"].append("本轮依据为记录到的阈下膜电位变化，没有内部神经脉冲。")
    if unqualified:
        result["limitations"].append("未通过验证的候选标签未作为感觉陈述。")
    return result


def sensation_text(decoded, reaction):
    return sensory_report(decoded, reaction)["text"]


def observation_sensation(observation):
    """Apply the same distribution qualification in reception and chat."""
    report = sensory_report(observation.get('decoded', {}), observation.get('reaction', {}))
    if observation.get('embodied_feedback') and report['status'] == 'decoded':
        report['status'] = 'candidate'
        adapted=observation.get('decoded',{}).get('feedback_adaptation',{}).get('applied',False)
        report['feedback_partially_validated']=bool(adapted)
        report['text'] = ('候选感觉读出（连续反馈仅有限验证）：' if adapted else '候选感觉读出（连续反馈试次尚未校准）：') + report['text']
        report['limitations'].append('部分标签经过有限模拟场景的分会话验证；不代表全词汇、所有环境或生理校准。' if adapted else '已有读出器在独立人工刺激上验证；未以连续感觉反馈试次重新验证。')
    return report


def _dataset(value):
    if isinstance(value, str) and re.fullmatch(r"(?:MaleCNS v1\.0|FlyWire (?:v)?(?:630|783))", value):
        return value
    return "unspecified"


def _ambiguous(report):
    channels = {}
    for item in report["feelings"]:
        channels.setdefault(item["channel"], set()).add(item["side"])
    for a, b in (("warm", "cold"), ("moist", "dry"), ("humid", "dry")):
        if a in channels and b in channels:
            if channels[a] & channels[b] or None in channels[a] | channels[b] or "bilateral" in channels[a] | channels[b]:
                return True
    return False
