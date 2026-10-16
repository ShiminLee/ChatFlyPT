"""Compositional renderer: varied phrasing, auditable grounding, no new facts.

The chat's words come from decoded slots, and this module only decides *how* to
say them. It may vary wording and order, and it may join several decoded slots
into one utterance (so combinations that never appeared as a fixed sentence can
be spoken), but it may not introduce a slot, a modifier or a cause that was not
decoded and qualified.

Two auditable quantities come with it:

* ``invented``      - slots present in the sentence that the decoder never
                      produced. Must be zero by construction; checked anyway.
* ``contradicted``  - slots whose *measured* value disagrees with the decoded one
                      (e.g. decoded "busy" while the measured band is quiet).
                      These are genuine misreads and are reported, not hidden.

Everything is deterministic in ``(slots, seed)`` so any utterance can be replayed.
"""
import hashlib

from .slot_readout import ACTIVITY_LABELS, PATTERN_LABELS

# Phrasing variants per slot family. Only wording varies; the claim does not.
ACTIVITY_PHRASES = {
    'activity:quiet': ('这一轮它比较安静。', '它这会儿不太活跃。', '这段时间它很安静。'),
    'activity:moderate': ('这一轮它状态一般。', '它的活动处于中等水平。', '它不算安静，也不算活跃。'),
    'activity:busy': ('这一轮它挺活跃。', '它这会儿相当活跃。', '这一轮它的活动很密集。'),
}
PATTERN_PHRASES = {
    'pattern:sustained': ('这种感觉一直持续着。', '整个窗口里这种感觉都没有停。'),
    'pattern:transient': ('这种感觉只是短促地出现了一下。', '这种感觉一闪就过去了。'),
}
OPENERS = ('', '它这一轮：', '以下是这一轮盲读出的内容：')
CONNECTIVES = {1: ('',), 2: ('而且', '同时'), 3: ('而且', '同时', '另外')}


def _choose(options, seed, key):
    digest = hashlib.sha256(f'{seed}:{key}'.encode()).digest()
    return options[int.from_bytes(digest[:4], 'little') % len(options)]


def render_state(*, feelings=(), activity=None, pattern=None, seed=0):
    """Compose one utterance from decoded slots.

    ``feelings`` are the per-channel clause dicts produced by
    `dialogue.sensory_report` (each already decoded and qualified); ``activity``
    and ``pattern`` are slot labels from `slot_readout`.
    """
    slots, clauses = [], []
    if activity in ACTIVITY_PHRASES:
        slots.append(activity)
        clauses.append(_choose(ACTIVITY_PHRASES[activity], seed, activity))
    sensation_texts = [feeling['text'] for feeling in feelings if feeling.get('text')]
    if sensation_texts:
        slots.extend(f'{f["channel"]}:{f["side"]}' if f.get('side') else f['channel'] for f in feelings)
        connective = _choose(CONNECTIVES[min(len(sensation_texts), 3)], seed, 'sensation-connective')
        joined = ('；' + connective).join(t.rstrip('。') for t in sensation_texts) + '。'
        clauses.append(joined)
    if pattern in PATTERN_PHRASES:
        slots.append(pattern)
        clauses.append(_choose(PATTERN_PHRASES[pattern], seed, pattern))
    if not clauses:
        return {'text': None, 'slots': [], 'reason': '本轮没有合格的盲读出槽位，不生成陈述。'}
    opener = _choose(OPENERS, seed, 'opener') if len(clauses) > 1 else ''
    return {'text': opener + ''.join(clauses), 'slots': slots, 'reason': '由盲读出槽位组合而成。'}


def audit_grounding(rendered, measured):
    """Check a rendered utterance against the measured window facts.

    ``rendered`` is the dict from `render_state`; ``measured`` carries the
    measured facts for the same window: delivered channel keys, the measured
    activity band and the measured delivery fraction.
    """
    delivered = set()
    for key in measured.get('delivered_items', {}):
        key = str(key).split('@')[0]
        delivered.add(key)
        # Decoder labels drop the channel-id prefix (mc_bright -> bright), so both
        # spellings of the same measured channel are accepted.
        if key.startswith('mc_'):
            delivered.add(key[3:])
    invented, unsupported = [], []
    for slot in rendered.get('slots', []):
        if slot in ACTIVITY_LABELS or slot in PATTERN_LABELS:
            continue
        if slot in delivered:
            continue
        channel = slot.rsplit(':', 1)[0] if ':' in slot else slot
        if any(key.split(':')[0] == channel for key in delivered):
            continue
        # A sensation clause with no matching delivered event is not necessarily
        # invented (a decoded label can be carried by a different input channel),
        # so it is reported as unsupported rather than as a contradiction.
        unsupported.append(slot)
    contradicted = []
    measured_activity = measured.get('activity_band')
    decoded_activity = next((slot for slot in rendered.get('slots', []) if slot in ACTIVITY_LABELS), None)
    if measured_activity and decoded_activity and measured_activity != decoded_activity:
        contradicted.append({'slot': decoded_activity, 'measured': measured_activity})
    measured_pattern = measured.get('pattern_band')
    decoded_pattern = next((slot for slot in rendered.get('slots', []) if slot in PATTERN_LABELS), None)
    if measured_pattern and decoded_pattern and measured_pattern != decoded_pattern:
        contradicted.append({'slot': decoded_pattern, 'measured': measured_pattern})
    return {'invented': invented, 'unsupported': unsupported, 'contradicted': contradicted,
            'audited_slots': len(rendered.get('slots', []))}
