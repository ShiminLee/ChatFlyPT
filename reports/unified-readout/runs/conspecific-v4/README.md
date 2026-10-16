# Unified MaleCNS environmental readout: conspecific-v4

This run adds the conspecific conditions (contact rhythm, cVA field, courtship-song field) on top of the v3 confound controls, so the corpus keeps the equal-geometry platform, darkness, far-source and unilateral controls that the previous batch introduced.

## Preregistered protocol

`protocol.json` declares 108 independent neural/body sessions and 228 windows of 500 ms, split by whole session into training, validation and test (76 windows each). Conditions:

- the 12 base conditions (bright, dark, yeast, dark_yeast, heat, cold, mouth_block, mouth_sugar, mouth_high_salt, leg_sugar, touch_left/right);
- the 12 v3 controls (equal-geometry sugar vs block platform under light and darkness, distant sugar, opposing-eye shade, fixed unilateral yeast/heat/cold sources);
- the 12 conspecific additions: `tap1x180_*`, `tap3x60_*`, `tap6x30_*` (equal 180 ms total contact time, three rhythms) and `pheromone_*`, `song_*`, `far_pheromone_control`, `far_song_control`.

Placement evidence for the two new fields is in `../conspecific-v4-placement.json`: both sources are non-solid, the initial body pose is bit-identical, zero contacts are recorded, and each initial field is unilateral (0.36 ipsilateral / 0.0 contralateral). cVA is sampled at the actual funiculus (Or67d ORNs) and song at the actual pedicel (Johnston's organ); leg chordotonal cells are never used as an auditory proxy.

## Frozen fit and inference rules

Unchanged from v3: Ridge (alpha 10) on internal spike counts and voltage-area statistics with every sensory and registered input cell excluded; targets require both measured receptor reception and delivered events inside the same window; validation alone selects thresholds and label qualification (>=2 positive and >=2 negative windows, precision >= 0.90, recall >= 0.75); the candidate is saved before the test split is opened; test failures are retained.

## Status

### Results (2026-09-29)

14 of 30 declared labels qualified, 16 abstain: `bright:left/right`, `yeast:right`, `warm:right`, `foreleg_touch:left/right`, `midleg_touch:left/right`, `hindleg_touch:left/right`, `leg_sweet_local:left/right`, `leg_appetitive_ascending:left/right`.

Held-out test (76 windows): exact-set accuracy 55/76 = 0.724, micro-F1 0.932, zero-input false positives 0. **Retained test failure: `foreleg_touch:left` has 2 false positives (precision 0.889); nothing was retuned to remove them.** `physical-test-summary.md`: 76 windows, 0 physics failures.

Compared with `retinal-axes-v3`: `yeast:right` is newly qualified (the union corpus carries more yeast windows), and the confound controls are preserved.

### The two new modalities are physically wired but abstain

The corpus proves the fields did reach their receptors: `mc_cva` in 9 windows and `mc_auditory` in 15 of 228 windows. They did not qualify because that is only 1-4 positive windows per split, below the preregistered gates: the fixed source sits at the receptor's initial position and the freely walking fly leaves the 0.20 mm field. `cva:*` and `auditory:*` therefore remain explicit abstentions; qualifying them needs a source that stays with the receptor, a larger field, or dedicated sessions - not a threshold change.

### Tap rhythm: delivery differs, the readout does not separate the rhythms

`reports/corpus/conspecific-v4-tap-rhythms.json` answers the preregistered rhythm question with equal total contact time (180 ms):

| rhythm | windows | measured contact bouts (min/median/max) | internal spikes (median) | touched-leg score (median) | windows spoken |
|---|---|---|---|---|---|
| `tap1x180` | 12 | 0 / 0 / 1 | 12,468 | 0.023 | 4 |
| `tap3x60` | 12 | 0 / 1 / 3 | 12,951 | 0.412 | 6 |
| `tap6x30` | 12 | 0 / 1 / 4 | 16,749 | 0.571 | 7 |

Two honest conclusions:

- The rhythm groups **overlap on every pairwise comparison, so the three rhythms are not distinguishable** by this readout on this corpus (12 windows per rhythm, correlated pairs within a session).
- There is a measured **delivery** difference worth recording: a single 180 ms press lost contact in half of its windows (median 0 bouts) because the probe is aimed once and the fly walks away, while tap trains re-aim at each tap and landed contact in most windows. Pulse trains are the more robust presentation of a brief conspecific contact, and the score median rises with tap count - a graded tendency, not an established rhythm code.

### Release

The candidate was verified over real HTTP (`scripts/check_unified_chat.py --decoder ...malecns-decoder.candidate.npz`) and then published by copying the locked candidate to `data/derived/unified-readout/malecns-decoder.npz` (sha256 `A7749B73...`). The post-release check on the live loader path reports `readout_model: general_sensory_unified`, environment updates that preserve neural/body state, a "左转" turn whose 4 measured contacts yield `foreleg_touch:right` with `requested_taps: 1` and `measured_contact_bouts: 1`, and the newly qualified `yeast:right` label firing in the yeast scene without false positives.

This remains finite, small-sample simulation validation with candidate labels; it is not biological calibration, and every unqualified label above is a genuine abstention.
