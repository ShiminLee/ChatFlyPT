# Unified MaleCNS environmental readout: retinal-axes-v3

This is a new corpus after replacing the previous bounding-box visual bridge with the author-atlas eye-axis map and repairing measurement of the merged left eye body. No receptors-v2 observations or decoder weights are reused.

## Preregistered protocol

`protocol.json` declares 72 independent neural/body sessions and 156 windows of 500 ms, split by whole session into training, validation and test (52 windows in each split). The original 84 windows retain their scene descriptions and seeds. The additional 72 windows use independent seeds:

- Same-sized ordinary block versus sugar platforms, plus sugar and ordinary block platforms in darkness, separate contact and visual appearance from candidate leg taste. The dark sugar platform matches the dark block geometry exactly.
- A distant visible sugar object is a no-planned-contact control. Actual contacts, if any, remain measured.
- Fixed attempted left/right visual, yeast-odor, warm and cold conditions test previously confounded laterality. Planned side is never used as a target. The freely moving animal can leave a source or encounter it with either side.

The intended source placements use actual compiled MuJoCo eye camera positions, funiculus body positions for odor, and arista body positions for temperature. Full placement evidence is in `../../retinal-axes-v3-placement.json` and `../../placement-checks-v3/`.

Odor is a 0.05 mm cube fixed 0.12 mm forward and 0.08 mm outward from each neutral funiculus. Its center distance is 0.144 mm and nearest cube point is 0.110 mm from the receptor position. Closer candidates were rejected: one contacted head/arista, and a second altered the warmup pose despite no final contact. The chosen cube produced zero contacts and exactly unchanged initial body positions; its saved warmup scan also showed identical qpos. The engineering odor range is 0.20 mm, initially yielding 0.0778 ipsilateral field and zero contralateral field. This is a seed-42 placement check, not a guarantee against contacts later in a freely moving trial.

Warm/cold sources are non-solid, fixed 0.08 mm outward and 0.04 mm upward from the neutral arista with range 0.16 mm. Initial ipsilateral field is 0.1945 and contralateral field zero. These fields are not calibrated concentrations or degrees Celsius. A 0.25 mm-wide fixed shade covers the opposite eye camera at initialization; it does not follow the fly. Whole-scene dark controls cover the standard arena.

## Frozen fit and inference rules

Training fits Ridge (alpha 10) using counts and voltage-area statistics from internal neurons only. Every sensory neuron and all registered input cells, including advanced proprioceptive populations, are excluded. Input text, environmental objects, position, intended side, measured receptor events and labels are never inference features. Targets are offline annotations requiring both measured receptor population reception and matching delivered events within the same sampling window.

Validation alone chooses label thresholds on the preregistered 0.20–0.80 grid and qualifies each label (at least two positive and two negative windows, precision at least 0.90, recall at least 0.75). The candidate file is saved before opening test records. Test failures are retained, not tuned away. True zero internal activity must remain silent. Unqualified labels cannot become chat words, and one unqualified label does not disable other qualified labels.

The model contract binds graph arrays, sensory/input populations, blindmask, actual inference code, neural/transduction/body code, spatial-map artifacts and all current author-atlas/camera/Retina dependencies. Frozen sources are copied into the corpus snapshot. Source changes invalidate the artifact. The 25 dedicated regression tests passed before collection.

No candidate is automatically published as the default. The parent task will test actual HTTP/session/environment behavior with the saved candidate before considering release. This classifier validation is finite, small-sample simulation validation; it is not biological sensory or behavior calibration, learned human language, or unrestricted intent decoding.

## Status

Protocol and placement were frozen before collection. Collection results and held-out metrics will be appended without rewriting this preregistration.

### Collection (2026-09-29)

All 72 sessions / 156 windows are complete. Shard 4 had died silently on 2026-09-24 after a MuJoCo worker exited without a traceback (6 concurrent collectors); it was resumed in place on 2026-09-29 with a single collector, which re-ran only the 8 missing sessions and left the 64 already-complete ones untouched (their `complete.json` corpus fingerprints still matched the frozen contract).

### Fit, gates and release (2026-09-29)

`evaluation.json` records 13 qualified / 13 unqualified labels of 26. Held-out test (52 windows): exact-set accuracy 37/52 = 0.712, micro-F1 0.931, **zero false positives**, `serving_matches_test_matrix: true`, zero-input silence enforced by the fit. `physical-test-summary.md` joins the frozen predictions with the real physical records: 52 windows, **0 physics failures**. Unqualified labels (`yeast:*`, `warm:left`, `cold:*`, `body_touch:*`, `wing_touch:*`, `sugar_specific:*`, `high_salt:*`) stay in `unqualified_words` and never become chat words.

The candidate was tested over real HTTP before release (`scripts/check_unified_chat.py --decoder .../malecns-decoder.candidate.npz`) and produced no false positives against the offline receptor labels. It was then published by copying the locked candidate to the default path `data/derived/unified-readout/malecns-decoder.npz`, and the same check was re-run without `--decoder` so the live loader path is exercised: `readout_model: general_sensory_unified`, environment updates preserving neural/body state, and the "左转" turn presenting a right-foreleg probe whose 4 measured contacts yielded `foreleg_touch:right`.

This remains finite, small-sample simulation validation with candidate labels; it is not biological calibration, and the unqualified labels above are genuine abstentions rather than untested guesses.
