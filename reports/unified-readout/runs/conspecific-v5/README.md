# Unified MaleCNS environmental readout: conspecific-v5

Union corpus: the base conditions, the v3 confound controls, the v4 conspecific conditions, and the v5 probe additions (three delivery fixes). 141 unique sessions / 294 windows are collected; the protocol file declares six conditions twice (see "Known authoring error" below).

## What this run was for

`conspecific-v4` left a specific diagnosis: the two conspecific modalities were physically wired but abstained because the fixed source sat at the receptor's initial position and the walking fly left the 0.20 mm field (only 9 cVA and 15 song windows of 228 carried an event). The probe adds three declared fixes, each with a measured path:

| condition family | what changed | measured delivery |
|---|---|---|
| `pheromone_plume_*`, `song_plume_*` | the **borrowed upstream plume grid** (flygym-gymnasium `plume_short.hdf5`, Apache-2.0) sampled at the funiculus/pedicel through `chatflypt/plume_field.py` | `mc_cva` 2/2 windows, `mc_auditory` 2/2 in every completed session |
| `pheromone_partner_*`, `song_partner_*` | a **following source**: the declared offset is preserved but re-anchored to the measured receptor pose each sample | `mc_cva` 2/2, `mc_auditory` 2/2 |
| `nudge_left/right`, `nudge_alternating` | multi-tap trains (3 x 60 ms, 140 ms spacing, strength 1.5) on the **opposite** foreleg, Ramdya-style nudging | `foreleg_touch` 2/2 (one alternating window missed contact) |

## Result

**18 of 30 labels qualified** (14 in v4). The four new ones are exactly the two conspecific modalities: `cva:left`, `cva:right`, `auditory:left`, `auditory:right`. The v4 test failure also disappeared: `foreleg_touch:left/right` are now 20/20 and 21/21 with zero false positives.

Held-out test (98 windows): exact-set accuracy 80/98 = 0.816, micro-F1 0.956, zero-input false positives 0, `physical-test-summary.md` 98 windows with 0 physics failures. One retained test false positive: `cva:left` (precision 0.889, 8/9) - not retuned.

Per-label held-out recall on the new modalities: `cva:right` 0.80, `cva:left` 0.889, `auditory:left` 0.818, `auditory:right` 0.75.

## Borrowed upstream work used here

- The plume **data and grid-indexing math** come from the vendored `flygym-gymnasium` (Apache-2.0, pinned): `plume_short.hdf5` (sha256 `40a76057…`, 60x320x480, upstream Navier-Stokes + Brownian inflow) with the upstream `frame = int(t*fps)`, `cell = mm/cell_mm`, NaN-off-grid convention, ported into `chatflypt/plume_field.py` with explicit engineering choices (arena-scale offset, 0.3 s record wrap, outside-grid = no input).
- The **continuous motor-side readout** shape (regress a continuous task vector, score it with a vector objective) is borrowed from the vendored `flyvis` (MIT) `DecoderGAVP`/`epe`; no flyvis code or weights are used.
- Our MaleCNS graph was **independently reproduced** by the vendored `flm` builder (MIT): 166,700 neurons / 25,582,938 annotated edges / 124,177,617 contacts all agree, and our 24,559,135 modelled edges equal annotated minus the 1,023,803 edges without a fast-transmitter model. Details: `research/borrowed-upstream-code.md`.

## Known authoring error (recorded, not hidden)

The protocol declares `far_pheromone_control` and `far_song_control` **twice per split**: the v4 extension already declared them, and the v5 probe extension added them again. The collector runs whichever declaration comes first and skips the second as already complete, so the corpus holds **one session per id** (141 unique of 147 declared). The frozen protocol file was *not* edited afterwards - that would invalidate every collected session - instead `collect()`/`fit()` now de-duplicate by id and print the duplicates, and the v5 extension no longer re-declares a condition the union plan already carries. Nothing about the collected sessions changes; only the declared count did.

## Release

Candidate verified over real HTTP and published by copying the locked candidate to `data/derived/unified-readout/malecns-decoder.npz` (sha256 `AB76541D…`). Post-release check on the live loader path: `readout_model: general_sensory_unified`, environment updates preserve neural/body state, "左转" yields `foreleg_touch:right` from 4 measured contacts.

This remains finite, small-sample simulation validation with candidate labels; the plume is not a cVA concentration, the following source is a different physical claim from a fixed source, and every unqualified label above is a genuine abstention.
