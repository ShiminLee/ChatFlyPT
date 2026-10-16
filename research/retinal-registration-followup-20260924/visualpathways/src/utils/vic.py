"""OL↔CB and binocular visual-input contribution (VIC) cached data.

Hosts visual-input contribution (VIC) loaders that measure how much of a
CB neuron's input traces back to the OL. Covers:

- OL → CB VIC: `get_ol_cb_vic_raw`, `get_ol_cb_vic_type`, `get_ol_type_vic`,
  `get_vcbn_types`
- Binocular / left-right CB VIC: `get_cb_vic_binocular`,
  `get_cb_vic_binocular_type`, `get_cb_vic_lr_homologue`
- Per-ROI synaptic VIC: `get_roi_syn_vic`
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.cache import cached_parquet
from utils.config import DATA_DIR, DATASET, SIDE_CHAR
from utils.core_data import (
    _SIDE_WORD,
    get_meta,
    get_ol_meta,
    grouped_effweight,
    is_cb_neuron,
)
from utils.ol_data import get_ol_stepsn_sum


def get_ol_type_vic(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-OL-instance median VIC: summed effective weight from visual inputs
    via `lat_flow_sum`, aggregated first per bodyId then median per instance.
    """
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_sum(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        side_word = _SIDE_WORD[side_char]
        inidx = ol_meta[
            (ol_meta.main_groups == "visual input") & (ol_meta.side == side_word)
        ].idx.values
        outidx = ol_meta.idx.values
        idx_to_zero = {i: 0 for i in ol_meta.idx}
        idx_to_bid = dict(zip(ol_meta.idx, ol_meta.bodyId))

        vic = grouped_effweight(
            stepsn, inidx, outidx, idx_to_zero, idx_to_bid,
            combining_method="mean",  # per-bodyId output ⇒ singleton groups
        )
        vic.columns = ["VIC"]
        vic = vic.reset_index(names="bodyId")
        vic["bodyId"] = vic["bodyId"].astype("int64")  # not astype(int): int32 on Windows
        vic = vic.merge(
            ol_meta[["bodyId", "instance"]].drop_duplicates(),
            on="bodyId", how="left",
        )
        return vic.groupby("instance")["VIC"].median().reset_index()

    return cached_parquet(
        f"{dataset}_OL_{side_char}_type_vic", _compute, force=force,
    )


def get_ol_cb_vic_raw(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-body summed effective input weight from visual inputs (VIC) across
    VPNs + CB neurons. Derived from `get_ol_cb_effweight_by_channel` by summing
    over its visual-input channel columns (the total VIC is the per-channel
    breakdown collapsed) — so both share one cached effective-weight build.

    Returns a long df with columns `bodyId, VIC, region, instance,
    main_groups`.
    """
    def _compute():
        table = get_ol_cb_effweight_by_channel(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        side_word = _SIDE_WORD[side_char]
        channels = [c for c in table.columns if c != "bodyId"]
        vic = pd.DataFrame({
            # .astype('int64'), not astype(int): int32 truncation on Windows
            "bodyId": table["bodyId"].astype("int64"),
            "VIC": table[channels].sum(axis=1).values,
        })
        # region: CB bodies -> "CB"; the rest (VPNs on this side) -> "{side} OL"
        cb_bodies = set(meta.loc[is_cb_neuron(meta), "bodyId"])
        vic["region"] = np.where(
            vic["bodyId"].isin(cb_bodies), "CB", f"{side_word} OL"
        )
        vic = vic.merge(
            meta[["bodyId", "instance", "main_groups"]].drop_duplicates(),
            on="bodyId", how="left",
        )
        vic = vic[~pd.isna(vic["instance"])]
        vic = vic[~vic["instance"].str.contains("unclear")]
        return vic.reset_index(drop=True)

    return cached_parquet(
        f"{dataset}_{side_char}_ol_cb_vic_raw", _compute, force=force,
    )


def get_ol_cb_vic_type(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-(instance, region) median VIC for VPN + CB neurons."""
    def _compute():
        vic = get_ol_cb_vic_raw(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        return (
            vic.groupby(["instance", "region"])["VIC"].median()
               .reset_index()
               .sort_values("VIC", ascending=False)
               .reset_index(drop=True)
        )

    return cached_parquet(
        f"{dataset}_{side_char}_ol_cb_vic_type", _compute, force=force,
    )


def get_vcbn_types(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    vic_thre: float = 5e-4, force: bool = False,
) -> pd.DataFrame:
    """CB instances with median VIC above `vic_thre` — the VCBN set."""
    def _compute():
        type_vic = get_ol_cb_vic_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        return type_vic[
            (type_vic["region"] == "CB") & (type_vic["VIC"] > vic_thre)
        ].reset_index(drop=True)

    tag = f"{vic_thre:.1e}".replace(".", "p").replace("-", "m")
    return cached_parquet(
        f"{dataset}_{side_char}_vcbn_types_{tag}", _compute, force=force,
    )


def get_cb_vic_binocular(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    fillna: float = 1e-6, force: bool = False,
) -> pd.DataFrame:
    """Merge per-body right + left VIC for CB neurons.

    Uses `get_ol_cb_vic_raw` on both sides, restricted to `region == 'CB'`,
    merges on bodyId (outer), fills missing VIC with `fillna` so log-scale
    plots stay finite. Returns per-body df with `VIC_r, VIC_l`.
    """
    def _compute():
        right = get_ol_cb_vic_raw(
            dataset=dataset, data_dir=data_dir, side_char="r",
        )
        left = get_ol_cb_vic_raw(
            dataset=dataset, data_dir=data_dir, side_char="l",
        )
        r_cb = right[right["region"] == "CB"].rename(columns={"VIC": "VIC_r"})
        l_cb = left[left["region"] == "CB"].rename(columns={"VIC": "VIC_l"})
        merged = r_cb.merge(
            l_cb[["bodyId", "VIC_l"]], on="bodyId", how="outer",
        )
        merged["VIC_r"] = merged["VIC_r"].fillna(fillna)
        merged["VIC_l"] = merged["VIC_l"].fillna(fillna)
        return merged.reset_index(drop=True)

    return cached_parquet(
        f"{dataset}_cb_vic_binocular", _compute, force=force,
    )


def get_cb_vic_binocular_type(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    fillna: float = 1e-6, force: bool = False,
) -> pd.DataFrame:
    """Per-instance median VIC_r / VIC_l for CB neurons."""
    def _compute():
        raw = get_cb_vic_binocular(
            dataset=dataset, data_dir=data_dir, fillna=fillna,
        )
        return raw.groupby("instance")[["VIC_r", "VIC_l"]].median().reset_index()

    return cached_parquet(
        f"{dataset}_cb_vic_binocular_type", _compute, force=force,
    )


def get_cb_vic_lr_homologue(
    *, dataset: str = DATASET, data_dir=DATA_DIR, force: bool = False,
) -> pd.DataFrame:
    """Cell-type-matched right + left CB VIC (homologue comparison).

    For each side, propagates side-local visual inputs via the side's
    `lat_flow_sum.npz` onto all CB bodies, aggregates per instance via
    `get_ol_cb_vic_raw`, then collapses hemisphere variants by stripping
    `_R`/`_L` from `instance` and taking the **max** VIC per cell_type.
    Inner-merge on cell_type gives one row per bilateral homologue.
    """
    def _compute():
        def _side_cell_type_vic(side_char, col):
            raw = get_ol_cb_vic_raw(
                dataset=dataset, data_dir=data_dir, side_char=side_char,
            )
            cb = raw[raw["region"] == "CB"].copy()
            per_inst = cb.groupby("instance")["VIC"].median().reset_index()
            per_inst["cell_type"] = [
                s[:-2] if s[-2:] in ("_R", "_L") else s
                for s in per_inst["instance"]
            ]
            idx = per_inst.groupby("cell_type")["VIC"].idxmax()
            out = per_inst.loc[idx, ["cell_type", "instance", "VIC"]].rename(
                columns={"instance": f"instance_{side_char}", "VIC": col},
            )
            return out.reset_index(drop=True)

        r = _side_cell_type_vic("r", "VIC_r")
        l = _side_cell_type_vic("l", "VIC_l")
        return r.merge(l, on="cell_type", how="inner")

    return cached_parquet(
        f"{dataset}_cb_vic_lr_homologue", _compute, force=force,
    )


def get_roi_syn_vic(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-synapse (bodyId, roi, VIC) records used by the coverage analysis.

    Combines `tbar_VIC_r.pkl` (continuous VIC on non-OL synapses)
    with the four OL-side ROIs (`ME({S})`, `AME({S})`, `LO({S})`, `LOP({S})`)
    drawn from `{dataset}_OL_{side_char}_roi_counts.pkl`, expanded to
    per-synapse rows with VIC=1 (inside-OL is trivially fully visual).
    """
    def _compute():
        side_upper = side_char.upper()
        tbar = pd.read_pickle(data_dir / "tbar_VIC_r.pkl").rename(
            columns={"vision": "VIC"}
        )

        roi_counts = pd.read_pickle(
            data_dir / f"{dataset}_OL_{side_char}_roi_counts.pkl"
        )
        ol_rois = [
            f"ME({side_upper})", f"AME({side_upper})",
            f"LO({side_upper})", f"LOP({side_upper})",
        ]
        ol_sub = roi_counts[roi_counts.roi.isin(ol_rois)].copy()
        ol_expanded = ol_sub.loc[ol_sub.index.repeat(ol_sub["pre"])].copy()
        ol_expanded["VIC"] = 1.0

        return pd.concat(
            [tbar[["bodyId", "roi", "VIC"]],
             ol_expanded[["bodyId", "roi", "VIC"]]],
            ignore_index=True,
        )

    return cached_parquet(
        f"{dataset}_OL_{side_char}_roi_syn_vic", _compute, force=force,
    )


def load_vp_cb(
    lr: str, *, thr_vic: float = 5e-4, vic_filter: bool = True, data_dir=DATA_DIR,
) -> pd.DataFrame:
    """Load the assembled VPN + VCBN table for side `lr` ('r' / 'l').

    Reads `vp_cb_hit_vic_{lr}.p` (written by
    `setup_data/assemble_vpn_vcbn.py`), optionally keeps only instances whose
    *median* VIC exceeds `thr_vic`, and renames the columns the panel code uses:
    `VIC -> vision`, `hitting_time -> ht`. Returns the (filtered) frame with all
    other columns preserved.
    """
    meta_cb_vpn = pd.read_pickle(Path(data_dir, f"vp_cb_hit_vic_{lr}.p"))
    if vic_filter:
        med = meta_cb_vpn.groupby("instance")["VIC"].median()
        keep = med[med > thr_vic].index
        meta_cb_vpn = meta_cb_vpn[meta_cb_vpn.instance.isin(keep)]
    return meta_cb_vpn.rename(columns={"VIC": "vision", "hitting_time": "ht"})


def get_ol_cb_effweight_by_channel(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-body effective input weight from *each* visual-input channel.

    Rows = target `bodyId` (VPN + CB, matching `get_ol_cb_vic_raw`'s output
    set); columns = the visual-input instances for the side (e.g. ``L1_R`` …
    ``HBeyelet_R``). Each cell is the summed effective weight from that single
    channel to that body, via the full `lat_flow_sum.npz`.

    This is the per-channel breakdown that `get_ol_cb_vic_raw` collapses (it
    sums *all* channels into one number per body); summing this table across
    its channel columns reproduces that total. It replaces the per-channel
    `effwt_visr_{ch}.pkl` files the removed `setup_data/cb_cache/s2_effwt_matrix`
    used to write. Built once and cached (small: bodies × ~8 channels).
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        side_word = _SIDE_WORD[side_char]
        idx_to_zero = {i: 0 for i in meta.index}
        idx_to_bid = dict(zip(meta.index, meta.bodyId))

        channels = sorted(
            meta[(meta.main_groups == "visual input") & (meta.side == side_word)]
            .instance.unique()
        )
        outidx = np.concatenate([
            meta[(meta.region == f"{side_word} OL") & (meta.main_groups == "VPN")].index.values,
            meta[is_cb_neuron(meta)].index.values,
        ])

        cols = {}
        for ch in channels:
            inidx = meta[meta.instance == ch].idx.values
            # inidx_map = idx_to_zero -> collapse the channel's neurons into one
            # group: one summed effective-weight value per target body.
            m = grouped_effweight(
                stepsn, inidx, outidx, idx_to_zero, idx_to_bid,
                combining_method="mean",
            )
            cols[ch] = m.iloc[:, 0]  # single input group; index = str(bodyId)
        table = pd.DataFrame(cols)
        # .astype('int64'), not astype(int): the latter truncates to int32
        # garbage for large bodyIds on Windows.
        table.index = table.index.astype("int64")
        table.index.name = "bodyId"
        return table.reset_index()

    return cached_parquet(
        f"{dataset}_{side_char}_ol_cb_effweight_by_channel", _compute, force=force,
    )


def get_effweight_channel(
    vi: str, meta_cb_vpn: pd.DataFrame, *, thr_vic: float = 5e-4,
    dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
) -> pd.DataFrame:
    """Per-channel VIC table for a single visual input `vi` (e.g. ``'L1_R'``).

    Selects channel `vi` from `get_ol_cb_effweight_by_channel` (one VIC value
    per target `bodyId`), joins the instance / ht / main_groups metadata from
    `meta_cb_vpn`, and keeps only instances with median VIC > `thr_vic`.
    Returns columns `bodyId, VIC, instance, ht, main_groups`.
    """
    table = get_ol_cb_effweight_by_channel(
        dataset=dataset, data_dir=data_dir, side_char=side_char,
    )
    df = pd.DataFrame({"bodyId": table["bodyId"].values, "VIC": table[vi].values})
    df = df.merge(
        meta_cb_vpn[["instance", "bodyId", "ht", "main_groups"]],
        on="bodyId", how="inner",
    )
    med = df.groupby("instance")["VIC"].median()
    keep = med[med > thr_vic].index
    return df[df.instance.isin(keep)]


