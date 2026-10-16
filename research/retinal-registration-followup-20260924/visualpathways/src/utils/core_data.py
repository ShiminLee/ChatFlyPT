"""Foundational cached data loaders shared across every downstream module.

Hosts:
- meta / OL-meta loaders (`get_meta`, `get_ol_meta`, `get_inventory_meta`)
- flow-time loaders (`get_flow`, `get_ol_flow_type`, `get_flow_ol`)
- full propagation matrix (`get_full_prop`)
- retinotopic sector map (`get_sector_map`)
- the `is_cb_neuron` CB-membership predicate
- the private helpers other utils modules import: `_load_meta_groups`,
  `_ensure_ol_sidecars`, `_grouped_paths`, `_swap_lr_suffix`, `_SIDE_WORD`,
  `_LAYER_EXAMPLE_TYPES`, `load_layer_example_types`,
  `_PARTICIPATION_EXAMPLE_TYPES`, `load_participation_example_types`.
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.cache import cached_parquet
from utils.config import DATA_DIR, DATASET, HIT_THRE, N_FLOW, PARAMS_DIR, SIDE_CHAR


_SIDE_WORD = {"r": "right", "l": "left"}

_LAYER_EXAMPLE_TYPES = ['Mi4', 'T4a']

_PARTICIPATION_EXAMPLE_TYPES = ['T4a']


def _load_meta_groups(dataset: str, data_dir) -> pd.DataFrame:
    """Load meta CSV and annotate with main_groups / region."""
    meta = pd.read_csv(data_dir / f"{dataset}_meta.csv")
    meta["cell_type"] = meta["cell_type"].fillna("unknown")
    meta["instance"] = meta["instance"].fillna("unknown")

    # sign=0 unless neurotransmitter is acetylcholine / gaba / glutamate
    idx_0 = meta[~meta.nt.isin(["acetylcholine", "gaba", "glutamate"])].index
    meta.loc[idx_0, "sign"] = 0

    # main group for each cell type (from Nern et al. SuppTable01)
    seed_types = ["L1", "L2", "L3", "R8", "R7", "R8d", "R7d", "HBeyelet"]
    table_df = pd.read_excel(PARAMS_DIR / "Nern-et-al_SuppTable01_Cell-types-and-counts.xlsx")
    table_df["main_groups"] = table_df["main groups"].replace(
        {"ONCN": "VIN", "ONIN": "VIN", "VPN": "VPN", "VCN": "VCN"}
    )
    table_df.loc[
        table_df[table_df["cell type"].isin(seed_types)].index, "main_groups"
    ] = "visual input"

    # if cell type exists on both sides, keep the right-side row
    both_sides = (
        table_df.assign(side=table_df["instance"].str[-2:])
        .groupby("cell type")["side"]
        .apply(lambda x: set(x) >= {"_L", "_R"})
    )
    table_df = table_df[
        ~(
            table_df["cell type"].isin(both_sides[both_sides].index)
            & table_df["instance"].str.endswith("_L")
        )
    ]
    meta = meta.merge(
        table_df[["cell type", "main_groups"]],
        how="left",
        left_on="cell_type",
        right_on="cell type",
    ).drop(columns=["cell type"])

    # R7/R8 are split into p/y in table_df — label their parents directly
    meta.loc[meta[meta.cell_type == "R7"].index, "main_groups"] = "visual input"
    meta.loc[meta[meta.cell_type == "R8"].index, "main_groups"] = "visual input"
    meta["main_groups"] = meta["main_groups"].fillna("nonOL")

    # region
    meta["region"] = "CB"
    meta.loc[
        meta[
            meta.superclass.isin(
                ["cb_intrinsic", "cb_motor", "cb_sensory", "sensory_descending", "descending_neuron"]
            )
        ].index,
        "region",
    ] = "CB"
    meta.loc[
        meta[
            meta.superclass.isin(
                [
                    "vnc_intrinsic", "vnc_motor", "vnc_sensory", "vnc_efferent",
                    "sensory_ascending", "efferent_ascending", "ascending_neuron",
                ]
            )
        ].index,
        "region",
    ] = "VNC"
    right_instances = list(table_df.instance.values) + ["R7_R", "R8_R"]
    meta.loc[meta[meta.instance.isin(right_instances)].index, "region"] = "right OL"
    left_instances = [
        re.sub("_TMP$", "_R", re.sub("_R$", "_L", re.sub("_L$", "_TMP", s)))
        for s in right_instances
    ]
    meta.loc[meta[meta.instance.isin(left_instances)].index, "region"] = "left OL"

    # collapse residual "other" main_groups to nonOL/CB
    idx_other = meta[meta.main_groups == "other"].index
    meta.loc[idx_other, "main_groups"] = "nonOL"
    meta.loc[idx_other, "region"] = "CB"

    return meta


def get_meta(*, dataset: str = DATASET, data_dir=DATA_DIR, force: bool = False) -> pd.DataFrame:
    """Cached meta with `main_groups`, `region`, cleaned `cell_type`/`instance`."""
    return cached_parquet(
        f"{dataset}_meta",
        lambda: _load_meta_groups(dataset, data_dir),
        force=force,
    )


def is_cb_neuron(meta: pd.DataFrame) -> pd.Series:
    """CB-analysis mask: anatomical CB-region neurons"""
    return meta["region"] == "CB"


def get_flow(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    side_char: str = SIDE_CHAR, n_flow: int = N_FLOW, hit_thre: float = HIT_THRE,
    force: bool = False,
) -> pd.DataFrame:
    """Per-neuron hitting times; adds per-instance `count`."""
    def _compute():
        fp = data_dir / f"{dataset}_{side_char}_flow_{n_flow}step_{hit_thre}thre_hit.csv"
        flow = pd.read_csv(fp)
        flow["count"] = flow.groupby("instance")["instance"].transform("count")
        return flow

    return cached_parquet(f"{dataset}_{side_char}_flow", _compute, force=force)


def get_flow_per_group(
    *, dataset: str = DATASET, data_dir=DATA_DIR,
    side_char: str = SIDE_CHAR, n_flow: int = N_FLOW, hit_thre: float = HIT_THRE,
    force: bool = False,
) -> pd.DataFrame:
    """Per-instance median hitting times. Reads `..._hit_per_group.csv`
    written by `fetch_instance_flow`."""
    def _compute():
        fp = data_dir / f"{dataset}_{side_char}_flow_{n_flow}step_{hit_thre}thre_hit_per_group.csv"
        return pd.read_csv(fp)

    return cached_parquet(f"{dataset}_{side_char}_flow_per_group", _compute, force=force)


def _ensure_ol_sidecars(
    dataset: str, data_dir, side_char: str,
    n_flow: int = N_FLOW, hit_thre: float = HIT_THRE,
):
    """Materialise `OL_{side_char}_*` sidecar files in `data/` if missing.

    Builds from the full-brain files: takes
    region == '{side} OL' & main_groups != 'other' & instance != 'R1-R6_R',
    reindexes to local 0..N-1 `idx`, keeps original full-brain idx as `idx_full`,
    and slices prop / lat_flow_0 / lat_flow_sum accordingly.
    """
    ol_meta_f = data_dir / f"{dataset}_OL_{side_char}_meta.csv"
    ol_flow_f = data_dir / f"{dataset}_OL_{side_char}_flow_{n_flow}step_{hit_thre}thre_hit.csv"
    ol_prop_f = data_dir / f"{dataset}_OL_{side_char}_prop.npz"
    ol_sum_f  = data_dir / f"{dataset}_OL_{side_char}_lat_flow_sum.npz"
    ol_0_f    = data_dir / f"{dataset}_OL_{side_char}_lat_flow_0.npz"

    targets = [ol_meta_f, ol_flow_f, ol_prop_f, ol_sum_f, ol_0_f]
    if all(p.exists() for p in targets):
        return

    side_word = _SIDE_WORD[side_char]
    meta = get_meta(dataset=dataset, data_dir=data_dir)
    idx_sub = meta[
        (meta.region == f"{side_word} OL")
        & (meta.main_groups != "other")
        & (meta.instance != "R1-R6_R")
    ].index.values
    idx_sub = np.sort(idx_sub)

    if not ol_meta_f.exists():
        meta_ol = (
            meta.set_index("idx").loc[idx_sub]
            .reset_index(names="idx_full")
            .reset_index(names="idx")
        )
        meta_ol.to_csv(ol_meta_f, index=False)

    if not ol_flow_f.exists():
        flow = pd.read_csv(
            data_dir / f"{dataset}_{side_char}_flow_{n_flow}step_{hit_thre}thre_hit.csv"
        )
        flow = flow.merge(
            meta[["idx", "sign", "main_groups", "cell_type"]], on="idx", how="right"
        )
        flow_sub = flow.set_index("idx").loc[idx_sub].reset_index()
        flow_ol = (
            flow_sub.groupby("instance")
            .agg({
                "cell_type": "first", "hitting_time": "median", "sign": "first",
                "main_groups": "first", "idx": "count",
            })
            .reset_index()
            .rename(columns={"idx": "count"})
        )
        flow_ol.to_csv(ol_flow_f, index=False)

    if not ol_prop_f.exists():
        prop = sp.load_npz(data_dir / f"{dataset}_prop.npz")
        sp.save_npz(ol_prop_f, prop[idx_sub][:, idx_sub])

    if not ol_0_f.exists():
        steps_0 = sp.load_npz(data_dir / f"{dataset}_{side_char}_lat_flow_0.npz")
        sp.save_npz(ol_0_f, steps_0[idx_sub][:, idx_sub])

    if not ol_sum_f.exists():
        steps_sum = sp.load_npz(data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz")
        sp.save_npz(ol_sum_f, steps_sum[idx_sub][:, idx_sub])


def get_ol_meta(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """OL-subset meta (sidecar in `data/`, built from full-brain files if missing)."""
    def _compute():
        _ensure_ol_sidecars(dataset, data_dir, side_char)
        return pd.read_csv(data_dir / f"{dataset}_OL_{side_char}_meta.csv")

    return cached_parquet(f"{dataset}_OL_{side_char}_meta", _compute, force=force)


def get_ol_flow_type(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    n_flow: int = N_FLOW, hit_thre: float = HIT_THRE, force: bool = False,
) -> pd.DataFrame:
    """Per-OL-instance median hitting time (sidecar in `data/`, built if missing)."""
    def _compute():
        _ensure_ol_sidecars(dataset, data_dir, side_char, n_flow=n_flow, hit_thre=hit_thre)
        return pd.read_csv(
            data_dir / f"{dataset}_OL_{side_char}_flow_{n_flow}step_{hit_thre}thre_hit.csv"
        )

    return cached_parquet(f"{dataset}_OL_{side_char}_flow_type", _compute, force=force)


def load_layer_example_types(side_char: str = SIDE_CHAR) -> list:
    """Cell types used as "layer example" pathway subjects."""
    suffix = '_R' if side_char == 'r' else '_L'
    return [t + suffix for t in _LAYER_EXAMPLE_TYPES]


def load_participation_example_types(side_char: str = SIDE_CHAR) -> list:
    """Cell types used as participation-pathway example subjects (Main Fig. 3)."""
    suffix = '_R' if side_char == 'r' else '_L'
    return [t + suffix for t in _PARTICIPATION_EXAMPLE_TYPES]


def _grouped_paths(stepsn, inidx, outidx, n_interm: int, idx_to_instance: dict,
                   outprop: bool = False) -> list:
    """Collect per-length `find_paths_of_length` results, grouped by instance.

    `outprop=False` (default) gives input-normalised edges (fraction of post's
    input coming from pre-type). `outprop=True` gives output-normalised edges
    (fraction of pre's output going to post-type).
    """
    from connectome_interpreter.path_finding import find_paths_of_length, group_paths
    all_paths = []
    for k in range(1, n_interm + 1):
        paths = find_paths_of_length(stepsn, inidx, outidx, k)
        if paths is None or paths.empty:
            continue
        paths = group_paths(
            paths, pre_group=idx_to_instance, post_group=idx_to_instance,
            combining_method='mean', outprop=outprop,
        )
        all_paths.append(paths)
    return all_paths


def grouped_effweight(
    stepsn, inidx, outidx, inidx_map, outidx_map,
    combining_method: str = "median",
) -> pd.DataFrame:
    """Grouped effective weight from a lateral-flow-sum matrix.

    Sums the effective weight over input neurons within each `inidx_map` group,
    then combines (``"median"`` / ``"mean"`` / ``"sum"``) over output neurons
    within each `outidx_map` group. Returns rows = output groups, cols = input
    groups. Applies no thresholding, sorting, or printing.

    Single primitive behind the visual-input effective-weight family
    (clustering feature vectors, per-body VIC): with
    ``combining_method="median"`` and instance→instance maps it reproduces the
    clustering feature vectors exactly; it is otherwise equivalent to
    `connectome_interpreter.result_summary` (whose default combine is
    ``"mean"``) without that function's display / sort / threshold side effects.
    Note ``"mean"`` vs ``"median"`` only differs when an output group has more
    than one neuron (i.e. output grouped by type, not bodyId).
    """
    if combining_method not in ("median", "mean", "sum"):
        raise ValueError(
            "combining_method must be 'median', 'mean', or 'sum'; "
            f"got {combining_method!r}"
        )
    # slice rows first: cheap for CSR and shrinks the matrix before the
    # expensive column indexing (columns-first scans all nonzeros)
    in_out_df = pd.DataFrame(
        data=stepsn[inidx, :][:, outidx].toarray(),
        index=[str(inidx_map[k]) for k in inidx],
        columns=outidx,
    )
    in_out_df = in_out_df.groupby(in_out_df.index).sum()
    in_out_df.columns = [str(outidx_map[k]) for k in in_out_df.columns]
    grouped = in_out_df.T.groupby(level=0)
    return getattr(grouped, combining_method)()


def get_full_prop(*, dataset: str = DATASET, data_dir=DATA_DIR):
    """Full sparse connectivity matrix."""
    return sp.load_npz(data_dir / f"{dataset}_prop.npz")


def _swap_lr_suffix(s: str) -> str:
    """Flip trailing `_L`↔`_R` suffix; leave anything else alone."""
    return re.sub("_TMP$", "_R", re.sub("_R$", "_L", re.sub("_L$", "_TMP", s)))


def get_inventory_meta(
    *, dataset: str = DATASET, data_dir=DATA_DIR, force: bool = False
) -> pd.DataFrame:
    """Meta filtered and `type` canonicalised for inventory pies.

    Drops left OL and 'other' regions, drops `*_unclear` and `R1-R6` rows, and
    collapses R7p/R7y/R8p/R8y variants into R7/R8.
    """
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir).copy()
        meta["type"] = meta["cell_type"]
        # meta = meta[(meta.region != "other") & (meta.region != "left OL")]
        meta = meta[(~meta.type.str.contains("unclear")) & (meta.type != "R1-R6")]
        meta["type"] = meta["type"].replace(
            {"R7p": "R7", "R7y": "R7", "R8p": "R8", "R8y": "R8"}
        )
        return meta

    return cached_parquet(f"{dataset}_inventory_meta", _compute, force=force)


def get_flow_ol(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-neuron flow restricted to OL instances, with per-instance median joined."""
    def _compute():
        meta = get_meta(dataset=dataset, data_dir=data_dir)
        flow = get_flow(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow = flow.merge(meta[["idx", "main_groups", "sign", "bodyId"]], on="idx", how="right")
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow = flow[flow.instance.isin(ol_meta.instance.unique())]
        flow_type = get_ol_flow_type(dataset=dataset, data_dir=data_dir, side_char=side_char)
        flow = flow.merge(
            flow_type[["instance", "hitting_time"]], on="instance", how="left", suffixes=("", "_median")
        )
        return flow

    return cached_parquet(f"{dataset}_OL_{side_char}_flow_neurons", _compute, force=force)


def get_sector_map(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Assign each visual-column coord to one of 5 retinotopic sectors."""
    def _compute():
        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        df = ol_meta[ol_meta.instance.str.startswith("L2")][["coords"]].drop_duplicates().reset_index(drop=True)
        df = df[~pd.isna(df["coords"])].copy()
        xy = df["coords"].str.split(",", expand=True).astype(int).values
        df["hex1_id"] = (xy[:, 1] - xy[:, 0]) / 2
        df["hex2_id"] = (xy[:, 0] + xy[:, 1]) / 2
        xy2 = xy.astype(float).copy()
        xy2[:, 0] = xy2[:, 0] - 1
        xy2[:, 1] = (xy2[:, 1] - 39) / np.sqrt(3) / 1.3

        df["sector"] = 0
        df.loc[df[(df["hex1_id"] > 21) & (df["hex2_id"] > 24)].index, "sector"] = 2
        df.loc[df[(df["hex1_id"] > 15) & (df["hex2_id"] <= 24)].index, "sector"] = 1
        df.loc[df[(df["hex1_id"] <= 21) & (df["hex2_id"] > 16)].index, "sector"] = 3
        df.loc[df[(df["hex1_id"] <= 15) & (df["hex2_id"] <= 16)].index, "sector"] = 4
        df.loc[df[xy2[:, 0] ** 2 + xy2[:, 1] ** 2 < 40].index, "sector"] = 5

        manual_5 = [
            "-5,47", "-5,45", "-5,33", "-5,31", "7,47", "7,45", "7,33", "7,31",
            "-3,51", "-1,53", "1,55", "3,53", "5,51",
            "-3,27", "-1,25", "1,23", "3,25", "5,27",
        ]
        df.loc[df.coords.isin(manual_5), "sector"] = 5
        return df[["coords", "sector"]].reset_index(drop=True)

    return cached_parquet(f"{dataset}_coords_to_sectors", _compute, force=force)
