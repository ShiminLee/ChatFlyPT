"""Download and assemble malecns meta + sparse matrices from neuprint.

Produces `malecns_{tag}_meta.csv`, `prop.npz`, `inprop.npz` in `cache/data/`.
Consumed by `setup_data/s1_malecns_data.ipynb`. Uses tot_weight normalization
(not upstream), and instance override from cell_type + side letter. The
dataset-agnostic lateral-flow builders live in `flow.py`.
"""
from pathlib import Path

import numpy as np
import pandas as pd
import scipy as sp
from scipy import spatial
from scipy.sparse import coo_matrix, csc_matrix

from neuprint import (
    Client,
    NeuronCriteria as NC,
    NotNull,
    SynapseCriteria as SC,
    fetch_adjacencies,
    fetch_neurons,
    fetch_synapses,
)


NT_SIGN = {
    "acetylcholine": 1,
    "glutamate": -1,
    "gaba": -1,
    "dopamine": 0,
    "serotonin": 0,
    "octopamine": 0,
    "histamine": 0,
    "unclear": 0,
}

SIDE_WORD = {"left": "L", "right": "R", "noside": "0"}


def _eyemap(data_dir: Path) -> Path:
    from utils.config import PARAMS_DIR
    return PARAMS_DIR


# === 1. Connectivity ===


def fetch_connectivity(client: Client | None = None) -> pd.DataFrame:
    """Download all Neuron→Neuron adjacencies (type IS NOT NULL on both ends),
    drop every LA(R)/LA(L) row, and collapse per bodyId pair into a flat
    edge list."""
    neu, adj = fetch_adjacencies(NC(type=NotNull), NC(type=NotNull), client=client)
    in_lamina = adj["roi"].isin(["LA(R)", "LA(L)"])

    # Optional, keep the L1/L2/L3 ↔ L4 reciprocal lamina-cartridge edges 
    # It will change tot_weight normalization and every downstream
    # cache, so it would require a full s1→s5 rebuild (and the FlyWire analog
    # `s1_flywirefafb_data.py` mirrors the drop-all behavior — update both).
    # type_by_bid = dict(zip(neu["bodyId"], neu["type"]))
    # pre_t = adj["bodyId_pre"].map(type_by_bid)
    # post_t = adj["bodyId_post"].map(type_by_bid)
    # keep_lamina = in_lamina & (
    #     (pre_t.isin(["L1", "L2", "L3"]) & (post_t == "L4"))
    #     | ((pre_t == "L4") & post_t.isin(["L1", "L2", "L3"]))
    # )
    
    adj = adj.loc[(~in_lamina), ["bodyId_pre", "bodyId_post", "weight"]]
    adj["weight"] = adj["weight"].astype(np.int32)
    return adj.groupby(["bodyId_pre", "bodyId_post"], as_index=False, sort=False)["weight"].sum()


def _remint_r78_bodyids(df: pd.DataFrame, side: str) -> pd.DataFrame:
    """Rewrite R7/R8 bodyIds in a filled-in edge list to synthetic 10-digit IDs.

    Convention: `1` + 5×marker + last 4 digits of the original bodyId.
    For the right OL, marker == r78_int. For the left OL, marker == r78_int - 2
    (7→5, 8→6) to avoid collision with the right-side synthetic IDs.
    """
    assert side in ("R", "L")
    df = df.copy()
    df["bodyId_pre"] = df["bodyId_pre"].astype(str)
    df["bodyId_post"] = df["bodyId_post"].astype(str)
    for inout in ("pre", "post"):
        for r78_int in (7, 8):
            marker = r78_int if side == "R" else r78_int - 2
            prefix_old = str(r78_int) * 6
            prefix_new = "1" + str(marker) * 5
            col = f"bodyId_{inout}"
            mask = df[col].str.startswith(prefix_old)
            df.loc[mask, col] = prefix_new + df.loc[mask, col].str[6:]
            df.loc[mask, f"type_{inout}"] = f"R{r78_int}"
            df.loc[mask, f"instance_{inout}"] = f"R{r78_int}_{side}"
    df["bodyId_pre"] = df["bodyId_pre"].astype(int)
    df["bodyId_post"] = df["bodyId_post"].astype(int)
    return df


def load_r78_fill(data_dir: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load and remint R7/R8 filled-in edge lists for the right and left OL."""
    em = _eyemap(data_dir)
    right = pd.read_csv(em / "edge_list_OLR_filled_in_082925_v1.csv.gz", compression="gzip")
    left = pd.read_csv(em / "edge_list_OLL_filled_in_082925_v1.csv.gz", compression="gzip")
    return _remint_r78_bodyids(right, "R"), _remint_r78_bodyids(left, "L")


def merge_r78_fill(
    conn_df: pd.DataFrame,
    color_df_R: pd.DataFrame,
    color_df_L: pd.DataFrame,
) -> pd.DataFrame:
    """Replace rows in `conn_df` whose (pre, post) appears in either filled-in
    edge list, then append the filled-in rows."""
    fills = pd.concat([color_df_R, color_df_L], ignore_index=True).drop_duplicates(
        subset=["bodyId_pre", "bodyId_post"], keep="last"
    )
    conn_keys = pd.MultiIndex.from_arrays([conn_df["bodyId_pre"], conn_df["bodyId_post"]])
    fill_keys = pd.MultiIndex.from_arrays([fills["bodyId_pre"], fills["bodyId_post"]])
    keep = ~conn_keys.isin(fill_keys)
    return pd.concat([conn_df.loc[keep], fills], ignore_index=True)


def get_sorted_nodes(conn_df: pd.DataFrame) -> list[int]:
    return sorted(set(conn_df["bodyId_pre"]) | set(conn_df["bodyId_post"]))


# === 2. Neuron metadata ===


def fetch_neuron_metadata(bodyIds, client: Client) -> pd.DataFrame:
    """Fetch bodyId, instance, cell_type, nt, superclass, subclass, class,
    dimorphism."""
    cql = f"""
        WITH {list(bodyIds)} as bodyIds
        MATCH(n:Neuron)
        WHERE n.bodyId in bodyIds
        RETURN n.bodyId as bodyId, n.instance as instance, n.type as cell_type,
               n.consensusNt as nt, n.superclass as superclass,
               n.subclass as subclass, n.class as class,
               n.dimorphism as dimorphism
    """
    neu_df = client.fetch_custom(cql)
    meta = pd.DataFrame({"bodyId": list(bodyIds)})
    return meta.merge(neu_df, on="bodyId", how="left")


def override_r78_metadata(
    meta: pd.DataFrame,
    color_df_R: pd.DataFrame,
    color_df_L: pd.DataFrame,
) -> pd.DataFrame:
    """Force R7/R8 cell_type, instance, nt, class, superclass on every row whose
    bodyId shows up in the R7/R8 fill edge list, regardless of what neuprint
    returned."""
    out = meta.copy()
    for side, color in (("R", color_df_R), ("L", color_df_L)):
        ref = (
            color[["bodyId_pre", "type_pre", "instance_pre"]]
            .drop_duplicates()
            .rename(
                columns={
                    "bodyId_pre": "bodyId",
                    "type_pre": "cell_type",
                    "instance_pre": "instance",
                }
            )
        )
        for r78_int in (7, 8):
            bids = ref.loc[ref.cell_type == f"R{r78_int}", "bodyId"].values
            mask = out["bodyId"].isin(bids)
            out.loc[mask, "cell_type"] = f"R{r78_int}"
            out.loc[mask, "instance"] = f"R{r78_int}_{side}"
            out.loc[mask, "nt"] = "histamine"
            out.loc[mask, "class"] = "visual"
            out.loc[mask, "superclass"] = "ol_sensory"
    return out


# === 3. Coordinates ===


def _load_pins(eyemap_dir: Path, roi_str: str) -> tuple[np.ndarray, int, np.ndarray]:
    assert roi_str in ("ME(R)", "ME(L)", "LO(R)", "LO(L)", "LOP(R)", "LOP(L)"), \
        f"Unsupported roi_str '{roi_str}'"
    if roi_str.endswith("(R)"):
        path = eyemap_dir / f"{roi_str[:-3]}_col_center_pins.pickle"
    else:
        path = eyemap_dir / f"{roi_str[:-3]}_L_col_center_pins.pickle"
    col_df = pd.read_pickle(path).dropna()
    col_ids = col_df.index.values
    n_bins = int((col_df.shape[1] - 3) / 3)
    pins = col_df.iloc[:, 3:].values.reshape((-1, 3))
    return col_ids, n_bins, pins


def _all_hex_df(eyemap_dir: Path) -> pd.DataFrame:
    me_df = pd.read_excel(eyemap_dir / "ME_columnar-cells_location.xlsx").convert_dtypes()
    return me_df[["hex1_id", "hex2_id"]].drop_duplicates().reset_index(drop=True)


def _find_hex_ids(xyz_df: pd.DataFrame, roi_str: str, eyemap_dir: Path) -> pd.DataFrame:
    col_ids, n_bins, pins = _load_pins(eyemap_dir, roi_str)
    tree = spatial.KDTree(pins)
    _, minid = tree.query(xyz_df[["x", "y", "z"]].values)
    col_df = _all_hex_df(eyemap_dir)
    col_df.index.name = "col_id"
    result = pd.DataFrame(
        col_ids[np.floor(minid / n_bins).astype(int)], columns=["col_id"]
    )
    return result.merge(col_df, how="left", on="col_id")


def _find_neuron_hex_ids(syn_df: pd.DataFrame, roi_str: str, eyemap_dir: Path) -> pd.DataFrame:
    """Majority-vote one hex coord per bodyId assuming all synapses lie in `roi_str`."""
    syn_df = syn_df.copy()
    hex_df = _find_hex_ids(syn_df, roi_str, eyemap_dir)
    syn_df["col_id"] = hex_df["col_id"].values
    target = pd.DataFrame(
        syn_df.groupby("bodyId")[["col_id"]].agg(lambda x: pd.Series.mode(x)[0])
    )
    col_df = _all_hex_df(eyemap_dir)
    col_df.index.name = "col_id"
    target.sort_values("col_id", inplace=True)
    target.reset_index(inplace=True)
    return target.merge(col_df, how="left", on="col_id")


def _coords_from_hex(hex1, hex2) -> pd.Series:
    x = pd.Series(hex2).astype(int).reset_index(drop=True) - pd.Series(hex1).astype(int).reset_index(drop=True)
    y = pd.Series(hex2).astype(int).reset_index(drop=True) + pd.Series(hex1).astype(int).reset_index(drop=True)
    return x.astype(str) + "," + y.astype(str)


def _majority_coords_for(
    bodyIds: list[int],
    roi: str,
    client: Client,
    eyemap_dir: Path,
) -> pd.Series:
    """Return Series mapping bodyId → 'x,y' coord via synapse majority in `roi`."""
    if not bodyIds:
        return pd.Series(dtype=str)
    syn_df = fetch_synapses(NC(bodyId=list(bodyIds)), SC(rois=[roi]), client=client)
    neu = _find_neuron_hex_ids(syn_df, roi_str=roi, eyemap_dir=eyemap_dir)
    neu["coords"] = _coords_from_hex(neu["hex1_id"].values, neu["hex2_id"].values).values
    return neu.set_index("bodyId")["coords"]


def _load_columnar_sheet(data_dir: Path, side: str) -> pd.DataFrame:
    """Long-form coord table from ME{_L}_columnar-cells_location + Mi1→T4 alignment."""
    assert side in ("r", "l")
    em = _eyemap(data_dir)
    S = side.upper()  # uppercase used by neuPrint ROI names + instance suffix + filenames
    col_xlsx = "ME_columnar-cells_location.xlsx" if side == "r" else "ME_L_columnar-cells_location.xlsx"
    t4_xlsx = "mi1_t4_alignment.xlsx" if side == "r" else "mi1_t4_alignment_L.xlsx"

    col = pd.read_excel(em / col_xlsx, dtype=str, engine="openpyxl")

    long = col.melt(["hex1_id", "hex2_id"], var_name="cell_type", value_name="bodyId")
    long["instance"] = long["cell_type"] + f"_{S}"

    t4 = pd.read_excel(em / t4_xlsx).convert_dtypes()
    t4["Mi1"] = t4["Mi1"].astype(str)
    t4 = t4[t4["valid_group"] == 1]
    merged = col.merge(t4, how="left", on="Mi1")
    merged = merged[["hex1_id", "hex2_id", "T4a", "T4b", "T4c", "T4d"]].drop_duplicates()
    merged = merged.melt(["hex1_id", "hex2_id"], var_name="cell_type", value_name="bodyId")
    merged = merged[~pd.isna(merged["bodyId"])]
    merged["instance"] = merged["cell_type"] + f"_{S}"

    out = pd.concat([long, merged], ignore_index=True)
    out = out[~pd.isna(out["bodyId"])].copy()
    out["coords"] = _coords_from_hex(out["hex1_id"].values, out["hex2_id"].values).values
    out["bodyId"] = out["bodyId"].astype(int)
    return out


def add_column_coords(meta: pd.DataFrame, data_dir: Path) -> pd.DataFrame:
    """Attach `coords` from the ME columnar sheets (both sides, Mi1+T4 alignment)."""
    col_df = pd.concat(
        [_load_columnar_sheet(data_dir, "r"), _load_columnar_sheet(data_dir, "l")],
        ignore_index=True,
    )
    return meta.merge(col_df[["bodyId", "coords"]].drop_duplicates(), on="bodyId", how="left")


def add_r78_dorsal_l3_coords(
    meta: pd.DataFrame,
    client: Client,
    data_dir: Path,
) -> pd.DataFrame:
    """Fill `coords` for R7/R8 (filled-in: parse bodyId digits; legacy: synapse
    majority), dorsal R7d/R8d (synapse majority), and L3_L (synapse majority)."""
    em = _eyemap(data_dir)
    out = meta.copy()

    for side, roi in (("r", "ME(R)"), ("l", "ME(L)")):
        S = side.upper()  # uppercase used in neuPrint ROI / instance suffix
        for r78_int in (7, 8):
            inst = f"R{r78_int}_{S}"
            marker = r78_int if side == "r" else r78_int - 2
            prefix = "1" + str(marker) * 5
            idx_inst = out.index[out["instance"] == inst]
            bid_str = out.loc[idx_inst, "bodyId"].astype(str)

            is_filled = bid_str.str.startswith(prefix).values
            idx_fill = idx_inst[is_filled]
            if len(idx_fill):
                bs = out.loc[idx_fill, "bodyId"].astype(str)
                out.loc[idx_fill, "coords"] = _coords_from_hex(
                    bs.str[6:8].values, bs.str[8:10].values
                ).values

            idx_old = idx_inst[~is_filled]
            bids_old = out.loc[idx_old, "bodyId"].tolist()
            if bids_old:
                cmap = _majority_coords_for(bids_old, roi, client, em)
                old_coords = out.loc[idx_old, "bodyId"].map(cmap)
                out.loc[idx_old, "coords"] = old_coords.fillna(out.loc[idx_old, "coords"]).values

        bids_dorsal = out.loc[
            out["instance"].isin([f"R7d_{S}", f"R8d_{S}"]), "bodyId"
        ].unique().tolist()
        if bids_dorsal:
            cmap = _majority_coords_for(bids_dorsal, roi, client, em)
            out["coords"] = out["coords"].fillna(out["bodyId"].map(cmap))

    bids_l3l = out.loc[out["instance"] == "L3_L", "bodyId"].unique().tolist()
    if bids_l3l:
        cmap = _majority_coords_for(bids_l3l, "ME(L)", client, em)
        out["coords"] = out["coords"].fillna(out["bodyId"].map(cmap))

    return out


# === 4. Meta finalize ===


def finalize_meta(meta: pd.DataFrame, sorted_nodes: list[int]) -> pd.DataFrame:
    """Add `sign`, `side`, `idx`; fill NaN `instance` with `cell_type + '_0'`;
    override `instance = cell_type + '_' + side_letter`; strip `(...)` and
    `_indeterminate` from both; sort by `idx`."""
    out = meta.copy()

    out["sign"] = out["nt"].map(NT_SIGN)

    na_idx = out.index[out["instance"].isna()]
    out.loc[na_idx, "instance"] = out.loc[na_idx, "cell_type"] + "_0"

    last = out["instance"].str[-1]
    out["side"] = np.where(last == "L", "left", np.where(last == "R", "right", "noside"))

    out["instance"] = out["cell_type"] + "_" + out["side"].map(SIDE_WORD)

    for c in ("instance", "cell_type"):
        out[c] = (
            out[c]
            .str.replace(r"\s*\(.*?\)", "", regex=True)
            .str.replace(r"_indeterminate", "", regex=True)
        )

    nodes_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
    out["idx"] = out["bodyId"].map(nodes_to_idx)
    out = out.sort_values("idx").reset_index(drop=True)
    return out


# === 5. Sparse matrices ===


def build_sparse_matrices(
    conn_df: pd.DataFrame, sorted_nodes: list[int]
) -> tuple[csc_matrix, csc_matrix]:
    """Build (prop, inprop) from the edge list.

    `prop` is raw weight as int32. `inprop` is `weight / tot_weight` per
    bodyId_post (per-post-neuron normalization — NOT `upstream`)
    as float32. Both are CSC sparse matrices of shape (N, N).
    """
    nodes_to_idx = {n: i for i, n in enumerate(sorted_nodes)}
    df = conn_df.copy()
    df["pre_idx"] = df["bodyId_pre"].map(nodes_to_idx)
    df["post_idx"] = df["bodyId_post"].map(nodes_to_idx)
    df["tot_weight"] = df.groupby("bodyId_post")["weight"].transform("sum")
    df["rel_input"] = df["weight"] / df["tot_weight"]
    n = len(sorted_nodes)
    row = df["pre_idx"].values
    col = df["post_idx"].values
    prop = coo_matrix((df["weight"].values, (row, col)), shape=(n, n)).tocsc().astype(np.int32)
    inprop = coo_matrix((df["rel_input"].values, (row, col)), shape=(n, n)).tocsc().astype(np.float32)
    return prop, inprop


def save_outputs(
    meta: pd.DataFrame,
    prop: csc_matrix,
    inprop: csc_matrix,
    data_dir: Path,
    tag: str,
) -> None:
    data_dir = Path(data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)
    meta.to_csv(data_dir / f"malecns_{tag}_meta.csv", index=False)
    sp.sparse.save_npz(data_dir / f"malecns_{tag}_prop.npz", prop)
    sp.sparse.save_npz(data_dir / f"malecns_{tag}_inprop.npz", inprop)


# === 6. Input RF data ===


_OL_ROIS = ["ME(R)", "LO(R)", "LOP(R)", "ME(L)", "LO(L)", "LOP(L)"]


def _fetch_input_neuropil(
    instance: str, client: Client, threshold: float = 0.05,
) -> pd.DataFrame:
    """ROIs receiving >= `threshold` of `instance`'s post synapses.

    Returns columns `[roi, instance, syn_frac]`.
    """
    assert 0 <= threshold <= 1
    rois_literal = ", ".join(f"'{r}'" for r in _OL_ROIS)
    cql = f"""
        UNWIND [{rois_literal}] as roi
        MATCH (n:Neuron)
        WHERE n.instance='{instance}'
        WITH
            distinct n
          , apoc.convert.fromJsonMap(n.roiInfo) as nri
          , coalesce(n.post, 0) as syn_total
          , roi
          , n.instance as instance
        with
            coalesce(nri[roi].post, 0) as syn_post
          , syn_total
          , roi
          , instance
          , n
        WITH
            distinct n.type as type
          , roi
          , instance
          , sum(syn_post) as syn_post
          , sum(syn_total) as syn_total
        WITH
          CASE
                WHEN syn_total> 0 THEN toFloat(syn_post)/ syn_total
                ELSE 0
            END AS syn_frac
          , roi
          , instance
        WHERE syn_frac >= {threshold}
        RETURN
            distinct roi
          , instance
          , syn_frac
    """
    return client.fetch_custom(cql)


def _fetch_post_syn_per_col_for_instance(
    instance: str, roi_str: str, client: Client,
) -> pd.DataFrame:
    """Per-column post-synapse counts for bodies in `instance` within `roi_str`.

    Returns columns `[column, roi, bodyId, synapse_count, synapse_frac]`.
    """
    assert roi_str in _OL_ROIS, f"Unsupported roi_str '{roi_str}'"
    m_to_s = "(m:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(ms:Synapse)"
    cql = f"""
    MATCH (n:Neuron)-[:Contains]->(:SynapseSet)-[:Contains]->(ns:Synapse)
    WHERE n.instance='{instance}'
        AND (exists(ns['{roi_str}']) and ns['{roi_str}'] IS NOT NULL)
        AND (exists(ns.olHex1) and ns.olHex1 IS NOT NULL)
        AND (exists(ns.olHex2) and ns.olHex2 IS NOT NULL)
        AND EXISTS {{{m_to_s}-[:SynapsesTo]->(ns)}}
    WITH n, ns, toString(ns.olHex1)+'_'+toString(ns.olHex2) AS col
    WITH {{bid: n.bodyId, col: col, syn: count(distinct ns)}} AS tmp_res
      , n.bodyId AS tmpbid
      , count(DISTINCT ns) AS syn_count
    WITH tmpbid, collect(tmp_res) as agg_res, sum(syn_count) as total_syn_count
    UNWIND agg_res as per_col
    RETURN
        per_col.col as column
      , '{roi_str}' as roi
      , per_col.bid as bodyId
      , per_col.syn as synapse_count
      , toFloat(per_col.syn)/total_syn_count as synapse_frac
    ORDER BY bodyId, synapse_count DESC
    """
    return client.fetch_custom(cql)


INPUT_SYN_PER_COL_COLS = [
    "bodyId", "instance", "roi", "column",
    "hex1_id", "hex2_id", "x", "y",
    "synapse_count", "synapse_frac",
]


def fetch_input_syn_per_col(
    instances,
    client: Client,
    rel_input_weight: float = 0.4,
) -> pd.DataFrame:
    """Per-(body, column) post-synapse fractions for each instance in its input
    neuropils.

    For every instance: (1) find ROIs receiving `>= rel_input_weight` of its
    post synapses via `_fetch_input_neuropil`, (2) for each such ROI fetch
    per-column post-synapse counts via `_fetch_post_syn_per_col_for_instance`,
    (3) derive `hex1_id`, `hex2_id`, `x = h2 - h1`, `y = (h2 + h1) / sqrt(3)`.
    Returns a flat DataFrame; consumed by `preprocessing.get_input_rf_raw_ol`.
    """
    frames = []
    for inst in instances:
        neuropils = _fetch_input_neuropil(inst, client, threshold=rel_input_weight)
        for roi in neuropils["roi"]:
            df = _fetch_post_syn_per_col_for_instance(inst, roi, client)
            if df.empty:
                continue
            df["instance"] = inst
            hex_cols = df["column"].str.split("_", expand=True).astype(int)
            df["hex1_id"] = hex_cols[0].values
            df["hex2_id"] = hex_cols[1].values
            df["x"] = df["hex2_id"] - df["hex1_id"]
            df["y"] = (df["hex2_id"] + df["hex1_id"]) / np.sqrt(3)
            frames.append(df[INPUT_SYN_PER_COL_COLS])
    if not frames:
        return pd.DataFrame(columns=INPUT_SYN_PER_COL_COLS)
    return pd.concat(frames, ignore_index=True)


def save_input_syn_per_col(df: pd.DataFrame, data_dir: Path, stem: str) -> Path:
    """Write per-(body, column) input synapse data to
    `{data_dir}/{stem}_input_syn_per_col.parquet`."""
    path = Path(data_dir) / f"{stem}_input_syn_per_col.parquet"
    df.to_parquet(path, index=False)
    return path


# === 7. ROI synapse counts ===


def fetch_roi_counts(instances, client: Client) -> pd.DataFrame:
    """Per-neuron per-ROI synapse counts for every bodyId in `instances`.

    Returns the `roi_counts_df` output of `neuprint.fetch_neurons`
    (`[bodyId, roi, pre, post, downstream, upstream, mito, size]`).
    """
    _, roi_counts = fetch_neurons(NC(instance=list(instances)), client=client)
    return roi_counts


# === 8. ROI adjacency (OL layer ROIs) ===


def _ol_layer_rois(side_char: str) -> list[str]:
    """Neuprint ROI names for the OL layers on the requested side."""
    assert side_char in ("r", "l"), f"side_char must be 'r' or 'l', got {side_char!r}"
    S = side_char.upper()
    me = [f"ME_{S}_layer_{i:02d}" for i in range(1, 11)]
    ame = [f"AME({S})"]
    lo = [f"LO_{S}_layer_{i}" for i in range(1, 8)]
    lop = [f"LOP_{S}_layer_{i}" for i in range(1, 5)]
    return me + ame + lo + lop


def fetch_roi_adj(
    instances, side_char: str, client: Client,
) -> pd.DataFrame:
    """Adjacency restricted to the OL layer ROIs on `side_char`.

    `pre` is any neuron in `instances`; `post` is any neuron with a type.
    Includes non-primary ROI assignments (`include_nonprimary=True`).
    """
    _, adj = fetch_adjacencies(
        NC(instance=list(instances)),
        NC(type=NotNull),
        rois=_ol_layer_rois(side_char),
        include_nonprimary=True,
        client=client,
    )
    return adj


def save_roi_pickles(
    roi_counts: pd.DataFrame,
    roi_adj: pd.DataFrame,
    data_dir: Path,
    stem: str,
) -> None:
    """Write `{stem}_roi_counts.pkl` and `{stem}_roi_adj.pkl` under `data_dir`."""
    data_dir = Path(data_dir)
    roi_counts.to_pickle(data_dir / f"{stem}_roi_counts.pkl")
    roi_adj.to_pickle(data_dir / f"{stem}_roi_adj.pkl")
