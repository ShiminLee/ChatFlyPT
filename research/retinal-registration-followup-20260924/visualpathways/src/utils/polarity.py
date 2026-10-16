"""ON/OFF polarity cross-checks against the fly-literature spreadsheet.

Hosts:
- Curated label loader: `get_polarity_experiments`
- OL-side feature-vector + polarity prediction: `get_ol_polarity_comparison`
- CB-side counterpart: `get_cb_polarity_comparison`
- Private helper `_split_polarity_types`, alias table `_POLARITY_INSTANCE_ALIASES`.
"""
import re

import numpy as np
import pandas as pd
import scipy.sparse as sp

from utils.cache import cached_parquet
from utils.config import DATA_DIR, DATASET, PARAMS_DIR, SIDE_CHAR
from utils.core_data import (
    _SIDE_WORD,
    get_meta,
    get_ol_flow_type,
    get_ol_meta,
    grouped_effweight,
)
from utils.ol_data import get_ol_stepsn_sum


def _split_polarity_types(cell) -> list[str]:
    """Split a 'Neuron type' cell from VCBNexperiments, dropping parts whose
    own name ends in '?' (e.g. `ER4d, ER3?` → `['ER4d']`)."""
    if pd.isna(cell):
        return []
    parts = [p.strip() for p in str(cell).split(",")]
    return [p for p in parts if p and not p.endswith("?")]


def _expand_wildcards(parts: list[str], universe) -> list[str]:
    """Resolve `*` wildcards in cell-type names against the dataset's types.

    A trailing `*` means "this prefix followed by anything" — e.g. `T4*`
    matches `T4a, T4b, T4c, T4d`. Each wildcard part is replaced by the bare
    cell types in `universe` whose name starts with the text before the `*`
    (sorted, deterministic). `*_unclear` variants are excluded (they are
    low-confidence type assignments, not real subtypes). Non-wildcard parts
    pass through unchanged; a wildcard that matches nothing contributes no
    types."""
    out: list[str] = []
    for p in parts:
        if "*" in p:
            prefix = p.split("*", 1)[0]
            out.extend(sorted(
                t for t in universe
                if t.startswith(prefix) and "unclear" not in t
            ))
        else:
            out.append(p)
    return out


def _side_cell_types(dataset, data_dir, side_char) -> set[str]:
    """Bare cell-type names (instance minus the `_R`/`_L` side suffix) present
    on the requested side of the dataset meta — the universe for `*`
    expansion (covers both OL and central-brain types)."""
    suffix = "_R" if side_char == "r" else "_L"
    side_word = _SIDE_WORD[side_char]
    meta = get_meta(dataset=dataset, data_dir=data_dir)
    return {
        i[:-2]
        for i in meta.loc[meta.side == side_word, "instance"].dropna().unique()
        if isinstance(i, str) and i.endswith(suffix)
    }


_CB_POLARITY_PREFIXES = ["DN", "PLP", "PVLP", "TuBu", "ER4d", "LNd_b"]


def _cb_polarity_cat(group_label: str) -> int:
    """Sort key: category rank for CB polarity bar ordering."""
    for i, prefix in enumerate(_CB_POLARITY_PREFIXES):
        if group_label.startswith(prefix):
            return i
    return len(_CB_POLARITY_PREFIXES)


# Manual bar order for the VPN polarity panel (Main 2d, right sub-panel /
# `main_2_d_2`): these types are pulled to immediately after `LC6`; every
# other bar keeps its spreadsheet order.
_VPN_ORDER_ANCHOR = "LC6"
_VPN_ORDER_AFTER = ["VS", "HS", "LC10a", "MeTu3c"]


def _reorder_after(df: pd.DataFrame, anchor: str, moved: list[str]) -> pd.DataFrame:
    """Row order of `df` with the `moved` cell types (matched on instance
    prefix = instance minus the side suffix) pulled to immediately after the
    `anchor` type; all other rows keep their order. Missing `moved` types are
    skipped; if `anchor` is absent they stay at their original tail position."""
    names = df["instance"].str[:-2]
    moved = [m for m in moved if (names == m).any()]
    moved_set = set(moved)
    order: list = []
    inserted = False
    for i in df.index[~names.isin(moved_set)]:
        order.append(i)
        if names[i] == anchor:
            for m in moved:
                order.extend(df.index[names == m].tolist())
            inserted = True
    if not inserted:
        for m in moved:
            order.extend(df.index[names == m].tolist())
    return df.loc[order].reset_index(drop=True)


def _abbreviate_group_label(parts: list[str]) -> str:
    """Abbreviate a group to 'PrefixFirst-Last' when all types share a prefix
    and have consecutive numeric or alphabetic suffixes. Falls back to
    comma-joining otherwise.

    Examples:
    - ['TuBu02','TuBu03','TuBu04','TuBu05'] -> 'TuBu02-05'
    - ['DNg02_a','DNg02_b',...,'DNg02_f']   -> 'DNg02_a-f'
    - ['PVLP008_a1',...,'PVLP008_a4']       -> 'PVLP008_a1-4'
    """
    if len(parts) <= 1:
        return parts[0] if parts else ""
    # Numeric suffix with letter-only prefix: TuBu02-05
    matches = [re.fullmatch(r"([A-Za-z_]+)(\d+)", p) for p in parts]
    if all(matches):
        prefixes = [m.group(1) for m in matches]
        suffixes = [m.group(2) for m in matches]
        if len(set(prefixes)) == 1:
            nums = [int(s) for s in suffixes]
            if nums == list(range(nums[0], nums[0] + len(nums))):
                return f"{prefixes[0]}{suffixes[0]}-{suffixes[-1]}"
    # Alphabetic single-char suffix: DNg02_a through DNg02_f
    matches_alpha = [re.fullmatch(r"(.+)_([a-z])$", p) for p in parts]
    if all(matches_alpha):
        prefixes = [m.group(1) for m in matches_alpha]
        suffixes = [m.group(2) for m in matches_alpha]
        if len(set(prefixes)) == 1:
            codes = [ord(c) for c in suffixes]
            if codes == list(range(codes[0], codes[0] + len(codes))):
                return f"{prefixes[0]}_{suffixes[0]}-{suffixes[-1]}"
    # Numeric suffix with any prefix: PVLP008_a1 through PVLP008_a4
    matches_any = [re.fullmatch(r"(.+?)(\d+)$", p) for p in parts]
    if all(matches_any):
        prefixes = [m.group(1) for m in matches_any]
        suffixes = [m.group(2) for m in matches_any]
        if len(set(prefixes)) == 1:
            nums = [int(s) for s in suffixes]
            if nums == list(range(nums[0], nums[0] + len(nums))):
                return f"{prefixes[0]}{suffixes[0]}-{suffixes[-1]}"
    return ", ".join(parts)


_POLARITY_INSTANCE_ALIASES: dict[str, str] = {}


def _build_polarity_labels(df, side_types, suffix, instance_map):
    """Spreadsheet rows -> per-(type, instance) polarity labels.

    Shared core of `get_polarity_experiments` (malecns) and the flywire
    adaptation in `make_figures/flywire_main_2d_4d_polarity`. Filters to
    `ON`/`OFF`, splits multi-type cells, expands `*` wildcards against
    `side_types`, drops cell types with conflicting polarity, builds
    `*`-stripped `group_label`s, and maps `type + suffix` to instance name(s)
    via `instance_map` (a callable that may return a list for 1->many remaps).
    Returns columns: `type, polarity, instance, group_label` (one row per
    instance). Callers add `in_ol`.
    """
    df = df[df["Contrast selectivity"].isin(["ON", "OFF"])].copy()
    df["_parts"] = df["Neuron type"].apply(_split_polarity_types)
    # Resolve `*` wildcards to the matching cell types in this dataset.
    df["_types"] = df["_parts"].apply(lambda p: _expand_wildcards(p, side_types))
    # Drop cell types labelled both ON and OFF anywhere (ambiguous polarity).
    _expl = df[["Contrast selectivity", "_types"]].explode("_types").dropna(subset=["_types"])
    ambiguous = set(
        _expl.groupby("_types")["Contrast selectivity"].nunique().pipe(lambda s: s[s > 1]).index
    )

    def _surviving(part):
        return [t for t in _expand_wildcards([part], side_types) if t not in ambiguous]

    # group_label is the original entry abbreviated, keeping only parts that
    # still contribute a non-ambiguous type. A wildcard part shows as its bare
    # prefix (e.g. `T4*` -> `T4`) — one averaged bar, not `T4*`.
    df["group_label"] = df["_parts"].apply(
        lambda parts: _abbreviate_group_label(
            [p.split("*", 1)[0] if "*" in p else p for p in parts if _surviving(p)]
        )
    )
    df["_types"] = df["_types"].apply(lambda ts: [t for t in ts if t not in ambiguous])
    df = df.explode("_types").dropna(subset=["_types"])
    df = df[df["_types"] != ""]
    df = df.rename(columns={"_types": "type"})
    df["instance"] = (df["type"].astype(str) + suffix).map(instance_map)
    df = df.explode("instance")
    return df[["type", "Contrast selectivity", "instance", "group_label"]].reset_index(drop=True)


def get_polarity_experiments(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    source_file: str = "TableS2_experimental_data.xlsx",
    sheet: str = "contrast selectivity",
    force: bool = False,
) -> pd.DataFrame:
    """Curated ON/OFF polarity labels from the experimental-data spreadsheet.

    Keeps rows whose `polarity` is exactly `'ON'` or `'OFF'` (drops NaN / any
    value with `'?'`). Splits comma-separated multi-type cells into one row
    per type; drops parts that themselves end in `'?'`. Resolves `*` wildcards
    (e.g. `T4*` → `T4a..T4d`) against the dataset's actual types. Cell types
    labelled with conflicting polarity (both `'ON'` and `'OFF'` somewhere in
    the sheet, e.g. `ER3*`) are ambiguous and dropped everywhere. Adds
    `instance` = `type + '_R'` (or `'_L'` if `side_char='l'`) and `in_ol`
    based on `get_ol_meta` membership. Applies `_POLARITY_INSTANCE_ALIASES`
    for types whose sheet name maps to a sub-variant in the dataset's meta.
    """
    def _compute():
        suffix = "_R" if side_char == "r" else "_L"
        df = pd.read_excel(PARAMS_DIR / source_file, sheet_name=sheet)
        side_types = _side_cell_types(dataset, data_dir, side_char)
        out = _build_polarity_labels(
            df, side_types, suffix, lambda x: _POLARITY_INSTANCE_ALIASES.get(x, x),
        )
        ol_instances = set(
            get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
            .instance.dropna().unique()
        )
        out["in_ol"] = out["instance"].isin(ol_instances)
        return out[["type", "Contrast selectivity", "instance", "in_ol", "group_label"]].reset_index(drop=True)

    return cached_parquet(
        f"{dataset}_OL_{side_char}_polarity_labels", _compute, force=force,
    )


def get_ol_polarity_comparison(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-entry effective-weight feature vectors from visual-input seeds plus
    predicted polarity (`'ON' if L1 > L2 else 'OFF'`) and match flag.

    Feature vectors are normalised per instance (rows that sum to zero stay
    zero), then averaged over the cell types sharing one spreadsheet entry, so
    each entry is a single bar (e.g. `T4*` -> one `T4` bar, the mean of
    T4a..T4d). Seed inputs are every instance with `hitting_time == 0` in
    `get_ol_flow_type` (includes L1/L2/L3, R7/R7p/R7y,
    R8/R8p/R8y, R7d/R8d, HBeyelet). Uses OL `lat_flow_sum` (fast, OL-only).
    """
    def _compute():
        labels = get_polarity_experiments(
            dataset=dataset, data_dir=data_dir, side_char=side_char
        )
        labels = labels[labels["in_ol"]].reset_index(drop=True)

        ol_meta = get_ol_meta(dataset=dataset, data_dir=data_dir, side_char=side_char)
        stepsn = get_ol_stepsn_sum(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        flow_type = get_ol_flow_type(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        in_instances = flow_type[flow_type.hitting_time == 0].instance.unique()
        inidx = ol_meta[ol_meta.instance.isin(in_instances)].idx.values
        outidx = ol_meta[ol_meta.instance.isin(labels["instance"].values)].idx.values

        idx_to_instance = dict(zip(ol_meta.idx, ol_meta.instance))
        feat = grouped_effweight(
            stepsn, inidx, outidx, idx_to_instance, idx_to_instance,
            combining_method="median",
        )
        feat = feat.loc[feat.index.isin(labels["instance"].values)]
        row_sum = feat.sum(axis=1)
        feat_norm = feat.div(row_sum.where(row_sum > 0, 1), axis=0)
        feat_norm = feat_norm.reset_index(names="instance")

        merged = labels.merge(feat_norm, on="instance", how="inner")
        merged = merged.merge(
            ol_meta[["instance", "main_groups"]].drop_duplicates("instance"),
            on="instance", how="left",
        )
        # Drop visual-input seeds themselves — they sit on the input side of the
        # flow so their own feature vectors collapse to self-loops, and L1/L2
        # are the rule's own predictors (comparing them is tautological).
        merged = merged[merged["main_groups"] != "visual input"]
        suffix = "_R" if side_char == "r" else "_L"

        # Average the cell types sharing one spreadsheet entry into a single bar
        # per group_label (e.g. T4* -> mean of T4a..T4d). Keep first-appearance
        # (spreadsheet) order.
        feature_cols = [c for c in feat_norm.columns if c != "instance"]
        grouped = (
            merged.groupby(["group_label", "Contrast selectivity"], sort=False)[feature_cols]
            .mean()
            .reset_index()
        )
        grouped["instance"] = grouped["group_label"] + suffix
        grouped["type"] = grouped["group_label"]
        grouped["main_groups"] = (
            merged[["group_label", "main_groups"]]
            .drop_duplicates("group_label")
            .set_index("group_label")["main_groups"]
            .reindex(grouped["group_label"].values)
            .values
        )
        l1_col, l2_col = f"L1{suffix}", f"L2{suffix}"
        l1 = grouped[l1_col] if l1_col in grouped.columns else 0.0
        l2 = grouped[l2_col] if l2_col in grouped.columns else 0.0
        grouped["predicted"] = np.where(l1 > l2, "ON", "OFF")
        grouped["match"] = grouped["predicted"] == grouped["Contrast selectivity"]
        # Custom VPN-panel order: VS, HS, LC10a, MeTu3c pulled after LC6.
        grouped = _reorder_after(grouped, _VPN_ORDER_ANCHOR, _VPN_ORDER_AFTER)
        return grouped.reset_index(drop=True)

    return cached_parquet(
        f"{dataset}_OL_{side_char}_polarity_comparison", _compute, force=force,
    )


def get_cb_polarity_comparison(
    *, dataset: str = DATASET, data_dir=DATA_DIR, side_char: str = SIDE_CHAR,
    force: bool = False,
) -> pd.DataFrame:
    """Per-CB-instance effective-weight feature vectors from visual-input
    seeds plus predicted polarity (`'ON' if L1 > L2 else 'OFF'`) and match
    flag.

    Uses the labels rows where `in_ol == False` (24 CB-only types) and
    propagates on the full `lat_flow_sum.npz` (not OL-subset). Seeds are the
    side's visual-input instances from the full meta.
    """
    def _compute():
        labels = get_polarity_experiments(
            dataset=dataset, data_dir=data_dir, side_char=side_char,
        )
        labels = labels[~labels["in_ol"]].reset_index(drop=True)

        meta = get_meta(dataset=dataset, data_dir=data_dir)
        stepsn = sp.load_npz(
            data_dir / f"{dataset}_{side_char}_lat_flow_sum.npz"
        )
        side_word = _SIDE_WORD[side_char]
        in_instances = meta[
            (meta.main_groups == "visual input") & (meta.side == side_word)
        ].instance.unique()
        inidx = meta[meta.instance.isin(in_instances)].idx.values
        outidx = meta[meta.instance.isin(labels["instance"].values)].idx.values
        idx_to_instance = dict(zip(meta.idx, meta.instance))

        # Step 1: normalize per output body (compute input-weight ratios per body)
        raw = pd.DataFrame(
            stepsn[:, outidx][inidx, :].toarray(),
            index=[idx_to_instance[k] for k in inidx],
            columns=outidx,
        )
        raw = raw.groupby(raw.index).sum()
        raw.columns = [idx_to_instance[k] for k in raw.columns]
        raw = raw.T  # rows=output bodies, cols=input instances
        body_sum = raw.sum(axis=1)
        raw_norm = raw.div(body_sum.where(body_sum > 0, 1), axis=0)

        # Step 2: mean over bodies of the same cell type
        type_norm = raw_norm.groupby(level=0).mean()
        type_norm = type_norm.reset_index(names="instance")

        merged = labels.merge(type_norm, on="instance", how="inner")
        merged = merged.merge(
            meta[["instance", "main_groups"]].drop_duplicates("instance"),
            on="instance", how="left",
        )

        # Step 3: mean over cell types sharing the same spreadsheet entry
        feature_cols = [c for c in type_norm.columns if c != "instance"]
        grouped = (
            merged.groupby(["group_label", "Contrast selectivity"])[feature_cols]
            .mean()
            .reset_index()
        )
        suffix = "_R" if side_char == "r" else "_L"
        grouped["instance"] = grouped["group_label"] + suffix
        grouped["main_groups"] = (
            merged[["group_label", "main_groups"]]
            .drop_duplicates("group_label")
            .set_index("group_label")["main_groups"]
            .reindex(grouped["group_label"].values)
            .values
        )
        l1_col, l2_col = f"L1{suffix}", f"L2{suffix}"
        l1 = grouped[l1_col] if l1_col in grouped.columns else 0.0
        l2 = grouped[l2_col] if l2_col in grouped.columns else 0.0
        grouped["predicted"] = np.where(l1 > l2, "ON", "OFF")
        grouped["match"] = grouped["predicted"] == grouped["Contrast selectivity"]
        grouped["_cat"] = grouped["group_label"].apply(_cb_polarity_cat)
        grouped = grouped.sort_values(["_cat", "group_label"]).drop(columns="_cat").reset_index(drop=True)
        return grouped

    return cached_parquet(
        f"{dataset}_{side_char}_cb_polarity_comparison", _compute, force=force,
    )
