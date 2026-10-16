"""Section 1 (Inventory) plot_* functions: dataset-composition overviews
(full brain / right OL / by main-group).
"""


from pathlib import Path


import pandas as pd


from utils.config import FIG_DATASET as DATASET, FIG_DIR


from utils.plotting_shared import _pie, _save


# %%
def plot_inventory_full_brain(
    meta: pd.DataFrame,
    colors: dict,
    *,
    dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Pie charts of neurons and cell types by region (full brain).

    The neuron pie keeps left/right OL as separate wedges; the types pie merges
    them into one "OL" wedge (see below). `colors`: {'region': series/dict of
    color by region}. Returns list of (fig, filename) pairs.
    """
    save_dir = save_dir if save_dir is not None else FIG_DIR / "inventory"
    m_all = meta[meta.region != "other"]
    m = meta[(meta.region != "other") & (meta.region != "left OL")]
    cmap = colors["region"]
    figs = []
    count_df = m_all.groupby("region")["bodyId"].nunique().to_frame(name="count").reset_index()
    fig = _pie(count_df, "count", "region", cmap, height=350, width=350)
    fig.update_layout(showlegend=False)
    figs.append((fig, f"{dataset}_neuron_region_pie.pdf"))

    # Types-by-region pie (main_1_d_3): merge left + right OL into a single
    # "OL" wedge, coloured like right OL. Cell types are bilaterally shared, so
    # nunique over the merged region ~= one side's type count (not the sum).
    m_types = m_all.assign(
        region=m_all["region"].replace({"left OL": "OL", "right OL": "OL"})
    )
    cmap_ol = {**cmap, "OL": cmap["right OL"]}
    count_type_df = m_types.groupby("region")["cell_type"].nunique().to_frame(name="count").reset_index()
    fig = _pie(count_type_df, "count", "region", cmap_ol, height=350, width=350)
    fig.update_layout(showlegend=False)
    figs.append((fig, f"{dataset}_types_region_pie_all.pdf"))

    # Sign pies at full brain (drop NaN sign rows)
    sign_cmap = colors["sign"]
    m_sign = m_all[~m_all["sign"].isna()]
    count_df = m_sign.groupby("sign")["bodyId"].nunique().to_frame(name="count").reset_index()
    fig = _pie(count_df, "count", "sign", sign_cmap, height=350, width=350)
    fig.update_layout(showlegend=False)
    figs.append((fig, f"{dataset}_neuron_sign_pie.pdf"))

    count_type_df = m_sign.groupby("sign")["cell_type"].nunique().to_frame(name="count").reset_index()
    fig = _pie(count_type_df, "count", "sign", sign_cmap, height=350, width=350)
    fig.update_layout(showlegend=False)
    figs.append((fig, f"{dataset}_types_sign_pie_all.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_inventory_right_ol(
    meta: pd.DataFrame,
    colors: dict,
    *,
    dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """Pie charts of right-OL neurons / cell types by main_groups and sign."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "inventory"
    m = meta[(meta.region == "right OL") & (meta.main_groups != "nonOL")]

    figs = []
    for val in ["main_groups", "sign"]:
        cmap = colors[val]
        count_df = m.groupby(val)["bodyId"].nunique().to_frame(name="count").reset_index()
        fig = _pie(
            count_df, "count", val, cmap,
            title=f"distribution of {count_df['count'].sum()} OL neurons",
            height=350, width=350,
        )
        fig.update_layout(showlegend=False)
        figs.append((fig, f"{dataset}_OL_r_neuron_{val}_pie.pdf"))

        count_type_df = m.groupby(val)["cell_type"].nunique().to_frame(name="count").reset_index()
        fig = _pie(
            count_type_df, "count", val, cmap,
            title=f"distribution of {count_type_df['count'].sum()} OL instances",
            height=350, width=350,
        )
        fig.update_layout(showlegend=False)
        figs.append((fig, f"{dataset}_OL_r_types_{val}_pie_all.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs


# %%
def plot_inventory_by_main_group(
    meta: pd.DataFrame,
    colors: dict,
    *,
    dataset: str = DATASET,
    save_dir: Path | None = None,
):
    """For each main group in right OL, pie chart of sign distribution."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "inventory"
    m = meta[(meta.region == "right OL") & (meta.main_groups != "other") & (meta.instance != "R1-R6_R")]

    cmap = colors["sign"]
    figs = []
    for main_group in m["main_groups"].unique():
        plot_df = m[m["main_groups"] == main_group]

        count_df = plot_df.groupby("sign")["bodyId"].nunique().to_frame(name="count").reset_index()
        fig = _pie(
            count_df, "count", "sign", cmap,
            title=f"distribution of {count_df['count'].sum()} {main_group} neurons",
        )
        figs.append((fig, f"{dataset}_OL_r_neuron_{main_group}_sign_pie.pdf"))

        count_type_df = plot_df.groupby("sign")["cell_type"].nunique().to_frame(name="count").reset_index()
        fig = _pie(
            count_type_df, "count", "sign", cmap,
            title=f"distribution of {count_type_df['count'].sum()} {main_group} cell types",
        )
        figs.append((fig, f"{dataset}_OL_r_types_{main_group}_sign_pie_all.pdf"))

    for fig, name in figs:
        _save(fig, save_dir, name)
    return figs
