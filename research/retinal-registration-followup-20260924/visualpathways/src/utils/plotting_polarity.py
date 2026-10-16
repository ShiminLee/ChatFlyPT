"""Section 5 (Polarity experimental comparisons) plot_* functions:
predicted-vs-experimental ON/OFF polarity stacked bars.
"""


from pathlib import Path


import pandas as pd


from utils.config import FIG_DATASET as DATASET, FIG_DIR


from utils.plotting_shared import _save, _stacked_bars


# %%
_POLARITY_DISPLAY_RENAMES = {"5thsLNv_LNd6": "5thsLNv"}


# %%
def plot_polarity_stacked_bars(
    comparison: pd.DataFrame, colored_seed_df: pd.DataFrame,
    *, dataset: str = DATASET, filename_stem: str = "polarity_predictions",
    save_dir: Path | None = None, width: int | None = None, height: int = 350,
    font_size: int = 18,
):
    """Stacked bars of normalised visual-input effective weight per instance,
    colored by visual-input seed. Instance labels on the x-axis are bolded
    when the ground-truth `polarity` column is `'OFF'` (by convention).
    Display names are shortened via `_POLARITY_DISPLAY_RENAMES` for ticktext
    only — the underlying `out type` category keeps the full name so the bar
    ordering is unambiguous. Width defaults to ~35 px per instance + padding."""
    save_dir = save_dir if save_dir is not None else FIG_DIR / "exp_comparison"
    if width is None:
        width = int(max(400, 35 * len(comparison) + 150) * 0.8)

    # `colored_seed_df` index is bare seed names (no suffix); match columns in the comparison
    suffix = "_R" if any(c.endswith("_R") for c in comparison.columns) else "_L"
    seed_cols = [s + suffix for s in colored_seed_df.index if s + suffix in comparison.columns]

    long = comparison[["instance"] + seed_cols].copy()
    long["out type"] = long["instance"].str[:-2]
    melted = long.melt(
        id_vars=["out type"], value_vars=seed_cols,
        var_name="in type", value_name="effective weight",
    )
    melted["in type"] = melted["in type"].str[:-2]

    cmap = colored_seed_df["color"].to_dict()
    fig = _stacked_bars(
        melted, "out type", "effective weight", "in type",
        cmap, xrot=-45, width=width, height=height, font_size=font_size,
    )

    # Re-imported live from the facade (not this module's own bare global) so
    # a caller's `plot._POLARITY_DISPLAY_RENAMES = {...}` override (e.g. the
    # SI-14 flywire display rename in paper_figures.py) takes effect here too.
    from utils.plotting import _POLARITY_DISPLAY_RENAMES

    order = comparison["instance"].str[:-2].tolist()
    labels = [
        f"<b>{_POLARITY_DISPLAY_RENAMES.get(t, t)}</b>" if p == "OFF"
        else _POLARITY_DISPLAY_RENAMES.get(t, t)
        for t, p in zip(order, comparison["Contrast selectivity"].values)
    ]
    fig.update_xaxes(
        categoryorder="array", categoryarray=order,
        tickvals=order, ticktext=labels,
    )
    _save(fig, save_dir, f"{dataset}_{filename_stem}.pdf", html=True)
    return fig
