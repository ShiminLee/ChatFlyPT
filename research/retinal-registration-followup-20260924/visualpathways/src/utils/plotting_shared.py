"""Shared plot helpers for the utils/plotting.py facade's per-section modules.

HTML-sidecar plumbing (`set_html_output`, `_save`), generic chart primitives
(`_scatter`, `_box*`, `_heatmatrix`, `_hist_*`, `_stacked_bars`, `_point_line`,
`_violin_median`, `_pie`, `_add_gaussian_ellipses`, `_eyesymbol`,
`_directedness_triangle`), and the shared pathway-diagram renderers
(`plot_pathway`, `plot_comparison_scatter`).
"""


from pathlib import Path


import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


# %%
_WRITE_HTML = False


def set_html_output(enabled: bool) -> None:
    """Enable or disable optional HTML sidecar outputs."""
    global _WRITE_HTML
    _WRITE_HTML = bool(enabled)


# %%
_SIDE_WORD = {"r": "right", "l": "left"}


# %%
def _pie(df: pd.DataFrame, val: str, col: str, cmap: dict, *, height=400, width=400, title=""):
    fig = px.pie(df, values=val, names=col, title=title, color=col, color_discrete_map=cmap)
    fig.update_traces(textposition="inside", texttemplate="%{percent:.0%}")
    fig.update_layout(
        width=width,
        height=height,
        plot_bgcolor="white",
        paper_bgcolor="white",
        font=dict(family="arial", size=18),
    )
    return fig


# %%
def _save(fig, save_dir: Path | None, filename: str, *, html: bool = False, rasterize: bool = False):
    if save_dir is None:
        return
    save_dir.mkdir(parents=True, exist_ok=True)
    if rasterize and filename.endswith('.pdf'):
        from io import BytesIO
        from PIL import Image
        png_bytes = fig.to_image(format='png', scale=4)
        Image.open(BytesIO(png_bytes)).save(save_dir / filename, 'PDF', resolution=300)
    else:
        fig.write_image(save_dir / filename)
    if html and _WRITE_HTML:
        stem = Path(filename).stem
        fig.write_html(save_dir / f"{stem}.html")


# %%
_AXIS = dict(
    showline=True, linewidth=1, linecolor="black", mirror=True,
    ticks="outside", ticklen=5, tickcolor="black",
)


# %%
def _hist_lines(df, val, col, bins, color_df, *, height=400, width=500, norm=True):
    fig = go.Figure()
    centers = (bins[:-1] + bins[1:]) / 2
    for group in df[col].unique():
        sub = df[df[col] == group]
        counts, _ = np.histogram(sub[val], bins=bins)
        if norm:
            counts = counts / counts.sum()
        fig.add_trace(go.Scatter(
            x=centers, y=counts, mode="markers+lines", name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        font=dict(family="arial", size=18), xaxis_title=val,
        xaxis_range=[bins[0], bins[-1]],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _hist_lines_v(df, val, col, bins, color_df, *, height=400, width=500, norm=True):
    """Vertical variant of `_hist_lines`: x-axis = fraction, y-axis = value."""
    fig = go.Figure()
    centers = (bins[:-1] + bins[1:]) / 2
    for group in df[col].unique():
        sub = df[df[col] == group]
        counts, _ = np.histogram(sub[val], bins=bins)
        if norm:
            counts = counts / counts.sum()
        fig.add_trace(go.Scatter(
            x=counts, y=centers, mode="markers+lines", name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        font=dict(family="arial", size=18), yaxis_title=val,
        yaxis_range=[bins[0], bins[-1]],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _hist_cumsum_lines(df, val, col, bins, color_df, *, height=500, width=400):
    """Cumulative-sum histogram by group, horizontal orientation."""
    fig = go.Figure()
    centers = (bins[:-1] + bins[1:]) / 2
    for group in df[col].unique():
        sub = df[df[col] == group]
        counts, _ = np.histogram(sub[val], bins=bins)
        if counts.sum() == 0:
            continue
        cum = np.cumsum(counts / counts.sum())
        fig.add_trace(go.Scatter(
            x=centers, y=cum, mode="markers+lines", name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        font=dict(family="arial", size=18),
        xaxis_title=val, yaxis_title="cum. frac.",
        xaxis_range=[bins[0], bins[-1]],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _hist_cumsum_lines_v(df, val, col, bins, color_df, *, height=500, width=400):
    """Vertical cumulative-sum histogram."""
    fig = go.Figure()
    centers = (bins[:-1] + bins[1:]) / 2
    for group in df[col].unique():
        sub = df[df[col] == group]
        counts, _ = np.histogram(sub[val], bins=bins)
        if counts.sum() == 0:
            continue
        cum = np.cumsum(counts / counts.sum())
        fig.add_trace(go.Scatter(
            x=cum, y=centers, mode="markers+lines", name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        font=dict(family="arial", size=18),
        xaxis_title="cum. frac.", yaxis_title=val,
        yaxis_range=[bins[0], bins[-1]],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _hist_sum_lines(df, val, sum_val, col, bins, color_df, *, height=400, width=500, norm=True):
    centers = (bins[:-1] + bins[1:]) / 2
    plot_df = df.copy()
    plot_df["val_bin"] = pd.cut(plot_df[val], bins=bins, labels=centers).astype(float)
    fig = go.Figure()
    for group in plot_df[col].unique():
        sub = plot_df[plot_df[col] == group]
        binned = sub.groupby("val_bin", observed=False)[sum_val].sum()
        binned = binned.reindex(centers, fill_value=0).reset_index()
        binned.columns = ["bin", "count"]
        if norm:
            binned["count"] = binned["count"] / binned["count"].sum()
        fig.add_trace(go.Scatter(
            x=binned["bin"], y=binned["count"], mode="markers+lines", name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        font=dict(family="arial", size=18), xaxis_title=val,
        xaxis_range=[bins[0], bins[-1]],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _scatter(plot_df, x_col, y_col, c_col, t_col, color_df, t_stars=(), *,
             height=400, width=400, marker="circle", size=7, show_labels=True):
    fig = go.Figure(go.Scatter(
        x=plot_df[x_col].values, y=plot_df[y_col].values, mode="markers",
        marker=dict(
            size=size,
            color=color_df.loc[plot_df[c_col].values, "color"].values,
            opacity=0.4, symbol=marker,
        ),
        text=plot_df[t_col],
    ))
    for t in t_stars:
        sub = plot_df[plot_df[t_col] == t]
        if sub.empty:
            continue
        fig.add_trace(go.Scatter(
            x=sub[x_col].values, y=sub[y_col].values,
            mode="markers+text" if show_labels else "markers",
            marker=dict(
                size=size,
                color=color_df.loc[sub[c_col].values, "color"].values,
                opacity=1, symbol=marker, line=dict(color="black", width=1),
            ),
            text=[s.replace("_R", "") for s in sub[t_col]],
            textposition="bottom center",
        ))
    fig.update_layout(
        xaxis_title=x_col, yaxis_title=y_col, width=width, height=height,
        font=dict(family="arial", size=18), showlegend=False,
        paper_bgcolor="white", plot_bgcolor="white",
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _box(plot_df, x_col, y_col, c_col, cmap, *, height=600, width=400):
    fig = px.box(plot_df, x=x_col, y=y_col, points="all",
                 color=c_col, color_discrete_map=cmap)
    fig.update_layout(
        font=dict(family="arial", size=18),
        xaxis_title=x_col, yaxis_title=y_col,
        xaxis_range=[0.9 * plot_df[x_col].min(), 1.1 * plot_df[x_col].max()],
        yaxis_tickvals=plot_df[y_col].values,
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width, showlegend=False,
    )
    fig.update_xaxes(title_standoff=0, tickangle=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _box_scatter(plot_df, x_col, y_col, c_col, color_df, *, height=350, width=450):
    """Grouped horizontal-by-group box (one box-per-category × group). Unlike
    `_box`, does not set per-row y-ticks — used when the group dimension has
    many rows."""
    fig = go.Figure()
    for group in plot_df[c_col].unique():
        sub = plot_df[plot_df[c_col] == group]
        fig.add_trace(go.Box(
            x=sub[x_col], y=sub[y_col], name=str(group),
            marker=dict(color=color_df.loc[group, "color"]),
            line=dict(color=color_df.loc[group, "color"]),
        ))
    fig.update_layout(
        xaxis_title=x_col, yaxis_title=y_col,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="arial", size=18),
        height=height, width=width, boxmode="group",
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _heatmatrix(
    mat, tick_labels, *, cmap="Greys", height=500, width=500, anno=None,
    x_tick_labels=None, y_tick_labels=None, font_size=18,
):
    x_tick_labels = list(tick_labels if x_tick_labels is None else x_tick_labels)
    y_tick_labels = list(tick_labels if y_tick_labels is None else y_tick_labels)
    n_x = len(x_tick_labels)
    n_y = len(y_tick_labels)
    fig = go.Figure(data=go.Heatmap(
        z=mat, x=x_tick_labels, y=y_tick_labels,
        colorscale=cmap, reversescale=False, showscale=True,
        text=anno, texttemplate="%{text}", textfont={"size": font_size},
    ))
    n_diag = min(n_x, n_y)
    fig.add_shape(type="line", x0=-0.5, y0=-0.5, x1=n_diag - 0.5, y1=n_diag - 0.5,
                  line=dict(color="gray", width=2))
    fig.update_layout(
        xaxis_title="post layer", yaxis_title="pre layer",
        xaxis_tickvals=list(range(n_x)), xaxis_ticktext=x_tick_labels,
        yaxis_tickvals=list(range(n_y)), yaxis_ticktext=y_tick_labels,
        paper_bgcolor="white", plot_bgcolor="white",
        font=dict(family="arial", size=font_size),
        height=height, width=width,
        xaxis=dict(scaleanchor="y", scaleratio=1),
        yaxis=dict(scaleanchor="x", scaleratio=1),
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _point_line(plot_df, x_col, y_col, *, x_thre=None, y_thre=None,
                height=500, width=600, color="black"):
    fig = go.Figure()
    if x_thre is not None:
        fig.add_vline(x=x_thre, line_width=1, line_dash="dash", line_color="gray")
    if y_thre is not None:
        fig.add_hline(y=y_thre, line_width=1, line_dash="dash", line_color="gray")
    fig.add_trace(go.Scatter(
        x=plot_df[x_col].values, y=plot_df[y_col].values,
        mode="markers+lines",
        marker=dict(size=8, color=color, opacity=0.7),
        line=dict(width=1, color=color),
    ))
    fig.update_layout(
        font=dict(family="arial", size=18),
        xaxis_title=x_col, yaxis_title=y_col,
        xaxis_range=[0.9 * plot_df[x_col].min(), 1.1 * plot_df[x_col].max()],
        yaxis_range=[0.9 * plot_df[y_col].min(), 1.1 * plot_df[y_col].max()],
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width, showlegend=False,
    )
    fig.update_xaxes(title_standoff=0, **_AXIS)
    fig.update_yaxes(title_standoff=2, **_AXIS)
    return fig


# %%
def _stacked_bars(
    plot_df, x_col, y_col, c_col, cmap, *,
    height=350, width=500, xrot=0, font_size=18,
    show_hlines=True, show_legend=True,
    show_xtitle=True, show_ytitle=True,
    show_xticklabels=True, show_yticklabels=True,
):
    """Stacked bar chart of a normalized quantity (0-1) split by category.

    Low-level primitive behind `plot_polarity_stacked_bars`: each x position
    gets one stacked bar whose segments are the `y_col` values broken down by
    `c_col`, expected to sum to ~1. Segment colors and stack/legend order
    follow `cmap`'s key order.

    Args:
        plot_df (pd.DataFrame): Long-form data; one row per (x, category) with its value.
        x_col (str): Column for the categorical x-axis positions (one bar per value).
        y_col (str): Column for the stacked segment heights.
        c_col (str): Column holding each row's category (the color/stack split).
        cmap (dict[str, str]): Ordered {category: color} map. Key order sets the
            stacking order, legend order, and `category_orders` for `c_col`.
        height (int): Figure height in px. Defaults to 350.
        width (int): Figure width in px. Defaults to 500.
        xrot (int): X tick-label rotation in degrees. Defaults to 0.
        font_size (int): Base font size in px (arial). Defaults to 18.
        show_hlines (bool): Draw the y=0 and y=1 reference lines. Defaults to True.
        show_legend (bool): Show the category legend. Defaults to True.
        show_xtitle (bool): Show the x-axis title. Defaults to True.
        show_ytitle (bool): Show the y-axis title. Defaults to True.
        show_xticklabels (bool): Show the x tick labels. Defaults to True.
        show_yticklabels (bool): Show the y tick labels (the 0/1 ticks).
            Defaults to True.

    Returns:
        go.Figure: The assembled stacked bar figure.
    """
    fig = px.bar(
        plot_df, x=x_col, y=y_col, color=c_col,
        color_discrete_map=cmap, category_orders={c_col: list(cmap.keys())},
    )
    if show_hlines:
        fig.add_hline(y=0, line_color="black", line_width=1)
        fig.add_hline(y=1, line_color="black", line_width=1)
    fig.update_layout(
        boxmode="group", xaxis_tickangle=xrot,
        yaxis_range=[-0.01, 1.01],
        font=dict(family="arial", size=font_size),
        height=height, width=width,
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=show_legend,
    )
    fig.update_xaxes(
        showgrid=False, tickvals=plot_df[x_col].values,
        title_text=x_col if show_xtitle else None,
        showticklabels=show_xticklabels,
    )
    fig.update_yaxes(
        showgrid=False, tickvals=[0, 1],
        title_text=y_col if show_ytitle else None,
        showticklabels=show_yticklabels,
    )
    return fig


# %%
def _violin_median(
    df, val_col, cat_col, *, labels=None, height=800, width=500,
    xaxis_title="", xaxis_tickvals=None, xaxis_range=None,
):
    """Horizontal violin of `val_col` across categories in `cat_col`, with a
    median marker per category. Categories appear top-to-bottom in the order
    they first occur in `df`."""
    cats = df[cat_col].drop_duplicates().tolist()
    pos = {c: i for i, c in enumerate(cats)}
    plot_df = df.copy()
    plot_df["__y"] = plot_df[cat_col].map(pos)
    medians = (
        plot_df.groupby(cat_col, observed=False)[val_col].median()
        .reindex(cats).reset_index()
    )
    medians["__y"] = medians[cat_col].map(pos)

    fig = px.violin(plot_df, x=val_col, y="__y", orientation="h")
    fig.update_traces(line_color="gray", marker=dict(size=3, opacity=0.4, color="gray"))
    fig.add_scatter(
        x=medians[val_col], y=medians["__y"], mode="markers",
        marker=dict(color="black", size=10), showlegend=False,
    )
    fig.update_layout(
        font=dict(family="arial", size=18),
        xaxis_title=xaxis_title, yaxis_title="",
        paper_bgcolor="white", plot_bgcolor="white",
        height=height, width=width, showlegend=False,
    )
    ticktext = labels if labels is not None else cats
    fig.update_yaxes(
        tickmode="array", tickvals=list(range(len(cats))), ticktext=ticktext,
        title_standoff=2, autorange="reversed", **_AXIS,
    )
    xkw = {"title_standoff": 0, **_AXIS}
    if xaxis_tickvals is not None:
        xkw["tickvals"] = xaxis_tickvals
    if xaxis_range is not None:
        xkw["range"] = xaxis_range
    fig.update_xaxes(**xkw)
    return fig


# %%
def _add_gaussian_ellipses(fig, params_df, *, example_bid=None, color="red", y_scale=1.0):
    """Overlay fitted 2D-Gaussian ellipses + centres onto an existing figure.

    `params_df` must have columns x0, y0, a, b, phi; if `example_bid` is given
    and matches `params_df['bodyId']`, that row is drawn with a thicker line.
    `y_scale` multiplies the y coordinate of every drawn point — pass
    `y_scale=np.sqrt(3)` to un-scale Gaussian fits (which were done in
    isotropic y/sqrt(3) units) back onto raw hex coords.
    """
    t = np.linspace(0, 2 * np.pi, 60)
    ct, st = np.cos(t), np.sin(t)
    fig.add_trace(go.Scatter(
        x=params_df["x0"].values, y=y_scale * params_df["y0"].values, mode="markers",
        marker=dict(color=color, size=6, symbol="cross"), showlegend=False,
    ))
    for _, row in params_df.iterrows():
        highlight = example_bid is not None and row.get("bodyId") == example_bid
        lw = 4 if highlight else 1
        cp, sp = np.cos(row["phi"]), np.sin(row["phi"])
        x = row["x0"] + row["a"] * cp * ct - row["b"] * sp * st
        y_fit = row["y0"] + row["b"] * cp * st + row["a"] * sp * ct
        fig.add_trace(go.Scatter(
            x=x, y=y_scale * y_fit, mode="lines",
            line=dict(color=color, width=lw), showlegend=False,
        ))
    return fig


# %%
def _eyesymbol(color_segments):
    """Stylised 5-segment eye pictogram: 4 outer colored wedges + colored inner
    ellipse. `color_segments`: 5 CSS color strings, ordered
    (sector1, sector2, sector3, sector4, center)."""
    a, b = 1.2, 2
    a2, b2 = 0.5, 0.8
    N = 200
    theta = np.linspace(-np.pi / 4, 7 * np.pi / 4, N)
    wedges = [
        (-np.pi / 4, np.pi / 4), (np.pi / 4, 3 * np.pi / 4),
        (3 * np.pi / 4, 5 * np.pi / 4), (5 * np.pi / 4, 7 * np.pi / 4),
    ]
    # index mapping: outer wedges (R, U, L, D) draw from sectors 3, 2, 1, 4
    wedge_colors = [color_segments[2], color_segments[1], color_segments[0], color_segments[3]]
    fig = go.Figure()
    for (start, end), color in zip(wedges, wedge_colors):
        mask = (theta >= start) & (theta <= end + 2 * np.pi / N)
        x = np.concatenate(([0], a * np.cos(theta[mask]), [0]))
        y = np.concatenate(([0], b * np.sin(theta[mask]), [0]))
        fig.add_trace(go.Scatter(
            x=x, y=y, fill="toself", fillcolor=color,
            line=dict(color="black", width=1), mode="lines", showlegend=False,
        ))
    fig.add_trace(go.Scatter(
        x=a2 * np.cos(theta), y=b2 * np.sin(theta),
        fill="toself", fillcolor=color_segments[4],
        line=dict(color="black", width=1.5), mode="lines", showlegend=False,
    ))
    fig.update_layout(
        width=500, height=500,
        xaxis=dict(scaleanchor="y", visible=False),
        yaxis=dict(visible=False),
        plot_bgcolor="white", margin=dict(l=0, r=0, t=0, b=0),
    )
    return fig


# %%
def _directedness_triangle(plot_df, c_col, t_col, cmap, t_stars=(), *, height=450, width=450):
    fig = px.scatter_ternary(
        plot_df, a="frac_la", b="frac_fb", c="frac_ff", hover_name=t_col,
        color=c_col, color_discrete_map=cmap,
    )
    fig.update_traces(marker=dict(size=7, opacity=0.4))
    for t in t_stars:
        sub = plot_df[plot_df[t_col] == t]
        if sub.empty:
            continue
        fig.add_trace(go.Scatterternary(
            a=sub["frac_la"].values, b=sub["frac_fb"].values, c=sub["frac_ff"].values,
            mode="markers+text",
            marker=dict(size=7, color=cmap[sub[c_col].values[0]], opacity=1,
                        line=dict(color="black", width=1)),
            text=[s.split("_")[0] for s in sub[t_col]],
            textposition="middle left",
        ))
    fig.update_layout(
        plot_bgcolor="white", paper_bgcolor="white",
        font=dict(family="arial", size=18),
        ternary=dict(
            sum=1,
            aaxis=dict(title="la", showgrid=False, showline=True, linecolor="black", tickvals=[0, 0.5, 1]),
            baxis=dict(title="fb", showgrid=False, showline=True, linecolor="black", tickvals=[0, 0.5, 1]),
            caxis=dict(title="ff", showgrid=False, showline=True, linecolor="black", tickvals=[0, 0.5, 1]),
        ),
        height=height, width=width, showlegend=False,
    )
    return fig


# %%
def _attach_flow_layers(paths_df: pd.DataFrame, flow_type: pd.DataFrame) -> pd.DataFrame:
    """Merge per-instance hitting times onto pre/post, then strip `_R`/`_L`."""
    out = paths_df.merge(
        flow_type[['instance', 'hitting_time']], how='left',
        left_on='pre', right_on='instance',
    ).rename(columns={'hitting_time': 'pre_layer'}).drop(columns=['instance'])
    out = out.merge(
        flow_type[['instance', 'hitting_time']], how='left',
        left_on='post', right_on='instance',
    ).rename(columns={'hitting_time': 'post_layer'}).drop(columns=['instance'])
    out['pre'] = [s[:-2] for s in out['pre']]
    out['post'] = [s[:-2] for s in out['post']]
    return out


# %%
def plot_pathway(
    paths_dict: dict, instance: str, flow_type: pd.DataFrame,
    *,
    thre_cumsum: float = 0.4, thre_step_min: float = 0.0,
    top_n: int | None = None,
    neuron_to_color: dict | None = None,
    neuron_to_sign: dict | None = None, sign_color_map: dict | None = None,
    figsize: tuple | None = None,
    frac_text_pos: tuple = (0.85, 0.8), frac_text_size: int = 18,
    node_size: int = 1000, node_text_size: int = 18,
    edge_text: bool = False, edge_text_size: int = 10,
    weight_decimals: int = 2, highlight: bool = True,
    save_path: Path | None = None,
    show: bool = True,
    show_frac: bool = True,
):
    """Render an example-neuron pathway diagram via `plot_paths`.

    `paths_dict` is the output of `pre.get_paths_to_instance` /
    `pre.get_participation_paths` / `pre.get_cb_paths_to_instance`.
    Filter thresholds are applied here (not cached). Returns
    `(conn_paths_frac, fig)` — the rendered displayed-path weight fraction
    (0.0 if no surviving paths) and the matplotlib figure (`None` when no
    figure was drawn, i.e. on the early-exit paths).

    When `top_n` is set, selection switches to the strongest `top_n` paths
    (via `filter_all_paths_to_top_n`) instead of the `thre_cumsum`
    cumulative-weight set; `thre_cumsum` is ignored in that mode while
    `thre_step_min` still applies.
    """
    from connectome_interpreter.utils import plot_paths
    import matplotlib.pyplot as plt

    all_paths = paths_dict['paths']
    hit_inst = paths_dict['hit_inst']
    if not all_paths:
        print(f'no paths for {instance}')
        return 0.0, None

    if figsize is None:
        figsize = (1.5 + hit_inst * 3, 8)

    if top_n is not None:
        from utils.path_filtering import filter_all_paths_to_top_n
        filtered, w_filter, w_all, _ = filter_all_paths_to_top_n(
            all_paths, top_n=top_n, thre_step_min=thre_step_min,
        )
    else:
        from utils.path_filtering import filter_all_paths_to_top_set
        filtered, w_filter, w_all, _ = filter_all_paths_to_top_set(
            all_paths, thre_cumsum=thre_cumsum, thre_step_min=thre_step_min,
        )
    conn_paths_frac = w_filter / w_all if w_all > 0 else 0.0

    if isinstance(filtered, list):
        dfs = [p for p in filtered if p is not None and not p.empty]
        if not dfs:
            print(f'no surviving paths for {instance} after filter')
            return conn_paths_frac, None
        paths_df = pd.concat(dfs, axis=0)
    else:
        if filtered is None or filtered.empty:
            print(f'no surviving paths for {instance} after filter')
            return conn_paths_frac, None
        paths_df = filtered

    paths_df = paths_df[['pre', 'post', 'weight']].drop_duplicates()
    paths_df = paths_df[paths_df['pre'] != paths_df['post']]
    paths_df = _attach_flow_layers(paths_df, flow_type)

    highlight_nodes = [instance[:-2]] if highlight else []
    # Pass show=False so plot_paths returns (fig, ax) instead of calling
    # plt.show() — otherwise the inline backend clears pyplot's current
    # figure and the text/savefig below land on a fresh empty figure.
    fig, ax = plot_paths(
        paths_df,
        neuron_to_color=neuron_to_color,
        neuron_to_sign=neuron_to_sign, sign_color_map=sign_color_map,
        interactive=False, node_size=node_size, node_text_size=node_text_size,
        weight_decimals=weight_decimals, edge_text=edge_text, edge_text_size=edge_text_size,
        highlight_nodes=highlight_nodes, save_plot=False, figsize=figsize,
        show=False,
    )
    if show_frac:
        ax.text(
            frac_text_pos[0], frac_text_pos[1], rf'$\mathit{{f}}$ = {conn_paths_frac: .2f}',
            transform=ax.transAxes, fontsize=frac_text_size,
        )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(
            str(save_path), bbox_inches='tight', dpi=300,
            format='pdf', transparent=True,
        )
    if show:
        plt.show()
    plt.close(fig)
    
    return conn_paths_frac, fig


# %%
def plot_comparison_scatter(
    df: pd.DataFrame, x_col: str, y_col: str,
    color_col: str, color_df: pd.DataFrame, *,
    xaxis_title: str, yaxis_title: str, filename_stem: str,
    save_dir: Path | None = None, stars: tuple = (), id_col: str = "instance",
    size: int = 12, width: int = 600, height: int = 600,
    diag_range: tuple | list | None = None, axis_range: tuple | list | None = None,
    log_axes: bool = False,
):
    """Scatter of a quantity from two models (x vs y), colored by `color_col`,
    with optional identity diagonal, log axes, and shared axis range."""
    fig = _scatter(
        df, x_col, y_col, color_col, id_col, color_df, stars,
        height=height, width=width, size=size,
    )
    if diag_range is not None:
        fig.add_trace(go.Scatter(
            x=list(diag_range), y=list(diag_range), mode="lines",
            line=dict(color="gray", width=2, dash="dash"), showlegend=False,
        ))
    fig.update_layout(xaxis_title=xaxis_title, yaxis_title=yaxis_title)
    if log_axes:
        fig.update_xaxes(type="log")
        fig.update_yaxes(type="log")
    if axis_range is not None:
        fig.update_xaxes(range=list(axis_range))
        fig.update_yaxes(range=list(axis_range))
    _save(fig, save_dir, f"{filename_stem}.pdf")
    return fig
