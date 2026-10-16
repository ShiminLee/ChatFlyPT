"""Reusable helpers extracted from `make_figures/paper_figures.ipynb`.

Self-contained, argument-driven functions the paper-figure notebook imports
rather than defining inline:

- Signed polarity-bar family (Main / SI polarity panels): `build_sign_map`,
  `compute_signed_totals`, `add_abs_l1_l2_prediction`, `build_cmp`,
  `group_cmp_like_figures`, `plot_signed_bars`.
- `_instance_effective_n` — per-instance effective number of inputs
  (exp of Shannon entropy of the normalised feature vector), joined to cluster.
- `_wrap` — wrap a label on hyphens / spaces for multi-line axis ticks.
- `plot_gaussian_params` — RF-ellipse-fit scatter, used by the VPN/VCBN
  galleries (former `utils/plotting_functions.py`, retired 2026-08-20).
"""
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from connectome_interpreter.path_finding import find_paths_of_length, group_paths
from connectome_interpreter import signed_effective_conn_from_paths

from utils import config, polarity

NR_HOPS_MAX = 10


def build_sign_map(meta_df):
    """Instance → ±1 with overrides: L1_R flipped to +1, sign==0 coerced to +1."""
    s = meta_df[['instance', 'sign']].drop_duplicates('instance').set_index('instance')['sign'].copy()
    if 'L1_R' in s.index:
        s.loc['L1_R'] = 1
    s[s == 0] = 1
    return s.astype(int).to_dict()


def compute_signed_totals(A, inidx, outidx, idx_to_instance, inst_to_sign,
                           nr_hops_max=NR_HOPS_MAX):
    total = None
    for k in range(1, nr_hops_max + 1):
        raw = find_paths_of_length(A, inidx, outidx, k, quiet=True)
        if raw is None or raw.empty:
            continue
        paths = group_paths(raw, idx_to_instance, idx_to_instance, combining_method='mean')
        tmp_e, tmp_i = signed_effective_conn_from_paths(paths, wide=True, idx_to_nt=inst_to_sign)
        tmp = tmp_e.sub(tmp_i, fill_value=0)
        print(f'  k={k}: {len(raw):,} path-edges, signed table {tmp.shape}')
        total = tmp if total is None else total.add(tmp, fill_value=0)
    return total


def add_abs_l1_l2_prediction(cmp_df, side_char=config.SIDE_CHAR):
    """Predict ON/OFF from absolute L1/L2 propagated weights."""
    cmp_df = cmp_df.copy()
    suffix = f'_{side_char.upper()}'
    l1_col, l2_col = f'L1{suffix}', f'L2{suffix}'
    l1 = cmp_df[l1_col].abs() if l1_col in cmp_df.columns else pd.Series(0.0, index=cmp_df.index)
    l2 = cmp_df[l2_col].abs() if l2_col in cmp_df.columns else pd.Series(0.0, index=cmp_df.index)
    cmp_df['predicted'] = np.where(l1 > l2, 'ON', 'OFF')
    cmp_df['match'] = cmp_df['predicted'] == cmp_df['Contrast selectivity']
    return cmp_df


def build_cmp(total_signed, labels_df, seed_instance_list, meta_df):
    """Row-normalized (absolute-value) signed feature table joined with polarity labels.
    Adds `main_groups`, `predicted`, `match` columns."""
    feat = total_signed.T
    feat = feat.reindex(index=labels_df['instance'].values,
                        columns=list(seed_instance_list), fill_value=0).fillna(0)
    abs_sum = feat.abs().sum(axis=1)
    feat_norm = feat.div(abs_sum.where(abs_sum > 0, 1), axis=0).reset_index(names='instance')
    merged = labels_df.merge(feat_norm, on='instance', how='inner').merge(
        meta_df[['instance', 'main_groups']].drop_duplicates('instance'),
        on='instance', how='left',
    )
    return add_abs_l1_l2_prediction(merged)


def group_cmp_like_figures(cmp_df, side_char=config.SIDE_CHAR, sort='cb'):
    """Average cell types per spreadsheet entry (one bar per group_label) and
    re-normalize by |sum|. `sort='cb'` uses the CB category
    order; `sort='ol'` keeps spreadsheet order with the VPN reorder
    (VS/HS/LC10a/MeTu3c after LC6), matching Main 2d."""
    suffix = f'_{side_char.upper()}'
    feature_cols = [c for c in cmp_df.columns if c.endswith(suffix) and c != 'instance']
    grouped = (
        cmp_df.groupby(['group_label', 'Contrast selectivity'], sort=False)[feature_cols]
        .mean()
        .reset_index()
    )
    abs_sum = grouped[feature_cols].abs().sum(axis=1)
    grouped[feature_cols] = grouped[feature_cols].div(abs_sum.where(abs_sum > 0, 1), axis=0)
    grouped['instance'] = grouped['group_label'] + suffix
    grouped['main_groups'] = (
        cmp_df[['group_label', 'main_groups']]
        .drop_duplicates('group_label')
        .set_index('group_label')['main_groups']
        .reindex(grouped['group_label'].values)
        .values
    )
    grouped = add_abs_l1_l2_prediction(grouped, side_char=side_char)
    if sort == 'cb':
        grouped['_cat'] = grouped['group_label'].apply(polarity._cb_polarity_cat)
        grouped = grouped.sort_values(['_cat', 'group_label']).drop(columns='_cat')
    elif sort == 'ol':
        grouped = polarity._reorder_after(
            grouped, polarity._VPN_ORDER_ANCHOR, polarity._VPN_ORDER_AFTER)
    return grouped.reset_index(drop=True)


def plot_signed_bars(cmp_df, colored_seed, filename_stem, width_scale=1.0, height_scale=1.0):
    from utils.plotting import _POLARITY_DISPLAY_RENAMES

    suffix = '_R'
    seed_cols = [s + suffix for s in colored_seed.index if s + suffix in cmp_df.columns]
    long = cmp_df[['instance'] + seed_cols].copy()
    long['out type'] = long['instance'].str[:-2]
    melted = long.melt(id_vars=['out type'], value_vars=seed_cols,
                        var_name='in type', value_name='effective weight')
    melted['in type'] = melted['in type'].str[:-2]
    cmap = colored_seed['color'].to_dict()
    width = int(max(400, 35 * len(cmp_df) + 150) * 0.8 * width_scale)
    height = int(350 * height_scale)
    fig = px.bar(melted, x='out type', y='effective weight', color='in type',
                 color_discrete_map=cmap, category_orders={'in type': list(cmap.keys())})
    fig.add_hline(y=0, line_color='black', line_width=1)
    fig.add_hline(y=1, line_color='black', line_width=1)
    fig.add_hline(y=-1, line_color='black', line_width=1)
    fig.update_layout(xaxis_tickangle=-45, yaxis_range=[-1.05, 1.05],
                       font=dict(family='arial', size=18), height=height, width=width,
                       plot_bgcolor='white', paper_bgcolor='white')
    order = cmp_df['instance'].str[:-2].tolist()
    labels_x = [
        f'<b>{_POLARITY_DISPLAY_RENAMES.get(t, t)}</b>' if p == 'OFF'
        else _POLARITY_DISPLAY_RENAMES.get(t, t)
        for t, p in zip(order, cmp_df['Contrast selectivity'])
    ]
    fig.update_xaxes(showgrid=False, categoryorder='array', categoryarray=order,
                      tickvals=order, ticktext=labels_x)
    fig.update_yaxes(showgrid=False, tickvals=[-1, 0, 1])
    out_dir = config.FIG_DIR / 'exp_comparison'
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = out_dir / f'{config.FIG_DATASET}_{filename_stem}'
    fig.write_image(str(stem) + '.pdf')
    fig.write_html(str(stem) + '.html')
    return fig


def _instance_effective_n(feature_df, cluster_df):
    """(per-instance eff-#inputs joined to cluster, n_inputs)."""
    in_cols = [c for c in feature_df.columns if c.endswith(('_L', '_R'))]
    vals = feature_df[in_cols].fillna(0.0)
    row_sum = vals.sum(1)
    valid = row_sum > 0
    p = vals[valid].div(row_sum[valid], axis=0)
    entropy = -(p * np.log(p.where(p > 0))).fillna(0.0).sum(1)
    df = pd.DataFrame({'instance': p.index, 'eff_n': np.exp(entropy).values})
    return df.merge(cluster_df[['instance', 'cluster']], on='instance'), len(in_cols)


def _wrap(name):
    return name.replace("-", "-\n").replace(" ", "\n")


def plot_gaussian_params(params_df, example_bid=None, height=500, width=500, fac=1):

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=params_df['x0'].values,
        y=fac*params_df['y0'].values,
        mode='markers',
        marker=dict(color='red', size=6, symbol='cross'),
        showlegend=False,
    ))
    for i in range(params_df.shape[0]):
        params_single_df = params_df.iloc[i]
        if params_single_df.bodyId==example_bid:
            line_width=4
            fig.add_trace(go.Scatter(
                x=[params_single_df['x0']],
                y=[fac*params_single_df['y0']],
                mode='markers',
                marker=dict(color='red', size=10, symbol='cross'),
                showlegend=False,
            ))
        else:
            line_width=1
        fig.add_trace(go.Scatter(
            x=params_single_df['x0'] +\
                params_single_df['a']*np.cos(params_single_df['phi'])*np.cos(np.linspace(0, 2*np.pi, 20)) - \
                params_single_df['b']*np.sin(params_single_df['phi'])*np.sin(np.linspace(0, 2*np.pi, 20)),
            y=fac*params_single_df['y0'] +\
                fac*params_single_df['b']*np.cos(params_single_df['phi'])*np.sin(np.linspace(0, 2*np.pi, 20)) + \
                fac*params_single_df['a']*np.sin(params_single_df['phi'])*np.cos(np.linspace(0, 2*np.pi, 20)),
            mode='lines',
            line=dict(color='red', width=line_width),
            showlegend=False,
        ))
    fig.update_yaxes(scaleanchor = "x", scaleratio = 1)
    fig.update_layout(
        font={
            "family":'Arial'
            , "size" : 14
        }
        , paper_bgcolor="rgba(255,255,255,255)"
        , plot_bgcolor="rgba(255,255,255,255)"
        , height=500
        , width=width
    )

    return fig
