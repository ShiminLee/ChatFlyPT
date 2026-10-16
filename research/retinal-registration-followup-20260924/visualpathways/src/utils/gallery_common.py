"""
Shared figure builders for the per-instance galleries.

Domain-specific (but data-parameterized) figure constructors used by both
``make_figures/gallery_vpn`` and ``make_figures/gallery_vcbn``. The generic
export/tiling primitives they rely on live in :mod:`utils.pdf_report`.
"""

from utils.pdf_report import add_segment_labels
from utils.plotting import _stacked_bars

# stacked-bar strip aspect (PDF points; also the px aspect for HTML)
BAR_W, BAR_H = 36, 260


def make_bar_fig(inst, feat_norm_long, cmap, *, height, font_size, width=None,
                 bar_w=BAR_W, bar_h=BAR_H):
    """Stacked VI effective-weight bar for one instance, sized for the target medium.

    PDF wants a tall (height=1200), thin (aspect bar_w/bar_h) figure scaled down into a
    36x260 pt box for crisp text; HTML wants a screen-sized figure with an explicit,
    wider `width` so plotly hover tooltips aren't clipped by a too-narrow graph div.
    `width=None` -> derive width from height by the PDF strip aspect.

    `feat_norm_long` is the melted (instance, VI, effective weight) table and `cmap`
    the VI->color dict; both are built once per gallery script and passed in.
    """
    plot_df = feat_norm_long[feat_norm_long.instance == inst]
    fig = _stacked_bars(
        plot_df, "instance", "effective weight", "VI",
        cmap, height=height,
        width=width if width is not None else round(height * bar_w / bar_h),
        show_legend=False, show_hlines=False,
        show_xtitle=False, show_ytitle=False,
        show_xticklabels=False, show_yticklabels=False,
    )
    add_segment_labels(fig, cmap, font_size=font_size)
    return fig
