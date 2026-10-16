"""Plotting functions grouped into sections (0..11) by figure group — facade.

This module used to hold all ~136 `plot_*` functions directly (5,482 lines).
It's now a thin re-exporting facade over one module per section
(`utils/plotting_<section>.py`, split 2026-08-20), so existing call sites
(`from utils import plotting as plot; plot.plot_*(...)`, including
private-looking `plot._helper`/`plot._CONSTANT` access) keep working
unchanged. Section 11 ("Model comparisons") has no functions and so has no
corresponding module. See each submodule's docstring for its section's scope.
"""
from utils.config import FIG_DATASET as DATASET  # noqa: F401  (used as `plot.DATASET`)

from utils.plotting_shared import (  # noqa: F401  section 0 — shared plot helpers
    set_html_output,
    _SIDE_WORD,
    _pie, _save, _AXIS,
    _hist_lines, _hist_lines_v, _hist_cumsum_lines, _hist_cumsum_lines_v, _hist_sum_lines,
    _scatter, _box, _box_scatter, _heatmatrix, _point_line, _stacked_bars, _violin_median,
    _add_gaussian_ellipses, _eyesymbol, _directedness_triangle, _attach_flow_layers,
    plot_pathway, plot_comparison_scatter,
)
# `_WRITE_HTML` is deliberately NOT re-exported here: it's mutated in place by
# `set_html_output()` inside `plotting_shared`, and readers elsewhere
# (e.g. `plotting_clusters._save_graph_html`) re-import it live from
# `utils.plotting_shared` rather than trusting a copy. A copy bound here
# would silently go stale after any `set_html_output()` call.

from utils.plotting_inventory import (  # noqa: F401  section 1 — Inventory
    plot_inventory_full_brain, plot_inventory_right_ol, plot_inventory_by_main_group,
)

from utils.plotting_flow import (  # noqa: F401  section 2 — Flow
    _FLOW_PLOT_TYPES, _FLOW_STAR_TYPES, _plot_types,
    plot_coords, plot_flow_sector_division, plot_flow_by_main_groups, plot_flow_by_sign,
    plot_flow_type_scatter_and_triangle, plot_flow_type_boxes, plot_flow_numerous_type_boxes,
    plot_flow_connectivity_heatmap, plot_flow_threshold_choice, plot_ol_layers_violin,
)

from utils.plotting_clusters import (  # noqa: F401  section 3 — Clusters
    _CLUSTER_BLUES, _VCBN_CC_TO_OC, _CLUSTER_TYPES, _OUT_TYPES, _PART_TYPES,
    _cluster_color_df, plot_clusters_dendrograms, plot_clusters_correlation_matrix,
    plot_clusters_example_features, plot_clusters_pathway_features, plot_clusters_scatter,
    plot_clusters_participation, plot_clusters_intra_connectivity, plot_clusters_pathway_connectivity,
    _prepare_flow_sankey_edges, _render_flow_sankey, plot_clusters_sankey,
    plot_visual_system_sankey, plot_vcn_input_pie, _save_graph_html, plot_clusters_graph,
)

from utils.plotting_lr_clustering import (  # noqa: F401  section 4 — LR clustering
    plot_lr_clustering_sweep, plot_lr_cluster_confusion,
)

from utils.plotting_polarity import (  # noqa: F401  section 5 — Polarity experimental comparisons
    _POLARITY_DISPLAY_RENAMES, plot_polarity_stacked_bars,
)
# `_POLARITY_DISPLAY_RENAMES` IS meant to be rebound externally (SI-14 flywire
# display rename in `paper_figures.py` does `plot._POLARITY_DISPLAY_RENAMES =
# {...}`) — `plot_polarity_stacked_bars` and `figure_helpers.plot_signed_bars`
# both re-import it fresh from this module at call time, so they always see
# the current value regardless of where they're defined.

from utils.plotting_rf import (  # noqa: F401  section 6 — Propagated RFs
    _RF_EXAMPLE_BID, _RF_STAR_EXAMPLES, _RF_STAR_OUTPUT, _RF_STAR_INTERNAL,
    plot_rf_example, plot_rf_estimated_overlay, plot_rf_positions_across_neurons,
    plot_input_rf_example, _rf_axis_update, plot_rf_boxes_by_celltype,
    plot_rf_scatters_by_main_groups, plot_rf_scatters_by_cluster, plot_rf_histograms,
    _SIZE_DIFF_BINS, _SIZE_DIFF_TICKS, _SIZE_DIFF_LABELS, _bin_size_diff,
    plot_rf_sizediff_internal, plot_rf_sizediff_output, plot_rf_connectivity_histograms,
    plot_rf_connectivity_boxes,
)

from utils.plotting_rf_experimental import (  # noqa: F401  section 7 — RF size experimental comparisons
    plot_rf_size_example_comparison, plot_rf_size_type_scatters, _mixed_color_df,
    _add_2x_guide_lines, plot_rf_size_vs_experiment, plot_rf_aspect_vs_experiment,
    plot_rf_size_vs_experiment_cc,
)

from utils.plotting_cb_vic import (  # noqa: F401  section 8 — CB VIC
    plot_vic_cumsum, plot_cumfrac_by_roi, plot_vic_layer_histogram, plot_vic_vs_layer_scatter,
)

from utils.plotting_cb_lr_clustering import (  # noqa: F401  section 9 — LR CB clustering
    _CB_CLUSTER_TYPES, _cb_cluster_labels, _cb_cluster_side,
    plot_cb_directedness_histograms, plot_cb_directedness_triangle,
    plot_cb_clusters_example_features, plot_cb_clusters_pathway_features,
    plot_ol_cb_cluster_connectivity, _CB_ROI_ORDER, _fmt_km, plot_cb_clusters_rois,
    plot_cb_participation_n_clu_out, plot_cb_clusters_graph, plot_out_cb_highres_sub,
    plot_rf_example_pos_par, plot_ol_layers_hit_diff, plot_cb_layer_hist_left,
    plot_cb_clusters_size_vs_cluster, plot_cb_clusters_scatter, plot_cb_clusters_intra_connectivity,
    plot_cb_lr_clustering_sweep, plot_cb_lr_cluster_confusion, _plot_layer_lr_scatter,
    plot_cb_layer_lr_homologue, plot_cb_layer_lr_comparison, plot_cb_vic_lr_homologue,
    plot_cb_vic_binocularity,
)

from utils.plotting_cb_rf import (  # noqa: F401  section 10 — CB RFs
    plot_cb_rf_r2_summary, plot_cb_rf_size_summary, plot_cb_ahull_size_summary,
    plot_cb_vcbn_ahull_size_comparison, plot_ol_ahull_size_comparison, plot_rf_size_per_layer,
    plot_propagated_arf_example, plot_combined_arf, plot_ol_to_cb_heatmap, _draw_sector_polar,
    plot_ol_roi_coverage_per_roi, plot_ol_roi_coverage_summary,
)
