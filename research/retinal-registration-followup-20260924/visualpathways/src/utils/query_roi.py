"""Primary-ROI lookup helper (`get_primary_rois`). Imported as a `utils`
submodule — expects `src` on `sys.path` (handled by the entry-point
scripts/notebooks)."""

import pandas as pd
import neuprint

from utils import olc_client
c = olc_client.connect(verbose=False)

def get_primary_rois(branch='CentralBrain', remove_unspecified=True):
    """Return the primary ROIs under a given branch of the ROI hierarchy.

    Parameters
    ----------
    branch : str
        Name of the branch under which to collect primary ROIs. Possible
        values are the keys of
        ``neuprint.queries.fetch_roi_hierarchy(include_subprimary=False)['CNS']``
        plus ``'CNS'`` itself. Pass an invalid value to have the valid
        options listed in the raised error message.
    remove_unspecified : bool
        If True, drop any ROI whose name contains 'unspecified'.

    Returns
    -------
    list of str
        Sorted list of primary ROI names under the requested branch.
    """
    hier = neuprint.queries.fetch_roi_hierarchy(
        include_subprimary=False, mark_primary=False)
    primary = set(neuprint.queries.fetch_primary_rois())

    valid_branches = set(hier['CNS'].keys()) | {'CNS'}
    if branch not in valid_branches:
        raise ValueError(
            f"Invalid branch '{branch}'. "
            f"Must be one of {sorted(valid_branches)}.")

    def find_subtree(tree, target):
        for name, children in tree.items():
            if name == target:
                return children or {}
            if children:
                hit = find_subtree(children, target)
                if hit is not None:
                    return hit
        return None

    def all_rois(tree):
        for name, children in tree.items():
            yield name
            if children:
                yield from all_rois(children)

    subtree = find_subtree(hier, branch)
    if subtree is None:
        raise ValueError(f"Branch '{branch}' not found in the ROI hierarchy.")

    rois = sorted(set(all_rois(subtree)) & primary)
    if remove_unspecified:
        rois = [roi for roi in rois if 'unspecified' not in roi]
    return rois

