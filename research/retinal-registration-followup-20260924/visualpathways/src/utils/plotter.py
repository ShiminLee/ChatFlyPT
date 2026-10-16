import pickle
import io
import os
import re
import time
import unicodedata
from pathlib import Path
import logging
from typing import Union
import datetime
import warnings

from dotenv import find_dotenv, load_dotenv
import pandas as pd
import numpy as np
from IPython.display import Image, display
import PIL
from PIL import ImageChops, ImageDraw, ImageFont
from matplotlib import font_manager
from sklearn.decomposition import PCA
from scipy.spatial.transform import Rotation as scipy_R # define rotation matrix
import alphashape #make alpha shape

import plotly.graph_objects as go
import plotly.express as px
import plotly.colors as po

from neuprint import NeuronCriteria as NC
import navis
import navis.interfaces.neuprint as neu
import cloudvolume as cv
import cloudfiles.monitoring as _cf_monitoring
import kaleido

from utils.ng_view import NG_View

logging.getLogger('navis').setLevel(logging.WARN)


def _patched_transmission_monitor_end_io(self, flight_id, num_bytes):
    # cloud-files truncates transfer start/end times to microseconds and feeds them to
    # an IntervalTree that rejects zero-length intervals. A transfer completing within
    # the same microsecond (routine for tiny, already-cached fetches like mesh/skeleton
    # info JSON) then raises ValueError("Null Interval objects not allowed"). We don't
    # use the bandwidth-monitoring stats this feeds, so nudge end_us forward by 1 to
    # keep the interval non-empty rather than patch the vendored intervaltree/cloud-files
    # packages on disk.
    end_us = int(time.monotonic() * 1e6)
    with self._lock:
        start_us = int(self._in_flight.pop(flight_id) * 1e6)
        if end_us <= start_us:
            end_us = start_us + 1
        self._in_flight_bytes -= num_bytes
        self._intervaltree.addi(start_us, end_us, [flight_id, num_bytes])
        self._total_bytes_landed += num_bytes


_cf_monitoring.TransmissionMonitor.end_io = _patched_transmission_monitor_end_io


def _ensure_kaleido_server():
    # `fig.to_image()` otherwise spins up and tears down a fresh headless-Chrome
    # subprocess on every single call (kaleido's "oneshot" mode) — slow across a
    # per-neuron gallery loop, and the Windows subprocess teardown occasionally
    # raises RuntimeError("Couldn't close or kill browser subprocess"). Starting
    # kaleido's persistent sync server once reuses one browser for the whole
    # session instead; kaleido registers its own atexit cleanup for it.
    kaleido.start_sync_server(silence_warnings=True)


def slugify(value, to_lower=True, allow_unicode=False):
    """
    Convert to ASCII if 'allow_unicode' is False. Convert spaces or repeated
    dashes to single dashes. Remove characters that aren't alphanumerics,
    underscores, or hyphens. Convert to lowercase. Also strip leading and
    trailing whitespace, dashes, and underscores.

    original source: https://docs.djangoproject.com/en/4.1/_modules/django/utils/text/#slugify
    """
    value = str(value)
    if allow_unicode:
        value = unicodedata.normalize("NFKC", value)
    else:
        value = (
            unicodedata.normalize("NFKD", value)
            .encode("ascii", "ignore")
            .decode("ascii")
        )
    value = re.sub(r"[\(\)]", "_", value)
    value = re.sub(r"[^\w\s-]", "", value)
    if to_lower:
        value = value.lower()
    return re.sub(r"[-\s]+", "-", value)

def get_synapses(
    body_id:int
  , ignore_cache:bool=False
) -> pd.DataFrame:
    """
    Retrieve synapses for neuron.

    For performance reasons, this function first attempts to load the synapses from a local cache.
    If this fails, the synapses are retrieved from neuprint and saved in the cache. The cache is
    located at the ${PROJECT_PATH}/cache/synapses. To force a reload, you can either set the
    ignore_cache flag or delete files from the cache.

    Parameters
    ----------
    body_id : int
        the body ID of a neuron
    ignore_cache : int, default = False
        When enabled, load the synapses from neuprint and overwrite the currently cached file.

    Returns
    -------
    """
    cachedir = Path(find_dotenv()).parent / "cache" / "synapses"
    if not cachedir.exists():
        cachedir.mkdir(parents=True)
    syn_file = cachedir / f"sy_{body_id}.pickle"
    if not ignore_cache and syn_file.is_file():
        with syn_file.open('rb') as syn_fh:
            syns = pickle.load(syn_fh)
    else:
        syns = neu.fetch_synapses(NC(bodyId=body_id))
        with syn_file.open('wb') as syn_fh:
            pickle.dump(syns, syn_fh)
    return syns


def get_skeleton(
    body_id:int
  , ignore_cache:bool=False
) -> neu.NeuronList:
    """
    Get a skeleton for a neuron.

    For performance reasons, this function first attempts to load the skeleton from a local cache.
    If this fails, the skeleton is retrieved from neuprint and saved in the cache. The cache is
    located at the PROJECT_PATH/cache/skeletons. To force a reload, you can either set the
    ignore_cache flag or delete files from the cache.

    Parameters
    ----------
    body_id : int
        the body ID of a neuron
    ignore_cache : bool, default = False
        When enabled, load the skeleton from neuprint and overwrite the currently cached file.

    Returns
    -------
    skel : NeuronList
        A skeleton for the neuron of body_id
    """
    cachedir = Path(find_dotenv()).parent / "cache" / "skeletons"
    if not cachedir.exists():
        cachedir.mkdir(parents=True)
    skel_file = cachedir / f"sk_{body_id}.pickle"
    if not ignore_cache and skel_file.is_file():
        with skel_file.open('rb') as skel_fh:
            skel = pickle.load(skel_fh)
    else:
        try:
            skel = neu.fetch_skeletons(NC(bodyId=body_id))
            with skel_file.open('wb') as skel_fh:
                pickle.dump(skel, skel_fh)
        except ValueError:
            skel = None
    return skel


def get_skeletons(
    body_ids:list[int]
  , ignore_cache:bool=False
) -> navis.NeuronList:
    """
    Get skeletons for a list of neurons.

    Retrieve the skeleton of a list of neurons via `get_skeleton()`. Set `ignore_cache` if all of
      the skeletons should be retrieved directly from the database and not your local cache.

    Parameters
    ----------
    body_ids : list of int
        the body IDs of neurons
    ignore_cache : bool, default = False
        When enabled, load the skeleton from neuprint and overwrite the currently cached file.

    Returns
    -------
    skeletons : NeuronList
        A skeleton for the neuron of body_id or a list of skeletons
    """
    skels = []
    for body_id in body_ids:
        skel = get_skeleton(body_id, ignore_cache=ignore_cache)
        skels.append(skel)
    return skels


def get_mesh(
    body_id:int
  , ignore_cache:bool=False
  , lod:int=None
  , save_cache:bool=True
) -> navis.NeuronList:
    """
    Retrieve mesh for neuron.

    For performance reasons, this function first attempts to load the mesh from a local cache.
    If this fails, the mesh is retrieved from neuprint and saved in the cache. The cache is
    located at the PROJECT_PATH/cache/meshes. To force a reload, you can either set the
    ignore_cache flag or delete files from the cache.

    Parameters
    ----------
    body_id : int
        the body ID of a neuron
    ignore_cache : bool, default = False
        When enabled, load the mesh from neuprint and overwrite the currently cached file.
    lod : int, default = None
        Level of detail for multi-resolution meshes. `None` fetches the finest mesh
        (e.g. ~700k vertices); a higher integer (e.g. `2`) fetches a coarser, much
        lighter mesh — useful when rendering many neurons at once. The returned
        coordinates are in the 8 nm-voxel frame regardless of `lod`. Cached per-LOD.
    save_cache : bool, default = True
        When disabled, a freshly-fetched mesh is not written to `cache/meshes` — useful
        for one-off fetches (e.g. large batches of body IDs used only once) that would
        otherwise fill the cache with entries you'll never reuse.

    Returns
    -------
    mesh : NeuronList
        A mesh for the neuron of body_id
    """

    assert os.environ.get('SEGMENTATION_SOURCE'),\
        "Please set the `SEGMENTATION_SOURCE` variable in your environment."
    cachedir = Path(find_dotenv()).parent / "cache" / "meshes"
    mesh_fn = cachedir / (f"ms_{body_id}.pickle" if lod is None else f"ms_{body_id}_lod{lod}.pickle")
    if not ignore_cache and mesh_fn.is_file():
        with mesh_fn.open('rb') as mesh_fh:
            mesh = pickle.load(mesh_fh)
    else:
        ## FIXME(@floesche): remove SEGMENTATION_SOURCE in the future
        mesh = neu.fetch_mesh_neuron(
            NC(bodyId=body_id)
          , seg_source=os.environ['SEGMENTATION_SOURCE']
          , lod=lod)
        if save_cache:
            cachedir.mkdir(parents=True, exist_ok=True)
            with mesh_fn.open('wb') as mesh_fh:
                pickle.dump(mesh, mesh_fh)
    return mesh


def get_meshes(
    body_ids:list[int]
  , ignore_cache:bool=False
  , lod:int=None
  , save_cache:bool=True
) -> navis.NeuronList:
    """
    Retrieve mesh for list of neurons.

    Retrieve the meshes of a list of neurons via `get_mesh()`. Set `ignore_cache` if all of the
      meshes should be retrieved directly from the database and not from your local cache.

    Parameters
    ----------
    body_id : list of int
        the body IDs of neuron(s)
    ignore_cache : bool, default = False
        When enabled, load the mesh from neuprint and overwrite the currently cached file.
    lod : int, default = None
        Level of detail passed to `get_mesh()`. `None` = finest mesh; a higher integer
        (e.g. `2`) fetches lighter meshes, useful for rendering many neurons at once.
    save_cache : bool, default = True
        Passed to `get_mesh()`. Set `False` to skip writing freshly-fetched meshes to
        `cache/meshes` (e.g. for large one-off batches you won't reuse).

    Returns
    -------
    mesh : NeuronList
        A mesh for the neuron of body_id or a list of meshes
    """
    meshes = []
    for body_id in body_ids:
        mesh = get_mesh(body_id, ignore_cache=ignore_cache, lod=lod, save_cache=save_cache)
        meshes.append(mesh)
    return meshes


def get_roi(
    roi_name:str
  , ignore_cache:bool=False
):
    """
    Retrieve ROI.

    For performance reasons, this function attempts to load the ROI from a local cache first.
    If this fails, the ROI is retrieved from neuprint and saved in the cache. The cache is
    located at the PROJECT_PATH/cache/rois. To force a reload, you can either set the
    `ignore_cache` flag or delete files from the cache folder.

    Parameters
    ----------
    roi_name : str
        the name of an ROI
    ignore_cache : bool, default=False
        When enabled, load the mesh from neuprint and overwrite the currently cached file.

    Returns
    -------
    roi : navis.Volume
        An ROI.
    """
    cachedir = Path(find_dotenv()).parent / "cache" / "rois"
    cachedir.mkdir(parents=True, exist_ok=True)
    roi_fn = cachedir / f"roi_{slugify(roi_name)}.pickle"
    if not ignore_cache and roi_fn.is_file():
        with roi_fn.open('rb') as roi_fh:
            roi = pickle.load(roi_fh)
    else:
        roi = neu.fetch_roi(roi_name)
        with roi_fn.open('wb') as roi_fh:
            pickle.dump(roi, roi_fh)
    return roi


def calc_orient(skeleton:pd.DataFrame) -> list[list[float]]:
    """
    Calculate ideal camera orientations for a Neuron.

    Based on the coordinates skeleton nodes, PCA finds the two primary orientations and returns an
      array with three vectors: the first two principal components and an orthogonal vector to
      these two. The first and the third vector are ideal to position the plotly scene camera.

    Parameters
    ----------
    skeleton : NeuronList
        A skeleton

    Returns
    -------
    orient : list[list[float]]
        three vectors in an array
    """
    if len(skeleton[0].edge_coords) == 0:
        return []
    dim_ed = skeleton[0].edge_coords.shape
    etg = skeleton[0].edge_coords.reshape(dim_ed[0]*dim_ed[1], dim_ed[2])
    etg = np.unique(etg, axis=0) # remove duplicates
    et_mn = etg.mean(axis=0)
    et_norm = etg-et_mn
    pca = PCA(n_components=2)
    pca.fit(et_norm)
    orient = pca.components_
    orient = np.append(
        orient
      , [np.cross(orient[0], orient[1])]
      , axis=0)
    return orient


def group_plotter(
    body_ids:list[int]
  , colors:list[tuple[int]]=None
  , plot_roi:str=None
  , shadow_rois:list[str]=None
  , prune_roi:str=None
  , camera_distance:float=2.0
  , ignore_cache:bool=False
  , plot_synapses:bool=True
  , plot_skeleton:bool=True
  , plot_mesh:bool=False
  , view:NG_View=NG_View.SVD
) -> go.Figure:
    """
    Plot a group of neurons.

    To plot a group of neurons, this functions attempts to retrieve the data from the local cache
    first. If that fails, it loads the skeleton and synapses from neuprint and then stores them in
    the local cache.

    Parameters
    ----------
    body_ids : list[int]
        List of neuron body IDs. The body IDs are given as numbers. The first neuron in this list
        will be used to determine the camera position (maximum visibility). The list can contain
        `None`.
    colors   : list[tuple[int]]
        List of colors. The list should be the same length or shorter than the list of body_ids.
        Each color is represented by a tuple with 4 entries in the range [0…1] representing (red,
        green, blue, alpha). If there are less colors than neurons, the neurons that have not
        received a color will be shown in gray (.5, .5, .5, .7). If the list of `body_ids` contains
        `None`, the associated color will be skipped. If no colors are given, the Plotlys Light24
        palette will be used.
    plot_roi : str, default=None
        Name of a ROI that should be plotted together with the Neurons
    shadow_rois : list[str], default=None
        Names of the ROIs that are used as a backdrop for the sliced neurons.
    prune_roi : str, default=None
        All parts of the neurons outside the ROI will be removed from the plot. Special value
            `slice` will cut a slice for the gallery view.
    camera_distance : float, default=2.0
        For smaller groups of neurons, a value between 1 and 2 should be reasonable.
    ignore_cache : bool, default=False

    Returns
    -------
    fig : go.Figure
        Plotly figure with the skeleton and synapses for all the body IDs. The layout is reasonable
        for this application and differs from the plotly default.
    """
    allowed_s_roi = ['ME(R)', 'LO(R)', 'LOP(R)']
    assert shadow_rois is None or set(shadow_rois) <= set(allowed_s_roi)\
      , f"List of shadow ROIs ({shadow_rois}) is different from  allowed ROIs: {allowed_s_roi}."

    fig = go.Figure()
    if not colors:
        colors = [po.hex_to_rgb(col) for col in px.colors.qualitative.Light24]
        colors = colors * int(np.ceil(len(body_ids) / len(colors)))
        colors = [ tuple([cval / 255 for cval in col]) + (1,) for col in colors]
    if plot_roi:
        roi = get_roi(plot_roi, ignore_cache=ignore_cache)
        f_roi = navis.plot3d(roi, inline=False)
        fig.add_trace(f_roi.data[0])
    if shadow_rois:
        for roi in shadow_rois:
            s_roi_layer_m, _ = get_shadow_roi_layers(roi, ignore_cache=ignore_cache)
            f_roi = navis.plot3d(s_roi_layer_m
                , color=['gray']*10
                , alpha=0.4, inline=False)
            for l_mesh in f_roi.data:
                fig.add_trace(l_mesh)
            _ , s_roi_b = get_shadow_roi(roi, ignore_cache=ignore_cache)
            fig.add_trace(
                go.Scatter3d(x=s_roi_b[:,0], y=s_roi_b[:,1], z=s_roi_b[:,2]
              , mode='lines', line={'color':'gray', 'width':3})
            )
    p_roi = None
    if prune_roi and prune_roi != "slice":
        if prune_roi == plot_roi:
            p_roi = roi
        else:
            p_roi = get_roi(prune_roi, ignore_cache=ignore_cache)
    ori = []
    for idx, body_id in enumerate(body_ids):
        if body_id is None:
            continue
        skel = get_skeleton(body_id=body_id, ignore_cache=ignore_cache)
        if skel is None or skel.empty or len(skel.nodes)==0:
            warnings.warn(
                f"Neuron with Body ID {body_id} does not have a skeleton. Skipping cell.")
            continue
        if plot_synapses:
            syns  = get_synapses(body_id=body_id, ignore_cache=ignore_cache)
        if plot_mesh:
            mesh = get_mesh(body_id=body_id, ignore_cache=ignore_cache)
        if prune_roi == "slice":
            vn = np.array([1, 2, 0])
            vn = vn / np.linalg.norm(vn)
            vt1 = np.array([20e3, 40e3, 35e3]) * 0.85
            vt2 = np.array([22e3, 44e3, 35e3]) * 0.95
            d1 = vt1 @ vn
            d2 = vt2 @ vn
            # get the cross section of a neuron
            if plot_skeleton:
                ind_ske_in = skel[0]\
                    .nodes[['x','y','z']]\
                    .apply(lambda row: row @ vn > d1 and row @ vn < d2, axis=1)
                skel = navis\
                    .subset_neuron(skel[0], ind_ske_in.values, inplace=False)
            if plot_mesh:
                msh_v = pd.DataFrame(mesh[0].vertices, columns=['x', 'y', 'z'])
                ind_msh_in = msh_v.apply(
                    lambda row: row @ vn > d1 and row @ vn < d2
                  , axis=1
                )
                mesh = navis.subset_neuron(mesh[0], ind_msh_in.values, inplace=False)
            if plot_synapses:
                syns_in = syns.apply(
                    lambda row: row[['x', 'y', 'z']] @ vn > d1 and row[['x', 'y', 'z']] @ vn < d2
                  , axis =1
                )
                syns = syns[syns_in]
        elif prune_roi:
            if plot_skeleton:
                skel = navis.in_volume(skel, p_roi)
            if plot_synapses:
                syns = syns[navis.in_volume(syns, p_roi)]
        if len(skel.nodes)==0:
            continue
        if colors and idx < len(colors):
            col = colors[idx]
        else:
            col = (0.5, 0.5, 0.5, 0.7) # Gray
        if len(skel.nodes)>0 and len(ori)==0:
            if view==NG_View.SVD:
                ori = calc_orient(skel)
            else:
                #r = scipy_r.from_quat(view.orientation)
                #ori = r.as_matrix()
                # FIXME(@floesche): use non-static view.
                ori = [[-4, 3, -4], [0, 0, 0], [31304.951684997053, 62609.90336999411, 0.0]]
        col_st = f'rgba({int(col[0]*255)},{int(col[1]*255)},{int(col[2]*255)},{col[3]})'
        if plot_skeleton:
            fig_skel = navis.plot3d(skel, inline=False, color=col[0:4])
            if fig_skel.data:
                fig_skel.data[0].hovertemplate = f"body_id: {body_id}<extra></extra>"
                fig_skel.data[0].name=f"ID {body_id}"
                fig_skel.data[0].legendgroup=idx
                fig_skel.data[0].line.color=col_st
                fig.add_trace(fig_skel.data[0])
        if plot_mesh:
            fig_mesh = navis.plot3d(mesh, inline=False, color=col[0:4])
            if fig_mesh.data:
                fig.add_trace(fig_mesh.data[0])
        if plot_synapses:
            fig.add_trace(go.Scatter3d(
                x=syns['x'], y=syns['y'], z=syns['z'],
                customdata=syns['bodyId'],
                legendgroup=idx,
                name=f"{body_id}",
                mode='markers',
                showlegend=False,
                hovertemplate = 'body_id: %{customdata}<extra></extra>',
                marker = {
                    "size":1.5,
                    "color": f"rgba(5,5,5,{col[3]})",
                    "line": {
                        "width":.5
                        , "color": col_st}
                }
            ))
    zoom = 2.0/camera_distance
    if len(ori) > 0:
        fig.update_scenes(
            camera={
                "up": {"x":ori[0][0], "y":ori[0][1], "z":ori[0][2]}
              , "eye": {
                    "x":ori[2][0]*camera_distance
                  , "y":ori[2][1]*camera_distance
                  , "z":ori[2][2]*camera_distance}
              , "center": {"x":0, "y":0, "z":0}
              , "projection": {"type": "orthographic"}}
            )
    fig.update_layout(
        margin={'l':0, 'r':0, 'b':0, 't':20}
      , showlegend=True
      , scene = {
            # "aspectmode": "manual",
            # "aspectratio": {'x': 1, 'y': 1, 'z':1},
            "xaxis" : {
                "showgrid": False
              , "gridcolor": 'rgba(0,0,0,0)'
              , "showbackground": False
              , "visible": False}
          , "yaxis" : {
                "showgrid": False
              , "gridcolor": 'rgba(0,0,0,0)'
              , "showbackground": False
              , "visible": False}
          , "zaxis" : {
                "showgrid":False
              , "gridcolor": 'rgba(0,0,0,0)'
              , "showbackground":False
              , "visible":False}
          , 'aspectmode': 'cube'   # TODO: figure out why this doesn't work on large plots
          , 'aspectratio': {"x":zoom, "y":zoom, "z":zoom}    # TODO: figure out why this
                                                             # doesn't work when ROIs are plotted
    })
    if prune_roi == 'slice':
        fig.update_layout(
            scene = {
                "xaxis" : {
                    "autorange": False
                  , "range": [5400, 46000] # [5437.458372673634, 45525.45722734322]
                }
              , "yaxis" : {
                    "autorange": False
                  , "range": [23000, 42000] #[23135.281677246094, 41176.317932128906]
                }
              , "zaxis" : {
                    "autorange": False
                  , "range":[23500, 43500] #[22801.495262524946, 43050.66538977197]
                }
            }
        )
    return fig


def get_shadow_roi_layers(
    roi_str:str
  , ignore_cache:bool=False
) -> tuple[list[navis.Volume], list[np.ndarray]]:
    """
    Get rough estimates of slices and boundaries for the layers in one of the 3 OL neuropils
    at a fixed location. The received shapes are smoothed by alpha shape. The slicing plane
    (and location) are arbitrariyl chosen.

    Warning: This is not precise and should only be used for quick estimations.

    Parameters
    ----------
    roi_str : str
        name of the neuropil to slice (either `ME(R)`, `LO(R)`, or `LOP(R)`).
    ignore_cache : bool, default=False
        if true, download the meshes from the original data source each time. Otherwise
        use a filesystem cache.

    Returns
    -------
    layer_m : list[navis.Volume]
        a list of thinly sliced volumes, one for each layer of the neuropil.
    layer_bd : list[np.ndarray]
        a list of numpy arrays that contain 3D points representing the outline around layers.
    """
    # load layer meshes and slice them
    vn = np.array([1, 2, 0]) # normal vector
    vn = vn / np.linalg.norm(vn) # normalize
    vt1 = np.array([20e3, 40e3, 35e3]) * 0.85 # translation vector
    if roi_str == 'ME(R)':
        layer = [None] * 10
        layer_sec = [None] * 10
        layer_m = [None] * 10
        layer_bd = [None] * 10
        # selected layers
        for i in [1,3,5,7,9]:
            layer[i-1] = get_roi(
                f'ME_R_layer_{i:02d}'
              , ignore_cache=ignore_cache)   # load layer meshes with a constructed variable name
            layer_sec[i-1] = layer[i-1].section(vn, vt1)
            layer_sec[i-1] = pd.DataFrame(
                layer_sec[i-1].vertices
              , columns=['x','y','z'])  # convert TrackedArray to dataframe
            pts = layer_sec[i-1].values
            layer_m[i-1], layer_bd[i-1] = alpha_plane(pts, vn, alpha=0.0004)
    elif roi_str == 'LO(R)':
        layer = [None] * 7
        layer_sec = [None] * 7
        layer_m = [None] * 7
        layer_bd = [None] * 7
        # selected layers
        for i in [1,3,5,7]:
            layer[i-1] = get_roi(
                f'LO_R_layer_{i}'
              , ignore_cache=ignore_cache) # load layer meshes with a constructed variable name
            layer_sec[i-1] = layer[i-1].section(vn, vt1)
            layer_sec[i-1] = pd.DataFrame(
                layer_sec[i-1].vertices
              , columns=['x','y','z'])  # convert TrackedArray to dataframe
            pts = layer_sec[i-1].values
            layer_m[i-1], layer_bd[i-1] = alpha_plane(pts, vn, alpha=0.0004)
    elif roi_str == 'LOP(R)':
        layer = [None] * 4
        layer_sec = [None] * 4
        layer_m = [None] * 4
        layer_bd = [None] * 4
        # selected layers
        for i in [1,3]:
            layer[i-1] = get_roi(
                f'LOP_R_layer_{i}'
              , ignore_cache=ignore_cache) # load layer meshes with a constructed variable name
            layer_sec[i-1] = layer[i-1].section(vn, vt1)
            layer_sec[i-1] = pd.DataFrame(
                layer_sec[i-1].vertices
              , columns=['x','y','z'])  # convert TrackedArray to dataframe
            pts = layer_sec[i-1].values
            layer_m[i-1], layer_bd[i-1] = alpha_plane(pts, vn, alpha=0.0004)
    return layer_m, layer_bd


def get_shadow_roi(
    roi_str:str
  , ignore_cache:bool=False
) -> tuple[navis.Volume, np.ndarray]:
    """
    Get rough estimates of slices and boundaries for one of the 3 OL neuropils at a fixed location.
    The received shapes are smoothed by alpha shape. The slicing plane (and location) are
    arbitrariyl chosen.

    Warning: This is not precise and should only be used for quick estimations.

    Parameters
    ----------
    roi_str : str
        name of the neuropil to slice (either `ME(R)`, `LO(R)`, or `LOP(R)`).
    ignore_cache : bool, default=False
        if true, download the meshes from the original data source each time. Otherwise
        use a filesystem cache.

    Returns
    -------
    m : navis.Volume
        a thinly sliced volume of the neuropil.
    bd : np.ndarray
        a numpy arrays that contain 3D points representing the outline around the neuropil.
    """
    vn = np.array([1, 2, 0]) # normal vector
    vn = vn / np.linalg.norm(vn) # normalize
    vt1 = np.array([20e3, 40e3, 35e3]) * 0.85 # translation vector
    bound = get_roi(roi_str, ignore_cache=ignore_cache)
    sec = bound.section(vn, vt1)
    # convert TrackedArray to dataframe
    sec = pd.DataFrame(sec.vertices, columns=['x','y','z'])
    m, bd = alpha_plane(sec.values, vn, alpha=0.0004)
    return m, bd


def show_figure(
    fig:go.Figure
  , width=1200, height=800
  , static=False
  , showlegend=True
) -> None:
    """
    Outputs figure either as dynamic (default) or static image

    Parameters
    ----------
    fig : go.Figure
        The figure to output
    width : int
        width of the output in px
    height : int
        height of the output in px
    static : bool, default=False
        shows go.Figure after resizing, generates png and displays it in IPython otherwise
    showlegend : bool, default=True
        show or remove legend in the figure
    """
    fig.update_layout(showlegend=showlegend)
    if static:
        _ensure_kaleido_server()
        img_bytes = fig.to_image(
            format="png"
          , width=width, height=height
          , scale=1)
        display(Image(img_bytes))
    else:
        fig.update_layout(
            width=width,
            height=height
        )
        fig.show(config={"displayModeBar":False})


def save_figure(
    fig:go.Figure
  , name:str
  , width=1200, height=800
  , showlegend=False
  , path:Path=None
  , replace:bool=False
  , transparent_bg:bool=False
) -> None:
    """
    Saves the figure to an image file.

    Parameters
    ----------
    fig : go.Figure
        figure to be saved
    name : str
        Name (string) used for the base filename. If the file already exists, `save_figure` will
            attach a current time stamp.
    width : int, default=1200
        width of the output image in px
    height : int, default=800
        height of the output image in px
    showlegend : bool, default=False
        enable legend
    path : Path, default=None
        If no path is given, output is saved in `results/cell_gallery/name*.png`
    replace : bool, default=False
        replace file instead of attaching timestamp
    """
    fig.update_layout(showlegend=showlegend)
    if transparent_bg:
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)'
          , plot_bgcolor='rgba(0,0,0,0)'
        )
    _ensure_kaleido_server()
    img_bytes = fig.to_image(
            format="png"
          , width=width, height=height
          , scale=1)
    image = PIL.Image.open(io.BytesIO(img_bytes))
    if path is None:
        path = Path(find_dotenv()).parent / "results" / "cell_gallery"
    path.mkdir(parents=True, exist_ok=True)
    img_fn = path / f"{name}.png"
    if img_fn.exists() and not replace:
        today_str = datetime.datetime.today().strftime("%Y-%m-%dT%H-%M-%S")
        img_fn = path / f"{name}_{today_str}.png"
    image.save(img_fn)


_anterior_font = None


def save_figure_anterior(
    fig:go.Figure
  , name:str
  , path:Path=None
  , label:str=None
  , width:int=1523, height:int=850
  , pad:int=12
  , font_size:int=22
) -> Path:
    """
    Save an anterior-view figure as a tight PNG.

    Wraps `save_figure()`, then (1) auto-trims the white margins plotly leaves
    around the rendered brain, and (2) optionally draws `label` inside the
    top-right corner of the cropped image — drawn *into the pixels* rather than as
    a plotly annotation, so it never shifts the crop or adds whitespace.

    The default `width`/`height` (1523x850) match the CNS frontal aspect ratio
    (the brain is wider than tall), minimizing wasted canvas before the trim.

    Parameters
    ----------
    fig : go.Figure
        figure to save (e.g. from `plot_cns(..., show_silhouette=True)`)
    name : str
        base filename (no extension); written as `{name}.png` (always replaced)
    path : Path, default=None
        output directory; defaults to `results/cell_gallery` (as `save_figure`)
    label : str, default=None
        text drawn in the top-right corner; omit for no label
    width, height : int
        render size in px before trimming
    pad : int, default=12
        white border (px) kept around the content after trimming
    font_size : int, default=22
        label font size

    Returns
    -------
    img_fn : Path
        path to the saved PNG
    """
    save_figure(fig, name=name, width=width, height=height, path=path, replace=True)
    if path is None:
        path = Path(find_dotenv()).parent / "results" / "cell_gallery"
    img_fn = Path(path) / f"{name}.png"

    im = PIL.Image.open(img_fn).convert("RGB")
    bbox = ImageChops.difference(im, PIL.Image.new("RGB", im.size, (255, 255, 255))).getbbox()
    if bbox:
        l, t, r, b = bbox
        im = im.crop((
            max(0, l - pad), max(0, t - pad)
          , min(im.width, r + pad), min(im.height, b + pad)))
    if label:
        global _anterior_font
        if _anterior_font is None or _anterior_font.size != font_size:
            _anterior_font = ImageFont.truetype(font_manager.findfont("DejaVu Sans"), font_size)
        draw = ImageDraw.Draw(im)
        tw = draw.textlength(label, font=_anterior_font)
        draw.text((im.width - tw - 10, 8), label, fill=(0, 0, 0), font=_anterior_font)
    im.save(img_fn)
    return img_fn


def alpha_plane(
    pts:np.ndarray
  , vn:np.ndarray
  , alpha:float=0.001
) -> Union[navis.Volume, np.ndarray]:
    """
    Make an alpha shape plane from a point set on a plane in 3D.

    First rotate the points to be on a plane parallel to the XY plane.
    Then generate a 2D alpha shape, and find the boundary points and triangle face indicies.
    Finally construct navis.Volume from original points.

    Parameters
    ----------
    pts : ndarray
        [x y z] points on a plane
    vn : ndarray
        [x y z] normal vector of the plane
    alpha : float, default=0.001
        alpha parameter for alpha shape

    Returns:
    -------
    m : navis.Volume
    pts : ndarray
        [x y z] points on the boundary
    """

    # define rotation to rotate the point to be in the plane parallel to the xy plane
    rot_mat = scipy_R.from_rotvec(
        np.cross(vn, np.array([0,0,1])) * np.arccos(vn @ np.array([0,0,1]))
    )

    pts_rot = rot_mat.apply(pts) # rotate the points
    offset = np.mean(pts_rot[:,2]) # mean of the last column (z coord) as offset
    pts_rot = np.delete(pts_rot, 2, 1) # romove the last column

    # https://alphashape.readthedocs.io/en/latest/alphashape.html
    ss = alphashape.alphasimplices(pts_rot)
    ind = []
    for point_indices, circumradius in ss:
        # print(point_indices, circumradius)
        if circumradius < 1.0 / alpha:
            ind.append(point_indices)
    ind = np.array(ind)

    # make mesh
    m = navis.Volume(vertices=pts, faces=ind)

    # get the boundary points
    alpha_shape = alphashape.alphashape(pts_rot, alpha) # alpha shape with alpha = 0.001
    alpha_shape_ext = np.array(alpha_shape.exterior.coords) # points on the boundary

    # add z coordinate  to alpha_shape_pts
    alpha_shape_pts = np.hstack((alpha_shape_ext, np.ones((alpha_shape_ext.shape[0],1))))
    alpha_shape_pts[:,2] = alpha_shape_pts[:,2] + offset # add offset
    # rotate back to 3d
    alpha_shape_pts = rot_mat.apply(alpha_shape_pts, inverse=True)

    # # DEBUG
    # # plot the mesh
    # fig_m = navis.plot3d(m,
    #     color='black', linewidth=2,
    #     inline=False, backend='plotly')

    # fig_m.show()

    return m, alpha_shape_pts


def plot_cns(
    bodyid:int|list[int]
    , celltype:str|list[str]
    , show_skeletons:bool=False
    , show_meshes:bool=False
    , show_shell:bool=False
    , show_outline:bool=False
    , show_silhouette:bool=False
    , zoom:float=3.0
    , palette:str='viridis'
    , color=None
    , scatter_kws:dict=None
    , mesh_lod:int=None
    , annotate:bool=True
    , inline:bool=False
) -> go.Figure:
    """
    Plot VPNs with a CNS outline as background, frotal view with slightly modified camera settings.

    Parameters
    ----------
    bodyid : int|list[int]
        body ID(s) of the VPNs. A single int is accepted and wrapped into a list.
    celltype : str|list[str]
        cell type(s) of the VPNs. A single str is accepted and wrapped into a list.
    show_skeletons : bool, default=False
        show the skeleton of the VPNs
    show_meshes : bool, default=False
        show the mesh of the VPNs
    show_shell : bool, default=False
        show the (translucent) shell of the CNS
    show_outline : bool, default=False
        show the outline of the CNS
    show_silhouette : bool, default=False
        show the CNS shell flattened to one plane as a flat, opaque gray silhouette
        (the esophageal foramen stays a real hole). See `get_silhouette_cns()`.
    zoom : float, default=3.0
        zooming factor
    palette : str, default='viridis'
        color palette for the VPNs (ignored when `color` is given)
    color : color-like, default=None
        a single explicit neuron color (e.g. `(0,0,0,1)` for black) passed straight
        to `navis.plot3d`. Takes precedence over `palette`. The CNS shell/outline/
        silhouette keep their own colors regardless.
    scatter_kws : dict, default=None
        plotly scatter plot settings
    mesh_lod : int, default=None
        level of detail for neuron meshes when `show_meshes=True` (passed to
        `get_meshes()`). `None` = finest; `2` is much lighter for many neurons.
    annotate : bool, default=True
        draw a `celltype:bodyid` text annotation in the top-right corner. Set
        `False` when you intend to add a label later (e.g. `save_figure_anterior`).
    inline : bool, default=False
        show figure inline

    Returns
    -------
    fig : go.Figure
        plotly figure

    """

    if isinstance(bodyid, (int, np.integer)):
        bodyid = [int(bodyid)]
    if isinstance(celltype, str):
        celltype = [celltype]

    if scatter_kws is None:
        scatter_kws = {'mode': 'lines', 'color': 'black', 'opacity':1}

    # range of cns
    range_x = [5000, 91000]
    range_y = [5000, 53000]
    range_z = [10000, 43000]
    range_ratio = np.concatenate([np.diff(range_x), np.diff(range_y), np.diff(range_z)])
    range_ratio = range_ratio / np.linalg.norm(range_ratio)
    # multiplicative factor ~ zooming
    range_ratio = range_ratio * zoom

    plotlist = []
    if show_skeletons:
        plotlist += [get_skeletons(bodyid)] # from madvisc.utils.plotter
    if show_meshes:
        plotlist += [get_meshes(bodyid, lod=mesh_lod, save_cache=False)]
    if show_shell:
        plotlist += [get_shell_cns()]
    if show_outline:
        plotlist += [get_outline_cns()]
    if show_silhouette:
        plotlist += [get_silhouette_cns()]

    # `color` (an explicit neuron color) takes precedence over `palette`. The CNS
    # shell/outline/silhouette are navis.Volumes and keep their own `.color`.
    plot3d_kws = {'color': color} if color is not None else {'palette': palette}
    fig = navis.plot3d(
        plotlist
      , scatter_kws=scatter_kws
      , inline=inline
      , **plot3d_kws)

    up_x, up_y, up_z = [-0.05, -1 ,0] / np.linalg.norm([-0.05, -1 ,0])       # camera up
    eye_x, eye_y, eye_z = [0, 0, -1]    # camera eye

    fig.update_traces(
        lighting={'ambient':1, 'diffuse':0}
      , selector={'type': 'mesh3d'}
    )

    # add annotation with celltype and bodyid
    if annotate:
        if len(celltype) > 1:
            anno = [f"{i}:{j}" for i, j in zip(list(celltype), [str(b) for b in bodyid])]
            annotext = '<br>'.join(anno)
        else:
            annotext = f"{celltype[0]}:{bodyid[0]}"

        fig.add_annotation({
            'font':{ 'color':'black', 'size': 18}
          , 'x': 1, 'y': 1
          , 'showarrow': False
          , 'text': annotext
          , 'textangle': 0
          , 'xanchor': 'right'
          , 'xref': "paper"
          , 'yref': "paper"
        })

    fig.update_layout(
        margin={'l':0, 'r':0, 'b':0, 't':0}
      , showlegend=False
      , scene = {
            "xaxis" : {
                "showgrid": False
              , "gridcolor": 'rgba(0,0,0,0)'
              , "showbackground": False
              , "visible": False
              , "autorange": False
              , "range": range_x # These are the boundaries of cns
            }
          , "yaxis" : {
                "showgrid": False
              , "gridcolor": 'rgba(0,0,0,0)'
              , "showbackground": False
              , "visible": False
              , "autorange": False
              , "range": range_y
            }
          , "zaxis" : {
                "showgrid": False
              , "gridcolor": 'rgba(0,0,0,0)'
              , "showbackground":False
              , "visible": False
              , "autorange": False
              , "range": range_z
            }
          , "camera":{
                "up": {"x":up_x, "y": up_y, "z": up_z} # canvas/page orientation
              , "center": {"x":0, "y": 0, "z": 0} # translation
              , "eye": {"x":eye_x, "y": eye_y, "z": eye_z} # view direction
              , "projection": {"type": "orthographic"}
            }
          , 'aspectmode':'manual'
          , 'aspectratio': {"x": range_ratio[0], "y": range_ratio[1], "z": range_ratio[2]}
        }

      , paper_bgcolor='#fff'
      , plot_bgcolor='#fff'
    )
    return fig


def get_shell_cns(ignore_cache:bool=False) -> cv.mesh.Mesh:
    """
    Retrieve shell of CNS using cloudvolume.

    Parameters:
    ----------
    ignore_cache : bool, default=False
        if true, download the meshes from the original data source each time. Otherwise
        use a filesystem cache.

    Returns:
    -------
    roi : cv.mesh.Mesh
        shell of CNS
    """

    load_dotenv()
    assert os.environ.get('SHELL_SOURCE'),\
        "Please set the `SHELL_SOURCE` variable in your environment."
    cachedir = Path(find_dotenv()).parent / "cache" / "cns_shell_cv"
    cachedir.mkdir(parents=True, exist_ok=True)
    roi_fn = cachedir / "cns_shell.pickle"
    if not ignore_cache and roi_fn.is_file():
        with roi_fn.open('rb') as roi_fh:
            roi = pickle.load(roi_fh)
    else:
        vol = cv.CloudVolume(
            f"precomputed://{os.environ['SHELL_SOURCE']}"
          , use_https=True
          , progress=False
        )
        roi = vol.mesh.get([1,2,3])
        roi.vertices = roi.vertices / 8 # from nm to voxels
        with roi_fn.open('wb') as roi_fh:
            pickle.dump(roi, roi_fh)
    roi.color = (1,1,1, 0.2)
    return roi


# z-plane the CNS shell is flattened onto for the silhouette (behind the neurons,
# in the 8 nm-voxel frame), and the default flat gray fill color.
_SILHOUETTE_Z = 42500
_SILHOUETTE_COLOR = (0.82, 0.82, 0.82, 1.0)
_silhouette_cns = None


def get_silhouette_cns(
    z_plane:float=_SILHOUETTE_Z
  , color:tuple=_SILHOUETTE_COLOR
) -> navis.Volume:
    """
    CNS shell flattened to a single plane: a uniform, opaque gray silhouette that
    still keeps the esophageal foramen as a real hole (background shows through).

    This is the "filled gray brain" backdrop for anterior-view neuron plots — an
    alternative to `get_shell_cns()` (a near-invisible translucent 3-D mesh) and
    `get_outline_cns()` (just the boundary contour). Plot it with flat shading
    (`plot_cns` already sets `lighting={'ambient':1,'diffuse':0}` on mesh3d traces)
    so it renders as a flat fill rather than a shaded surface. Cached for the session.

    Parameters
    ----------
    z_plane : float, default=42500
        plane (8 nm-voxel frame) the shell vertices are collapsed onto; sits behind
        the neurons for the frontal/anterior camera.
    color : tuple, default=(0.52, 0.52, 0.52, 1.0)
        RGBA fill color of the silhouette.

    Returns
    -------
    sil : navis.Volume
        the flattened CNS shell, colored `color`
    """
    global _silhouette_cns
    if _silhouette_cns is None:
        shell = get_shell_cns()
        verts = np.asarray(shell.vertices, dtype=float).copy()
        verts[:, 2] = z_plane
        _silhouette_cns = navis.Volume(vertices=verts, faces=np.asarray(shell.faces))
    _silhouette_cns.color = color
    return _silhouette_cns


def get_outline_cns(ignore_cache:bool=False) -> cv.mesh.Mesh:
    """
    Retrieve outline of CNS using cloudvolume.

    Parameters:
    ----------
    ignore_cache : bool, default=False
        if true, download the meshes from the original data source each time. Otherwise
        use a filesystem cache.

    Returns:
    -------
    bd : cv.mesh.Mesh
        outline of CNS
    """

    load_dotenv()
    assert os.environ.get('SHELL_SOURCE'),\
        "Please set the `SHELL_SOURCE` variable in your environment."
    cachedir = Path(find_dotenv()).parent / "cache" / "cns_outline_cv"
    cachedir.mkdir(parents=True, exist_ok=True)
    bd_fn = cachedir / "cns_outline.pickle"
    if not ignore_cache and bd_fn.is_file():
        with bd_fn.open('rb') as bd_fh:
            bd = pickle.load(bd_fh)
    else:
        vol = cv.CloudVolume(
            f"precomputed://{os.environ['SHELL_SOURCE']}"
          , use_https=True
          , progress=False
        )
        roi = vol.mesh.get([1,2,3])
        roi.vertices = roi.vertices / 8 # from nm to voxels
        roi.vertices[:,2] = 30000 # This needs to be in the z range of the plot
        pts = pd.DataFrame(roi.vertices, columns=['x','y','z'])  #convert TrackedArray to dataframe
        vn = np.array([0, 0, 1])
        _, bd = alpha_plane(pts, vn, alpha=0.003) # alpha shape boundary, 0.003 looks ok
        with bd_fn.open('wb') as bd_fh:
            pickle.dump(bd, bd_fh)

    return bd
