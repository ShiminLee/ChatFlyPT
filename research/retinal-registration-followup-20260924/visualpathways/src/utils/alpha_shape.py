"""
Generic 2D concave-hull (alpha shape) geometry utilities.

These are domain-agnostic geometric helpers: computing an alpha shape (concave
hull) from a set of 2D points, ordering its boundary components, the triangle
circumradius test used by the alpha criterion, and a ray/boundary intersection
helper. They depend only on numpy/scipy/shapely and are used by the receptive
field metrics (:mod:`utils.ol_rf`).
"""

import warnings
from collections import defaultdict

import numpy as np
import shapely
from scipy.spatial import Delaunay


def alpha_shape_2d(points, alpha=0.5, min_alpha=0.0, alpha_reduction=0.1):
    """
    Compute the alpha shape (concave hull) of a set of 2D points.

    Parameters:
    -----------
    points : array-like, shape (n, 2)
        Array of 2D points
    alpha : float
        Alpha value to control the shape's tightness.
        - Smaller alpha -> tighter fit (more concave)
        - Larger alpha -> looser fit (approaches convex hull)
        - alpha = 0 returns convex hull
    min_alpha : float, default=0.0
        Lower bound for alpha. While the boundary has multiple disconnected
        components, alpha is reduced by ``alpha_reduction`` and retried, but it
        is never lowered below ``min_alpha``.
    alpha_reduction : float, default=0.1
        Amount to reduce alpha by on each retry

    Returns:
    --------
    edge_points : list of numpy arrays
        List of boundary components, each as (n, 2) array of (x, y) coordinates
    boundary_edges : set of tuples
        Set of edges (i, j) where i and j are indices in original points array
    areas : list of float
        List of areas corresponding to each boundary component in edge_points
    used_alpha : float
        The alpha value actually used to produce the returned boundary (may be
        smaller than the input ``alpha`` if it was reduced over retries)
    """

    points = np.array(points)

    if len(points) < 4:
        # Return convex hull for fewer than 4 points
        return points.tolist(), set(), [0.0], alpha

    if min_alpha > alpha:
        raise ValueError("min_alpha must be <= alpha")

    current_alpha = alpha

    while True:
        # Compute Delaunay triangulation
        tri = Delaunay(points)

        # Find boundary edges (edges that appear in only one triangle)
        edge_count = defaultdict(int)
        for simplex in tri.simplices:
            pts = points[simplex]
            circumradius = compute_circumradius_triangle(pts)

            if current_alpha == 0 or circumradius < 1.0 / current_alpha:
                for i in range(3):
                    edge = tuple(sorted([simplex[i], simplex[(i + 1) % 3]]))
                    edge_count[edge] += 1

        # Keep only boundary edges (those that appear once)
        boundary_edges = {edge for edge, count in edge_count.items() if count == 1}

        # Order the boundary points
        if boundary_edges:
            edge_points = order_boundaries(boundary_edges, points)
        else:
            edge_points = []

        # Check if we have multiple disconnected components
        if len(edge_points) > 1:
            if alpha_reduction > 0 and current_alpha - alpha_reduction >= min_alpha:
                current_alpha = round(current_alpha - alpha_reduction, 10)
                # warnings.warn(f"Multiple boundary components found. Retrying with alpha={current_alpha:.3f}")
                continue
            else:
                warnings.warn(f"Still have {len(edge_points)} components at alpha={current_alpha:.3f} (min_alpha={min_alpha})")

        # Compute areas for each boundary component
        areas = []
        for boundary in edge_points:
            if len(boundary) >= 3:
                # Use shoelace formula for polygon area
                x = boundary[:, 0]
                y = boundary[:, 1]
                area = 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))
                areas.append(float(area))
            else:
                areas.append(0.0)

        # edge_points = pd.DataFrame(edge_points, columns=['h', 'v'])

        return edge_points, boundary_edges, areas, current_alpha


def compute_circumradius_triangle(triangle_points):
    """
    Compute the circumradius of a triangle.

    Parameters:
    -----------
    triangle_points : array-like, shape (3, 2)
        The three vertices of the triangle

    Returns:
    --------
    radius : float
        The circumradius of the triangle
    """
    pts = np.array(triangle_points)
    a, b, c = pts[0], pts[1], pts[2]

    # Calculate side lengths
    a_len = np.linalg.norm(b - c)
    b_len = np.linalg.norm(a - c)
    c_len = np.linalg.norm(a - b)

    # Calculate area using cross product
    area = 0.5 * abs(np.cross(b - a, c - a))

    if area == 0:
        return float('inf')

    # Circumradius formula: R = (abc) / (4 * Area)
    circumradius = (a_len * b_len * c_len) / (4.0 * area)

    return circumradius


def order_boundaries(edges, points):
    """Find and order all disconnected boundary components."""
    if not edges:
        return []

    visited_edges = set()
    visited_nodes = set()
    all_boundaries = []
    all_boundaries_idx = []

    # Find all connected components
    for start_edge in edges:
        if start_edge in visited_edges:
            continue

        # Trace this component
        boundary = []
        current = start_edge[0]
        prev = None
        start = current

        while True:
            # find all edges that contain current node and a node that's not in visited_nodes
            current_edges = [e for e in edges if
                            current in e and
                            not (set(e) <= visited_nodes) and
                            e not in visited_edges]

            # pick the first in current_edges, get the node that's not current
            if not current_edges:
                break

            # check current_edges
            current_edge = current_edges[0]
            neighbor = current_edge[0] if current_edge[1] == current else current_edge[1]

            # update
            prev = current
            current = neighbor
            boundary.append(current)
            visited_edges.add(current_edge)
            visited_nodes.add(current)

        # add the start node as the first element of the boundary
        boundary.insert(0, start)
        visited_nodes.add(start)
        ordered_points = [(points[i][0], points[i][1]) for i in boundary]

        all_boundaries.append(np.array(ordered_points))
        all_boundaries_idx.append(boundary)

    return all_boundaries
