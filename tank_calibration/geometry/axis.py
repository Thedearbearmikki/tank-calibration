"""
Axis determination for a vertical cylindrical tank based on point cloud data.

The tank axis is computed as a best-fit line through centers of horizontal
cross-sections obtained from the point cloud.

Workflow:
1. Slice point cloud by height (Z)
2. Fit circle in XY for each slice → get centers
3. Fit 3D line (PCA/SVD) through centers → tank axis

Author: tank-calibration project
"""

from dataclasses import dataclass
from typing import List, Tuple

import numpy as np


# =========================
# Data structures
# =========================

@dataclass
class TankAxis:
    """
    Representation of tank axis in 3D space.

    Attributes
    ----------
    point : np.ndarray
        A point lying on the axis, shape (3,)
    direction : np.ndarray
        Unit direction vector of the axis, shape (3,)
    """
    point: np.ndarray
    direction: np.ndarray


# =========================
# Geometry helpers
# =========================

def slice_points_by_height(
    points: np.ndarray,
    dz: float,
    min_points: int = 30,
) -> List[np.ndarray]:
    """
    Split point cloud into horizontal slices.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud, shape (N, 3)
    dz : float
        Slice thickness along Z axis
    min_points : int
        Minimum number of points per slice

    Returns
    -------
    List[np.ndarray]
        List of point arrays per slice
    """
    z_min = points[:, 2].min()
    z_max = points[:, 2].max()

    slices: List[np.ndarray] = []
    z = z_min

    while z < z_max:
        mask = (points[:, 2] >= z) & (points[:, 2] < z + dz)
        if np.count_nonzero(mask) >= min_points:
            slices.append(points[mask])
        z += dz

    return slices


def fit_circle_xy(points_xy: np.ndarray) -> Tuple[float, float, float]:
    """
    Algebraic least-squares fit of a circle in XY plane.

    Parameters
    ----------
    points_xy : np.ndarray
        Array of XY points, shape (N, 2)

    Returns
    -------
    cx, cy, r : float
        Circle center coordinates and radius
    """
    x = points_xy[:, 0]
    y = points_xy[:, 1]

    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    b = x ** 2 + y ** 2

    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    cx = c[0]
    cy = c[1]
    r = np.sqrt(c[2] + cx ** 2 + cy ** 2)

    return cx, cy, r


def compute_section_centers(
    slices: List[np.ndarray],
) -> np.ndarray:
    """
    Compute XY circle centers for each height slice.

    Parameters
    ----------
    slices : list of np.ndarray
        List of point arrays per slice

    Returns
    -------
    np.ndarray
        Array of center points (cx, cy, z_mean), shape (M, 3)
    """
    centers = []

    for pts in slices:
        xy = pts[:, :2]
        z_mean = pts[:, 2].mean()

        cx, cy, _ = fit_circle_xy(xy)
        centers.append([cx, cy, z_mean])

    return np.asarray(centers)


def fit_axis_pca(centers: np.ndarray) -> TankAxis:
    """
    Fit a 3D line through section centers using PCA (SVD).

    Parameters
    ----------
    centers : np.ndarray
        Array of center points, shape (M, 3)

    Returns
    -------
    TankAxis
        Fitted tank axis
    """
    if centers.shape[0] < 2:
        raise ValueError("At least two section centers are required to fit axis.")

    mean_point = centers.mean(axis=0)

    # Principal direction
    _, _, vt = np.linalg.svd(centers - mean_point)
    direction = vt[0]
    direction /= np.linalg.norm(direction)

    return TankAxis(point=mean_point, direction=direction)


# =========================
# Public API
# =========================

def compute_tank_axis(
    points: np.ndarray,
    dz: float = 0.3,
    min_points_per_slice: int = 30,
) -> TankAxis:
    """
    Compute tank axis from point cloud.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud, shape (N, 3)
    dz : float, optional
        Height of slicing layer (meters), default 0.3
    min_points_per_slice : int, optional
        Minimum points per slice, default 30

    Returns
    -------
    TankAxis
        Computed tank axis
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    slices = slice_points_by_height(
        points,
        dz=dz,
        min_points=min_points_per_slice,
    )

    if len(slices) < 3:
        raise RuntimeError(
            "Insufficient number of valid slices to determine tank axis."
        )

    centers = compute_section_centers(slices)
    return fit_axis_pca(centers)
