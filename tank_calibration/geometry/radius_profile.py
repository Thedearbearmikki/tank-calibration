"""
Radius profile computation for a vertical cylindrical tank.

Computes radius statistics per horizontal slice:
- mean radius
- min / max radius
- ovality (r_max - r_min)

Assumes point cloud is already transformed into tank frame:
- tank axis aligned with Z
- center near (0, 0)

Author: tank-calibration project
"""

from dataclasses import dataclass
from typing import List

import numpy as np


# =========================
# Data structures
# =========================

@dataclass
class RadiusSlice:
    """
    Radius statistics for one height slice.
    """
    z_mean: float
    r_mean: float
    r_min: float
    r_max: float
    ovality: float
    n_points: int


@dataclass
class RadiusProfile:
    """
    Full radius profile along tank height.
    """
    slices: List[RadiusSlice]

    def as_arrays(self):
        """
        Convert profile to numpy arrays.

        Returns
        -------
        dict of np.ndarray
        """
        return {
            "z": np.array([s.z_mean for s in self.slices]),
            "r_mean": np.array([s.r_mean for s in self.slices]),
            "r_min": np.array([s.r_min for s in self.slices]),
            "r_max": np.array([s.r_max for s in self.slices]),
            "ovality": np.array([s.ovality for s in self.slices]),
            "n": np.array([s.n_points for s in self.slices]),
        }


# =========================
# Core logic
# =========================

def compute_radius_profile(
    points_tank: np.ndarray,
    dz: float = 0.3,
    min_points_per_slice: int = 30,
) -> RadiusProfile:
    """
    Compute radius profile along tank height.

    Parameters
    ----------
    points_tank : np.ndarray
        Point cloud in tank frame, shape (N, 3)
    dz : float, optional
        Height slice thickness (meters), default 0.3
    min_points_per_slice : int, optional
        Minimum number of points per slice

    Returns
    -------
    RadiusProfile
        Radius profile data
    """
    if points_tank.ndim != 2 or points_tank.shape[1] != 3:
        raise ValueError("points_tank must have shape (N, 3)")

    z_min = points_tank[:, 2].min()
    z_max = points_tank[:, 2].max()

    slices: List[RadiusSlice] = []
    z = z_min

    while z < z_max:
        mask = (points_tank[:, 2] >= z) & (points_tank[:, 2] < z + dz)
        pts = points_tank[mask]

        if pts.shape[0] < min_points_per_slice:
            z += dz
            continue

        x = pts[:, 0]
        y = pts[:, 1]
        r = np.sqrt(x ** 2 + y ** 2)

        slice_data = RadiusSlice(
            z_mean=pts[:, 2].mean(),
            r_mean=float(r.mean()),
            r_min=float(r.min()),
            r_max=float(r.max()),
            ovality=float(r.max() - r.min()),
            n_points=pts.shape[0],
        )

        slices.append(slice_data)
        z += dz

    if not slices:
        raise RuntimeError("No valid radius slices computed.")

    return RadiusProfile(slices=slices)
