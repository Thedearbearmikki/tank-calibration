"""
Global cylinder fitting for tank geometry.

Fits an approximating vertical cylinder to point cloud
already transformed into tank frame.

Approximates:
- center (x0, y0)
- radius R

Method:
- nonlinear least squares on radial distances

Author: tank-calibration project
"""

from dataclasses import dataclass
import numpy as np
from scipy.optimize import least_squares


# =========================
# Data structure
# =========================

@dataclass
class CylinderFit:
    x0: float
    y0: float
    r: float
    rmse: float
    n_points: int


# =========================
# Core logic
# =========================

def fit_cylinder_xy(points_tank: np.ndarray) -> CylinderFit:
    """
    Fit vertical cylinder to tank-frame point cloud.

    Parameters
    ----------
    points_tank : np.ndarray
        Point cloud in tank frame, shape (N, 3)

    Returns
    -------
    CylinderFit
        Fitted cylinder parameters
    """
    if points_tank.ndim != 2 or points_tank.shape[1] != 3:
        raise ValueError("points_tank must have shape (N, 3)")

    x = points_tank[:, 0]
    y = points_tank[:, 1]

    # Initial guess: centroid + mean radius
    x0_init = np.mean(x)
    y0_init = np.mean(y)
    r_init = np.mean(np.sqrt((x - x0_init) ** 2 + (y - y0_init) ** 2))

    def residuals(params):
        x0, y0, r = params
        return np.sqrt((x - x0) ** 2 + (y - y0) ** 2) - r

    res = least_squares(
        residuals,
        x0=[x0_init, y0_init, r_init],
        method="lm",
    )

    x0, y0, r = res.x
    rmse = float(np.sqrt(np.mean(res.fun ** 2)))

    return CylinderFit(
        x0=float(x0),
        y0=float(y0),
        r=float(r),
        rmse=rmse,
        n_points=points_tank.shape[0],
    )
