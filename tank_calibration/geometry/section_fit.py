"""
Horizontal section fitting for tank geometry.

Provides:
- circle fitting (mandatory)
- ellipse fitting (optional)

Assumes points are already transformed into tank frame:
- Z axis aligned with tank axis
- center approximately near (0, 0)

Author: tank-calibration project
"""

from dataclasses import dataclass
from typing import Optional

import numpy as np


# =========================
# Data structures
# =========================

@dataclass
class CircleFit:
    cx: float
    cy: float
    r: float
    residual_std: float


@dataclass
class EllipseFit:
    cx: float
    cy: float
    a: float
    b: float
    angle_rad: float
    residual_std: float


# =========================
# Circle fitting
# =========================

def fit_circle_xy(
    points_xy: np.ndarray,
) -> CircleFit:
    """
    Least-squares circle fit in XY plane.

    Parameters
    ----------
    points_xy : np.ndarray
        Array of shape (N, 2)

    Returns
    -------
    CircleFit
        Fitted circle parameters
    """
    if points_xy.ndim != 2 or points_xy.shape[1] != 2:
        raise ValueError("points_xy must have shape (N, 2)")

    x = points_xy[:, 0]
    y = points_xy[:, 1]

    A = np.column_stack((2 * x, 2 * y, np.ones_like(x)))
    b = x ** 2 + y ** 2

    c, *_ = np.linalg.lstsq(A, b, rcond=None)

    cx, cy = c[0], c[1]
    r = np.sqrt(c[2] + cx ** 2 + cy ** 2)

    residuals = np.sqrt((x - cx) ** 2 + (y - cy) ** 2) - r
    residual_std = float(np.std(residuals))

    return CircleFit(
        cx=float(cx),
        cy=float(cy),
        r=float(r),
        residual_std=residual_std,
    )


# =========================
# Ellipse fitting (optional)
# =========================

def fit_ellipse_xy(
    points_xy: np.ndarray,
) -> EllipseFit:
    """
    Direct least-squares ellipse fit (Fitzgibbon method).

    Parameters
    ----------
    points_xy : np.ndarray
        Array of shape (N, 2)

    Returns
    -------
    EllipseFit
        Fitted ellipse parameters
    """
    x = points_xy[:, 0][:, np.newaxis]
    y = points_xy[:, 1][:, np.newaxis]

    D = np.hstack([x * x, x * y, y * y, x, y, np.ones_like(x)])
    S = np.dot(D.T, D)

    C = np.zeros((6, 6))
    C[0, 2] = C[2, 0] = 2
    C[1, 1] = -1

    eigvals, eigvecs = np.linalg.eig(np.linalg.inv(S) @ C)
    a = eigvecs[:, np.argmax(np.real(eigvals))].real

    A, B, Cc, Dd, Ee, Ff = a

    # Center
    denom = B ** 2 - 4 * A * Cc
    cx = (2 * Cc * Dd - B * Ee) / denom
    cy = (2 * A * Ee - B * Dd) / denom

    # Axes and rotation
    angle = 0.5 * np.arctan2(B, A - Cc)

    up = 2 * (A * cx ** 2 + Cc * cy ** 2 + B * cx * cy - Ff)
    down1 = A + Cc + np.sqrt((A - Cc) ** 2 + B ** 2)
    down2 = A + Cc - np.sqrt((A - Cc) ** 2 + B ** 2)

    a_len = np.sqrt(up / down1)
    b_len = np.sqrt(up / down2)

    # Residuals
    xp = (points_xy[:, 0] - cx)
    yp = (points_xy[:, 1] - cy)

    cos_t = np.cos(angle)
    sin_t = np.sin(angle)

    xr = xp * cos_t + yp * sin_t
    yr = -xp * sin_t + yp * cos_t

    residuals = (xr / a_len) ** 2 + (yr / b_len) ** 2 - 1
    residual_std = float(np.std(residuals))

    return EllipseFit(
        cx=float(cx),
        cy=float(cy),
        a=float(a_len),
        b=float(b_len),
        angle_rad=float(angle),
        residual_std=residual_std,
    )
