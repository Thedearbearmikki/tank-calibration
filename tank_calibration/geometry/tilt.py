"""
Tank tilt computation based on tank axis.

Computes:
- tilt angle
- tilt degree (tan of angle, per GOST)
- tilt azimuth in horizontal plane

Author: tank-calibration project
"""

from dataclasses import dataclass
from typing import Tuple

import numpy as np

from .axis import TankAxis


# =========================
# Data structures
# =========================

@dataclass
class TankTilt:
    """
    Tank tilt parameters.

    Attributes
    ----------
    angle_rad : float
        Tilt angle in radians
    angle_deg : float
        Tilt angle in degrees
    degree : float
        Tilt degree (tan(angle))
    azimuth_rad : float
        Azimuth of tilt direction in radians
    azimuth_deg : float
        Azimuth of tilt direction in degrees
    """
    angle_rad: float
    angle_deg: float
    degree: float
    azimuth_rad: float
    azimuth_deg: float


# =========================
# Core calculations
# =========================

def compute_tank_tilt(axis: TankAxis) -> TankTilt:
    """
    Compute tank tilt parameters from tank axis.

    Parameters
    ----------
    axis : TankAxis
        Tank axis definition

    Returns
    -------
    TankTilt
        Computed tilt parameters
    """
    v = axis.direction / np.linalg.norm(axis.direction)

    vx, vy, vz = v

    # Safety check
    if vz <= 0:
        raise ValueError("Tank axis Z component must be positive.")

    # Tilt angle (from vertical)
    angle_rad = np.arccos(vz)
    angle_deg = np.degrees(angle_rad)

    # Tilt degree (GOST definition: tan(beta))
    degree = np.sqrt(vx ** 2 + vy ** 2) / vz

    # Tilt azimuth (direction of maximum slope)
    azimuth_rad = np.arctan2(vy, vx)
    azimuth_deg = np.degrees(azimuth_rad)

    if azimuth_deg < 0:
        azimuth_deg += 360.0

    return TankTilt(
        angle_rad=angle_rad,
        angle_deg=angle_deg,
        degree=degree,
        azimuth_rad=azimuth_rad,
        azimuth_deg=azimuth_deg,
    )


# =========================
# GOST helpers
# =========================

def check_tilt_gost(
    tilt: TankTilt,
    max_degree: float = 0.01,
) -> bool:
    """
    Check tilt against GOST limit.

    Parameters
    ----------
    tilt : TankTilt
        Computed tilt parameters
    max_degree : float, optional
        Maximum allowed tilt degree (default 0.01)

    Returns
    -------
    bool
        True if tilt is within allowed limit
    """
    return tilt.degree <= max_degree
