"""
Coordinate transformation into tank reference frame.

Transforms a point cloud so that:
- tank axis passes through origin
- tank axis aligns with global Z axis

Author: tank-calibration project
"""

from typing import Tuple

import numpy as np

from .axis import TankAxis


# =========================
# Rotation helpers
# =========================

def rotation_matrix_from_vectors(
    v_from: np.ndarray,
    v_to: np.ndarray,
) -> np.ndarray:
    """
    Compute rotation matrix that rotates v_from to v_to.

    Parameters
    ----------
    v_from : np.ndarray
        Initial unit vector, shape (3,)
    v_to : np.ndarray
        Target unit vector, shape (3,)

    Returns
    -------
    np.ndarray
        Rotation matrix, shape (3, 3)
    """
    v_from = v_from / np.linalg.norm(v_from)
    v_to = v_to / np.linalg.norm(v_to)

    cross = np.cross(v_from, v_to)
    dot = np.dot(v_from, v_to)

    if np.isclose(dot, 1.0):
        return np.eye(3)

    if np.isclose(dot, -1.0):
        # 180-degree rotation: choose arbitrary orthogonal axis
        axis = np.array([1.0, 0.0, 0.0])
        if np.allclose(v_from, axis):
            axis = np.array([0.0, 1.0, 0.0])
        axis = axis - axis.dot(v_from) * v_from
        axis /= np.linalg.norm(axis)

        return rotation_matrix_axis_angle(axis, np.pi)

    s = np.linalg.norm(cross)
    k = np.array([
        [0, -cross[2], cross[1]],
        [cross[2], 0, -cross[0]],
        [-cross[1], cross[0], 0],
    ])

    return np.eye(3) + k + k @ k * ((1 - dot) / (s ** 2))


def rotation_matrix_axis_angle(
    axis: np.ndarray,
    angle: float,
) -> np.ndarray:
    """
    Rodrigues rotation formula.

    Parameters
    ----------
    axis : np.ndarray
        Rotation axis (unit vector), shape (3,)
    angle : float
        Rotation angle in radians

    Returns
    -------
    np.ndarray
        Rotation matrix, shape (3, 3)
    """
    axis = axis / np.linalg.norm(axis)

    k = np.array([
        [0, -axis[2], axis[1]],
        [axis[2], 0, -axis[0]],
        [-axis[1], axis[0], 0],
    ])

    return (
        np.eye(3)
        + np.sin(angle) * k
        + (1 - np.cos(angle)) * (k @ k)
    )


# =========================
# Public API
# =========================

def transform_to_tank_frame(
    points: np.ndarray,
    axis: TankAxis,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Transform point cloud into tank coordinate system.

    Parameters
    ----------
    points : np.ndarray
        Input point cloud, shape (N, 3)
    axis : TankAxis
        Tank axis definition

    Returns
    -------
    points_tank : np.ndarray
        Transformed points, shape (N, 3)
    R : np.ndarray
        Applied rotation matrix, shape (3, 3)
    """
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("points must have shape (N, 3)")

    # 1. Translation
    translated = points - axis.point

    # 2. Rotation: axis.direction → Z
    z_axis = np.array([0.0, 0.0, 1.0])
    R = rotation_matrix_from_vectors(axis.direction, z_axis)

    # 3. Apply rotation
    points_rotated = (R @ translated.T).T

    return points_rotated, R
