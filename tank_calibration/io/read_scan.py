"""
Reading and preprocessing laser scan point cloud data.

Expected input format (CSV-like):
label, X, Y, Z

Example:
p1,41.723,31.885,3.442
"""

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd


def read_scan(
    filepath: str | Path,
    delimiter: str = ",",
) -> np.ndarray:
    """
    Read point cloud from scan file.

    Parameters
    ----------
    filepath : str or Path
        Path to scan file.
    delimiter : str
        Field delimiter (default: ',')

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with columns [X, Y, Z]
    """

    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scan file not found: {filepath}")

    # Read raw file
    df = pd.read_csv(
        filepath,
        delimiter=delimiter,
        header=None,
        names=["label", "x", "y", "z"],
        engine="python",
    )

    # Drop rows with missing coordinates
    df = df.dropna(subset=["x", "y", "z"])

    # Convert to numeric (coerce errors)
    for col in ["x", "y", "z"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["x", "y", "z"])

    points = df[["x", "y", "z"]].to_numpy(dtype=float)

    return points


def scan_summary(points: np.ndarray) -> dict:
    """
    Calculate basic statistics of point cloud.

    Parameters
    ----------
    points : np.ndarray
        Array of shape (N, 3)

    Returns
    -------
    dict
        Summary statistics
    """

    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError("Points array must have shape (N, 3)")

    summary = {
        "num_points": points.shape[0],
        "x_min": float(points[:, 0].min()),
        "x_max": float(points[:, 0].max()),
        "y_min": float(points[:, 1].min()),
        "y_max": float(points[:, 1].max()),
        "z_min": float(points[:, 2].min()),
        "z_max": float(points[:, 2].max()),
    }

    return summary
