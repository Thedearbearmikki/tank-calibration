from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def read_scan(
    filepath: str | Path,
    delimiter: str = ",",
    drop_service_points: bool = True,
) -> np.ndarray:
    """
    Read laser scan point cloud from CSV-like file.

    Expected format per line:
        label, x, y, z,   (trailing comma allowed)

    Wall points carry purely numeric labels (1, 2, 3, ...). Points with
    letter labels (st1, m1, luk, kor, p0, a1, ...) are service points:
    scanner stations, marks, hatch, etc. They distort the geometry fit
    and are dropped by default.

    Parameters
    ----------
    filepath : str or Path
        Path to scan file
    delimiter : str, optional
        Field delimiter, default ','
    drop_service_points : bool, optional
        Keep only points with numeric labels, default True.
        If the file has no numeric labels at all, all points are kept.

    Returns
    -------
    np.ndarray
        Array of shape (N, 3) with columns [X, Y, Z]
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Scan file not found: {filepath}")

    # Read raw file (allow extra empty column)
    df = pd.read_csv(
        filepath,
        delimiter=delimiter,
        header=None,
        engine="python",
    )

    if df.shape[1] < 4:
        raise ValueError("Scan file must contain at least 4 columns: label, x, y, z")

    if drop_service_points:
        labels = df.iloc[:, 0].astype(str).str.strip()
        is_wall = labels.str.fullmatch(r"\d+")
        if is_wall.any():
            df = df[is_wall]

    # Take columns explicitly: x, y, z
    df_xyz = df.iloc[:, 1:4]

    # Convert to numeric, drop invalid rows
    df_xyz = df_xyz.apply(pd.to_numeric, errors="coerce")
    df_xyz = df_xyz.dropna()

    points = df_xyz.to_numpy(dtype=float)

    if points.shape[0] == 0:
        raise ValueError("Scan file contains no valid XYZ points")

    return points
