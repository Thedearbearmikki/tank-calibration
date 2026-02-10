from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


def read_scan(
    filepath: str | Path,
    delimiter: str = ",",
) -> np.ndarray:
    """
    Read laser scan point cloud from CSV-like file.

    Expected format per line:
        label, x, y, z,   (trailing comma allowed)

    Parameters
    ----------
    filepath : str or Path
        Path to scan file
    delimiter : str, optional
        Field delimiter, default ','

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

    # Take columns explicitly: x, y, z
    df_xyz = df.iloc[:, 1:4]

    # Convert to numeric, drop invalid rows
    df_xyz = df_xyz.apply(pd.to_numeric, errors="coerce")
    df_xyz = df_xyz.dropna()

    points = df_xyz.to_numpy(dtype=float)

    if points.shape[0] == 0:
        raise ValueError("Scan file contains no valid XYZ points")

    return points
