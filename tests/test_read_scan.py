from pathlib import Path

import numpy as np

from tank_calibration.io.read_scan import read_scan

DATA = Path(__file__).resolve().parent.parent / "data"


def _write(tmp_path, text):
    f = tmp_path / "scan.csv"
    f.write_text(text)
    return f


def test_service_points_dropped(tmp_path):
    f = _write(tmp_path, "st1,50,50,3,\n1,1,2,3,\nluk,9,9,9,\n2,4,5,6,\n")
    pts = read_scan(f)
    np.testing.assert_array_equal(pts, [[1, 2, 3], [4, 5, 6]])


def test_service_points_kept_when_disabled(tmp_path):
    f = _write(tmp_path, "st1,50,50,3,\n1,1,2,3,\n")
    assert read_scan(f, drop_service_points=False).shape == (2, 3)


def test_file_without_numeric_labels_kept_whole():
    pts = read_scan(DATA / "examples" / "example_scan.txt")
    assert pts.shape == (5, 3)


def test_real_scan_drops_36_service_points():
    assert read_scan(DATA / "scan.csv").shape == (1193, 3)
