from tank_calibration.io.read_scan import read_scan
from tank_calibration.geometry.axis import compute_tank_axis
from tank_calibration.geometry.transform import transform_to_tank_frame
from tank_calibration.geometry.tilt import compute_tank_tilt
from tank_calibration.geometry.radius_profile import compute_radius_profile
from tank_calibration.geometry.cylinder_fit import fit_cylinder_xy


def main():
    points = read_scan("data/scan.csv")

    axis = compute_tank_axis(points)
    points_tank, _ = transform_to_tank_frame(points, axis)

    cyl = fit_cylinder_xy(points_tank)

    print("Cylinder fit:")
    print("  X0   =", cyl.x0)
    print("  Y0   =", cyl.y0)
    print("  R    =", cyl.r)
    print("  RMSE =", cyl.rmse)

    tilt = compute_tank_tilt(axis)
    profile = compute_radius_profile(points_tank)

    print()
    print("Axis:", axis)
    print("Tilt:", tilt)
    print("Radius slices:", len(profile.slices))


if __name__ == "__main__":
    main()
