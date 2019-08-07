import numpy as np
from numpy.testing import assert_allclose
from pvlib import horizon, tools


def test_grid_lat_lon():
    grid_radius = 50
    grid_step = .2
    lat = 24.21
    lon = -35.52
    lat_grid, lon_grid = horizon.grid_lat_lon(lat, lon,
                                              grid_radius=grid_radius,
                                              grid_step=grid_step)

    assert(lat_grid.shape[0] == 101)
    assert(lat_grid.shape == lon_grid.shape)
    assert_allclose(lat_grid[50][50], lat)
    assert_allclose(lon_grid[49][51], lon - grid_step)
    assert_allclose(lat_grid[30][72], lat + 22*grid_step)


def test_elev_calc():
    pt1 = np.array((71.23, -34.70, 1234))
    pt2 = np.array((71.12, -34.16, 124))
    pt3 = np.array((71.29, -35.23, 30044))

    test_pts = np.vstack([pt1, pt2])
    reverse_test_pts = np.vstack([pt2, pt1])

    expected_bearings = np.array([121.9895, 302.5061])
    expected_elevs = np.array([-2.8654, 2.6593])

    expected13 = np.array([[289.8132], [54.8663]])

    act_bearings, act_elevs = horizon.elevation_angle_calc(test_pts,
                                                           reverse_test_pts)
    assert_allclose(act_bearings, expected_bearings, rtol=1e-3)
    assert_allclose(act_elevs, expected_elevs, rtol=1e-3)

    actual13 = horizon.elevation_angle_calc(pt1, pt3)
    assert_allclose(expected13, actual13, rtol=1e-3)


def test_calculate_horizon_points():
    test_lat_grid = np.array([[1, 2, 3],
                              [1, 2, 3],
                              [1, 2, 3]])

    test_lon_grid = np.array([[-3, -3, -3],
                              [-2, -2, -2],
                              [-1, -1, -1]])

    test_elev_grid = np.array([[15, 18, 43],
                               [212, 135, 1],
                               [36, 145, 5]])

    dirs, elevs = horizon.calculate_horizon_points(test_lat_grid,
                                                   test_lon_grid,
                                                   test_elev_grid,
                                                   sampling_method="grid")

    expected_dirs = np.array([0, 90, 45, 270, 135])
    rounded_dirs = np.round(dirs).astype(int)
    assert(dirs.shape == elevs.shape)
    assert(np.all(np.in1d(expected_dirs, rounded_dirs)))

    dirs, elevs = horizon.calculate_horizon_points(test_lat_grid,
                                                   test_lon_grid,
                                                   test_elev_grid,
                                                   sampling_method="triangles",
                                                   sampling_param=5)
    assert(dirs.shape == elevs.shape)

    dirs, _ = horizon.calculate_horizon_points(test_lat_grid,
                                               test_lon_grid,
                                               test_elev_grid,
                                               sampling_method="interpolator",
                                               sampling_param=(10, 10))
    assert(dirs.shape[0] == 100)


def test_sample_using_grid():
    test_lat_grid = np.array([[1, 1, 3],
                              [2, 1, 1],
                              [3, 1, 5]])

    test_lon_grid = np.array([[1, 1, -3],
                              [2, -1, 1],
                              [3, -15, 5]])

    test_elev_grid = np.array([[15, 18, 43],
                               [212, 135, 1],
                               [36, 145, 5]])

    samples = horizon.sample_using_grid(test_lat_grid,
                                        test_lon_grid,
                                        test_elev_grid)
    assert(len(samples) == 8)


def test_sample_using_triangles():
    test_lat_grid = np.array([[1, 1, 3],
                              [2, 1, 1],
                              [3, 1, 5]])

    test_lon_grid = np.array([[1, 1, -3],
                              [2, -1, 1],
                              [3, -15, 5]])

    test_elev_grid = np.array([[15, 18, 43],
                               [212, 135, 1],
                               [36, 145, 5]])
    samples = horizon.sample_using_triangles(test_lat_grid,
                                             test_lon_grid,
                                             test_elev_grid,
                                             samples_per_triangle=2)
    assert(len(samples) == 32)


def test_using_interpolator():
    test_lat_grid = np.array([[1, 2, 3],
                              [1, 2, 3],
                              [1, 2, 3]])

    test_lon_grid = np.array([[-3, -3, -3],
                              [-2, -2, -2],
                              [-1, -1, -1]])

    test_elev_grid = np.array([[15, 18, 43],
                               [212, 135, 1],
                               [36, 145, 5]])
    samples = horizon.sample_using_interpolator(test_lat_grid,
                                                test_lon_grid,
                                                test_elev_grid,
                                                num_samples=(5, 5))
    assert(len(samples) == 25)


def test_uniformly_sample_triangle():
    pt1 = np.array((71.23, -34.70, 1234))
    pt2 = np.array((69.12, -38.16, 124))
    pt3 = np.array((78.23, -36.23, 344))
    points = horizon.uniformly_sample_triangle(pt1, pt2, pt3, 5)

    p1 = tools.lle_to_xyz(pt1)
    p2 = tools.lle_to_xyz(pt2)
    p3 = tools.lle_to_xyz(pt3)
    area = 0.5 * np.linalg.norm(np.cross(p2-p1, p3-p1))

    for point in points:
        print(point)
        p = tools.lle_to_xyz(point)
        alpha = 0.5 * np.linalg.norm(np.cross(p2-p, p3-p)) / area
        beta = 0.5 * np.linalg.norm(np.cross(p3-p, p1-p)) / area
        gamma = 1 - alpha - beta
        assert(0 <= alpha <= 1)
        assert(0 <= beta <= 1)
        assert(0 <= gamma <= 1)


def test_filter_points():
    test_azimuths = np.array([23, 23.05, 22.56, 55])
    bogus_horizon = np.array([10, 8, 14, 2])
    filtered_azimuths, filtered_angles = horizon.filter_points(test_azimuths,
                                                               bogus_horizon,
                                                               bin_size=1)
    assert(filtered_azimuths.shape[0] == filtered_angles.shape[0])
    assert(filtered_angles[0] == 14)

    filtered_azimuths, filtered_angles = horizon.filter_points(test_azimuths,
                                                               bogus_horizon,
                                                               bin_size=.2)
    assert(filtered_azimuths.shape[0] == 3)
    assert(filtered_angles[1] == 10)


def test_collection_plane_elev_angle():
    surface_tilts = np.array([0, 5, 20, 38, 89])
    surface_azimuths = np.array([0, 90, 180, 235, 355])
    directions_easy = np.array([78, 270, 0, 145, 355])
    directions_hard = np.array([729, 220, 60, 115, 3545])

    expected_easy = np.array([0, 5, 20, 0, 0])
    expected_hard = np.array([0, 3.21873120519, 10.3141048156,
                              21.3377447931, 0])
    elevs_easy = horizon.collection_plane_elev_angle(surface_tilts,
                                                     surface_azimuths,
                                                     directions_easy)
    assert_allclose(elevs_easy, expected_easy)

    elevs_hard = horizon.collection_plane_elev_angle(surface_tilts,
                                                     surface_azimuths,
                                                     directions_hard)
    assert_allclose(elevs_hard, expected_hard)


def test_calculate_dtf():
    num_points = 360
    test_azimuths = np.arange(0, num_points, dtype=np.float64)
    zero_horizon = np.zeros(num_points)
    max_horizon = np.full((num_points), 90.0)
    uniform_horizon = np.full((num_points), 7.0)
    random_horizon = np.random.random(num_points) * 7

    surface_tilts = np.array([0, 5, 20, 38, 89])
    surface_azimuths = np.array([0, 90, 180, 235, 355])

    adjusted = horizon.calculate_dtf(test_azimuths,
                                     zero_horizon,
                                     surface_tilts,
                                     surface_azimuths)
    expected = (1 + tools.cosd(surface_tilts)) * 0.5
    assert_allclose(adjusted, expected, atol=2e-3)

    adjusted = horizon.calculate_dtf(test_azimuths,
                                     max_horizon,
                                     surface_tilts,
                                     surface_azimuths)
    expected = np.zeros(5)
    assert_allclose(adjusted, expected, atol=1e-7)

    adjusted = horizon.calculate_dtf(test_azimuths,
                                     random_horizon,
                                     surface_tilts,
                                     surface_azimuths)
    min_random_dtf = horizon.calculate_dtf(test_azimuths,
                                           uniform_horizon,
                                           surface_tilts,
                                           surface_azimuths)
    max_random_dtf = (1 + tools.cosd(surface_tilts)) * 0.5

    mask = np.logical_and((adjusted >= min_random_dtf),
                          (adjusted <= max_random_dtf))
    assert(np.all(mask))
