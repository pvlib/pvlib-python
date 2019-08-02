import numpy as np
from numpy.testing import assert_allclose
from pvlib import horizon, tools


def test_grid_lat_lon():
    grid_size = 100
    grid_step = .2
    lat = 24.21
    lon = -35.52
    grid = horizon.grid_lat_lon(lat, lon,
                                grid_size=grid_size,
                                grid_step=grid_step)

    assert(grid.shape[0] == 101)
    assert(grid[50][50][0] == lat)
    assert(grid[49][51][1] == lon + grid_step)


def test_dip_calc():
    pt1 = np.array((71.23, -34.70, 1234))
    pt2 = np.array((71.12, -34.16, 124))
    pt3 = np.array((71.29, -35.23, 30044))

    expected12 = (121.9895, -2.8654)
    expected21 = (302.5061, 2.6593)
    expected13 = (289.8132, 54.8663)

    actual12 = horizon.dip_calc(pt1, pt2)
    actual13 = horizon.dip_calc(pt1, pt3)
    actual21 = horizon.dip_calc(pt2, pt1)
    assert_allclose(expected12, actual12, rtol=1e-3)
    assert_allclose(expected13, actual13, rtol=1e-3)
    assert_allclose(expected21, actual21, rtol=1e-3)


def test_calculate_horizon_points():
    pass


def test_sample_using_grid():
    test_grid = np.array([[[1, 1, 3], [1, 2, 8], [1, 3, 8]],
                          [[2, 1, 1], [2, 2, 2], [2, 3, 1]],
                          [[3, 1, 5], [3, 2, 7], [3, 3, 9]]])
    samples = horizon.sample_using_grid(test_grid)
    assert(len(samples) == 8)


def test_sample_using_triangles():
    test_grid = np.array([[[1, 1, 3], [1, 2, 8], [1, 3, 8]],
                          [[2, 1, 1], [2, 2, 2], [2, 3, 1]],
                          [[3, 1, 5], [3, 2, 7], [3, 3, 9]]])
    samples = horizon.sample_using_triangles(test_grid, samples_per_triangle=2)
    assert(len(samples) == 32)


def test_using_interpolator():
    test_grid = np.array([[[1, 1, 3], [1, 2, 8], [1, 3, 8]],
                          [[2, 1, 1], [2, 2, 2], [2, 3, 1]],
                          [[3, 1, 5], [3, 2, 7], [3, 3, 9]]])
    samples = horizon.sample_using_interpolator(test_grid, num_samples=(5, 5))
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
        p = tools.lle_to_xyz(point)
        alpha = 0.5 * np.linalg.norm(np.cross(p2-p, p3-p)) / area
        beta = 0.5 * np.linalg.norm(np.cross(p3-p, p1-p)) / area
        gamma = 1 - alpha - beta
        assert(0 <= alpha <= 1)
        assert(0 <= beta <= 1)
        assert(0 <= gamma <= 1)


def test_filter_points():
    bogus_horizon = [(23, 10), (23.05, 8), (22.56, 14), (55, 2)]
    filtered = horizon.filter_points(bogus_horizon, bin_size=1)
    assert(len(filtered) == 2)
    assert(filtered[0][1] == 14)

    filtered = horizon.filter_points(bogus_horizon, bin_size=.2)
    assert(len(filtered) == 3)
    assert(filtered[1][1] == 10)


def test_collection_plane_dip_angle():
    surface_tilts = np.array([0, 5, 20, 38, 89])
    surface_azimuths = np.array([0, 90, 180, 235, 355])
    directions_easy = np.array([78, 270, 0, 145, 355])
    directions_hard = np.array([729, 220, 60, 115, 3545])

    expected_easy = np.array([0, 5, 20, 0, 0])
    expected_hard = np.array([0, 3.21873120519, 10.3141048156,
                              21.3377447931, 0])
    dips_easy = horizon.collection_plane_dip_angle(surface_tilts,
                                                   surface_azimuths,
                                                   directions_easy)
    assert_allclose(dips_easy, expected_easy)

    dips_hard = horizon.collection_plane_dip_angle(surface_tilts,
                                                   surface_azimuths,
                                                   directions_hard)
    assert_allclose(dips_hard, expected_hard)


def test_calculate_dtf():
    zero_horizon = []
    max_horizon = []
    for i in range(-180, 181):
        zero_horizon.append((i, 0.0))
        max_horizon.append((i, 90.0))

    surface_tilts = np.array([0, 5, 20, 38, 89])
    surface_azimuths = np.array([0, 90, 180, 235, 355])

    adjusted = horizon.calculate_dtf(zero_horizon,
                                     surface_tilts,
                                     surface_azimuths)
    expected = (1 + tools.cosd(surface_tilts)) * 0.5
    assert_allclose(adjusted, expected, atol=2e-3)

    adjusted = horizon.calculate_dtf(max_horizon,
                                     surface_tilts,
                                     surface_azimuths)
    expected = np.zeros(5)
    assert_allclose(adjusted, expected, atol=1e-7)
