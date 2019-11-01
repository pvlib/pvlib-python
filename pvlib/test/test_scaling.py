import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_almost_equal

from pvlib import scaling

# All expected_xxxxxx variable results computed in Matlab code

# Sample positions
lat = np.array((9.99, 10, 10.01))
lon = np.array((4.99, 5, 5.01))
# Sample cloud speed
cloud_speed = 5
# Generate a sample clear_sky_index and time vector.
clear_sky_index = np.ones(10000)
clear_sky_index[5000:5005] = np.array([1, 1, 1.1, 0.9, 1])
time = np.arange(0, len(clear_sky_index))
# Sample dt
dt = 1

# Expected distance and positions for sample lat/lon given above
expect_dist = np.array((1560.6, 3121.3, 1560.6))
expect_xpos = np.array([554863.4, 555975.4, 557087.3])
expect_ypos = np.array([1106611.8, 1107719.5, 1108827.2])

# Expected timescales for dt = 1
expect_tmscale = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Expected wavelet for indices 5000:5004 using clear_sky_index above
expect_wavelet = np.array([[-0.025, 0.05, 0. ,-0.05, 0.025],
                           [ 0.025, 0.  , 0. , 0.  ,-0.025],
                           [ 0.,    0.  , 0. , 0.  , 0.]])

# Expected smoothed clear sky index for indices 5000:5004 using inputs above
expect_cs_smooth = np.array([1., 1.0289, 1., 0.9711, 1.])


def test_latlon_to_dist_zero():
    lat = 0
    lon = 0
    xpos_e = 0
    ypos_e = 0
    xpos, ypos = scaling._latlon_to_dist(lon, lat)
    assert_almost_equal(xpos, xpos_e, decimal=1)
    assert_almost_equal(ypos, ypos_e, decimal=1)


def test_latlon_to_dist_single():
    # Must test against central value, because latlon_to_dist uses the mean
    xpos, ypos = scaling._latlon_to_dist(lon[1], lat[1])
    assert_almost_equal(xpos, expect_xpos[1], decimal=1)
    assert_almost_equal(ypos, expect_ypos[1], decimal=1)


def test_latlon_to_dist_array():
    xpos, ypos = scaling._latlon_to_dist(lon, lat)
    assert_almost_equal(xpos, expect_xpos, decimal=1)
    assert_almost_equal(ypos, expect_ypos, decimal=1)


def test_compute_distances_invalid():
    with pytest.raises(ValueError):
        scaling._compute_distances(0, 0, method='invalid')


def test_compute_distances_discrete_zero():
    lat = np.array((0, 0))
    lon = np.array((0, 0))
    assert_almost_equal(scaling._compute_distances(lon, lat, 'discrete'), 0)


def test_compute_distances_discrete_array():

    dist = scaling._compute_distances(lon, lat, 'discrete')
    assert_almost_equal(dist, expect_dist, decimal=1)


def test_compute_wavelet_series():
    csi_series = pd.Series(clear_sky_index, index=time)
    wavelet, tmscale = scaling._compute_wavelet(csi_series)
    assert_almost_equal(tmscale, expect_tmscale)
    assert_almost_equal(wavelet[0:3, 5000:5005], expect_wavelet)


def test_compute_wavelet_array():
    wavelet, tmscale = scaling._compute_wavelet(clear_sky_index, dt)
    assert_almost_equal(tmscale, expect_tmscale)
    assert_almost_equal(wavelet[0:3, 5000:5005], expect_wavelet)


def test_compute_wavelet_array_invalid():
    with pytest.raises(ValueError):
        scaling._compute_wavelet(clear_sky_index)


def test_wvm_series():
    csi_series = pd.Series(clear_sky_index, index=time)
    cs_sm, _, _ = scaling.wvm(csi_series, lat, lon, cloud_speed, "discrete")
    assert_almost_equal(cs_sm[5000:5005], expect_cs_smooth, decimal=4)


def test_compute_wvm_array():
    cs_sm, _, _ = scaling.wvm(clear_sky_index, lat, lon, cloud_speed,
                              "discrete", dt=dt)
    assert_almost_equal(cs_sm[5000:5005], expect_cs_smooth, decimal=4)


def test_compute_wvm_array_invalid():
    with pytest.raises(ValueError):
        scaling.wvm(clear_sky_index, lat, lon, cloud_speed, "discrete")
