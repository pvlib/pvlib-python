import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_almost_equal

from pvlib import scaling
from conftest import requires_scipy


# Sample positions
lat = np.array((9.99, 10, 10.01))
lon = np.array((4.99, 5, 5.01))
coordinates = np.array([(lati, loni) for (lati, loni) in zip(lat, lon)])

# Sample cloud speed
cloud_speed = 5
# Generate a sample clear_sky_index and time vector.
clear_sky_index = np.ones(10000)
clear_sky_index[5000:5005] = np.array([1, 1, 1.1, 0.9, 1])
time = np.arange(0, len(clear_sky_index))

# Sample dt
dt = 1

# Expected positions for sample lat/lon given above (calculated manually)
expect_xpos = np.array([554863.4, 555975.4, 557087.3])
expect_ypos = np.array([1110838.8, 1111950.8, 1113062.7])

# Sample positions based on the previous lat/lon
positions = np.array([pt for pt in zip(expect_xpos, expect_ypos)])

# Expected timescales for dt = 1
expect_tmscale = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]

# Expected wavelet for indices 5000:5004 using clear_sky_index above (Matlab)
expect_wavelet = np.array([[-0.025, 0.05, 0., -0.05, 0.025],
                           [0.025, 0., 0., 0., -0.025],
                           [0., 0., 0., 0., 0.]])

# Expected smoothed clear sky index for indices 5000:5004 (Matlab)
expect_cs_smooth = np.array([1., 1.0289, 1., 0.9711, 1.])


def test_latlon_to_xy_zero():
    coord = [0, 0]
    pos_e = [0, 0]
    pos = scaling.latlon_to_xy(coord)
    assert_almost_equal(pos, pos_e, decimal=1)


def test_latlon_to_xy_single():
    # Must test against central value, because latlon_to_dist uses the mean
    coord = (lat[1], lon[1])
    pos = scaling.latlon_to_xy(coord)
    assert_almost_equal(pos, (expect_xpos[1], expect_ypos[1]), decimal=1)


def test_latlon_to_xy_array():
    pos = scaling.latlon_to_xy(coordinates)
    assert_almost_equal(pos, positions, decimal=1)


def test_latlon_to_xy_list():
    pos = scaling.latlon_to_xy(coordinates.tolist())
    assert_almost_equal(pos, positions, decimal=1)


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


@requires_scipy
def test_wvm_series():
    csi_series = pd.Series(clear_sky_index, index=time)
    cs_sm, _, _ = scaling.wvm(csi_series, positions, cloud_speed)
    assert_almost_equal(cs_sm[5000:5005], expect_cs_smooth, decimal=4)


@requires_scipy
def test_wvm_array():
    cs_sm, _, _ = scaling.wvm(clear_sky_index, positions, cloud_speed, dt=dt)
    assert_almost_equal(cs_sm[5000:5005], expect_cs_smooth, decimal=4)


@requires_scipy
def test_wvm_series_xyaslist():
    csi_series = pd.Series(clear_sky_index, index=time)
    cs_sm, _, _ = scaling.wvm(csi_series, positions.tolist(), cloud_speed)
    assert_almost_equal(cs_sm[5000:5005], expect_cs_smooth, decimal=4)


@requires_scipy
def test_wvm_invalid():
    with pytest.raises(ValueError):
        scaling.wvm(clear_sky_index, positions, cloud_speed)
