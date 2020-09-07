import numpy as np
import pandas as pd

import pytest
from numpy.testing import assert_almost_equal

from pvlib import scaling


# Sample cloud speed
cloud_speed = 5

# Sample dt
dt = 1


@pytest.fixture
def coordinates():
    # Sample positions in lat/lon
    lat = np.array((9.99, 10, 10.01))
    lon = np.array((4.99, 5, 5.01))
    coordinates = np.array([(lati, loni) for (lati, loni) in zip(lat, lon)])
    return coordinates


@pytest.fixture
def clear_sky_index():
    # Generate a sample clear_sky_index
    clear_sky_index = np.ones(10000)
    clear_sky_index[5000:5005] = np.array([1, 1, 1.1, 0.9, 1])
    return clear_sky_index


@pytest.fixture
def time(clear_sky_index):
    # Sample time vector
    return np.arange(0, len(clear_sky_index))


@pytest.fixture
def positions():
    # Sample positions based on the previous lat/lon (calculated manually)
    expect_xpos = np.array([554863.4, 555975.4, 557087.3])
    expect_ypos = np.array([1110838.8, 1111950.8, 1113062.7])
    return np.array([pt for pt in zip(expect_xpos, expect_ypos)])


@pytest.fixture
def expect_tmscale():
    # Expected timescales for dt = 1
    return [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]


@pytest.fixture
def expect_wavelet():
    # Expected wavelet for indices 5000:5004 for clear_sky_index above (Matlab)
    return np.array([[-0.025, 0.05, 0., -0.05, 0.025],
                     [0.025, 0., 0., 0., -0.025],
                     [0., 0., 0., 0., 0.]])


@pytest.fixture
def expect_cs_smooth():
    # Expected smoothed clear sky index for indices 5000:5004 (Matlab)
    return np.array([1., 1.0289, 1., 0.9711, 1.])


def test_latlon_to_xy_zero():
    coord = [0, 0]
    pos_e = [0, 0]
    pos = scaling.latlon_to_xy(coord)
    assert_almost_equal(pos, pos_e, decimal=1)


def test_latlon_to_xy_single(coordinates, positions):
    # Must test against central value, because latlon_to_xy uses the mean
    coord = coordinates[1]
    pos = scaling.latlon_to_xy(coord)
    assert_almost_equal(pos, positions[1], decimal=1)


def test_latlon_to_xy_array(coordinates, positions):
    pos = scaling.latlon_to_xy(coordinates)
    assert_almost_equal(pos, positions, decimal=1)


def test_latlon_to_xy_list(coordinates, positions):
    pos = scaling.latlon_to_xy(coordinates.tolist())
    assert_almost_equal(pos, positions, decimal=1)


def test_compute_wavelet_series(clear_sky_index, time,
                                expect_tmscale, expect_wavelet):
    csi_series = pd.Series(clear_sky_index, index=time)
    wavelet, tmscale = scaling._compute_wavelet(csi_series)
    assert_almost_equal(tmscale, expect_tmscale)
    assert_almost_equal(wavelet[0:3, 5000:5005], expect_wavelet)


def test_compute_wavelet_series_numindex(clear_sky_index, time,
                                         expect_tmscale, expect_wavelet):
    dtindex = pd.to_datetime(time, unit='s')
    csi_series = pd.Series(clear_sky_index, index=dtindex)
    wavelet, tmscale = scaling._compute_wavelet(csi_series)
    assert_almost_equal(tmscale, expect_tmscale)
    assert_almost_equal(wavelet[0:3, 5000:5005], expect_wavelet)


def test_compute_wavelet_array(clear_sky_index,
                               expect_tmscale, expect_wavelet):
    wavelet, tmscale = scaling._compute_wavelet(clear_sky_index, dt)
    assert_almost_equal(tmscale, expect_tmscale)
    assert_almost_equal(wavelet[0:3, 5000:5005], expect_wavelet)


def test_compute_wavelet_array_invalid(clear_sky_index):
    with pytest.raises(ValueError):
        scaling._compute_wavelet(clear_sky_index)


def test_wvm_series(clear_sky_index, time, positions, expect_cs_smooth):
    csi_series = pd.Series(clear_sky_index, index=time)
    cs_sm, _, _ = scaling.wvm(csi_series, positions, cloud_speed)
    assert_almost_equal(cs_sm[5000:5005], expect_cs_smooth, decimal=4)


def test_wvm_array(clear_sky_index, positions, expect_cs_smooth):
    cs_sm, _, _ = scaling.wvm(clear_sky_index, positions, cloud_speed, dt=dt)
    assert_almost_equal(cs_sm[5000:5005], expect_cs_smooth, decimal=4)


def test_wvm_series_xyaslist(clear_sky_index, time, positions,
                             expect_cs_smooth):
    csi_series = pd.Series(clear_sky_index, index=time)
    cs_sm, _, _ = scaling.wvm(csi_series, positions.tolist(), cloud_speed)
    assert_almost_equal(cs_sm[5000:5005], expect_cs_smooth, decimal=4)


def test_wvm_invalid(clear_sky_index, positions):
    with pytest.raises(ValueError):
        scaling.wvm(clear_sky_index, positions, cloud_speed)
