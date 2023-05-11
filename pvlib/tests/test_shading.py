import numpy as np
import pandas as pd

from pandas.testing import assert_series_equal
import pytest

from pvlib import shading


@pytest.fixture
def surface_tilt():
    idx = pd.date_range('2019-01-01', freq='h', periods=3)
    return pd.Series([0, 20, 90], index=idx)


@pytest.fixture
def masking_angle(surface_tilt):
    # masking angles for the surface_tilt fixture,
    # assuming GCR=0.5 and height=0.25
    return pd.Series([0.0, 11.20223712, 20.55604522], index=surface_tilt.index)


@pytest.fixture
def average_masking_angle(surface_tilt):
    # average masking angles for the surface_tilt fixture, assuming GCR=0.5
    return pd.Series([0.0, 7.20980655, 13.779867461], index=surface_tilt.index)


@pytest.fixture
def shading_loss(surface_tilt):
    # diffuse shading loss values for the average_masking_angle fixture
    return pd.Series([0, 0.00395338, 0.01439098], index=surface_tilt.index)


def test_masking_angle_series(surface_tilt, masking_angle):
    # series inputs and outputs
    masking_angle_actual = shading.masking_angle(surface_tilt, 0.5, 0.25)
    assert_series_equal(masking_angle_actual, masking_angle)


def test_masking_angle_scalar(surface_tilt, masking_angle):
    # scalar inputs and outputs, including zero
    for tilt, angle in zip(surface_tilt, masking_angle):
        masking_angle_actual = shading.masking_angle(tilt, 0.5, 0.25)
        assert np.isclose(masking_angle_actual, angle)


def test_masking_angle_zero_gcr(surface_tilt):
    # scalar inputs and outputs, including zero
    for tilt in surface_tilt:
        masking_angle_actual = shading.masking_angle(tilt, 0, 0.25)
        assert np.isclose(masking_angle_actual, 0)


def test_masking_angle_passias_series(surface_tilt, average_masking_angle):
    # pandas series inputs and outputs
    masking_angle_actual = shading.masking_angle_passias(surface_tilt, 0.5)
    assert_series_equal(masking_angle_actual, average_masking_angle)


def test_masking_angle_passias_scalar(surface_tilt, average_masking_angle):
    # scalar inputs and outputs, including zero
    for tilt, angle in zip(surface_tilt, average_masking_angle):
        masking_angle_actual = shading.masking_angle_passias(tilt, 0.5)
        assert np.isclose(masking_angle_actual, angle)


def test_sky_diffuse_passias_series(average_masking_angle, shading_loss):
    # pandas series inputs and outputs
    actual_loss = shading.sky_diffuse_passias(average_masking_angle)
    assert_series_equal(shading_loss, actual_loss)


def test_sky_diffuse_passias_scalar(average_masking_angle, shading_loss):
    # scalar inputs and outputs
    for angle, loss in zip(average_masking_angle, shading_loss):
        actual_loss = shading.sky_diffuse_passias(angle)
        assert np.isclose(loss, actual_loss)


@pytest.fixture
def expected_fs():
    # trivial case, 80% gcr, no slope, trackers & psz at 45-deg
    z0 = np.sqrt(2*0.8*0.8)
    # another trivial case, 60% gcr, no slope, trackers & psz at 60-deg
    z1 = 2*0.6
    # 30-deg isosceles, 60% gcr, no slope, 30-deg trackers, psz at 60-deg
    z2 = 0.6*np.sqrt(3)
    z = np.array([z0, z1, z2])
    return 1 - 1/z


def test_tracker_shade_fraction(expected_fs):
    """closes gh1690"""
    fs = shading.tracker_shaded_fraction(45.0, 0.8, 45.0)
    assert np.isclose(fs, expected_fs[0])
    # same trivial case with 40%, shadow is only 0.565-m long < 1-m r2r P
    zero_fs = shading.tracker_shaded_fraction(45.0, 0.4, 45.0)
    assert np.isclose(zero_fs, 0)
    # test vectors
    tracker_theta = [45.0, 60.0, 30.0]
    gcr = [0.8, 0.6, 0.6]
    psz = [45.0, 60.0, 60.0]
    slope = [0]*3
    fs_vec = shading.tracker_shaded_fraction(
        tracker_theta, gcr, psz, slope)
    assert np.allclose(fs_vec, expected_fs)


def test_linear_shade_loss(expected_fs):
    loss = shading.linear_shade_loss(expected_fs[0], 0.2)
    assert np.isclose(loss, 0.09289321881345258)
    # if no diffuse, shade fraction is the loss
    loss_no_df = shading.linear_shade_loss(expected_fs[0], 0)
    assert np.isclose(loss_no_df, expected_fs[0])
    # if all diffuse, no shade loss
    no_loss = shading.linear_shade_loss(expected_fs[0], 1.0)
    assert np.isclose(no_loss, 0)
    vec_loss = shading.linear_shade_loss(expected_fs, 0.2)
    expected_loss = np.array([0.09289322, 0.13333333, 0.03019964])
    assert np.allclose(vec_loss, expected_loss)
