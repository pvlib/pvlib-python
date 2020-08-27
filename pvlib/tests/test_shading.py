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
