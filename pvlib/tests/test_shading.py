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
    # masking angle values for the surface_tilt fixture, assuming GCR=0.5
    return pd.Series([0.0, 7.20980655, 13.779867461], index=surface_tilt.index)


@pytest.fixture
def shading_loss(surface_tilt):
    # diffuse shading loss values for the masking_angle fixture
    return pd.Series([0, 0.00395338, 0.01439098], index=surface_tilt.index)


def test_passias_masking_angle_series(surface_tilt, masking_angle):
    # pandas series inputs and outputs
    masking_angle_actual = shading.passias_masking_angle(surface_tilt, 0.5)
    assert_series_equal(masking_angle_actual, masking_angle)


def test_passias_masking_angle_scalar(surface_tilt, masking_angle):
    # scalar inputs and outputs, including zero
    for tilt, angle in zip(surface_tilt, masking_angle):
        masking_angle_actual = shading.passias_masking_angle(tilt, 0.5)
        assert np.isclose(masking_angle_actual, angle)


def test_passias_sky_diffuse_series(masking_angle, shading_loss):
    # pandas series inputs and outputs
    actual_loss = shading.passias_sky_diffuse(masking_angle)
    assert_series_equal(shading_loss, actual_loss)


def test_passias_sky_diffuse_scalar(masking_angle, shading_loss):
    # scalar inputs and outputs
    for angle, loss in zip(masking_angle, shading_loss):
        actual_loss = shading.passias_sky_diffuse(angle)
        assert np.isclose(loss, actual_loss)
