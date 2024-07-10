import numpy as np
import pandas as pd
import pytest
from pvlib import windspeed

from .conftest import assert_series_equal
from numpy.testing import assert_allclose


@pytest.mark.parametrize(
    'wind_speed_measured,height_measured,height_desired,wind_speed_calc',
    [
        (10, -2, 5, np.nan),
        (-10, 2, 5, np.nan),
        (5, 4, 5, 5.067393209486324),
        (7, 6, 10, 7.2178684911195905),
        (10, 8, 20, 10.565167835216586),
        (12, 10, 30, 12.817653329393977),])
def test_wind_speed_at_height_hellmann(wind_speed_measured,
                                       height_measured,
                                       height_desired,
                                       wind_speed_calc):
    result = windspeed.wind_speed_at_height_hellmann(
        wind_speed_measured,
        height_measured,
        height_desired,
        surface_type='unstable_air_above_open_water_surface')
    assert_allclose(result, wind_speed_calc)


@pytest.fixture
def times():
    return pd.date_range(start="2015-01-01 00:00", end="2015-01-01 05:00",
                         freq="1h")


@pytest.fixture
def wind_speeds_measured(times):
    return pd.Series([10, -10, 5, 7, 10, 12], index=times)


@pytest.fixture
def heights_measured(times):
    return np.array([-2, 2, 4, 6, 8, 10])


@pytest.fixture
def heights_desired():
    return np.array([5, 5, 5, 10, 20, 30])


@pytest.fixture
def wind_speeds_calc(times):
    return pd.Series([np.nan, np.nan, 5.067393209486324, 7.2178684911195905,
                      10.565167835216586, 12.817653329393977], index=times)


def test_wind_speed_at_height_hellmann_ndarray(wind_speeds_measured,
                                               heights_measured,
                                               heights_desired,
                                               wind_speeds_calc):
    result = windspeed.wind_speed_at_height_hellmann(
        wind_speeds_measured.to_numpy(),
        heights_measured,
        heights_desired,
        surface_type='unstable_air_above_open_water_surface')
    assert_allclose(wind_speeds_calc.to_numpy(), result)


def test_wind_speed_at_height_hellmann_series(wind_speeds_measured,
                                              heights_measured,
                                              heights_desired,
                                              wind_speeds_calc):
    result = windspeed.wind_speed_at_height_hellmann(
        wind_speeds_measured,
        heights_measured,
        heights_desired,
        surface_type='unstable_air_above_open_water_surface')
    assert_series_equal(wind_speeds_calc, result)


def test_wind_speed_at_height_hellmann_invalid():
    with pytest.raises(ValueError, match='Either a `surface_type` has to be '
                       'chosen or an exponent'):
        # no exponent or surface_type given
        windspeed.wind_speed_at_height_hellmann(wind_speed_measured=10,
                                                height_measured=5,
                                                height_desired=10)
    with pytest.raises(KeyError, match='not_an_exponent'):
        # invalid surface_type
        windspeed.wind_speed_at_height_hellmann(wind_speed_measured=10,
                                                height_measured=5,
                                                height_desired=10,
                                                surface_type='not_an_exponent')
