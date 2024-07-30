import numpy as np
import pandas as pd
from pvlib import floating

from .conftest import assert_series_equal
from numpy.testing import assert_allclose
import pytest


@pytest.mark.parametrize('air_temp,water_temp', [
    (-15, np.nan),
    (-5, 1.25),
    (40, 35),
])
def test_stream_temperature_stefan(air_temp, water_temp):
    result = floating.stream_temperature_stefan(air_temp)
    assert_allclose(result, water_temp)


@pytest.fixture
def times():
    return pd.date_range(start="2015-01-01 00:00", end="2015-01-07 00:00",
                         freq="1d")


@pytest.fixture
def air_temps(times):
    return pd.Series([-15, -5, 2.5, 15, 20, 30, 40], index=times)


@pytest.fixture
def water_temps_expected(times):
    return pd.Series([np.nan, 1.25, 6.875, 16.25, 20, 27.5, 35], index=times)


def test_stream_temperature_stefan_ndarray(air_temps, water_temps_expected):
    result = floating.stream_temperature_stefan(temp_air=air_temps.to_numpy())
    assert_allclose(water_temps_expected.to_numpy(), result)


def test_stream_temperature_stefan_series(air_temps, water_temps_expected):
    result = floating.stream_temperature_stefan(temp_air=air_temps)
    assert_series_equal(water_temps_expected, result)
