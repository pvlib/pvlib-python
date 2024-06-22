import numpy as np
import pandas as pd
from pvlib import floating

from .conftest import assert_series_equal
from numpy.testing import assert_allclose


def test_daily_stream_temperature_stefan_default():
    result = floating.daily_stream_temperature_stefan(temp_air=15)
    assert_allclose(result, 16.25, 0.01)


def test_daily_stream_temperature_stefan_negative_temp():
    result = floating.daily_stream_temperature_stefan(temp_air=-15)
    assert_allclose(result, 0, 0.1)


def test_daily_stream_temperature_stefan_ndarray():
    air_temps = np.array([-5, 2.5, 20, 30, 40])
    result = floating.daily_stream_temperature_stefan(temp_air=air_temps)
    expected = np.array([1.25, 6.875, 20, 27.5, 35])
    assert_allclose(expected, result, atol=1e-3)


def test_daily_stream_temperature_stefan_series():
    times = pd.date_range(start="2015-01-01 00:00", end="2015-01-05 00:00",
                          freq="1d")
    air_temps = pd.Series([-5, 2.5, 20, 30, 40], index=times)

    result = floating.daily_stream_temperature_stefan(temp_air=air_temps)
    expected = pd.Series([1.25, 6.875, 20, 27.5, 35], index=times)
    assert_series_equal(expected, result, atol=1e-3)
