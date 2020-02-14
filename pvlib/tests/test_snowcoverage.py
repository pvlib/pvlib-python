import numpy as np
import pandas as pd

from pandas.util.testing import assert_series_equal

from pvlib import snowcoverage
from pvlib.tools import sind


def test_snow_nrel_fully_covered():
    dt = pd.date_range(start="2019-1-1 12:00:00", end="2019-1-1 18:00:00",
                       freq='1h')
    snowfall_data = pd.Series([1, 5, .6, 4, .23, -5, 19], index=dt)
    expected = pd.Series([False, True, False, True, False, False, True],
                         index=dt)
    actual_snowfall = snowcoverage.snow_nrel_fully_covered(snowfall_data)
    assert_series_equal(actual_snowfall, expected)


def test_snow_nrel_hourly():
    surface_tilt = 45
    sliding_coefficient = 0.197
    dt = pd.date_range(start="2019-1-1 10:00:00", end="2019-1-1 17:00:00",
                       freq='1h')
    poa_irradiance = pd.Series([400, 200, 100, 1234, 134, 982, 100, 100],
                               index=dt)
    temp_air = pd.Series([10, 2, 10, 1234, 34, 982, 10, 10], index=dt)
    snowfall_data = pd.Series([1, .5, .6, .4, .23, -5, .1, .1], index=dt)
    snow_coverage = snowcoverage.snow_nrel(
        snowfall_data, poa_irradiance, temp_air, surface_tilt,
        threshold_snowfall=0.6)

    slide_amt = sliding_coefficient * sind(surface_tilt)
    covered = np.append(np.array([0., 0.]),
                        1.0 - slide_amt * np.array([0, 1, 2, 3, 4, 5]))
    expected = pd.Series(covered, index=dt)
    assert_series_equal(snow_coverage, expected)


def test_snow_nrel_subhourly():
    surface_tilt = 45
    sliding_coefficient = 0.197
    dt = pd.date_range(start="2019-1-1 11:00:00", end="2019-1-1 14:00:00",
                       freq='15T')
    poa_irradiance = pd.Series([400, 200, 100, 1234, 134, 982, 100, 100, 100,
                                100, 1000, 1000, 0],
                               index=dt)
    temp_air = pd.Series([10, 2, 10, 1234, 34, 982, 10, 10, 10, 10, 10, 10,
                             10], index=dt)
    snowfall_data = pd.Series([1, .5, .6, .4, .23, -5, .1, .1, 0., 1., 0., 0.,
                               0.], index=dt)
    snow_coverage = snowcoverage.snow_nrel(
        snowfall_data, poa_irradiance, temp_air, surface_tilt)
    slide_amt = sliding_coefficient * sind(surface_tilt) * 0.25
    covered = np.append(np.array([0., 1., 1., 1.]),
                        1.0 - slide_amt * np.array([1, 2, 3, 4, 5]))
    covered = np.append(covered, np.array([1., 1., 1., 1. - slide_amt]))
    expected = pd.Series(covered, index=dt)
    assert_series_equal(snow_coverage, expected)


def test_snow_nrel_dc_loss():
    num_strings = 8
    snow_coverage = pd.Series([1, 1, .5, .6, .2, .4, 0])
    expected = pd.Series([1, 1, .5, .625, .25, .5, 0])
    actual = snowcoverage.snow_nrel_dc_loss(snow_coverage, num_strings)
    assert_series_equal(expected, actual)
