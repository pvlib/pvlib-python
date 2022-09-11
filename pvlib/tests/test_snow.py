import numpy as np
import pandas as pd

from .conftest import assert_series_equal

from pvlib import snow
from pvlib.tools import sind


def test_fully_covered_nrel():
    dt = pd.date_range(start="2019-1-1 12:00:00", end="2019-1-1 18:00:00",
                       freq='1h')
    snowfall_data = pd.Series([1, 5, .6, 4, .23, -5, 19], index=dt)
    expected = pd.Series([False, True, False, True, False, False, True],
                         index=dt)
    fully_covered = snow.fully_covered_nrel(snowfall_data)
    assert_series_equal(expected, fully_covered)


def test_coverage_nrel_hourly():
    surface_tilt = 45
    slide_amount_coefficient = 0.197
    dt = pd.date_range(start="2019-1-1 10:00:00", end="2019-1-1 17:00:00",
                       freq='1h')
    poa_irradiance = pd.Series([400, 200, 100, 1234, 134, 982, 100, 100],
                               index=dt)
    temp_air = pd.Series([10, 2, 10, 1234, 34, 982, 10, 10], index=dt)
    snowfall_data = pd.Series([1, .5, .6, .4, .23, -5, .1, .1], index=dt)
    snow_coverage = snow.coverage_nrel(
        snowfall_data, poa_irradiance, temp_air, surface_tilt,
        threshold_snowfall=0.6)

    slide_amt = slide_amount_coefficient * sind(surface_tilt)
    covered = 1.0 - slide_amt * np.array([0, 1, 2, 3, 4, 5, 6, 7])
    expected = pd.Series(covered, index=dt)
    assert_series_equal(expected, snow_coverage)


def test_coverage_nrel_subhourly():
    surface_tilt = 45
    slide_amount_coefficient = 0.197
    dt = pd.date_range(start="2019-1-1 11:00:00", end="2019-1-1 14:00:00",
                       freq='15T')
    poa_irradiance = pd.Series([400, 200, 100, 1234, 134, 982, 100, 100, 100,
                                100, 100, 100, 0],
                               index=dt)
    temp_air = pd.Series([10, 2, 10, 1234, 34, 982, 10, 10, 10, 10, -10, -10,
                          10], index=dt)
    snowfall_data = pd.Series([1, .5, .6, .4, .23, -5, .1, .1, 0., 1., 0., 0.,
                               0.], index=dt)
    snow_coverage = snow.coverage_nrel(
        snowfall_data, poa_irradiance, temp_air, surface_tilt)
    slide_amt = slide_amount_coefficient * sind(surface_tilt) * 0.25
    covered = np.append(np.array([1., 1., 1., 1.]),
                        1.0 - slide_amt * np.array([1, 2, 3, 4, 5]))
    covered = np.append(covered, np.array([1., 1., 1., 1. - slide_amt]))
    expected = pd.Series(covered, index=dt)
    assert_series_equal(expected, snow_coverage)


def test_fully_covered_nrel_irregular():
    # test when frequency is not specified and can't be inferred
    dt = pd.DatetimeIndex(["2019-1-1 11:00:00", "2019-1-1 14:30:00",
                           "2019-1-1 15:07:00", "2019-1-1 14:00:00"])
    snowfall_data = pd.Series([1, .5, .6, .4], index=dt)
    snow_coverage = snow.fully_covered_nrel(snowfall_data,
                                            threshold_snowfall=0.5)
    covered = np.array([False, False, True, False])
    expected = pd.Series(covered, index=dt)
    assert_series_equal(expected, snow_coverage)


def test_coverage_nrel_initial():
    surface_tilt = 45
    slide_amount_coefficient = 0.197
    dt = pd.date_range(start="2019-1-1 10:00:00", end="2019-1-1 17:00:00",
                       freq='1h')
    poa_irradiance = pd.Series([400, 200, 100, 1234, 134, 982, 100, 100],
                               index=dt)
    temp_air = pd.Series([10, 2, 10, 1234, 34, 982, 10, 10], index=dt)
    snowfall_data = pd.Series([0, .5, .6, .4, .23, -5, .1, .1], index=dt)
    snow_coverage = snow.coverage_nrel(
        snowfall_data, poa_irradiance, temp_air, surface_tilt,
        initial_coverage=0.5, threshold_snowfall=1.)
    slide_amt = slide_amount_coefficient * sind(surface_tilt)
    covered = 0.5 - slide_amt * np.array([0, 1, 2, 3, 4, 5, 6, 7])
    covered = np.where(covered < 0, 0., covered)
    expected = pd.Series(covered, index=dt)
    assert_series_equal(expected, snow_coverage)


def test_dc_loss_nrel():
    num_strings = 8
    snow_coverage = pd.Series([1, 1, .5, .6, .2, .4, 0])
    expected = pd.Series([1, 1, .5, .625, .25, .5, 0])
    actual = snow.dc_loss_nrel(snow_coverage, num_strings)
    assert_series_equal(expected, actual)


def test__townsend_effective_snow():
    snow_total = np.array([25.4, 25.4, 12.7, 2.54, 0, 0, 0, 0, 0, 0, 12.7,
                           25.4])
    snow_events = np.array([2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 3])
    expected = np.array([19.05, 19.05, 12.7, 0, 0, 0, 0, 0, 0, 0, 9.525,
                         254 / 15])
    actual = snow._townsend_effective_snow(snow_total, snow_events)
    np.testing.assert_allclose(expected, actual, rtol=1e-07)


def test_loss_townsend():
    snow_total = np.array([25.4, 25.4, 12.7, 2.54, 0, 0, 0, 0, 0, 0, 12.7,
                           25.4])
    snow_events = np.array([2, 2, 1, 0, 0, 0, 0, 0, 0, 0, 2, 3])
    surface_tilt = 20
    relative_humidity = np.array([80, 80, 80, 80, 80, 80, 80, 80, 80, 80,
                                  80, 80])
    temp_air = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    poa_global = np.array([350000, 350000, 350000, 350000, 350000, 350000,
                           350000, 350000, 350000, 350000, 350000, 350000])
    angle_of_repose = 40
    slant_height = 2.54
    lower_edge_height = 0.254
    expected = np.array([0.07696253, 0.07992262, 0.06216201, 0.01715392, 0, 0,
                         0, 0, 0, 0, 0.02643821, 0.06068194])
    actual = snow.loss_townsend(snow_total, snow_events, surface_tilt,
                                relative_humidity, temp_air,
                                poa_global, slant_height,
                                lower_edge_height, angle_of_repose)
    np.testing.assert_allclose(expected, actual, rtol=1e-05)
