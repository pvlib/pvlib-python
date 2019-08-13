import numpy as np
import pandas as pd

from numpy.testing import assert_almost_equal

from pandas.util.testing import assert_series_equal

from pvlib import snowcoverage


def test_snow_slide_amount():
    tilt = 45
    sc = 2
    expected_specified = 2**.5
    expected_default = 1.97/2*2**.5
    actual_default = snowcoverage.snow_slide_amount(tilt)
    assert_almost_equal(expected_default, actual_default)
    actual_specified = snowcoverage.snow_slide_amount(tilt,
                                                      sliding_coefficient=sc)
    assert_almost_equal(expected_specified, actual_specified)


def test_snow_slide_event():
    poa_irradiance = pd.Series([400, 200, 100, 1234, 134, 982])
    temperature = pd.Series([10, 2, -10, 1234, 34, -982])
    expected = pd.Series([True, True, False, True, True, False])
    actual = snowcoverage.snow_slide_event(poa_irradiance, temperature)
    assert_series_equal(expected, actual)


def test_fully_covered_panel():
    snowfall_data = pd.Series([1, 5, .6, 4, .23, -5, 19])
    expected_snowfall = pd.Series([True, True, False, True,
                                   False, False, True])
    actual_snowfall = snowcoverage.fully_covered_panel(snowfall_data)
    assert_series_equal(actual_snowfall, expected_snowfall)

    snowdepth_data = pd.Series([.5, 1, 2, 4, .23, 4, 19])
    expected_snowdepth = pd.Series([False, False, True, True,
                                    False, True, True])
    actual = snowcoverage.fully_covered_panel(snowdepth_data,
                                              snow_data_type="snow_depth")
    assert_series_equal(actual, expected_snowdepth)


def test_snow_coverage_model():
    tilt = 45
    poa_irradiance = pd.Series([400, 200, 100, 1234, 134, 982, 100, 100])
    temperature = pd.Series([10, 2, 10, 1234, 34, 982, 10, 10])
    slide = snowcoverage.snow_slide_amount(tilt)*.1

    snowfall_data = pd.Series([1, .5, .6, .4, .23, -5, .1, .1])
    snow_coverage = snowcoverage.snow_coverage_model(snowfall_data, "snowfall",
                                                     poa_irradiance,
                                                     temperature,
                                                     tilt)
    expected = pd.Series(1-slide*np.array([1, 2, 3, 4, 5, 6, 7, 1.0/slide]))
    assert_series_equal(snow_coverage, expected)

    snowdepth_data = pd.Series([.5, 1, 2, 4, .23, 4, 11, .1])
    snow_coverage = snowcoverage.snow_coverage_model(snowdepth_data,
                                                     "snow_depth",
                                                     poa_irradiance,
                                                     temperature,
                                                     tilt)
    expected = pd.Series(1-slide*np.array([1.0/slide, 1.0/slide, 1,
                                           1, 2, 1, 1, 2]))
    assert_series_equal(snow_coverage, expected)


def test_DC_loss_factor():
    num_strings = 8
    snow_coverage = pd.Series([1, 1, .5, .6, .2, .4, 0])
    expected = pd.Series([1, 1, .5, .625, .25, .5, 0])
    actual = snowcoverage.DC_loss_factor(snow_coverage, num_strings)
    assert_series_equal(expected, actual)
