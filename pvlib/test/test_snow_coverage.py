from __future__ import division

import datetime
from collections import OrderedDict
import warnings

import numpy as np
from numpy import array, nan
import pandas as pd

import pytest
from numpy.testing import assert_almost_equal, assert_allclose

from pandas.util.testing import assert_frame_equal, assert_series_equal

from pvlib import snowcoverage
from pvlib._deprecation import pvlibDeprecationWarning


from conftest import (fail_on_pvlib_version, needs_numpy_1_10, pandas_0_22,
                      requires_ephem, requires_numba)


def test_snow_slide_amount():
    tilt = 45
    sliding_coefficient = 2
    expected_specified = 2**.5
    expected_default = 1.97/2*2**.5
    actual_default = snowcoverage.snow_slide_amount(tilt)
    assert_almost_equal(expected_default, actual_default)
    actual_specified = snowcoverage.snow_slide_amount(tilt, sliding_coefficient=sliding_coefficient)
    assert_almost_equal(expected_specified, actual_specified)


def test_snow_slide_event():
    poa_irradiance = pd.Series([400, 200, 100, 1234, 134, 982])
    temperature = pd.Series([10, 2, -10, 1234, 34, -982])
    expected = pd.Series([True, True, False, True, True, False])
    actual = snowcoverage.snow_slide_event(poa_irradiance, temperature)
    assert_series_equal(expected, actual)


def test_snow_covered_panel():
    snowfall_data = pd.Series([1, 5, .6, 4, .23, -5, 19])
    expected_snowfall = pd.Series([1, 1, np.nan, 1, np.nan, np.nan, 1])
    actual_snowfall = snowcoverage.snow_covered_panel(snowfall_data)
    assert_series_equal(actual_snowfall, expected_snowfall)

    snowdepth_data = pd.Series([.5, 1, 2, 4, .23, 4, 19])
    expected_snowdepth = pd.Series([np.nan, np.nan, 1, 1, np.nan, 1, 1])
    actual_snowdepth = snowcoverage.snow_covered_panel(snowdepth_data, snow_data_type="snow_depth")
    assert_series_equal(actual_snowdepth, expected_snowdepth)


def test_snow_coverage_model():
    pass