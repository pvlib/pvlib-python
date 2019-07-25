# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 10:07:25 2019

@author: cwhanse
"""
import pandas as pd

import pytest
from pandas.util.testing import assert_frame_equal
from numpy.testing import assert_allclose

from pvlib import celltemp, pvsystem
from pvlib._deprecation import pvlibDeprecationWarning

from conftest import fail_on_pvlib_version


@pytest.fixture
def celltemp_sapm_default():
    return celltemp.TEMP_MODEL_PARAMS['sapm']['open_rack_cell_glassback']


def test_sapm_celltemp(celltemp_sapm_default):
    a, b, deltaT = celltemp_sapm_default
    default = celltemp.sapm(900, 20, 5, a, b, deltaT)
    assert_allclose(default['temp_cell'], 43.509, 3)
    assert_allclose(default['temp_module'], 40.809, 3)


def test_sapm_celltemp_with_index(celltemp_sapm_default):
    a, b, deltaT = celltemp_sapm_default
    times = pd.date_range(start='2015-01-01', end='2015-01-02', freq='12H')
    temps = pd.Series([0, 10, 5], index=times)
    irrads = pd.Series([0, 500, 0], index=times)
    winds = pd.Series([10, 5, 0], index=times)
    pvtemps = celltemp.sapm(irrads, temps, winds, a, b, deltaT)
    expected = pd.DataFrame({'temp_cell': [0., 23.06066166, 5.],
                             'temp_module': [0., 21.56066166, 5.]},
                            index=times)
    assert_frame_equal(expected, pvtemps)


def test_pvsyst_celltemp_default():
    default = celltemp.pvsyst(900, 20, 5)
    assert_allclose(default['temp_cell'], 45.137, 0.001)


def test_pvsyst_celltemp_non_model():
    tup_non_model = pvsystem.pvsyst_celltemp(900, 20, wind_speed=5.0,
                                             constant_loss_factor=23.5,
                                             wind_loss_factor=6.25, eta_m=0.1)
    assert_allclose(tup_non_model['temp_cell'], 33.315, 0.001)

    list_non_model = pvsystem.pvsyst_celltemp(900, 20, wind_speed=5.0,
                                             constant_loss_factor=26.5,
                                             wind_loss_factor=7.68, eta_m=0.1)
    assert_allclose(list_non_model['temp_cell'], 31.233, 0.001)


def test_pvsyst_celltemp_with_index():
    times = pd.date_range(start="2015-01-01", end="2015-01-02", freq="12H")
    temps = pd.Series([0, 10, 5], index=times)
    irrads = pd.Series([0, 500, 0], index=times)
    winds = pd.Series([10, 5, 0], index=times)

    pvtemps = pvsystem.pvsyst_celltemp(irrads, temps, wind_speed=winds)
    expected = pd.DataFrame([0.0, 23.96551, 5.0], index=times,
                            columns=['temp_cell'])
    assert_frame_equal(expected, pvtemps)


@fail_on_pvlib_version('0.8')
def test_deprecated_07():
    with pytest.warns(pvlibDeprecationWarning):
        pvsystem.sapm_celltemp(1000, 25, 1, -3.47, -0.0594, 3)
    with pytest.warns(pvlibDeprecationWarning):
        pvsystem.pvsyst_celltemp(1000, 25)
