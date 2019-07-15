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


def test_sapm_celltemp():
    default = celltemp.sapm(900, 20, 5)
    assert_allclose(default['temp_cell'], 43.509, 3)
    assert_allclose(default['temp_module'], 40.809, 3)
    assert_frame_equal(default, celltemp.sapm(900, 20, 5, [-3.47, -.0594, 3]))


def test_sapm_celltemp_dict_like():
    default = celltemp.sapm(900, 20, 5)
    assert_allclose(default['temp_cell'], 43.509, 3)
    assert_allclose(default['temp_module'], 40.809, 3)
    model = {'a': -3.47, 'b': -.0594, 'deltaT': 3}
    assert_frame_equal(default, celltemp.sapm(900, 20, 5, model))
    model = pd.Series(model)
    assert_frame_equal(default, celltemp.sapm(900, 20, 5, model))


def test_sapm_celltemp_with_index():
    times = pd.date_range(start='2015-01-01', end='2015-01-02', freq='12H')
    temps = pd.Series([0, 10, 5], index=times)
    irrads = pd.Series([0, 500, 0], index=times)
    winds = pd.Series([10, 5, 0], index=times)

    pvtemps = celltemp.sapm(irrads, temps, winds)

    expected = pd.DataFrame({'temp_cell': [0., 23.06066166, 5.],
                             'temp_module': [0., 21.56066166, 5.]},
                            index=times)

    assert_frame_equal(expected, pvtemps)


def test_PVSystem_sapm_celltemp(mocker):
    racking_model = 'roof_mount_cell_glassback'

    system = pvsystem.PVSystem(racking_model=racking_model)
    mocker.spy(celltemp, 'sapm')
    temps = 25
    irrads = 1000
    winds = 1
    out = system.sapm_celltemp(irrads, temps, winds)
    celltemp.sapm.assert_called_once_with(
        irrads, temps, winds, model=racking_model)
    assert isinstance(out, pd.DataFrame)
    assert out.shape == (1, 2)


def test_pvsyst_celltemp_default():
    default = celltemp.pvsyst(900, 20, 5)
    assert_allclose(default['temp_cell'], 45.137, 0.001)


def test_pvsyst_celltemp_non_model():
    tup_non_model = pvsystem.pvsyst_celltemp(900, 20, 5, 0.1,
                                             model=(23.5, 6.25))
    assert_allclose(tup_non_model['temp_cell'], 33.315, 0.001)

    list_non_model = pvsystem.pvsyst_celltemp(900, 20, 5, 0.1,
                                              model=[26.5, 7.68])
    assert_allclose(list_non_model['temp_cell'], 31.233, 0.001)


def test_pvsyst_celltemp_model_wrong_type():
    with pytest.raises(TypeError):
        pvsystem.pvsyst_celltemp(
            900, 20, 5, 0.1,
            model={"won't": 23.5, "work": 7.68})


def test_pvsyst_celltemp_model_non_option():
    with pytest.raises(KeyError):
        pvsystem.pvsyst_celltemp(
            900, 20, 5, 0.1,
            model="not_an_option")


def test_pvsyst_celltemp_with_index():
    times = pd.date_range(start="2015-01-01", end="2015-01-02", freq="12H")
    temps = pd.Series([0, 10, 5], index=times)
    irrads = pd.Series([0, 500, 0], index=times)
    winds = pd.Series([10, 5, 0], index=times)

    pvtemps = pvsystem.pvsyst_celltemp(irrads, temps, wind_speed=winds)
    expected = pd.DataFrame([0.0, 23.96551, 5.0], index=times,
                            columns=['temp_cell'])
    assert_frame_equal(expected, pvtemps)


def test_PVSystem_pvsyst_celltemp(mocker):
    racking_model = 'insulated'
    alpha_absorption = 0.85
    eta_m = 0.17
    module_parameters = {}
    module_parameters['alpha_absorption'] = alpha_absorption
    module_parameters['eta_m'] = eta_m
    system = pvsystem.PVSystem(racking_model=racking_model,
                               module_parameters=module_parameters)
    mocker.spy(celltemp, 'pvsyst')
    irrad = 800
    temp = 45
    wind = 0.5
    out = system.pvsyst_celltemp(irrad, temp, wind_speed=wind)
    celltemp.pvsyst.assert_called_once_with(
        irrad, temp, wind, eta_m, alpha_absorption, racking_model)
    assert isinstance(out, pd.DataFrame)
    assert all(out['temp_cell'] < 90) and all(out['temp_cell'] > 70)


@fail_on_pvlib_version('0.7')
def test_deprecated_07():
    with pytest.warns(pvlibDeprecationWarning):
        pvsystem.sapm_celltemp(1000, 25, 1)
    with pytest.warns(pvlibDeprecationWarning):
        pvsystem.pvsyst_celltemp(1000, 25)
