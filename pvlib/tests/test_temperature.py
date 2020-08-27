import pandas as pd
import numpy as np

import pytest
from conftest import assert_series_equal
from numpy.testing import assert_allclose

from pvlib import temperature


@pytest.fixture
def sapm_default():
    return temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][
        'open_rack_glass_glass']


def test_sapm_cell(sapm_default):
    default = temperature.sapm_cell(900, 20, 5, sapm_default['a'],
                                    sapm_default['b'], sapm_default['deltaT'])
    assert_allclose(default, 43.509, 3)


def test_sapm_module(sapm_default):
    default = temperature.sapm_module(900, 20, 5, sapm_default['a'],
                                      sapm_default['b'])
    assert_allclose(default, 40.809, 3)


def test_sapm_cell_from_module(sapm_default):
    default = temperature.sapm_cell_from_module(50, 900,
                                                sapm_default['deltaT'])
    assert_allclose(default, 50 + 900 / 1000 * sapm_default['deltaT'])


def test_sapm_ndarray(sapm_default):
    temps = np.array([0, 10, 5])
    irrads = np.array([0, 500, 0])
    winds = np.array([10, 5, 0])
    cell_temps = temperature.sapm_cell(irrads, temps, winds, sapm_default['a'],
                                       sapm_default['b'],
                                       sapm_default['deltaT'])
    module_temps = temperature.sapm_module(irrads, temps, winds,
                                           sapm_default['a'],
                                           sapm_default['b'])
    expected_cell = np.array([0., 23.06066166, 5.])
    expected_module = np.array([0., 21.56066166, 5.])
    assert_allclose(expected_cell, cell_temps, 3)
    assert_allclose(expected_module, module_temps, 3)


def test_sapm_series(sapm_default):
    times = pd.date_range(start='2015-01-01', end='2015-01-02', freq='12H')
    temps = pd.Series([0, 10, 5], index=times)
    irrads = pd.Series([0, 500, 0], index=times)
    winds = pd.Series([10, 5, 0], index=times)
    cell_temps = temperature.sapm_cell(irrads, temps, winds, sapm_default['a'],
                                       sapm_default['b'],
                                       sapm_default['deltaT'])
    module_temps = temperature.sapm_module(irrads, temps, winds,
                                           sapm_default['a'],
                                           sapm_default['b'])
    expected_cell = pd.Series([0., 23.06066166, 5.], index=times)
    expected_module = pd.Series([0., 21.56066166, 5.], index=times)
    assert_series_equal(expected_cell, cell_temps)
    assert_series_equal(expected_module, module_temps)


def test_pvsyst_cell_default():
    result = temperature.pvsyst_cell(900, 20, 5)
    assert_allclose(result, 45.137, 0.001)


def test_pvsyst_cell_kwargs():
    result = temperature.pvsyst_cell(900, 20, wind_speed=5.0, u_c=23.5,
                                     u_v=6.25, eta_m=0.1)
    assert_allclose(result, 33.315, 0.001)


def test_pvsyst_cell_ndarray():
    temps = np.array([0, 10, 5])
    irrads = np.array([0, 500, 0])
    winds = np.array([10, 5, 0])
    result = temperature.pvsyst_cell(irrads, temps, wind_speed=winds)
    expected = np.array([0.0, 23.96551, 5.0])
    assert_allclose(expected, result, 3)


def test_pvsyst_cell_series():
    times = pd.date_range(start="2015-01-01", end="2015-01-02", freq="12H")
    temps = pd.Series([0, 10, 5], index=times)
    irrads = pd.Series([0, 500, 0], index=times)
    winds = pd.Series([10, 5, 0], index=times)

    result = temperature.pvsyst_cell(irrads, temps, wind_speed=winds)
    expected = pd.Series([0.0, 23.96551, 5.0], index=times)
    assert_series_equal(expected, result)


def test_faiman_default():
    result = temperature.faiman(900, 20, 5)
    assert_allclose(result, 35.203, 0.001)


def test_faiman_kwargs():
    result = temperature.faiman(900, 20, wind_speed=5.0, u0=22.0, u1=6.)
    assert_allclose(result, 37.308, 0.001)


def test_faiman_list():
    temps = [0, 10, 5]
    irrads = [0, 500, 0]
    winds = [10, 5, 0]
    result = temperature.faiman(irrads, temps, wind_speed=winds)
    expected = np.array([0.0, 18.446, 5.0])
    assert_allclose(expected, result, 3)


def test_faiman_ndarray():
    temps = np.array([0, 10, 5])
    irrads = np.array([0, 500, 0])
    winds = np.array([10, 5, 0])
    result = temperature.faiman(irrads, temps, wind_speed=winds)
    expected = np.array([0.0, 18.446, 5.0])
    assert_allclose(expected, result, 3)


def test_faiman_series():
    times = pd.date_range(start="2015-01-01", end="2015-01-02", freq="12H")
    temps = pd.Series([0, 10, 5], index=times)
    irrads = pd.Series([0, 500, 0], index=times)
    winds = pd.Series([10, 5, 0], index=times)

    result = temperature.faiman(irrads, temps, wind_speed=winds)
    expected = pd.Series([0.0, 18.446, 5.0], index=times)
    assert_series_equal(expected, result)


def test__temperature_model_params():
    params = temperature._temperature_model_params('sapm',
                                                   'open_rack_glass_glass')
    assert params == temperature.TEMPERATURE_MODEL_PARAMETERS['sapm'][
        'open_rack_glass_glass']
    with pytest.raises(KeyError):
        temperature._temperature_model_params('sapm', 'not_a_parameter_set')
