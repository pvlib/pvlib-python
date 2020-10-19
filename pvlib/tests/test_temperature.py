import pandas as pd
import numpy as np

import pytest
from conftest import DATA_DIR, assert_series_equal
from numpy.testing import assert_allclose

from pvlib import temperature, location, irradiance, iam


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


def _read_pvwatts_8760(filename):
    df = pd.read_csv(filename,
                     skiprows=17,  # ignore location/simulation metadata
                     skipfooter=1,  # ignore "Totals" row
                     engine='python')
    df['Year'] = 2019
    df.index = pd.to_datetime(df[['Year', 'Month', 'Day', 'Hour']])
    return df


@pytest.mark.parametrize('filename,inoct', [
    ('pvwatts_8760_rackmount.csv', 45),
    ('pvwatts_8760_roofmount.csv', 49),
])
def test_fuentes(filename, inoct):
    # Test against data exported from pvwatts.nrel.gov
    data = _read_pvwatts_8760(DATA_DIR / filename)
    data = data.iloc[:24*7, :]  # just use one week
    inputs = {
        'poa_global': data['Plane of Array Irradiance (W/m^2)'],
        'temp_air': data['Ambient Temperature (C)'],
        'wind_speed': data['Wind Speed (m/s)'],
        'noct_installed': inoct,
    }
    expected_tcell = data['Cell Temperature (C)']
    expected_tcell.name = 'tmod'
    actual_tcell = temperature.fuentes(**inputs)
    # the SSC implementation of PVWatts diverges from the Fuentes model at
    # at night by setting Tcell=Tamb when POA=0. This not only means that
    # nighttime values are slightly different (Fuentes models cooling to sky
    # at night), but because of the thermal inertia, there is a transient
    # error after dawn as well. Test each case separately:
    is_night = inputs['poa_global'] == 0
    is_dawn = is_night.shift(1) & ~is_night
    is_daytime = (inputs['poa_global'] > 0) & ~is_dawn
    # the accuracy is probably higher than 3 digits here, but the PVWatts
    # export data has low precision so can only test up to 3 digits
    assert_series_equal(expected_tcell[is_daytime].round(3),
                        actual_tcell[is_daytime].round(3))
    # use lower precision for dawn times to accommodate the dawn transient
    error = actual_tcell[is_dawn] - expected_tcell[is_dawn]
    assert (error.abs() < 0.1).all()
    # sanity check on night values -- Fuentes not much lower than PVWatts
    night_difference = expected_tcell[is_night] - actual_tcell[is_night]
    assert night_difference.max() < 6
    assert night_difference.min() > 0


def test__calculate_radiative_heat():
    # TODO placeholder until final model is validated
    q = temperature._calculate_radiative_heat(
        module_area=2.47,
        view_factor=0.5,
        emissivity=0.5,
        temperature1=30 + 273.15,
        temperature2=10 + 273.15
    )
    assert round(q, 5) == 70.65021


def test_hayes():
    # TODO placeholder until final model is validated

    # simulate psm3 data
    data_psm3 = [
        {'Latitude': 39.66, 'Longitude': -105.207},
        pd.DataFrame(
            data={
                'DNI': [0, 163, 133, 189],
                'DHI': [0, 4, 12, 16],
                'GHI': [0, 7, 16, 25],
                'Temperature': [-13.2, -13.1, -13.1, -13],
                'Wind Speed': [1.6, 1.7, 1.7, 1.7]
            },
            index=pd.date_range('2019-01-01 07:25:00',
                                '2019-01-01 07:40:00',
                                freq='5min')
        )
    ]

    # data preparation
    module_tilt = 30
    module_azimuth = 180
    site = location.Location(
        latitude=data_psm3[0]['Latitude'],
        longitude=data_psm3[0]['Longitude'],
        tz='MST'
    )
    solar_position = site.get_solarposition(times=data_psm3[1].index)
    poa_global = irradiance.get_total_irradiance(
        surface_tilt=module_tilt,
        surface_azimuth=module_azimuth,
        dni=data_psm3[1]['DNI'],
        ghi=data_psm3[1]['GHI'],
        dhi=data_psm3[1]['DHI'],
        solar_zenith=solar_position['apparent_zenith'],
        solar_azimuth=solar_position['azimuth']
    )['poa_global']
    temp_air = data_psm3[1]['Temperature']
    wind_speed = data_psm3[1]['Wind Speed']

    # 1. Calculate module temp with new model
    aoi = irradiance.aoi(module_tilt, module_azimuth,
                         solar_position['zenith'],
                         solar_position['azimuth'])
    poa_effective = poa_global.multiply(iam.ashrae(aoi))
    module_efficiency = 0.176
    module_area = 2.47  # m^2
    module_weight = 34.5
    tmod_hayes = temperature.hayes(poa_effective, temp_air, wind_speed,
                                   module_efficiency, module_area,
                                   module_weight, module_tilt)

    assert [round(t, 2) for t in tmod_hayes.values] == \
           [-13.20, -14.81, -15.98, -16.85]
