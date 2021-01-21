import sys

import numpy as np
import pandas as pd

from pvlib import iam, modelchain, pvsystem, temperature
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.tracking import SingleAxisTracker
from pvlib.location import Location
from pvlib._deprecation import pvlibDeprecationWarning

from conftest import assert_series_equal, assert_frame_equal
import pytest

from conftest import fail_on_pvlib_version, requires_tables


@pytest.fixture(scope='function')
def sapm_dc_snl_ac_system(sapm_module_params, cec_inverter_parameters,
                          sapm_temperature_cs5p_220m):
    module = 'Canadian_Solar_CS5P_220M___2009_'
    module_parameters = sapm_module_params.copy()
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module=module,
                      module_parameters=module_parameters,
                      temperature_model_parameters=temp_model_params,
                      inverter_parameters=cec_inverter_parameters)
    return system


@pytest.fixture
def cec_dc_snl_ac_system(cec_module_cs5p_220m, cec_inverter_parameters,
                         sapm_temperature_cs5p_220m):
    module_parameters = cec_module_cs5p_220m.copy()
    module_parameters['b'] = 0.05
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module=module_parameters['Name'],
                      module_parameters=module_parameters,
                      temperature_model_parameters=temp_model_params,
                      inverter_parameters=cec_inverter_parameters)
    return system


@pytest.fixture
def cec_dc_snl_ac_arrays(cec_module_cs5p_220m, cec_inverter_parameters,
                         sapm_temperature_cs5p_220m):
    module_parameters = cec_module_cs5p_220m.copy()
    module_parameters['b'] = 0.05
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    array_one = pvsystem.Array(
        surface_tilt=32.2, surface_azimuth=180,
        module=module_parameters['Name'],
        module_parameters=module_parameters.copy(),
        temperature_model_parameters=temp_model_params.copy()
    )
    array_two = pvsystem.Array(
        surface_tilt=42.2, surface_azimuth=220,
        module=module_parameters['Name'],
        module_parameters=module_parameters.copy(),
        temperature_model_parameters=temp_model_params.copy()
    )
    system = PVSystem(
        arrays=[array_one, array_two],
        inverter_parameters=cec_inverter_parameters
    )
    return system


@pytest.fixture
def cec_dc_native_snl_ac_system(cec_module_cs5p_220m, cec_inverter_parameters,
                                sapm_temperature_cs5p_220m):
    module_parameters = cec_module_cs5p_220m.copy()
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module=module_parameters['Name'],
                      module_parameters=module_parameters,
                      temperature_model_parameters=temp_model_params,
                      inverter_parameters=cec_inverter_parameters)
    return system


@pytest.fixture
def pvsyst_dc_snl_ac_system(pvsyst_module_params, cec_inverter_parameters,
                            sapm_temperature_cs5p_220m):
    module = 'PVsyst test module'
    module_parameters = pvsyst_module_params
    module_parameters['b'] = 0.05
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module=module,
                      module_parameters=module_parameters,
                      temperature_model_parameters=temp_model_params,
                      inverter_parameters=cec_inverter_parameters)
    return system


@pytest.fixture
def pvsyst_dc_snl_ac_arrays(pvsyst_module_params, cec_inverter_parameters,
                            sapm_temperature_cs5p_220m):
    module = 'PVsyst test module'
    module_parameters = pvsyst_module_params
    module_parameters['b'] = 0.05
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    array_one = pvsystem.Array(
        surface_tilt=32.2, surface_azimuth=180,
        module=module,
        module_parameters=module_parameters.copy(),
        temperature_model_parameters=temp_model_params.copy()
    )
    array_two = pvsystem.Array(
        surface_tilt=42.2, surface_azimuth=220,
        module=module,
        module_parameters=module_parameters.copy(),
        temperature_model_parameters=temp_model_params.copy()
    )
    system = PVSystem(
        arrays=[array_one, array_two],
        inverter_parameters=cec_inverter_parameters
    )
    return system


@pytest.fixture
def cec_dc_adr_ac_system(sam_data, cec_module_cs5p_220m,
                         sapm_temperature_cs5p_220m):
    module_parameters = cec_module_cs5p_220m.copy()
    module_parameters['b'] = 0.05
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    inverters = sam_data['adrinverter']
    inverter = inverters['Zigor__Sunzet_3_TL_US_240V__CEC_2011_'].copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module=module_parameters['Name'],
                      module_parameters=module_parameters,
                      temperature_model_parameters=temp_model_params,
                      inverter_parameters=inverter)
    return system


@pytest.fixture
def pvwatts_dc_snl_ac_system(cec_inverter_parameters):
    module_parameters = {'pdc0': 220, 'gamma_pdc': -0.003}
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module_parameters=module_parameters,
                      inverter_parameters=cec_inverter_parameters)
    return system


@pytest.fixture(scope="function")
def pvwatts_dc_pvwatts_ac_system(sapm_temperature_cs5p_220m):
    module_parameters = {'pdc0': 220, 'gamma_pdc': -0.003}
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    inverter_parameters = {'pdc0': 220, 'eta_inv_nom': 0.95}
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module_parameters=module_parameters,
                      temperature_model_parameters=temp_model_params,
                      inverter_parameters=inverter_parameters)
    return system


@pytest.fixture(scope="function")
def pvwatts_dc_pvwatts_ac_system_arrays(sapm_temperature_cs5p_220m):
    module_parameters = {'pdc0': 220, 'gamma_pdc': -0.003}
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    inverter_parameters = {'pdc0': 220, 'eta_inv_nom': 0.95}
    array_one = pvsystem.Array(
        surface_tilt=32.2, surface_azimuth=180,
        module_parameters=module_parameters.copy(),
        temperature_model_parameters=temp_model_params.copy()
    )
    array_two = pvsystem.Array(
        surface_tilt=42.2, surface_azimuth=220,
        module_parameters=module_parameters.copy(),
        temperature_model_parameters=temp_model_params.copy()
    )
    system = PVSystem(
        arrays=[array_one, array_two], inverter_parameters=inverter_parameters)
    return system


@pytest.fixture(scope="function")
def pvwatts_dc_pvwatts_ac_faiman_temp_system():
    module_parameters = {'pdc0': 220, 'gamma_pdc': -0.003}
    temp_model_params = {'u0': 25.0, 'u1': 6.84}
    inverter_parameters = {'pdc0': 220, 'eta_inv_nom': 0.95}
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module_parameters=module_parameters,
                      temperature_model_parameters=temp_model_params,
                      inverter_parameters=inverter_parameters)
    return system


@pytest.fixture(scope="function")
def pvwatts_dc_pvwatts_ac_pvsyst_temp_system():
    module_parameters = {'pdc0': 220, 'gamma_pdc': -0.003}
    temp_model_params = {'u_c': 29.0, 'u_v': 0.0, 'eta_m': 0.1,
                         'alpha_absorption': 0.9}
    inverter_parameters = {'pdc0': 220, 'eta_inv_nom': 0.95}
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module_parameters=module_parameters,
                      temperature_model_parameters=temp_model_params,
                      inverter_parameters=inverter_parameters)
    return system


@pytest.fixture(scope="function")
def pvwatts_dc_pvwatts_ac_fuentes_temp_system():
    module_parameters = {'pdc0': 220, 'gamma_pdc': -0.003}
    temp_model_params = {'noct_installed': 45}
    inverter_parameters = {'pdc0': 220, 'eta_inv_nom': 0.95}
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module_parameters=module_parameters,
                      temperature_model_parameters=temp_model_params,
                      inverter_parameters=inverter_parameters)
    return system


@pytest.fixture(scope="function")
def system_no_aoi(cec_module_cs5p_220m, sapm_temperature_cs5p_220m,
                  cec_inverter_parameters):
    module_parameters = cec_module_cs5p_220m.copy()
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    inverter_parameters = cec_inverter_parameters.copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module_parameters=module_parameters,
                      temperature_model_parameters=temp_model_params,
                      inverter_parameters=inverter_parameters)
    return system


@pytest.fixture
def system_no_temp(cec_module_cs5p_220m, cec_inverter_parameters):
    module_parameters = cec_module_cs5p_220m.copy()
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    inverter_parameters = cec_inverter_parameters.copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module_parameters=module_parameters,
                      inverter_parameters=inverter_parameters)
    return system


@pytest.fixture
def location():
    return Location(32.2, -111, altitude=700)


@pytest.fixture
def weather():
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    weather = pd.DataFrame({'ghi': [500, 0], 'dni': [800, 0], 'dhi': [100, 0]},
                           index=times)
    return weather


@pytest.fixture
def total_irrad(weather):
    return pd.DataFrame({'poa_global': [800., 500.],
                         'poa_direct': [500., 300.],
                         'poa_diffuse': [300., 200.]}, index=weather.index)


@pytest.fixture(scope='function')
def sapm_dc_snl_ac_system_Array(sapm_module_params, cec_inverter_parameters,
                                sapm_temperature_cs5p_220m):
    module = 'Canadian_Solar_CS5P_220M___2009_'
    module_parameters = sapm_module_params.copy()
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    array_one = pvsystem.Array(surface_tilt=32, surface_azimuth=180,
                               albedo=0.2, module=module,
                               module_parameters=module_parameters,
                               temperature_model_parameters=temp_model_params,
                               modules_per_string=1,
                               strings=1)
    array_two = pvsystem.Array(surface_tilt=15, surface_azimuth=180,
                               albedo=0.2, module=module,
                               module_parameters=module_parameters,
                               temperature_model_parameters=temp_model_params,
                               modules_per_string=1,
                               strings=1)
    return PVSystem(arrays=[array_one, array_two],
                    inverter_parameters=cec_inverter_parameters)


@pytest.fixture(scope='function')
def sapm_dc_snl_ac_system_same_arrays(sapm_module_params,
                                      cec_inverter_parameters,
                                      sapm_temperature_cs5p_220m):
    """A system with two identical arrays."""
    module = 'Canadian_Solar_CS5P_220M___2009_'
    module_parameters = sapm_module_params.copy()
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    array_one = pvsystem.Array(surface_tilt=32.2, surface_azimuth=180,
                               module=module,
                               module_parameters=module_parameters,
                               temperature_model_parameters=temp_model_params,
                               modules_per_string=1,
                               strings=1)
    array_two = pvsystem.Array(surface_tilt=32.2, surface_azimuth=180,
                               module=module,
                               module_parameters=module_parameters,
                               temperature_model_parameters=temp_model_params,
                               modules_per_string=1,
                               strings=1)
    return PVSystem(arrays=[array_one, array_two],
                    inverter_parameters=cec_inverter_parameters)


def test_ModelChain_creation(sapm_dc_snl_ac_system, location):
    ModelChain(sapm_dc_snl_ac_system, location)


def test_with_sapm(sapm_dc_snl_ac_system, location, weather):
    mc = ModelChain.with_sapm(sapm_dc_snl_ac_system, location)
    assert mc.dc_model == mc.sapm
    mc.run_model(weather)


def test_with_pvwatts(pvwatts_dc_pvwatts_ac_system, location, weather):
    mc = ModelChain.with_pvwatts(pvwatts_dc_pvwatts_ac_system, location)
    assert mc.dc_model == mc.pvwatts_dc
    assert mc.temperature_model == mc.sapm_temp
    mc.run_model(weather)


@pytest.mark.parametrize('strategy, expected', [
    (None, (32.2, 180)), ('None', (32.2, 180)), ('flat', (0, 180)),
    ('south_at_latitude_tilt', (32.2, 180))
])
def test_orientation_strategy(strategy, expected, sapm_dc_snl_ac_system,
                              location):
    mc = ModelChain(sapm_dc_snl_ac_system, location,
                    orientation_strategy=strategy)

    # the || accounts for the coercion of 'None' to None
    assert (mc.orientation_strategy == strategy or
            mc.orientation_strategy is None)
    assert sapm_dc_snl_ac_system.surface_tilt == expected[0]
    assert sapm_dc_snl_ac_system.surface_azimuth == expected[1]


def test_run_model_with_irradiance(sapm_dc_snl_ac_system, location):
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni': 900, 'ghi': 600, 'dhi': 150},
                              index=times)
    ac = mc.run_model(irradiance).results.ac

    expected = pd.Series(np.array([187.80746494643176, -0.02]),
                         index=times)
    assert_series_equal(ac, expected)


@pytest.fixture(scope='function')
def multi_array_sapm_dc_snl_ac_system(
        sapm_temperature_cs5p_220m, sapm_module_params,
        cec_inverter_parameters):
    module_parameters = sapm_module_params
    temp_model_parameters = sapm_temperature_cs5p_220m.copy()
    inverter_parameters = cec_inverter_parameters
    array_one = pvsystem.Array(
        surface_tilt=32.2, surface_azimuth=180,
        module_parameters=module_parameters,
        temperature_model_parameters=temp_model_parameters
    )
    array_two = pvsystem.Array(
        surface_tilt=32.2, surface_azimuth=220,
        module_parameters=module_parameters,
        temperature_model_parameters=temp_model_parameters
    )
    two_array_system = PVSystem(
        arrays=[array_one, array_two],
        inverter_parameters=inverter_parameters
    )
    array_one_system = PVSystem(
        arrays=[array_one],
        inverter_parameters=inverter_parameters
    )
    array_two_system = PVSystem(
        arrays=[array_two],
        inverter_parameters=inverter_parameters
    )
    return {'two_array_system': two_array_system,
            'array_one_system': array_one_system,
            'array_two_system': array_two_system}


def test_run_model_from_irradiance_arrays_no_loss(
        multi_array_sapm_dc_snl_ac_system, location):
    mc_both = ModelChain(
        multi_array_sapm_dc_snl_ac_system['two_array_system'],
        location,
        aoi_model='no_loss',
        spectral_model='no_loss',
        losses_model='no_loss'
    )
    mc_one = ModelChain(
        multi_array_sapm_dc_snl_ac_system['array_one_system'],
        location,
        aoi_model='no_loss',
        spectral_model='no_loss',
        losses_model='no_loss'
    )
    mc_two = ModelChain(
        multi_array_sapm_dc_snl_ac_system['array_two_system'],
        location,
        aoi_model='no_loss',
        spectral_model='no_loss',
        losses_model='no_loss'
    )
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni': 900, 'ghi': 600, 'dhi': 150},
                              index=times)
    mc_one.run_model(irradiance)
    mc_two.run_model(irradiance)
    mc_both.run_model(irradiance)
    assert_frame_equal(
        mc_both.results.dc[0],
        mc_one.results.dc
    )
    assert_frame_equal(
        mc_both.results.dc[1],
        mc_two.results.dc
    )


@pytest.mark.parametrize("input_type", [tuple, list])
def test_run_model_from_irradiance_arrays_no_loss_input_type(
        multi_array_sapm_dc_snl_ac_system, location, input_type):
    mc_both = ModelChain(
        multi_array_sapm_dc_snl_ac_system['two_array_system'],
        location,
        aoi_model='no_loss',
        spectral_model='no_loss',
        losses_model='no_loss'
    )
    mc_one = ModelChain(
        multi_array_sapm_dc_snl_ac_system['array_one_system'],
        location,
        aoi_model='no_loss',
        spectral_model='no_loss',
        losses_model='no_loss'
    )
    mc_two = ModelChain(
        multi_array_sapm_dc_snl_ac_system['array_two_system'],
        location,
        aoi_model='no_loss',
        spectral_model='no_loss',
        losses_model='no_loss'
    )
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni': 900, 'ghi': 600, 'dhi': 150},
                              index=times)
    mc_one.run_model(irradiance)
    mc_two.run_model(irradiance)
    mc_both.run_model(input_type((irradiance, irradiance)))
    assert_frame_equal(
        mc_both.results.dc[0], mc_one.results.dc
    )
    assert_frame_equal(
        mc_both.results.dc[1], mc_two.results.dc
    )


@pytest.mark.parametrize('inverter', ['adr'])
def test_ModelChain_invalid_inverter_params_arrays(
        inverter, sapm_dc_snl_ac_system_same_arrays,
        location, adr_inverter_parameters):
    inverter_params = {'adr': adr_inverter_parameters}
    sapm_dc_snl_ac_system_same_arrays.inverter_parameters = \
        inverter_params[inverter]
    with pytest.raises(ValueError,
                       match=r'Only sandia and pvwatts inverter models'):
        ModelChain(sapm_dc_snl_ac_system_same_arrays, location)


@pytest.mark.parametrize("input_type", [tuple, list])
def test_prepare_inputs_multi_weather(
        sapm_dc_snl_ac_system_Array, location, input_type):
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    weather = pd.DataFrame({'ghi': 1, 'dhi': 1, 'dni': 1},
                           index=times)
    mc.prepare_inputs(input_type((weather, weather)))
    num_arrays = sapm_dc_snl_ac_system_Array.num_arrays
    assert len(mc.results.total_irrad) == num_arrays


def test_prepare_inputs_no_irradiance(sapm_dc_snl_ac_system, location):
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    weather = pd.DataFrame()
    with pytest.raises(ValueError):
        mc.prepare_inputs(weather)


def test_prepare_inputs_arrays_one_missing_irradiance(
        sapm_dc_snl_ac_system_Array, location):
    """If any of the input DataFrames is missing a column then a
    ValueError is raised."""
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    weather = pd.DataFrame(
        {'ghi': [1], 'dhi': [1], 'dni': [1]}
    )
    weather_incomplete = pd.DataFrame(
        {'ghi': [1], 'dhi': [1]}
    )
    with pytest.raises(ValueError,
                       match=r"Incomplete input data\. .*"):
        mc.prepare_inputs((weather, weather_incomplete))
    with pytest.raises(ValueError,
                       match=r"Incomplete input data\. .*"):
        mc.prepare_inputs((weather_incomplete, weather))


@pytest.mark.parametrize("input_type", [tuple, list])
def test_prepare_inputs_weather_wrong_length(
        sapm_dc_snl_ac_system_Array, location, input_type):
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    weather = pd.DataFrame({'ghi': [1], 'dhi': [1], 'dni': [1]})
    with pytest.raises(ValueError,
                       match="Input must be same length as number of Arrays "
                             r"in system\. Expected 2, got 1\."):
        mc.prepare_inputs(input_type((weather,)))
    with pytest.raises(ValueError,
                       match="Input must be same length as number of Arrays "
                             r"in system\. Expected 2, got 3\."):
        mc.prepare_inputs(input_type((weather, weather, weather)))


def test_ModelChain_times_error_arrays(sapm_dc_snl_ac_system_Array, location):
    """ModelChain.times is assigned a single index given multiple weather
    DataFrames.
    """
    error_str = r"Input DataFrames must have same index\."
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    irradiance = {'ghi': [1, 2], 'dhi': [1, 2], 'dni': [1, 2]}
    times_one = pd.date_range(start='1/1/2020', freq='6H', periods=2)
    times_two = pd.date_range(start='1/1/2020 00:15', freq='6H', periods=2)
    weather_one = pd.DataFrame(irradiance, index=times_one)
    weather_two = pd.DataFrame(irradiance, index=times_two)
    with pytest.raises(ValueError, match=error_str):
        mc.prepare_inputs((weather_one, weather_two))
    # test with overlapping, but differently sized indices.
    times_three = pd.date_range(start='1/1/2020', freq='6H', periods=3)
    irradiance_three = irradiance
    irradiance_three['ghi'].append(3)
    irradiance_three['dhi'].append(3)
    irradiance_three['dni'].append(3)
    weather_three = pd.DataFrame(irradiance_three, index=times_three)
    with pytest.raises(ValueError, match=error_str):
        mc.prepare_inputs((weather_one, weather_three))


def test_ModelChain_times_arrays(sapm_dc_snl_ac_system_Array, location):
    """ModelChain.times is assigned a single index given multiple weather
    DataFrames.
    """
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    irradiance_one = {'ghi': [1, 2], 'dhi': [1, 2], 'dni': [1, 2]}
    irradiance_two = {'ghi': [2, 1], 'dhi': [2, 1], 'dni': [2, 1]}
    times = pd.date_range(start='1/1/2020', freq='6H', periods=2)
    weather_one = pd.DataFrame(irradiance_one, index=times)
    weather_two = pd.DataFrame(irradiance_two, index=times)
    mc.prepare_inputs((weather_one, weather_two))
    assert mc.times.equals(times)
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    mc.prepare_inputs(weather_one)
    assert mc.times.equals(times)


@pytest.mark.parametrize("missing", ['dhi', 'ghi', 'dni'])
def test_prepare_inputs_missing_irrad_component(
        sapm_dc_snl_ac_system, location, missing):
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    weather = pd.DataFrame({'dhi': [1, 2], 'dni': [1, 2], 'ghi': [1, 2]})
    weather.drop(columns=missing, inplace=True)
    with pytest.raises(ValueError):
        mc.prepare_inputs(weather)


@pytest.mark.parametrize('ac_model', ['sandia', 'pvwatts'])
@pytest.mark.parametrize("input_type", [tuple, list])
def test_run_model_arrays_weather(sapm_dc_snl_ac_system_same_arrays,
                                  pvwatts_dc_pvwatts_ac_system_arrays,
                                  location, ac_model, input_type):
    system = {'sandia': sapm_dc_snl_ac_system_same_arrays,
              'pvwatts': pvwatts_dc_pvwatts_ac_system_arrays}
    mc = ModelChain(system[ac_model], location, aoi_model='no_loss',
                    spectral_model='no_loss')
    times = pd.date_range('20200101 1200-0700', periods=2, freq='2H')
    weather_one = pd.DataFrame({'dni': [900, 800],
                                'ghi': [600, 500],
                                'dhi': [150, 100]},
                               index=times)
    weather_two = pd.DataFrame({'dni': [500, 400],
                                'ghi': [300, 200],
                                'dhi': [75, 65]},
                               index=times)
    mc.run_model(input_type((weather_one, weather_two)))
    assert (mc.results.dc[0] != mc.results.dc[1]).all().all()
    assert not mc.results.ac.empty


def test_run_model_perez(sapm_dc_snl_ac_system, location):
    mc = ModelChain(sapm_dc_snl_ac_system, location,
                    transposition_model='perez')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni': 900, 'ghi': 600, 'dhi': 150},
                              index=times)
    ac = mc.run_model(irradiance).results.ac

    expected = pd.Series(np.array([187.94295642, -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_run_model_gueymard_perez(sapm_dc_snl_ac_system, location):
    mc = ModelChain(sapm_dc_snl_ac_system, location,
                    airmass_model='gueymard1993',
                    transposition_model='perez')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni': 900, 'ghi': 600, 'dhi': 150},
                              index=times)
    ac = mc.run_model(irradiance).results.ac

    expected = pd.Series(np.array([187.94317405, -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_run_model_with_weather_sapm_temp(sapm_dc_snl_ac_system, location,
                                          weather, mocker):
    # test with sapm cell temperature model
    weather['wind_speed'] = 5
    weather['temp_air'] = 10
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    mc.temperature_model = 'sapm'
    m_sapm = mocker.spy(sapm_dc_snl_ac_system, 'sapm_celltemp')
    mc.run_model(weather)
    assert m_sapm.call_count == 1
    # assert_called_once_with cannot be used with series, so need to use
    # assert_series_equal on call_args
    assert_series_equal(m_sapm.call_args[0][1], weather['temp_air'])  # temp
    assert_series_equal(m_sapm.call_args[0][2], weather['wind_speed'])  # wind
    assert not mc.results.ac.empty


def test_run_model_with_weather_pvsyst_temp(sapm_dc_snl_ac_system, location,
                                            weather, mocker):
    # test with pvsyst cell temperature model
    weather['wind_speed'] = 5
    weather['temp_air'] = 10
    sapm_dc_snl_ac_system.racking_model = 'freestanding'
    sapm_dc_snl_ac_system.temperature_model_parameters = \
        temperature._temperature_model_params('pvsyst', 'freestanding')
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    mc.temperature_model = 'pvsyst'
    m_pvsyst = mocker.spy(sapm_dc_snl_ac_system, 'pvsyst_celltemp')
    mc.run_model(weather)
    assert m_pvsyst.call_count == 1
    assert_series_equal(m_pvsyst.call_args[0][1], weather['temp_air'])
    assert_series_equal(m_pvsyst.call_args[0][2], weather['wind_speed'])
    assert not mc.results.ac.empty


def test_run_model_with_weather_faiman_temp(sapm_dc_snl_ac_system, location,
                                            weather, mocker):
    # test with faiman cell temperature model
    weather['wind_speed'] = 5
    weather['temp_air'] = 10
    sapm_dc_snl_ac_system.temperature_model_parameters = {
        'u0': 25.0, 'u1': 6.84
    }
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    mc.temperature_model = 'faiman'
    m_faiman = mocker.spy(sapm_dc_snl_ac_system, 'faiman_celltemp')
    mc.run_model(weather)
    assert m_faiman.call_count == 1
    assert_series_equal(m_faiman.call_args[0][1], weather['temp_air'])
    assert_series_equal(m_faiman.call_args[0][2], weather['wind_speed'])
    assert not mc.results.ac.empty


def test_run_model_with_weather_fuentes_temp(sapm_dc_snl_ac_system, location,
                                             weather, mocker):
    weather['wind_speed'] = 5
    weather['temp_air'] = 10
    sapm_dc_snl_ac_system.temperature_model_parameters = {
        'noct_installed': 45
    }
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    mc.temperature_model = 'fuentes'
    m_fuentes = mocker.spy(sapm_dc_snl_ac_system, 'fuentes_celltemp')
    mc.run_model(weather)
    assert m_fuentes.call_count == 1
    assert_series_equal(m_fuentes.call_args[0][1], weather['temp_air'])
    assert_series_equal(m_fuentes.call_args[0][2], weather['wind_speed'])
    assert not mc.results.ac.empty


def test_run_model_tracker(sapm_dc_snl_ac_system, location, weather, mocker):
    system = SingleAxisTracker(
        module_parameters=sapm_dc_snl_ac_system.module_parameters,
        temperature_model_parameters=(
            sapm_dc_snl_ac_system.temperature_model_parameters
        ),
        inverter_parameters=sapm_dc_snl_ac_system.inverter_parameters)
    mocker.spy(system, 'singleaxis')
    mc = ModelChain(system, location)
    mc.run_model(weather)
    assert system.singleaxis.call_count == 1
    assert (mc.results.tracking.columns == ['tracker_theta',
                                            'aoi',
                                            'surface_azimuth',
                                            'surface_tilt']).all()
    assert mc.results.ac[0] > 0
    assert np.isnan(mc.results.ac[1])


def test__assign_total_irrad(sapm_dc_snl_ac_system, location, weather,
                             total_irrad):
    data = pd.concat([weather, total_irrad], axis=1)
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    mc._assign_total_irrad(data)
    assert_frame_equal(mc.results.total_irrad, total_irrad)


def test_prepare_inputs_from_poa(sapm_dc_snl_ac_system, location,
                                 weather, total_irrad):
    data = pd.concat([weather, total_irrad], axis=1)
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    mc.prepare_inputs_from_poa(data)
    weather_expected = weather.copy()
    weather_expected['temp_air'] = 20
    weather_expected['wind_speed'] = 0
    # order as expected
    weather_expected = weather_expected[
        ['ghi', 'dhi', 'dni', 'wind_speed', 'temp_air']]
    # weather attribute
    assert_frame_equal(mc.weather, weather_expected)
    # total_irrad attribute
    assert_frame_equal(mc.results.total_irrad, total_irrad)


@pytest.mark.parametrize("input_type", [tuple, list])
def test_prepare_inputs_from_poa_multi_data(
        sapm_dc_snl_ac_system_Array, location, total_irrad, weather,
        input_type):
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    poa = pd.concat([weather, total_irrad], axis=1)
    mc.prepare_inputs_from_poa(input_type((poa, poa)))
    num_arrays = sapm_dc_snl_ac_system_Array.num_arrays
    assert len(mc.results.total_irrad) == num_arrays


@pytest.mark.parametrize("input_type", [tuple, list])
def test_prepare_inputs_from_poa_wrong_number_arrays(
        sapm_dc_snl_ac_system_Array, location, total_irrad, weather,
        input_type):
    len_error = r"Input must be same length as number of Arrays in system\. " \
                r"Expected 2, got [0-9]+\."
    type_error = r"Input must be a tuple of length 2, got .*\."
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    poa = pd.concat([weather, total_irrad], axis=1)
    with pytest.raises(TypeError, match=type_error):
        mc.prepare_inputs_from_poa(poa)
    with pytest.raises(ValueError, match=len_error):
        mc.prepare_inputs_from_poa(input_type((poa,)))
    with pytest.raises(ValueError, match=len_error):
        mc.prepare_inputs_from_poa(input_type((poa, poa, poa)))


def test_prepare_inputs_from_poa_arrays_different_indices(
        sapm_dc_snl_ac_system_Array, location, total_irrad, weather):
    error_str = r"Input DataFrames must have same index\."
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    poa = pd.concat([weather, total_irrad], axis=1)
    with pytest.raises(ValueError, match=error_str):
        mc.prepare_inputs_from_poa((poa, poa.shift(periods=1, freq='6H')))


def test_prepare_inputs_from_poa_arrays_missing_column(
        sapm_dc_snl_ac_system_Array, location, weather, total_irrad):
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    poa = pd.concat([weather, total_irrad], axis=1)
    with pytest.raises(ValueError, match=r"Incomplete input data\. "
                                         r"Data needs to contain .*\. "
                                         r"Detected data in element 1 "
                                         r"contains: .*"):
        mc.prepare_inputs_from_poa((poa, poa.drop(columns='poa_global')))


def test__prepare_temperature(sapm_dc_snl_ac_system, location, weather,
                              total_irrad):
    data = weather.copy()
    data[['poa_global', 'poa_diffuse', 'poa_direct']] = total_irrad
    mc = ModelChain(sapm_dc_snl_ac_system, location, aoi_model='no_loss',
                    spectral_model='no_loss')
    # prepare_temperature expects mc.total_irrad and mc.weather to be set
    mc._assign_weather(data)
    mc._assign_total_irrad(data)
    mc._prepare_temperature(data)
    expected = pd.Series([48.928025, 38.080016], index=data.index)
    assert_series_equal(mc.results.cell_temperature, expected)
    data['module_temperature'] = [40., 30.]
    mc._prepare_temperature(data)
    expected = pd.Series([42.4, 31.5], index=data.index)
    assert_series_equal(mc.results.cell_temperature, expected)
    data['cell_temperature'] = [50., 35.]
    mc._prepare_temperature(data)
    assert_series_equal(mc.results.cell_temperature, data['cell_temperature'])


def test__prepare_temperature_arrays_weather(sapm_dc_snl_ac_system_same_arrays,
                                             location, weather,
                                             total_irrad):
    data = weather.copy()
    data[['poa_global', 'poa_direct', 'poa_diffuse']] = total_irrad
    data_two = data.copy()
    mc = ModelChain(sapm_dc_snl_ac_system_same_arrays, location,
                    aoi_model='no_loss', spectral_model='no_loss')
    # prepare_temperature expects mc.total_irrad and mc.weather to be set
    mc._assign_weather((data, data_two))
    mc._assign_total_irrad((data, data_two))
    mc._prepare_temperature((data, data_two))
    expected = pd.Series([48.928025, 38.080016], index=data.index)
    assert_series_equal(mc.results.cell_temperature[0], expected)
    assert_series_equal(mc.results.cell_temperature[1], expected)
    data['module_temperature'] = [40., 30.]
    mc._prepare_temperature((data, data_two))
    expected = pd.Series([42.4, 31.5], index=data.index)
    assert (mc.results.cell_temperature[1] != expected).all()
    assert_series_equal(mc.results.cell_temperature[0], expected)
    data['cell_temperature'] = [50., 35.]
    mc._prepare_temperature((data, data_two))
    assert_series_equal(
        mc.results.cell_temperature[0], data['cell_temperature'])
    data_two['module_temperature'] = [40., 30.]
    mc._prepare_temperature((data, data_two))
    assert_series_equal(mc.results.cell_temperature[1], expected)
    assert_series_equal(
        mc.results.cell_temperature[0], data['cell_temperature'])
    data_two['cell_temperature'] = [10.0, 20.0]
    mc._prepare_temperature((data, data_two))
    assert_series_equal(
        mc.results.cell_temperature[1], data_two['cell_temperature'])
    assert_series_equal(
        mc.results.cell_temperature[0], data['cell_temperature'])


@pytest.mark.parametrize('temp_params,temp_model',
                         [({'a': -3.47, 'b': -.0594, 'deltaT': 3},
                           ModelChain.sapm_temp),
                          ({'u_c': 29.0, 'u_v': 0},
                           ModelChain.pvsyst_temp),
                          ({'u0': 25.0, 'u1': 6.84},
                           ModelChain.faiman_temp),
                          ({'noct_installed': 45},
                           ModelChain.fuentes_temp)])
def test_temperature_models_arrays_multi_weather(
        temp_params, temp_model,
        sapm_dc_snl_ac_system_same_arrays,
        location, weather, total_irrad):
    sapm_dc_snl_ac_system_same_arrays.temperature_model_parameters = \
        temp_params
    # set air temp so it does not default to the same value for both arrays
    weather['temp_air'] = 25
    weather_one = weather
    weather_two = weather.copy() * 0.5
    mc = ModelChain(sapm_dc_snl_ac_system_same_arrays, location,
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.prepare_inputs((weather_one, weather_two))
    temp_model(mc)
    assert (mc.results.cell_temperature[0]
            != mc.results.cell_temperature[1]).all()


def test_run_model_from_poa(sapm_dc_snl_ac_system, location, total_irrad):
    mc = ModelChain(sapm_dc_snl_ac_system, location, aoi_model='no_loss',
                    spectral_model='no_loss')
    ac = mc.run_model_from_poa(total_irrad).results.ac
    expected = pd.Series(np.array([149.280238, 96.678385]),
                         index=total_irrad.index)
    assert_series_equal(ac, expected)


@pytest.mark.parametrize("input_type", [tuple, list])
def test_run_model_from_poa_arrays(sapm_dc_snl_ac_system_Array, location,
                                   weather, total_irrad, input_type):
    data = weather.copy()
    data[['poa_global', 'poa_diffuse', 'poa_direct']] = total_irrad
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location, aoi_model='no_loss',
                    spectral_model='no_loss')
    mc.run_model_from_poa(input_type((data, data)))
    # arrays have different orientation, but should give same dc power
    # because we are the same passing POA irradiance and air
    # temperature.
    assert_frame_equal(mc.results.dc[0], mc.results.dc[1])


def test_run_model_from_poa_tracking(sapm_dc_snl_ac_system, location,
                                     total_irrad):
    system = SingleAxisTracker(
        module_parameters=sapm_dc_snl_ac_system.module_parameters,
        temperature_model_parameters=(
            sapm_dc_snl_ac_system.temperature_model_parameters
        ),
        inverter_parameters=sapm_dc_snl_ac_system.inverter_parameters)
    mc = ModelChain(system, location, aoi_model='no_loss',
                    spectral_model='no_loss')
    ac = mc.run_model_from_poa(total_irrad).results.ac
    assert (mc.results.tracking.columns == ['tracker_theta',
                                            'aoi',
                                            'surface_azimuth',
                                            'surface_tilt']).all()
    expected = pd.Series(np.array([149.280238, 96.678385]),
                         index=total_irrad.index)
    assert_series_equal(ac, expected)


def test_run_model_from_effective_irradiance(sapm_dc_snl_ac_system, location,
                                             weather, total_irrad):
    data = weather.copy()
    data[['poa_global', 'poa_diffuse', 'poa_direct']] = total_irrad
    data['effective_irradiance'] = data['poa_global']
    mc = ModelChain(sapm_dc_snl_ac_system, location, aoi_model='no_loss',
                    spectral_model='no_loss')
    ac = mc.run_model_from_effective_irradiance(data).results.ac
    expected = pd.Series(np.array([149.280238, 96.678385]),
                         index=data.index)
    assert_series_equal(ac, expected)


def test_run_model_from_effective_irradiance_no_poa_global(
        sapm_dc_snl_ac_system, location, weather, total_irrad):
    data = weather.copy()
    data['effective_irradiance'] = total_irrad['poa_global']
    mc = ModelChain(sapm_dc_snl_ac_system, location, aoi_model='no_loss',
                    spectral_model='no_loss')
    ac = mc.run_model_from_effective_irradiance(data).results.ac
    expected = pd.Series(np.array([149.280238, 96.678385]),
                         index=data.index)
    assert_series_equal(ac, expected)


def test_run_model_from_effective_irradiance_poa_global_differs(
        sapm_dc_snl_ac_system, location, weather, total_irrad):
    data = weather.copy()
    data[['poa_global', 'poa_diffuse', 'poa_direct']] = total_irrad
    data['effective_irradiance'] = data['poa_global'] * 0.8
    mc = ModelChain(sapm_dc_snl_ac_system, location, aoi_model='no_loss',
                    spectral_model='no_loss')
    ac = mc.run_model_from_effective_irradiance(data).results.ac
    expected = pd.Series(np.array([118.302801, 76.099841]),
                         index=data.index)
    assert_series_equal(ac, expected)


@pytest.mark.parametrize("input_type", [tuple, list])
def test_run_model_from_effective_irradiance_arrays_error(
        sapm_dc_snl_ac_system_Array, location, weather, total_irrad,
        input_type):
    data = weather.copy()
    data[['poa_global', 'poa_diffuse', 'poa_direct']] = total_irrad
    data['effetive_irradiance'] = data['poa_global']
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    len_error = r"Input must be same length as number of Arrays in system\. " \
                r"Expected 2, got [0-9]+\."
    type_error = r"Input must be a tuple of length 2, got DataFrame\."
    with pytest.raises(TypeError, match=type_error):
        mc.run_model_from_effective_irradiance(data)
    with pytest.raises(ValueError, match=len_error):
        mc.run_model_from_effective_irradiance(input_type((data,)))
    with pytest.raises(ValueError, match=len_error):
        mc.run_model_from_effective_irradiance(input_type((data, data, data)))
    with pytest.raises(ValueError,
                       match=r"Input DataFrames must have same index\."):
        mc.run_model_from_effective_irradiance(
            (data, data.shift(periods=1, freq='6H'))
        )


@pytest.mark.parametrize("input_type", [tuple, list])
def test_run_model_from_effective_irradiance_arrays(
        sapm_dc_snl_ac_system_Array, location, weather, total_irrad,
        input_type):
    data = weather.copy()
    data[['poa_global', 'poa_diffuse', 'poa_direct']] = total_irrad
    data['effective_irradiance'] = data['poa_global']
    data['cell_temperature'] = 40
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    mc.run_model_from_effective_irradiance(input_type((data, data)))
    # arrays have different orientation, but should give same dc power
    # because we are the same passing effective irradiance and cell
    # temperature.
    assert_frame_equal(mc.results.dc[0], mc.results.dc[1])
    # test that unequal inputs create unequal results
    data_two = data.copy()
    data_two['effective_irradiance'] = data['poa_global'] * 0.5
    mc.run_model_from_effective_irradiance(input_type((data, data_two)))
    assert (mc.results.dc[0] != mc.results.dc[1]).all().all()


def test_run_model_from_effective_irradiance_minimal_input(
        sapm_dc_snl_ac_system, sapm_dc_snl_ac_system_Array,
        location, total_irrad):
    data = pd.DataFrame({'effective_irradiance': total_irrad['poa_global'],
                         'cell_temperature': 40},
                        index=total_irrad.index)
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    mc.run_model_from_effective_irradiance(data)
    # make sure, for a single Array, the result is the correct type and value
    assert_series_equal(mc.results.cell_temperature, data['cell_temperature'])
    assert not mc.results.dc.empty
    assert not mc.results.ac.empty
    # test with multiple arrays
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    mc.run_model_from_effective_irradiance((data, data))
    assert_frame_equal(mc.results.dc[0], mc.results.dc[1])
    assert not mc.results.ac.empty


def test_run_model_from_effective_irradiance_missing_poa(
        sapm_dc_snl_ac_system_Array, location, total_irrad):
    data_incomplete = pd.DataFrame(
        {'effective_irradiance': total_irrad['poa_global'],
         'poa_global': total_irrad['poa_global']},
        index=total_irrad.index)
    data_complete = pd.DataFrame(
        {'effective_irradiance': total_irrad['poa_global'],
         'cell_temperature': 30},
        index=total_irrad.index)
    mc = ModelChain(sapm_dc_snl_ac_system_Array, location)
    with pytest.raises(ValueError,
                       match="you must provide 'poa_global' for all Arrays"):
        mc.run_model_from_effective_irradiance(
            (data_complete, data_incomplete))


def poadc(mc):
    mc.results.dc = mc.results.total_irrad['poa_global'] * 0.2
    mc.results.dc.name = None  # assert_series_equal will fail without this


@pytest.mark.parametrize('dc_model', [
    'sapm', 'cec', 'desoto', 'pvsyst', 'singlediode', 'pvwatts_dc'])
def test_infer_dc_model(sapm_dc_snl_ac_system, cec_dc_snl_ac_system,
                        pvsyst_dc_snl_ac_system, pvwatts_dc_pvwatts_ac_system,
                        location, dc_model, weather, mocker):
    dc_systems = {'sapm': sapm_dc_snl_ac_system,
                  'cec': cec_dc_snl_ac_system,
                  'desoto': cec_dc_snl_ac_system,
                  'pvsyst': pvsyst_dc_snl_ac_system,
                  'singlediode': cec_dc_snl_ac_system,
                  'pvwatts_dc': pvwatts_dc_pvwatts_ac_system}
    dc_model_function = {'sapm': 'sapm',
                         'cec': 'calcparams_cec',
                         'desoto': 'calcparams_desoto',
                         'pvsyst': 'calcparams_pvsyst',
                         'singlediode': 'calcparams_desoto',
                         'pvwatts_dc': 'pvwatts_dc'}
    temp_model_function = {'sapm': 'sapm',
                           'cec': 'sapm',
                           'desoto': 'sapm',
                           'pvsyst': 'pvsyst',
                           'singlediode': 'sapm',
                           'pvwatts_dc': 'sapm'}
    temp_model_params = {'sapm': {'a': -3.40641, 'b': -0.0842075, 'deltaT': 3},
                         'pvsyst': {'u_c': 29.0, 'u_v': 0}}
    system = dc_systems[dc_model]
    system.temperature_model_parameters = temp_model_params[
        temp_model_function[dc_model]]
    # remove Adjust from model parameters for desoto, singlediode
    if dc_model in ['desoto', 'singlediode']:
        system.module_parameters.pop('Adjust')
    m = mocker.spy(pvsystem, dc_model_function[dc_model])
    mc = ModelChain(system, location,
                    aoi_model='no_loss', spectral_model='no_loss',
                    temperature_model=temp_model_function[dc_model])
    mc.run_model(weather)
    assert m.call_count == 1
    assert isinstance(mc.results.dc, (pd.Series, pd.DataFrame))


@pytest.mark.parametrize('dc_model', ['cec', 'desoto', 'pvsyst'])
def test_singlediode_dc_arrays(location, dc_model,
                               cec_dc_snl_ac_arrays,
                               pvsyst_dc_snl_ac_arrays,
                               weather):
    systems = {'cec': cec_dc_snl_ac_arrays,
               'pvsyst': pvsyst_dc_snl_ac_arrays,
               'desoto': cec_dc_snl_ac_arrays}
    temp_sapm = {'a': -3.40641, 'b': -0.0842075, 'deltaT': 3}
    temp_pvsyst = {'u_c': 29.0, 'u_v': 0}
    temp_model_params = {'cec': temp_sapm,
                         'desoto': temp_sapm,
                         'pvsyst': temp_pvsyst}
    temp_model = {'cec': 'sapm', 'desoto': 'sapm', 'pvsyst': 'pvsyst'}
    system = systems[dc_model]
    system.temperature_model_parameters = temp_model_params[dc_model]
    if dc_model == 'desoto':
        for module_parameters in system.module_parameters:
            module_parameters.pop('Adjust')
    mc = ModelChain(system, location,
                    aoi_model='no_loss', spectral_model='no_loss',
                    temperature_model=temp_model[dc_model])
    mc.run_model(weather)
    assert isinstance(mc.results.dc, tuple)
    assert len(mc.results.dc) == system.num_arrays
    for dc in mc.results.dc:
        assert isinstance(dc, (pd.Series, pd.DataFrame))


@pytest.mark.parametrize('dc_model', ['sapm', 'cec', 'cec_native'])
def test_infer_spectral_model(location, sapm_dc_snl_ac_system,
                              cec_dc_snl_ac_system,
                              cec_dc_native_snl_ac_system, dc_model):
    dc_systems = {'sapm': sapm_dc_snl_ac_system,
                  'cec': cec_dc_snl_ac_system,
                  'cec_native': cec_dc_native_snl_ac_system}
    system = dc_systems[dc_model]
    mc = ModelChain(system, location,
                    orientation_strategy='None', aoi_model='physical')
    assert isinstance(mc, ModelChain)


@pytest.mark.parametrize('temp_model', [
    'sapm_temp', 'faiman_temp', 'pvsyst_temp', 'fuentes_temp'])
def test_infer_temp_model(location, sapm_dc_snl_ac_system,
                          pvwatts_dc_pvwatts_ac_pvsyst_temp_system,
                          pvwatts_dc_pvwatts_ac_faiman_temp_system,
                          pvwatts_dc_pvwatts_ac_fuentes_temp_system,
                          temp_model):
    dc_systems = {'sapm_temp': sapm_dc_snl_ac_system,
                  'pvsyst_temp': pvwatts_dc_pvwatts_ac_pvsyst_temp_system,
                  'faiman_temp': pvwatts_dc_pvwatts_ac_faiman_temp_system,
                  'fuentes_temp': pvwatts_dc_pvwatts_ac_fuentes_temp_system}
    system = dc_systems[temp_model]
    mc = ModelChain(system, location,
                    orientation_strategy='None', aoi_model='physical',
                    spectral_model='no_loss')
    assert temp_model == mc.temperature_model.__name__
    assert isinstance(mc, ModelChain)


def test_infer_temp_model_invalid(location, sapm_dc_snl_ac_system):
    sapm_dc_snl_ac_system.temperature_model_parameters.pop('a')
    with pytest.raises(ValueError):
        ModelChain(sapm_dc_snl_ac_system, location,
                   orientation_strategy='None', aoi_model='physical',
                   spectral_model='no_loss')


def test_temperature_model_inconsistent(location, sapm_dc_snl_ac_system):
    with pytest.raises(ValueError):
        ModelChain(sapm_dc_snl_ac_system, location,
                   orientation_strategy='None', aoi_model='physical',
                   spectral_model='no_loss', temperature_model='pvsyst')


def test_dc_model_user_func(pvwatts_dc_pvwatts_ac_system, location, weather,
                            mocker):
    m = mocker.spy(sys.modules[__name__], 'poadc')
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, dc_model=poadc,
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.run_model(weather)
    assert m.call_count == 1
    assert isinstance(mc.results.ac, (pd.Series, pd.DataFrame))
    assert not mc.results.ac.empty


def test_pvwatts_dc_multiple_strings(pvwatts_dc_pvwatts_ac_system, location,
                                     weather, mocker):
    system = pvwatts_dc_pvwatts_ac_system
    m = mocker.spy(system, 'scale_voltage_current_power')
    mc1 = ModelChain(system, location,
                     aoi_model='no_loss', spectral_model='no_loss')
    mc1.run_model(weather)
    assert m.call_count == 1
    system.arrays[0].modules_per_string = 2
    mc2 = ModelChain(system, location,
                     aoi_model='no_loss', spectral_model='no_loss')
    mc2.run_model(weather)
    assert isinstance(mc2.results.ac, (pd.Series, pd.DataFrame))
    assert not mc2.results.ac.empty
    expected = pd.Series(data=[2., np.nan], index=mc2.results.dc.index,
                         name='p_mp')
    assert_series_equal(mc2.results.dc / mc1.results.dc, expected)


def acdc(mc):
    mc.results.ac = mc.results.dc


@pytest.mark.parametrize('ac_model', ['sandia', 'adr',
                                      'pvwatts', 'sandia_multi',
                                      'pvwatts_multi'])
def test_ac_models(sapm_dc_snl_ac_system, cec_dc_adr_ac_system,
                   pvwatts_dc_pvwatts_ac_system, location, ac_model,
                   weather, mocker):
    ac_systems = {'sandia': sapm_dc_snl_ac_system,
                  'sandia_multi': sapm_dc_snl_ac_system,
                  'adr': cec_dc_adr_ac_system,
                  'pvwatts': pvwatts_dc_pvwatts_ac_system,
                  'pvwatts_multi': pvwatts_dc_pvwatts_ac_system}
    ac_method_name = {'sandia': 'snlinverter',
                      'sandia_multi': 'sandia_multi',
                      'adr': 'adrinverter',
                      'pvwatts': 'pvwatts_ac',
                      'pvwatts_multi': 'pvwatts_multi'}
    system = ac_systems[ac_model]

    mc = ModelChain(system, location, ac_model=ac_model,
                    aoi_model='no_loss', spectral_model='no_loss')
    m = mocker.spy(system, ac_method_name[ac_model])
    mc.run_model(weather)
    assert m.call_count == 1
    assert isinstance(mc.results.ac, pd.Series)
    assert not mc.results.ac.empty
    assert mc.results.ac[1] < 1


def test_ac_model_user_func(pvwatts_dc_pvwatts_ac_system, location, weather,
                            mocker):
    m = mocker.spy(sys.modules[__name__], 'acdc')
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, ac_model=acdc,
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.run_model(weather)
    assert m.call_count == 1
    assert_series_equal(mc.results.ac, mc.results.dc)
    assert not mc.results.ac.empty


def test_ac_model_not_a_model(pvwatts_dc_pvwatts_ac_system, location, weather):
    exc_text = 'not a valid AC power model'
    with pytest.raises(ValueError, match=exc_text):
        ModelChain(pvwatts_dc_pvwatts_ac_system, location,
                   ac_model='not_a_model', aoi_model='no_loss',
                   spectral_model='no_loss')


def test_infer_ac_model_invalid_params(location):
    # only the keys are relevant here, using arbitrary values
    module_parameters = {'pdc0': 1, 'gamma_pdc': 1}
    system = pvsystem.PVSystem(
        arrays=[pvsystem.Array(
            module_parameters=module_parameters
        )],
        inverter_parameters={'foo': 1, 'bar': 2}
    )
    with pytest.raises(ValueError, match='could not infer AC model'):
        ModelChain(system, location)


def constant_aoi_loss(mc):
    mc.results.aoi_modifier = 0.9


@pytest.mark.parametrize('aoi_model', [
    'sapm', 'ashrae', 'physical', 'martin_ruiz'
])
def test_aoi_models(sapm_dc_snl_ac_system, location, aoi_model,
                    weather, mocker):
    mc = ModelChain(sapm_dc_snl_ac_system, location, dc_model='sapm',
                    aoi_model=aoi_model, spectral_model='no_loss')
    m = mocker.spy(sapm_dc_snl_ac_system, 'get_iam')
    mc.run_model(weather=weather)
    assert m.call_count == 1
    assert isinstance(mc.results.ac, pd.Series)
    assert not mc.results.ac.empty
    assert mc.results.ac[0] > 150 and mc.results.ac[0] < 200
    assert mc.results.ac[1] < 1


def test_aoi_model_no_loss(sapm_dc_snl_ac_system, location, weather):
    mc = ModelChain(sapm_dc_snl_ac_system, location, dc_model='sapm',
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.run_model(weather)
    assert mc.results.aoi_modifier == 1.0
    assert not mc.results.ac.empty
    assert mc.results.ac[0] > 150 and mc.results.ac[0] < 200
    assert mc.results.ac[1] < 1


def test_aoi_model_user_func(sapm_dc_snl_ac_system, location, weather, mocker):
    m = mocker.spy(sys.modules[__name__], 'constant_aoi_loss')
    mc = ModelChain(sapm_dc_snl_ac_system, location, dc_model='sapm',
                    aoi_model=constant_aoi_loss, spectral_model='no_loss')
    mc.run_model(weather)
    assert m.call_count == 1
    assert mc.results.aoi_modifier == 0.9
    assert not mc.results.ac.empty
    assert mc.results.ac[0] > 140 and mc.results.ac[0] < 200
    assert mc.results.ac[1] < 1


@pytest.mark.parametrize('aoi_model', [
    'sapm', 'ashrae', 'physical', 'martin_ruiz'
])
def test_infer_aoi_model(location, system_no_aoi, aoi_model):
    for k in iam._IAM_MODEL_PARAMS[aoi_model]:
        system_no_aoi.module_parameters.update({k: 1.0})
    mc = ModelChain(system_no_aoi, location,
                    orientation_strategy='None',
                    spectral_model='no_loss')
    assert isinstance(mc, ModelChain)


def test_infer_aoi_model_invalid(location, system_no_aoi):
    exc_text = 'could not infer AOI model'
    with pytest.raises(ValueError, match=exc_text):
        ModelChain(system_no_aoi, location, orientation_strategy='None',
                   spectral_model='no_loss')


def constant_spectral_loss(mc):
    mc.results.spectral_modifier = 0.9


@pytest.mark.parametrize('spectral_model', [
    'sapm', 'first_solar', 'no_loss', constant_spectral_loss
])
def test_spectral_models(sapm_dc_snl_ac_system, location, spectral_model,
                         weather):
    # add pw to weather dataframe
    weather['precipitable_water'] = [0.3, 0.5]
    mc = ModelChain(sapm_dc_snl_ac_system, location, dc_model='sapm',
                    aoi_model='no_loss', spectral_model=spectral_model)
    spectral_modifier = mc.run_model(weather).results.spectral_modifier
    assert isinstance(spectral_modifier, (pd.Series, float, int))


def constant_losses(mc):
    mc.losses = 0.9
    mc.results.dc *= mc.losses


def test_losses_models_pvwatts(pvwatts_dc_pvwatts_ac_system, location, weather,
                               mocker):
    age = 1
    pvwatts_dc_pvwatts_ac_system.losses_parameters = dict(age=age)
    m = mocker.spy(pvsystem, 'pvwatts_losses')
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, dc_model='pvwatts',
                    aoi_model='no_loss', spectral_model='no_loss',
                    losses_model='pvwatts')
    mc.run_model(weather)
    assert m.call_count == 1
    m.assert_called_with(age=age)
    assert isinstance(mc.results.ac, (pd.Series, pd.DataFrame))
    assert not mc.results.ac.empty
    # check that we're applying correction to dc
    # GH 696
    dc_with_loss = mc.results.dc
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, dc_model='pvwatts',
                    aoi_model='no_loss', spectral_model='no_loss',
                    losses_model='no_loss')
    mc.run_model(weather)
    assert not np.allclose(mc.results.dc, dc_with_loss, equal_nan=True)


def test_losses_models_pvwatts_arrays(multi_array_sapm_dc_snl_ac_system,
                                      location, weather):
    age = 1
    system_both = multi_array_sapm_dc_snl_ac_system['two_array_system']
    system_both.losses_parameters = dict(age=age)
    mc = ModelChain(system_both, location,
                    aoi_model='no_loss', spectral_model='no_loss',
                    losses_model='pvwatts')
    mc.run_model(weather)
    dc_with_loss = mc.results.dc
    mc = ModelChain(system_both, location,
                    aoi_model='no_loss', spectral_model='no_loss',
                    losses_model='no_loss')
    mc.run_model(weather)
    assert not np.allclose(mc.results.dc[0], dc_with_loss[0], equal_nan=True)
    assert not np.allclose(mc.results.dc[1], dc_with_loss[1], equal_nan=True)
    assert not mc.results.ac.empty


def test_losses_models_ext_def(pvwatts_dc_pvwatts_ac_system, location, weather,
                               mocker):
    m = mocker.spy(sys.modules[__name__], 'constant_losses')
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, dc_model='pvwatts',
                    aoi_model='no_loss', spectral_model='no_loss',
                    losses_model=constant_losses)
    mc.run_model(weather)
    assert m.call_count == 1
    assert isinstance(mc.results.ac, (pd.Series, pd.DataFrame))
    assert mc.losses == 0.9
    assert not mc.results.ac.empty


def test_losses_models_no_loss(pvwatts_dc_pvwatts_ac_system, location, weather,
                               mocker):
    m = mocker.spy(pvsystem, 'pvwatts_losses')
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, dc_model='pvwatts',
                    aoi_model='no_loss', spectral_model='no_loss',
                    losses_model='no_loss')
    assert mc.losses_model == mc.no_extra_losses
    mc.run_model(weather)
    assert m.call_count == 0
    assert mc.losses == 1


def test_invalid_dc_model_params(sapm_dc_snl_ac_system, cec_dc_snl_ac_system,
                                 pvwatts_dc_pvwatts_ac_system, location):
    kwargs = {'dc_model': 'sapm', 'ac_model': 'snlinverter',
              'aoi_model': 'no_loss', 'spectral_model': 'no_loss',
              'temperature_model': 'sapm', 'losses_model': 'no_loss'}
    sapm_dc_snl_ac_system.module_parameters.pop('A0')  # remove a parameter
    with pytest.raises(ValueError):
        ModelChain(sapm_dc_snl_ac_system, location, **kwargs)

    kwargs['dc_model'] = 'singlediode'
    cec_dc_snl_ac_system.module_parameters.pop('a_ref')  # remove a parameter
    with pytest.raises(ValueError):
        ModelChain(cec_dc_snl_ac_system, location, **kwargs)

    kwargs['dc_model'] = 'pvwatts'
    kwargs['ac_model'] = 'pvwatts'
    pvwatts_dc_pvwatts_ac_system.module_parameters.pop('pdc0')
    with pytest.raises(ValueError):
        ModelChain(pvwatts_dc_pvwatts_ac_system, location, **kwargs)


@pytest.mark.parametrize('model', [
    'dc_model', 'ac_model', 'aoi_model', 'spectral_model',
    'temperature_model', 'losses_model'
])
def test_invalid_models(model, sapm_dc_snl_ac_system, location):
    kwargs = {'dc_model': 'pvwatts', 'ac_model': 'pvwatts',
              'aoi_model': 'no_loss', 'spectral_model': 'no_loss',
              'temperature_model': 'sapm', 'losses_model': 'no_loss'}
    kwargs[model] = 'invalid'
    with pytest.raises(ValueError):
        ModelChain(sapm_dc_snl_ac_system, location, **kwargs)


def test_bad_get_orientation():
    with pytest.raises(ValueError):
        modelchain.get_orientation('bad value')


# tests for PVSystem with multiple Arrays
def test_with_sapm_pvsystem_arrays(sapm_dc_snl_ac_system_Array, location,
                                   weather):
    mc = ModelChain.with_sapm(sapm_dc_snl_ac_system_Array, location,
                              ac_model='sandia_multi')
    assert mc.dc_model == mc.sapm
    assert mc.ac_model == mc.sandia_multi_inverter
    mc.run_model(weather)
    assert mc.results


def test_ModelChain_no_extra_kwargs(sapm_dc_snl_ac_system, location):
    with pytest.raises(TypeError, match="arbitrary_kwarg"):
        ModelChain(sapm_dc_snl_ac_system, location, arbitrary_kwarg='value')


@fail_on_pvlib_version('1.0')
def test_ModelChain_attributes_deprecated_10(sapm_dc_snl_ac_system, location):
    match = 'Use ModelChain.results'
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    with pytest.warns(pvlibDeprecationWarning, match=match):
        mc.aoi
    with pytest.warns(pvlibDeprecationWarning, match=match):
        mc.aoi = 5


def test_basic_chain_required(sam_data, cec_inverter_parameters,
                              sapm_temperature_cs5p_220m):
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    latitude = 32
    longitude = -111
    altitude = 700
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    with pytest.raises(ValueError):
        dc, ac = modelchain.basic_chain(
            times, latitude, longitude, module_parameters, temp_model_params,
            cec_inverter_parameters, altitude=altitude
        )


@requires_tables
def test_basic_chain_alt_az(sam_data, cec_inverter_parameters,
                            sapm_temperature_cs5p_220m):
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    surface_tilt = 0
    surface_azimuth = 0
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters,  temp_model_params,
                                    cec_inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth)

    expected = pd.Series(np.array([111.621405, -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


@requires_tables
def test_basic_chain_strategy(sam_data, cec_inverter_parameters,
                              sapm_temperature_cs5p_220m):
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    altitude = 700
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    dc, ac = modelchain.basic_chain(
        times, latitude, longitude, module_parameters, temp_model_params,
        cec_inverter_parameters, orientation_strategy='south_at_latitude_tilt',
        altitude=altitude)

    expected = pd.Series(np.array([178.382754, -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


@requires_tables
def test_basic_chain_altitude_pressure(sam_data, cec_inverter_parameters,
                                       sapm_temperature_cs5p_220m):
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    altitude = 700
    surface_tilt = 0
    surface_azimuth = 0
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    temp_model_params = sapm_temperature_cs5p_220m.copy()
    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, temp_model_params,
                                    cec_inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth,
                                    pressure=93194)

    expected = pd.Series(np.array([113.190045, -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, temp_model_params,
                                    cec_inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth,
                                    altitude=altitude)

    expected = pd.Series(np.array([113.189814, -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


@pytest.mark.parametrize('strategy, strategy_str', [
    ('south_at_latitude_tilt', 'south_at_latitude_tilt'),
    (None, 'None')])  # GitHub issue 352
def test_ModelChain___repr__(sapm_dc_snl_ac_system, location, strategy,
                             strategy_str):

    mc = ModelChain(sapm_dc_snl_ac_system, location,
                    orientation_strategy=strategy, name='my mc')

    expected = '\n'.join([
        'ModelChain: ',
        '  name: my mc',
        '  orientation_strategy: ' + strategy_str,
        '  clearsky_model: ineichen',
        '  transposition_model: haydavies',
        '  solar_position_method: nrel_numpy',
        '  airmass_model: kastenyoung1989',
        '  dc_model: sapm',
        '  ac_model: snlinverter',
        '  aoi_model: sapm_aoi_loss',
        '  spectral_model: sapm_spectral_loss',
        '  temperature_model: sapm_temp',
        '  losses_model: no_extra_losses'
    ])

    assert mc.__repr__() == expected


def test_complete_irradiance_clean_run(sapm_dc_snl_ac_system, location):
    """The DataFrame should not change if all columns are passed"""
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    times = pd.date_range('2010-07-05 9:00:00', periods=2, freq='H')
    i = pd.DataFrame(
        {'dni': [2, 3], 'dhi': [4, 6], 'ghi': [9, 5]}, index=times)

    mc.complete_irradiance(i)

    assert_series_equal(mc.weather['dni'],
                        pd.Series([2, 3], index=times, name='dni'))
    assert_series_equal(mc.weather['dhi'],
                        pd.Series([4, 6], index=times, name='dhi'))
    assert_series_equal(mc.weather['ghi'],
                        pd.Series([9, 5], index=times, name='ghi'))


@requires_tables
def test_complete_irradiance(sapm_dc_snl_ac_system, location):
    """Check calculations"""
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    times = pd.date_range('2010-07-05 7:00:00-0700', periods=2, freq='H')
    i = pd.DataFrame({'dni': [49.756966, 62.153947],
                      'ghi': [372.103976116, 497.087579068],
                      'dhi': [356.543700, 465.44400]}, index=times)

    with pytest.warns(UserWarning):
        mc.complete_irradiance(i[['ghi', 'dni']])
    assert_series_equal(mc.weather['dhi'],
                        pd.Series([356.543700, 465.44400],
                                  index=times, name='dhi'))

    with pytest.warns(UserWarning):
        mc.complete_irradiance(i[['dhi', 'dni']])
    assert_series_equal(mc.weather['ghi'],
                        pd.Series([372.103976116, 497.087579068],
                                  index=times, name='ghi'))

    mc.complete_irradiance(i[['dhi', 'ghi']])
    assert_series_equal(mc.weather['dni'],
                        pd.Series([49.756966, 62.153947],
                                  index=times, name='dni'))


@pytest.mark.filterwarnings("ignore:This function is not safe at the moment")
@pytest.mark.parametrize("input_type", [tuple, list])
@requires_tables
def test_complete_irradiance_arrays(
        sapm_dc_snl_ac_system_same_arrays, location, input_type):
    """ModelChain.complete_irradiance can accept a tuple of weather
    DataFrames."""
    times = pd.date_range(start='2020-01-01 0700-0700', periods=2, freq='H')
    weather = pd.DataFrame({'dni': [2, 3],
                            'dhi': [4, 6],
                            'ghi': [9, 5]}, index=times)
    mc = ModelChain(sapm_dc_snl_ac_system_same_arrays, location)
    with pytest.raises(ValueError,
                       match=r"Input DataFrames must have same index\."):
        mc.complete_irradiance(input_type((weather, weather[1:])))
    mc.complete_irradiance(input_type((weather, weather)))
    for mc_weather in mc.weather:
        assert_series_equal(mc_weather['dni'],
                            pd.Series([2, 3], index=times, name='dni'))
        assert_series_equal(mc_weather['dhi'],
                            pd.Series([4, 6], index=times, name='dhi'))
        assert_series_equal(mc_weather['ghi'],
                            pd.Series([9, 5], index=times, name='ghi'))
    mc = ModelChain(sapm_dc_snl_ac_system_same_arrays, location)
    mc.complete_irradiance(input_type((weather[['ghi', 'dhi']],
                                       weather[['dhi', 'dni']])))
    assert 'dni' in mc.weather[0].columns
    assert 'ghi' in mc.weather[1].columns
    mc.complete_irradiance(input_type((weather, weather[['ghi', 'dni']])))
    assert_series_equal(mc.weather[0]['dhi'],
                        pd.Series([4, 6], index=times, name='dhi'))
    assert_series_equal(mc.weather[0]['ghi'],
                        pd.Series([9, 5], index=times, name='ghi'))
    assert_series_equal(mc.weather[0]['dni'],
                        pd.Series([2, 3], index=times, name='dni'))
    assert 'dhi' in mc.weather[1].columns


@pytest.mark.parametrize("input_type", [tuple, list])
def test_complete_irradiance_arrays_wrong_length(
        sapm_dc_snl_ac_system_same_arrays, location, input_type):
    mc = ModelChain(sapm_dc_snl_ac_system_same_arrays, location)
    times = pd.date_range(start='2020-01-01 0700-0700', periods=2, freq='H')
    weather = pd.DataFrame({'dni': [2, 3],
                            'dhi': [4, 6],
                            'ghi': [9, 5]}, index=times)
    error_str = "Input must be same length as number " \
                r"of Arrays in system\. Expected 2, got [0-9]+\."
    with pytest.raises(ValueError, match=error_str):
        mc.complete_irradiance(input_type((weather,)))
    with pytest.raises(ValueError, match=error_str):
        mc.complete_irradiance(input_type((weather, weather, weather)))


def test_unknown_attribute(sapm_dc_snl_ac_system, location):
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    with pytest.raises(AttributeError):
        mc.unknown_attribute


def test_inconsistent_array_params(location,
                                   sapm_module_params,
                                   cec_module_params):
    module_error = ".* selected for the DC model but one or more Arrays are " \
                   "missing one or more required parameters"
    temperature_error = "could not infer temperature model from " \
                        r"system\.temperature_model_parameters\. Check " \
                        r"that all Arrays in system\.arrays have " \
                        r"parameters for the same temperature model\. " \
                        r"Common temperature model parameters: .*"
    different_module_system = pvsystem.PVSystem(
        arrays=[
            pvsystem.Array(
                module_parameters=sapm_module_params),
            pvsystem.Array(
                module_parameters=cec_module_params),
            pvsystem.Array(
                module_parameters=cec_module_params)]
    )
    with pytest.raises(ValueError, match=module_error):
        ModelChain(different_module_system, location, dc_model='cec')
    different_temp_system = pvsystem.PVSystem(
        arrays=[
            pvsystem.Array(
                module_parameters=cec_module_params,
                temperature_model_parameters={'a': 1,
                                              'b': 1,
                                              'deltaT': 1}),
            pvsystem.Array(
                module_parameters=cec_module_params,
                temperature_model_parameters={'a': 2,
                                              'b': 2,
                                              'deltaT': 2}),
            pvsystem.Array(
                module_parameters=cec_module_params,
                temperature_model_parameters={'b': 3, 'deltaT': 3})]
    )
    with pytest.raises(ValueError, match=temperature_error):
        ModelChain(different_temp_system, location,
                   ac_model='sandia_multi',
                   aoi_model='no_loss', spectral_model='no_loss',
                   temperature_model='sapm')


def test_modelchain__common_keys():
    dictionary = {'a': 1, 'b': 1}
    series = pd.Series(dictionary)
    assert {'a', 'b'} == modelchain._common_keys(
        {'a': 1, 'b': 1}
    )
    assert {'a', 'b'} == modelchain._common_keys(
        pd.Series({'a': 1, 'b': 1})
    )
    assert {'a', 'b'} == modelchain._common_keys(
        (dictionary, series)
    )
    no_a = dictionary.copy()
    del no_a['a']
    assert {'b'} == modelchain._common_keys(
        (dictionary, no_a)
    )
    assert {'b'} == modelchain._common_keys(
        (series, pd.Series(no_a))
    )
    assert {'b'} == modelchain._common_keys(
        (series, no_a)
    )


def test__irrad_for_celltemp():
    total_irrad = pd.DataFrame(index=[0, 1], columns=['poa_global'],
                               data=[10., 20.])
    empty = total_irrad.drop('poa_global', axis=1)
    effect_irrad = pd.Series(index=total_irrad.index, data=[5., 8.])
    # test with single array inputs
    poa = modelchain._irrad_for_celltemp(total_irrad, effect_irrad)
    assert_series_equal(poa, total_irrad['poa_global'])
    poa = modelchain._irrad_for_celltemp(empty, effect_irrad)
    assert_series_equal(poa, effect_irrad)
    # test with tuples
    poa = modelchain._irrad_for_celltemp(
        (total_irrad, total_irrad), (effect_irrad, effect_irrad))
    assert len(poa) == 2
    assert_series_equal(poa[0], total_irrad['poa_global'])
    assert_series_equal(poa[1], total_irrad['poa_global'])
    poa = modelchain._irrad_for_celltemp(
        (empty, empty), (effect_irrad, effect_irrad))
    assert len(poa) == 2
    assert_series_equal(poa[0], effect_irrad)
    assert_series_equal(poa[1], effect_irrad)
