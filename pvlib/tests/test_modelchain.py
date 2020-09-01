import sys

import numpy as np
import pandas as pd

from pvlib import iam, modelchain, pvsystem, temperature
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.tracking import SingleAxisTracker
from pvlib.location import Location
from pvlib._deprecation import pvlibDeprecationWarning

from conftest import assert_series_equal
import pytest

from conftest import fail_on_pvlib_version, requires_scipy


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
    ac = mc.run_model(irradiance).ac

    expected = pd.Series(np.array([187.80746494643176, -0.02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_prepare_inputs_no_irradiance(sapm_dc_snl_ac_system, location):
    mc = ModelChain(sapm_dc_snl_ac_system, location)
    weather = pd.DataFrame()
    with pytest.raises(ValueError):
        mc.prepare_inputs(weather)


def test_run_model_perez(sapm_dc_snl_ac_system, location):
    mc = ModelChain(sapm_dc_snl_ac_system, location,
                    transposition_model='perez')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni': 900, 'ghi': 600, 'dhi': 150},
                              index=times)
    ac = mc.run_model(irradiance).ac

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
    ac = mc.run_model(irradiance).ac

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
    assert not mc.ac.empty


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
    assert not mc.ac.empty


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
    assert not mc.ac.empty


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
    assert (mc.tracking.columns == ['tracker_theta', 'aoi', 'surface_azimuth',
                                    'surface_tilt']).all()
    assert mc.ac[0] > 0
    assert np.isnan(mc.ac[1])


def poadc(mc):
    mc.dc = mc.total_irrad['poa_global'] * 0.2
    mc.dc.name = None  # assert_series_equal will fail without this


@pytest.mark.parametrize('dc_model', [
    'sapm',
    pytest.param('cec', marks=requires_scipy),
    pytest.param('desoto', marks=requires_scipy),
    pytest.param('pvsyst', marks=requires_scipy),
    pytest.param('singlediode', marks=requires_scipy),
    'pvwatts_dc'])
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
    m = mocker.spy(system, dc_model_function[dc_model])
    mc = ModelChain(system, location,
                    aoi_model='no_loss', spectral_model='no_loss',
                    temperature_model=temp_model_function[dc_model])
    mc.run_model(weather)
    assert m.call_count == 1
    assert isinstance(mc.dc, (pd.Series, pd.DataFrame))


@pytest.mark.parametrize('dc_model', [
    'sapm',
    pytest.param('cec', marks=requires_scipy),
    pytest.param('cec_native', marks=requires_scipy)])
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
    'sapm_temp', 'faiman_temp',
    pytest.param('pvsyst_temp', marks=requires_scipy)])
def test_infer_temp_model(location, sapm_dc_snl_ac_system,
                          pvwatts_dc_pvwatts_ac_pvsyst_temp_system,
                          pvwatts_dc_pvwatts_ac_faiman_temp_system,
                          temp_model):
    dc_systems = {'sapm_temp': sapm_dc_snl_ac_system,
                  'pvsyst_temp': pvwatts_dc_pvwatts_ac_pvsyst_temp_system,
                  'faiman_temp': pvwatts_dc_pvwatts_ac_faiman_temp_system}
    system = dc_systems[temp_model]
    mc = ModelChain(system, location,
                    orientation_strategy='None', aoi_model='physical',
                    spectral_model='no_loss')
    assert temp_model == mc.temperature_model.__name__
    assert isinstance(mc, ModelChain)


@requires_scipy
def test_infer_temp_model_invalid(location, sapm_dc_snl_ac_system):
    sapm_dc_snl_ac_system.temperature_model_parameters.pop('a')
    with pytest.raises(ValueError):
        ModelChain(sapm_dc_snl_ac_system, location,
                   orientation_strategy='None', aoi_model='physical',
                   spectral_model='no_loss')


# ModelChain.infer_temperature_model. remove or statement in v0.9
@requires_scipy
@fail_on_pvlib_version('0.9')
def test_infer_temp_model_no_params(location, system_no_temp, weather):
    mc = ModelChain(system_no_temp, location, aoi_model='physical',
                    spectral_model='no_loss')
    match = "Reverting to deprecated default: SAPM cell temperature"
    with pytest.warns(pvlibDeprecationWarning, match=match):
        mc.run_model(weather)


@requires_scipy
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
    assert isinstance(mc.ac, (pd.Series, pd.DataFrame))
    assert not mc.ac.empty


def acdc(mc):
    mc.ac = mc.dc


@pytest.mark.parametrize('ac_model', [
    'sandia', pytest.param('adr', marks=requires_scipy), 'pvwatts'])
def test_ac_models(sapm_dc_snl_ac_system, cec_dc_adr_ac_system,
                   pvwatts_dc_pvwatts_ac_system, location, ac_model,
                   weather, mocker):
    ac_systems = {'sandia': sapm_dc_snl_ac_system,
                  'adr': cec_dc_adr_ac_system,
                  'pvwatts': pvwatts_dc_pvwatts_ac_system}
    ac_method_name = {'sandia': 'snlinverter',
                      'adr': 'adrinverter',
                      'pvwatts': 'pvwatts_ac'}
    system = ac_systems[ac_model]

    mc = ModelChain(system, location, ac_model=ac_model,
                    aoi_model='no_loss', spectral_model='no_loss')
    m = mocker.spy(system, ac_method_name[ac_model])
    mc.run_model(weather)
    assert m.call_count == 1
    assert isinstance(mc.ac, pd.Series)
    assert not mc.ac.empty
    assert mc.ac[1] < 1


# TODO in v0.9: remove this test for a deprecation warning
@pytest.mark.parametrize('ac_model', [
    'snlinverter', pytest.param('adrinverter', marks=requires_scipy)])
def test_ac_models_deprecated(sapm_dc_snl_ac_system, cec_dc_adr_ac_system,
                              location, ac_model, weather):
    ac_systems = {'snlinverter': sapm_dc_snl_ac_system,
                  'adrinverter': cec_dc_adr_ac_system}
    system = ac_systems[ac_model]
    warn_txt = "ac_model = '" + ac_model + "' is deprecated and will be" +\
               " removed in v0.9"
    with pytest.warns(pvlibDeprecationWarning, match=warn_txt):
        ModelChain(system, location, ac_model=ac_model,
                   aoi_model='no_loss', spectral_model='no_loss')


def test_ac_model_user_func(pvwatts_dc_pvwatts_ac_system, location, weather,
                            mocker):
    m = mocker.spy(sys.modules[__name__], 'acdc')
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, ac_model=acdc,
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.run_model(weather)
    assert m.call_count == 1
    assert_series_equal(mc.ac, mc.dc)
    assert not mc.ac.empty


def test_ac_model_not_a_model(pvwatts_dc_pvwatts_ac_system, location, weather):
    exc_text = 'not a valid AC power model'
    with pytest.raises(ValueError, match=exc_text):
        ModelChain(pvwatts_dc_pvwatts_ac_system, location,
                   ac_model='not_a_model', aoi_model='no_loss',
                   spectral_model='no_loss')


def constant_aoi_loss(mc):
    mc.aoi_modifier = 0.9


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
    assert isinstance(mc.ac, pd.Series)
    assert not mc.ac.empty
    assert mc.ac[0] > 150 and mc.ac[0] < 200
    assert mc.ac[1] < 1


def test_aoi_model_no_loss(sapm_dc_snl_ac_system, location, weather):
    mc = ModelChain(sapm_dc_snl_ac_system, location, dc_model='sapm',
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.run_model(weather)
    assert mc.aoi_modifier == 1.0
    assert not mc.ac.empty
    assert mc.ac[0] > 150 and mc.ac[0] < 200
    assert mc.ac[1] < 1


def test_aoi_model_user_func(sapm_dc_snl_ac_system, location, weather, mocker):
    m = mocker.spy(sys.modules[__name__], 'constant_aoi_loss')
    mc = ModelChain(sapm_dc_snl_ac_system, location, dc_model='sapm',
                    aoi_model=constant_aoi_loss, spectral_model='no_loss')
    mc.run_model(weather)
    assert m.call_count == 1
    assert mc.aoi_modifier == 0.9
    assert not mc.ac.empty
    assert mc.ac[0] > 140 and mc.ac[0] < 200
    assert mc.ac[1] < 1


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
    mc.spectral_modifier = 0.9


@requires_scipy
@pytest.mark.parametrize('spectral_model', [
        'sapm', 'first_solar', 'no_loss', constant_spectral_loss
])
def test_spectral_models(sapm_dc_snl_ac_system, location, spectral_model,
                         weather):
    # add pw to weather dataframe
    weather['precipitable_water'] = [0.3, 0.5]
    mc = ModelChain(sapm_dc_snl_ac_system, location, dc_model='sapm',
                    aoi_model='no_loss', spectral_model=spectral_model)
    spectral_modifier = mc.run_model(weather).spectral_modifier
    assert isinstance(spectral_modifier, (pd.Series, float, int))


def constant_losses(mc):
    mc.losses = 0.9
    mc.dc *= mc.losses


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
    assert isinstance(mc.ac, (pd.Series, pd.DataFrame))
    assert not mc.ac.empty
    # check that we're applying correction to dc
    # GH 696
    dc_with_loss = mc.dc
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, dc_model='pvwatts',
                    aoi_model='no_loss', spectral_model='no_loss',
                    losses_model='no_loss')
    mc.run_model(weather)
    assert not np.allclose(mc.dc, dc_with_loss, equal_nan=True)


def test_losses_models_ext_def(pvwatts_dc_pvwatts_ac_system, location, weather,
                               mocker):
    m = mocker.spy(sys.modules[__name__], 'constant_losses')
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, dc_model='pvwatts',
                    aoi_model='no_loss', spectral_model='no_loss',
                    losses_model=constant_losses)
    mc.run_model(weather)
    assert m.call_count == 1
    assert isinstance(mc.ac, (pd.Series, pd.DataFrame))
    assert mc.losses == 0.9
    assert not mc.ac.empty


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


@fail_on_pvlib_version('0.9')
@pytest.mark.parametrize('ac_model', ['snlinverter', 'adrinverter'])
def test_deprecated_09(sapm_dc_snl_ac_system, cec_dc_adr_ac_system,
                       location, ac_model, weather):
    # ModelChain.ac_model = 'snlinverter' or 'adrinverter' deprecated in v0.8,
    # removed in v0.9
    ac_systems = {'snlinverter': sapm_dc_snl_ac_system,
                  'adrinverter': cec_dc_adr_ac_system}
    system = ac_systems[ac_model]
    warn_txt = "ac_model = '" + ac_model + "' is deprecated and will be" +\
               " removed in v0.9"
    with pytest.warns(pvlibDeprecationWarning, match=warn_txt):
        ModelChain(system, location, ac_model=ac_model,
                   aoi_model='no_loss', spectral_model='no_loss')


@requires_scipy
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


@requires_scipy
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


@requires_scipy
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


@requires_scipy
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


@requires_scipy
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


@requires_scipy
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
