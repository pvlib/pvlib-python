import sys
try:
    from unittest.mock import ANY
except ImportError:
    # python 2
    from mock import ANY

import numpy as np
import pandas as pd

from pvlib import modelchain, pvsystem
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.tracking import SingleAxisTracker
from pvlib.location import Location
from pvlib._deprecation import pvlibDeprecationWarning

from pandas.util.testing import assert_series_equal
import pytest

from test_pvsystem import sam_data, pvsyst_module_params
from conftest import fail_on_pvlib_version, requires_scipy, requires_tables


@pytest.fixture
def system(sam_data):
    modules = sam_data['sandiamod']
    module = 'Canadian_Solar_CS5P_220M___2009_'
    module_parameters = modules[module].copy()
    inverters = sam_data['cecinverter']
    inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'].copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module=module,
                      module_parameters=module_parameters,
                      inverter_parameters=inverter)
    return system


@pytest.fixture
def cec_dc_snl_ac_system(sam_data):
    modules = sam_data['cecmod']
    module = 'Canadian_Solar_CS5P_220M'
    module_parameters = modules[module].copy()
    module_parameters['b'] = 0.05
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    inverters = sam_data['cecinverter']
    inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'].copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module=module,
                      module_parameters=module_parameters,
                      inverter_parameters=inverter)
    return system


@pytest.fixture
def cec_dc_native_snl_ac_system(sam_data):
    module = 'Canadian_Solar_CS5P_220M'
    module_parameters = sam_data['cecmod'][module].copy()
    inverters = sam_data['cecinverter']
    inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'].copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module=module,
                      module_parameters=module_parameters,
                      inverter_parameters=inverter)
    return system


@pytest.fixture
def pvsyst_dc_snl_ac_system(sam_data, pvsyst_module_params):
    module = 'PVsyst test module'
    module_parameters = pvsyst_module_params
    module_parameters['b'] = 0.05
    inverters = sam_data['cecinverter']
    inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'].copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module=module,
                      module_parameters=module_parameters,
                      inverter_parameters=inverter)
    return system


@pytest.fixture
def cec_dc_adr_ac_system(sam_data):
    modules = sam_data['cecmod']
    module = 'Canadian_Solar_CS5P_220M'
    module_parameters = modules[module].copy()
    module_parameters['b'] = 0.05
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    inverters = sam_data['adrinverter']
    inverter = inverters['Zigor__Sunzet_3_TL_US_240V__CEC_2011_'].copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module=module,
                      module_parameters=module_parameters,
                      inverter_parameters=inverter)
    return system


@pytest.fixture
def pvwatts_dc_snl_ac_system(sam_data):
    module_parameters = {'pdc0': 220, 'gamma_pdc': -0.003}
    inverters = sam_data['cecinverter']
    inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'].copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module_parameters=module_parameters,
                      inverter_parameters=inverter)
    return system


@pytest.fixture
def pvwatts_dc_pvwatts_ac_system(sam_data):
    module_parameters = {'pdc0': 220, 'gamma_pdc': -0.003}
    inverter_parameters = {'eta_inv_nom': 0.95}
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


def test_ModelChain_creation(system, location):
    mc = ModelChain(system, location)


@pytest.mark.parametrize('strategy, expected', [
    (None, (32.2, 180)), ('None', (32.2, 180)), ('flat', (0, 180)),
    ('south_at_latitude_tilt', (32.2, 180))
])
def test_orientation_strategy(strategy, expected, system, location):
    mc = ModelChain(system, location, orientation_strategy=strategy)

    # the || accounts for the coercion of 'None' to None
    assert (mc.orientation_strategy == strategy or
            mc.orientation_strategy is None)
    assert system.surface_tilt == expected[0]
    assert system.surface_azimuth == expected[1]


@requires_scipy
def test_run_model(system, location):
    mc = ModelChain(system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')

    with pytest.warns(pvlibDeprecationWarning):
        ac = mc.run_model(times).ac

    expected = pd.Series(np.array([  183.522449305,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=1)


def test_run_model_with_irradiance(system, location):
    mc = ModelChain(system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni': 900, 'ghi': 600, 'dhi': 150},
                              index=times)
    ac = mc.run_model(times, weather=irradiance).ac

    expected = pd.Series(np.array([  1.90054749e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_run_model_perez(system, location):
    mc = ModelChain(system, location, transposition_model='perez')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni': 900, 'ghi': 600, 'dhi': 150},
                              index=times)
    ac = mc.run_model(times, weather=irradiance).ac

    expected = pd.Series(np.array([  190.194545796,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_run_model_gueymard_perez(system, location):
    mc = ModelChain(system, location, airmass_model='gueymard1993',
                    transposition_model='perez')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni': 900, 'ghi': 600, 'dhi': 150},
                              index=times)
    ac = mc.run_model(times, weather=irradiance).ac

    expected = pd.Series(np.array([  190.194760203,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_run_model_with_weather(system, location, weather, mocker):
    mc = ModelChain(system, location)
    m = mocker.spy(system, 'sapm_celltemp')
    weather['wind_speed'] = 5
    weather['temp_air'] = 10
    mc.run_model(weather.index, weather=weather)
    assert m.call_count == 1
    # assert_called_once_with cannot be used with series, so need to use
    # assert_series_equal on call_args
    assert_series_equal(m.call_args[0][1], weather['wind_speed'])  # wind
    assert_series_equal(m.call_args[0][2], weather['temp_air'])  # temp
    assert not mc.ac.empty


def test_run_model_tracker(system, location, weather, mocker):
    system = SingleAxisTracker(module_parameters=system.module_parameters,
                               inverter_parameters=system.inverter_parameters)
    mocker.spy(system, 'singleaxis')
    mc = ModelChain(system, location)
    mc.run_model(weather.index, weather=weather)
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
def test_infer_dc_model(system, cec_dc_snl_ac_system, pvsyst_dc_snl_ac_system,
                        pvwatts_dc_pvwatts_ac_system, location, dc_model,
                        weather, mocker):
    dc_systems = {'sapm': system,
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
    system = dc_systems[dc_model]
    # remove Adjust from model parameters for desoto, singlediode
    if dc_model in ['desoto', 'singlediode']:
        system.module_parameters.pop('Adjust')
    m = mocker.spy(system, dc_model_function[dc_model])
    mc = ModelChain(system, location,
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.run_model(weather.index, weather=weather)
    assert m.call_count == 1
    assert isinstance(mc.dc, (pd.Series, pd.DataFrame))


@pytest.mark.parametrize('dc_model', [
    'sapm',
    pytest.param('cec', marks=requires_scipy),
    pytest.param('cec_native', marks=requires_scipy)])
def test_infer_spectral_model(location, system, cec_dc_snl_ac_system,
                              cec_dc_native_snl_ac_system, dc_model):
    dc_systems = {'sapm': system,
                  'cec': cec_dc_snl_ac_system,
                  'cec_native': cec_dc_native_snl_ac_system}
    system = dc_systems[dc_model]
    mc = ModelChain(system, location,
                    orientation_strategy='None', aoi_model='physical')
    assert isinstance(mc, ModelChain)


def test_dc_model_user_func(pvwatts_dc_pvwatts_ac_system, location, weather,
                            mocker):
    m = mocker.spy(sys.modules[__name__], 'poadc')
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, dc_model=poadc,
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.run_model(weather.index, weather=weather)
    assert m.call_count == 1
    assert isinstance(mc.ac, (pd.Series, pd.DataFrame))
    assert not mc.ac.empty


def acdc(mc):
    mc.ac = mc.dc


@pytest.mark.parametrize('ac_model', [
    'snlinverter', pytest.param('adrinverter', marks=requires_scipy),
    'pvwatts'])
def test_ac_models(system, cec_dc_adr_ac_system, pvwatts_dc_pvwatts_ac_system,
                   location, ac_model, weather, mocker):
    ac_systems = {'snlinverter': system, 'adrinverter': cec_dc_adr_ac_system,
                  'pvwatts': pvwatts_dc_pvwatts_ac_system}
    system = ac_systems[ac_model]

    mc = ModelChain(system, location, ac_model=ac_model,
                    aoi_model='no_loss', spectral_model='no_loss')
    if ac_model == 'pvwatts':
        ac_model += '_ac'
    m = mocker.spy(system, ac_model)
    mc.run_model(weather.index, weather=weather)
    assert m.call_count == 1
    assert isinstance(mc.ac, pd.Series)
    assert not mc.ac.empty
    assert mc.ac[1] < 1


def test_ac_model_user_func(pvwatts_dc_pvwatts_ac_system, location, weather,
                            mocker):
    m = mocker.spy(sys.modules[__name__], 'acdc')
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, ac_model=acdc,
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.run_model(weather.index, weather=weather)
    assert m.call_count == 1
    assert_series_equal(mc.ac, mc.dc)
    assert not mc.ac.empty


def constant_aoi_loss(mc):
    mc.aoi_modifier = 0.9


@pytest.mark.parametrize('aoi_model, method', [
    ('sapm', 'sapm_aoi_loss'), ('ashrae', 'ashraeiam'),
    ('physical', 'physicaliam')])
def test_aoi_models(system, location, aoi_model, method, weather, mocker):
    mc = ModelChain(system, location, dc_model='sapm',
                    aoi_model=aoi_model, spectral_model='no_loss')
    m = mocker.spy(system, method)
    mc.run_model(weather.index, weather=weather)
    assert m.call_count == 1
    assert isinstance(mc.ac, pd.Series)
    assert not mc.ac.empty
    assert mc.ac[0] > 150 and mc.ac[0] < 200
    assert mc.ac[1] < 1


def test_aoi_model_no_loss(system, location, weather):
    mc = ModelChain(system, location, dc_model='sapm',
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.run_model(weather.index, weather=weather)
    assert mc.aoi_modifier == 1.0
    assert not mc.ac.empty
    assert mc.ac[0] > 150 and mc.ac[0] < 200
    assert mc.ac[1] < 1


def test_aoi_model_user_func(system, location, weather, mocker):
    m = mocker.spy(sys.modules[__name__], 'constant_aoi_loss')
    mc = ModelChain(system, location, dc_model='sapm',
                    aoi_model=constant_aoi_loss, spectral_model='no_loss')
    mc.run_model(weather.index, weather=weather)
    assert m.call_count == 1
    assert mc.aoi_modifier == 0.9
    assert not mc.ac.empty
    assert mc.ac[0] > 140 and mc.ac[0] < 200
    assert mc.ac[1] < 1


def constant_spectral_loss(mc):
    mc.spectral_modifier = 0.9


@requires_scipy
@pytest.mark.parametrize('spectral_model', [
        'sapm', 'first_solar', 'no_loss', constant_spectral_loss
])
def test_spectral_models(system, location, spectral_model, weather):
    # add pw to weather dataframe
    weather['precipitable_water'] = [0.3, 0.5]
    mc = ModelChain(system, location, dc_model='sapm',
                    aoi_model='no_loss', spectral_model=spectral_model)
    spectral_modifier = mc.run_model(times=weather.index,
                                     weather=weather).spectral_modifier
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
    mc.run_model(weather.index, weather=weather)
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
    mc.run_model(weather.index, weather=weather)
    assert not np.allclose(mc.dc, dc_with_loss, equal_nan=True)


def test_losses_models_ext_def(pvwatts_dc_pvwatts_ac_system, location, weather,
                               mocker):
    m = mocker.spy(sys.modules[__name__], 'constant_losses')
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, dc_model='pvwatts',
                    aoi_model='no_loss', spectral_model='no_loss',
                    losses_model=constant_losses)
    mc.run_model(weather.index, weather=weather)
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
    mc.run_model(weather.index, weather=weather)
    assert m.call_count == 0
    assert mc.losses == 1


def test_invalid_dc_model_params(system, cec_dc_snl_ac_system,
                                 pvwatts_dc_pvwatts_ac_system, location):
    kwargs = {'dc_model': 'sapm', 'ac_model': 'snlinverter',
              'aoi_model': 'no_loss', 'spectral_model': 'no_loss',
              'temp_model': 'sapm', 'losses_model': 'no_loss'}
    system.module_parameters.pop('A0')  # remove a parameter
    with pytest.raises(ValueError):
        ModelChain(system, location, **kwargs)

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
    'dc_model', 'ac_model', 'aoi_model', 'spectral_model', 'losses_model',
    'temp_model', 'losses_model'
])
def test_invalid_models(model, system, location):
    kwargs = {'dc_model': 'pvwatts', 'ac_model': 'pvwatts',
              'aoi_model': 'no_loss', 'spectral_model': 'no_loss',
              'temp_model': 'sapm', 'losses_model': 'no_loss'}
    kwargs[model] = 'invalid'
    with pytest.raises(ValueError):
        ModelChain(system, location, **kwargs)


def test_bad_get_orientation():
    with pytest.raises(ValueError):
        modelchain.get_orientation('bad value')


@fail_on_pvlib_version('0.7')
def test_deprecated_07():
    # explicit system creation call because fail_on_pvlib_version
    # does not support decorators.
    # does not matter what the parameters are, just fake it until we make it
    module_parameters = {'R_sh_ref': 1, 'a_ref': 1, 'I_o_ref': 1,
                         'alpha_sc': 1, 'I_L_ref': 1, 'R_s': 1}
    system = PVSystem(module_parameters=module_parameters)
    with pytest.warns(pvlibDeprecationWarning):
        ModelChain(system, location,
                   dc_model='singlediode',  # this should fail after 0.7
                   aoi_model='no_loss', spectral_model='no_loss',
                   ac_model='snlinverter')


@requires_tables
@fail_on_pvlib_version('0.7')
def test_deprecated_clearsky_07():
    # explicit system creation call because fail_on_pvlib_version
    # does not support decorators.
    system = PVSystem(module_parameters={'pdc0': 1, 'gamma_pdc': -0.003})
    location = Location(32.2, -110.9)
    mc = ModelChain(system, location, dc_model='pvwatts', ac_model='pvwatts',
                    aoi_model='no_loss', spectral_model='no_loss')
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    with pytest.warns(pvlibDeprecationWarning):
        mc.prepare_inputs(times=times)


@requires_scipy
def test_basic_chain_required(sam_data):
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    latitude = 32
    longitude = -111
    altitude = 700
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    inverters = sam_data['cecinverter']
    inverter_parameters = inverters[
        'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']
    with pytest.raises(ValueError):
        dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                        module_parameters, inverter_parameters,
                                        altitude=altitude)


@requires_scipy
def test_basic_chain_alt_az(sam_data):
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    surface_tilt = 0
    surface_azimuth = 0
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    inverters = sam_data['cecinverter']
    inverter_parameters = inverters[
        'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth)

    expected = pd.Series(np.array([  115.40352679,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=1)


@requires_scipy
def test_basic_chain_strategy(sam_data):
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    altitude = 700
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    inverters = sam_data['cecinverter']
    inverter_parameters = inverters[
        'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

    dc, ac = modelchain.basic_chain(
        times, latitude, longitude, module_parameters, inverter_parameters,
        orientation_strategy='south_at_latitude_tilt', altitude=altitude)

    expected = pd.Series(np.array([  183.522449305,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=1)


@requires_scipy
def test_basic_chain_altitude_pressure(sam_data):
    times = pd.date_range(start='20160101 1200-0700',
                          end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    altitude = 700
    surface_tilt = 0
    surface_azimuth = 0
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    inverters = sam_data['cecinverter']
    inverter_parameters = inverters[
        'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth,
                                    pressure=93194)

    expected = pd.Series(np.array([  116.595664887,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=1)

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth,
                                    altitude=altitude)

    expected = pd.Series(np.array([  116.595664887,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=1)


@pytest.mark.parametrize('strategy, strategy_str', [
    ('south_at_latitude_tilt', 'south_at_latitude_tilt'),
    (None, 'None')])  # GitHub issue 352
def test_ModelChain___repr__(system, location, strategy, strategy_str):

    mc = ModelChain(system, location, orientation_strategy=strategy,
                    name='my mc')

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
        '  temp_model: sapm_temp',
        '  losses_model: no_extra_losses'
    ])

    assert mc.__repr__() == expected


@requires_scipy
def test_complete_irradiance_clean_run(system, location):
    """The DataFrame should not change if all columns are passed"""
    mc = ModelChain(system, location)
    times = pd.date_range('2010-07-05 9:00:00', periods=2, freq='H')
    i = pd.DataFrame(
        {'dni': [2, 3], 'dhi': [4, 6], 'ghi': [9, 5]}, index=times)

    mc.complete_irradiance(times, weather=i)

    assert_series_equal(mc.weather['dni'],
                        pd.Series([2, 3], index=times, name='dni'))
    assert_series_equal(mc.weather['dhi'],
                        pd.Series([4, 6], index=times, name='dhi'))
    assert_series_equal(mc.weather['ghi'],
                        pd.Series([9, 5], index=times, name='ghi'))


@requires_scipy
def test_complete_irradiance(system, location):
    """Check calculations"""
    mc = ModelChain(system, location)
    times = pd.date_range('2010-07-05 7:00:00-0700', periods=2, freq='H')
    i = pd.DataFrame({'dni': [49.756966, 62.153947],
                      'ghi': [372.103976116, 497.087579068],
                      'dhi': [356.543700, 465.44400]}, index=times)

    with pytest.warns(UserWarning):
        mc.complete_irradiance(times, weather=i[['ghi', 'dni']])
    assert_series_equal(mc.weather['dhi'],
                        pd.Series([356.543700, 465.44400],
                                  index=times, name='dhi'))

    with pytest.warns(UserWarning):
        mc.complete_irradiance(times, weather=i[['dhi', 'dni']])
    assert_series_equal(mc.weather['ghi'],
                        pd.Series([372.103976116, 497.087579068],
                                  index=times, name='ghi'))

    mc.complete_irradiance(times, weather=i[['dhi', 'ghi']])
    assert_series_equal(mc.weather['dni'],
                        pd.Series([49.756966, 62.153947],
                                  index=times, name='dni'))
