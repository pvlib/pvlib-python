import numpy as np
import pandas as pd
from numpy import nan

from pvlib import modelchain, pvsystem
from pvlib.modelchain import ModelChain
from pvlib.pvsystem import PVSystem
from pvlib.tracking import SingleAxisTracker
from pvlib.location import Location

from pandas.util.testing import assert_series_equal, assert_frame_equal
import pytest

from test_pvsystem import sam_data
from conftest import requires_scipy


@pytest.fixture
def system(sam_data):
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_'].copy()
    inverters = sam_data['cecinverter']
    inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'].copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module_parameters=module_parameters,
                      inverter_parameters=inverter)
    return system


@pytest.fixture
def cec_dc_snl_ac_system(sam_data):
    modules = sam_data['cecmod']
    module_parameters = modules['Canadian_Solar_CS5P_220M'].copy()
    module_parameters['b'] = 0.05
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    inverters = sam_data['cecinverter']
    inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'].copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
                      module_parameters=module_parameters,
                      inverter_parameters=inverter)
    return system


@pytest.fixture
def cec_dc_adr_ac_system(sam_data):
    modules = sam_data['cecmod']
    module_parameters = modules['Canadian_Solar_CS5P_220M'].copy()
    module_parameters['b'] = 0.05
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    inverters = sam_data['adrinverter']
    inverter = inverters['Zigor__Sunzet_3_TL_US_240V__CEC_2011_'].copy()
    system = PVSystem(surface_tilt=32.2, surface_azimuth=180,
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


@pytest.fixture()
def location():
    return Location(32.2, -111, altitude=700)


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
    ac = mc.run_model(times).ac

    expected = pd.Series(np.array([  183.522449305,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


def test_run_model_with_irradiance(system, location):
    mc = ModelChain(system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni':900, 'ghi':600, 'dhi':150},
                              index=times)
    ac = mc.run_model(times, weather=irradiance).ac

    expected = pd.Series(np.array([  1.90054749e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_run_model_perez(system, location):
    mc = ModelChain(system, location, transposition_model='perez')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni':900, 'ghi':600, 'dhi':150},
                              index=times)
    ac = mc.run_model(times, weather=irradiance).ac

    expected = pd.Series(np.array([  190.194545796,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_run_model_gueymard_perez(system, location):
    mc = ModelChain(system, location, airmass_model='gueymard1993',
                    transposition_model='perez')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni':900, 'ghi':600, 'dhi':150},
                              index=times)
    ac = mc.run_model(times, weather=irradiance).ac

    expected = pd.Series(np.array([  190.194760203,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


@requires_scipy
def test_run_model_with_weather(system, location):
    mc = ModelChain(system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    weather = pd.DataFrame({'wind_speed':5, 'temp_air':10}, index=times)
    ac = mc.run_model(times, weather=weather).ac

    expected = pd.Series(np.array([  201.691634921,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


@requires_scipy
def test_run_model_tracker(system, location):
    system = SingleAxisTracker(module_parameters=system.module_parameters,
                               inverter_parameters=system.inverter_parameters)
    mc = ModelChain(system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    ac = mc.run_model(times).ac

    expected = pd.Series(np.array([119.067713606,  nan]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)

    expected = pd.DataFrame(np.
        array([[ 54.82513187,  90.        ,  11.0039221 ,  11.0039221 ],
               [         nan,   0.        ,   0.        ,          nan]]),
        columns=['aoi', 'surface_azimuth', 'surface_tilt', 'tracker_theta'],
        index=times)
    assert_frame_equal(mc.tracking, expected, check_less_precise=2)


def poadc(mc):
    mc.dc = mc.total_irrad['poa_global'] * 0.2
    mc.dc.name = None  # assert_series_equal will fail without this

@requires_scipy
@pytest.mark.parametrize('dc_model, expected', [
    ('sapm', [181.604438144, -2.00000000e-02]),
    ('singlediode', [181.044109596, -2.00000000e-02]),
    ('pvwatts', [190.028186986, 0]),
    (poadc, [189.183065667, 0])  # user supplied function
])
def test_dc_models(system, cec_dc_snl_ac_system, pvwatts_dc_pvwatts_ac_system,
                   location, dc_model, expected):

    dc_systems = {'sapm': system, 'singlediode': cec_dc_snl_ac_system,
                  'pvwatts': pvwatts_dc_pvwatts_ac_system,
                  poadc: pvwatts_dc_pvwatts_ac_system}

    system = dc_systems[dc_model]

    mc = ModelChain(system, location, dc_model=dc_model,
                    aoi_model='no_loss', spectral_model='no_loss')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    ac = mc.run_model(times).ac

    expected = pd.Series(np.array(expected), index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


def acdc(mc):
    mc.ac = mc.dc

@requires_scipy
@pytest.mark.parametrize('ac_model, expected', [
    ('snlinverter', [181.604438144, -2.00000000e-02]),
    ('adrinverter', [np.nan, -25.00000000e-02]),
    ('pvwatts', [190.028186986, 0]),
    (acdc, [199.845296258, 0])  # user supplied function
])
def test_ac_models(system, cec_dc_adr_ac_system, pvwatts_dc_pvwatts_ac_system,
                   location, ac_model, expected):

    ac_systems = {'snlinverter': system, 'adrinverter': cec_dc_adr_ac_system,
                  'pvwatts': pvwatts_dc_pvwatts_ac_system,
                  acdc: pvwatts_dc_pvwatts_ac_system}

    system = ac_systems[ac_model]

    mc = ModelChain(system, location, ac_model=ac_model,
                    aoi_model='no_loss', spectral_model='no_loss')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    ac = mc.run_model(times).ac

    expected = pd.Series(np.array(expected), index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


def constant_aoi_loss(mc):
    mc.aoi_modifier = 0.9

@requires_scipy
@pytest.mark.parametrize('aoi_model, expected', [
    ('sapm', [182.784057666, -2.00000000e-02]),
    ('ashrae', [180.825930547, -2.00000000e-02]),
    ('physical', [181.453077805, -2.00000000e-02]),
    ('no_loss', [181.604438144, -2.00000000e-02]),
    (constant_aoi_loss, [164.997043305, -2e-2])
])
def test_aoi_models(system, location, aoi_model, expected):
    mc = ModelChain(system, location, dc_model='sapm',
                    aoi_model=aoi_model, spectral_model='no_loss')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    ac = mc.run_model(times).ac

    expected = pd.Series(np.array(expected), index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


def constant_spectral_loss(mc):
    mc.spectral_modifier = 0.9

@requires_scipy
@pytest.mark.parametrize('spectral_model, expected', [
    ('sapm', [182.338436597, -2.00000000e-02]),
    pytest.mark.xfail(raises=NotImplementedError)
    (('first_solar', [179.371460714, -2.00000000e-02])),
    ('no_loss', [181.604438144, -2.00000000e-02]),
    (constant_spectral_loss, [163.061569511, -2e-2])
])
def test_spectral_models(system, location, spectral_model, expected):
    mc = ModelChain(system, location, dc_model='sapm',
                    aoi_model='no_loss', spectral_model=spectral_model)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    ac = mc.run_model(times).ac

    expected = pd.Series(np.array(expected), index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


def constant_losses(mc):
    mc.losses = 0.9
    mc.ac *= mc.losses

@requires_scipy
@pytest.mark.parametrize('losses_model, expected', [
    ('pvwatts', [163.280464174, 0]),
    ('no_loss', [190.028186986, 0]),
    (constant_losses, [171.025368287, 0])
])
def test_losses_models(pvwatts_dc_pvwatts_ac_system, location, losses_model,
                       expected):
    mc = ModelChain(pvwatts_dc_pvwatts_ac_system, location, dc_model='pvwatts',
                    aoi_model='no_loss', spectral_model='no_loss',
                    losses_model=losses_model)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    ac = mc.run_model(times).ac

    expected = pd.Series(np.array(expected), index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


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
        mc = ModelChain(system, location, **kwargs)


def test_bad_get_orientation():
    with pytest.raises(ValueError):
        modelchain.get_orientation('bad value')


@requires_scipy
def test_basic_chain_required(sam_data):
    times = pd.DatetimeIndex(start='20160101 1200-0700',
                             end='20160101 1800-0700', freq='6H')
    latitude = 32
    longitude = -111
    altitude = 700
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    inverters = sam_data['cecinverter']
    inverter_parameters = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']
    with pytest.raises(ValueError):
        dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                        module_parameters, inverter_parameters,
                                        altitude=altitude)


@requires_scipy
def test_basic_chain_alt_az(sam_data):
    times = pd.DatetimeIndex(start='20160101 1200-0700',
                             end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    altitude = 700
    surface_tilt = 0
    surface_azimuth = 0
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    inverters = sam_data['cecinverter']
    inverter_parameters = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth)

    expected = pd.Series(np.array([  115.40352679,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


@requires_scipy
def test_basic_chain_strategy(sam_data):
    times = pd.DatetimeIndex(start='20160101 1200-0700',
                             end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    altitude = 700
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    inverters = sam_data['cecinverter']
    inverter_parameters = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    orientation_strategy='south_at_latitude_tilt',
                                    altitude=altitude)

    expected = pd.Series(np.array([  183.522449305,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


@requires_scipy
def test_basic_chain_altitude_pressure(sam_data):
    times = pd.DatetimeIndex(start='20160101 1200-0700',
                             end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    altitude = 700
    surface_tilt = 0
    surface_azimuth = 0
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    inverters = sam_data['cecinverter']
    inverter_parameters = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth,
                                    pressure=93194)

    expected = pd.Series(np.array([  116.595664887,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth,
                                    altitude=altitude)

    expected = pd.Series(np.array([  116.595664887,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


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
def test_weather_irradiance_input(system, location):
    """Test will raise a warning and should be removed in future versions."""
    mc = ModelChain(system, location)
    times = pd.date_range('2012-06-01 12:00:00', periods=2, freq='H')
    i = pd.DataFrame({'dni': [2, 3], 'dhi': [4, 6], 'ghi': [9, 5]}, index=times)
    w = pd.DataFrame({'wind_speed': [11, 5], 'temp_air': [30, 32]}, index=times)
    mc.run_model(times, irradiance=i, weather=w)

    assert_series_equal(mc.weather['dni'],
                        pd.Series([2, 3], index=times, name='dni'))
    assert_series_equal(mc.weather['wind_speed'],
                        pd.Series([11, 5], index=times, name='wind_speed'))


@requires_scipy
def test_complete_irradiance_clean_run(system, location):
    """The DataFrame should not change if all columns are passed"""
    mc = ModelChain(system, location)
    times = pd.date_range('2010-07-05 9:00:00', periods=2, freq='H')
    i = pd.DataFrame({'dni': [2, 3], 'dhi': [4, 6], 'ghi': [9, 5]}, index=times)

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

    mc.complete_irradiance(times, weather=i[['ghi', 'dni']])
    assert_series_equal(mc.weather['dhi'],
                        pd.Series([356.543700, 465.44400],
                                  index=times, name='dhi'))

    mc.complete_irradiance(times, weather=i[['dhi', 'dni']])
    assert_series_equal(mc.weather['ghi'],
                        pd.Series([372.103976116, 497.087579068],
                                  index=times, name='ghi'))

    mc.complete_irradiance(times, weather=i[['dhi', 'ghi']])
    assert_series_equal(mc.weather['dni'],
                        pd.Series([49.756966, 62.153947],
                                  index=times, name='dni'))
