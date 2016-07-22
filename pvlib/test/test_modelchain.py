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
    module = modules['Canadian_Solar_CS5P_220M___2009_'].copy()
    inverters = sam_data['cecinverter']
    inverter = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'].copy()

    system = PVSystem(module_parameters=module,
                      inverter_parameters=inverter)
    return system


@pytest.fixture()
def location():
    return Location(32.2, -111, altitude=700)


def test_ModelChain_creation(system, location):
    mc = ModelChain(system, location)


def test_orientation_strategy(system, location):
    strategies = {}

@pytest.mark.parametrize('strategy,expected', [
    (None, (0, 180)), ('None', (0, 180)), ('flat', (0, 180)),
    ('south_at_latitude_tilt', (32.2, 180))
])
def test_orientation_strategy(strategy, expected, system, location):
    mc = ModelChain(system, location, orientation_strategy=strategy)

    # the || accounts for the coercion of 'None' to None
    assert (mc.orientation_strategy == strategy or
            mc.orientation_strategy == None)
    assert system.surface_tilt == expected[0]
    assert system.surface_azimuth == expected[1]


@requires_scipy
def test_run_model(system, location):
    mc = ModelChain(system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    ac = mc.run_model(times).ac

    expected = pd.Series(np.array([  1.82033564e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


def test_run_model_with_irradiance(system, location):
    mc = ModelChain(system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni':900, 'ghi':600, 'dhi':150},
                              index=times)
    ac = mc.run_model(times, irradiance=irradiance).ac

    expected = pd.Series(np.array([  1.90054749e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_run_model_perez(system, location):
    mc = ModelChain(system, location, transposition_model='perez')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni':900, 'ghi':600, 'dhi':150},
                              index=times)
    ac = mc.run_model(times, irradiance=irradiance).ac

    expected = pd.Series(np.array([  190.194545796,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_run_model_gueymard_perez(system, location):
    mc = ModelChain(system, location, airmass_model='gueymard1993',
                    transposition_model='perez')
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame({'dni':900, 'ghi':600, 'dhi':150},
                              index=times)
    ac = mc.run_model(times, irradiance=irradiance).ac

    expected = pd.Series(np.array([  190.194760203,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


@requires_scipy
def test_run_model_with_weather(system, location):
    mc = ModelChain(system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    weather = pd.DataFrame({'wind_speed':5, 'temp_air':10}, index=times)
    ac = mc.run_model(times, weather=weather).ac

    expected = pd.Series(np.array([  1.99952400e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


@requires_scipy
def test_run_model_tracker(system, location):
    system = SingleAxisTracker(module_parameters=system.module_parameters,
                               inverter_parameters=system.inverter_parameters)
    mc = ModelChain(system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    ac = mc.run_model(times).ac

    expected = pd.Series(np.array([  121.421719,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)

    expected = pd.DataFrame(np.
        array([[ 54.82513187,  90.        ,  11.0039221 ,  11.0039221 ],
               [         nan,   0.        ,   0.        ,          nan]]),
        columns=['aoi', 'surface_azimuth', 'surface_tilt', 'tracker_theta'],
        index=times)
    assert_frame_equal(mc.tracking, expected, check_less_precise=2)


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

    expected = pd.Series(np.array([  1.14490928477e+02,  -2.00000000e-02]),
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

    expected = pd.Series(np.array([  1.82033563543e+02,  -2.00000000e-02]),
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

    expected = pd.Series(np.array([  1.15771428788e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth,
                                    altitude=altitude)

    expected = pd.Series(np.array([  1.15771428788e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected, check_less_precise=2)


def test_ModelChain___repr__(system, location):

    strategy = 'south_at_latitude_tilt'

    mc = ModelChain(system, location, orientation_strategy=strategy)

    assert mc.__repr__() == ('ModelChain for: PVSystem with tilt:32.2 and '+
    'azimuth: 180 with Module: None and Inverter: None '+
    'orientation_startegy: south_at_latitude_tilt clearsky_model: '+
    'ineichen transposition_model: haydavies solar_position_method: '+
    'nrel_numpy airmass_model: kastenyoung1989')
