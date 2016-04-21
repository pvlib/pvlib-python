import numpy as np
import pandas as pd
from numpy import nan

from pvlib import modelchain, pvsystem
from pvlib.modelchain import ModelChain, SAPM, SingleDiode
from pvlib.pvsystem import PVSystem
from pvlib.tracking import SingleAxisTracker
from pvlib.location import Location

from pandas.util.testing import assert_series_equal
from nose.tools import raises

from . import incompatible_conda_linux_py3

# should store this test data locally, but for now...
sam_data = {}
def retrieve_sam_network():
    sam_data['cecmod'] = pvsystem.retrieve_sam('cecmod')
    sam_data['sandiamod'] = pvsystem.retrieve_sam('sandiamod')
    sam_data['cecinverter'] = pvsystem.retrieve_sam('cecinverter')


def get_sapm_module_parameters():
    # limit network usage
    try:
        modules = sam_data['sandiamod']
    except KeyError:
        retrieve_sam_network()
        modules = sam_data['sandiamod']

    module = 'Canadian_Solar_CS5P_220M___2009_'
    module_parameters = modules[module].copy()

    return module_parameters


def get_cec_module_parameters():
    # limit network usage
    try:
        modules = sam_data['cecmod']
    except KeyError:
        retrieve_sam_network()
        modules = sam_data['cecmod']

    module = 'Canadian_Solar_CS5P_220M'
    module_parameters = modules[module].copy()
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677

    return module_parameters


def get_cec_inverter_parameters():

    inverters = sam_data['cecinverter']
    inverter = 'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'
    inverter_parameters = inverters[inverter].copy()

    return inverter_parameters


def sapm_setup():

    module_parameters = get_sapm_module_parameters()

    inverter_parameters = get_cec_inverter_parameters()

    system = PVSystem(module_parameters=module_parameters,
                      inverter_parameters=inverter_parameters)

    location = Location(32.2, -111, altitude=700)

    mc = SAPM(system, location)

    return mc


def singlediode_setup():

    module_parameters = get_cec_module_parameters()

    inverter_parameters = get_cec_inverter_parameters()

    system = PVSystem(module_parameters=module_parameters,
                      inverter_parameters=inverter_parameters)

    location = Location(32.2, -111, altitude=700)

    mc = SingleDiode(system, location)

    return mc


def test_ModelChain_creation():
    system = PVSystem()
    location = Location(32.2, -111, altitude=700)
    ModelChain(system, location)


def test_orientation_strategy():
    strategies = {None: (0, 180), 'None': (0, 180),
                  'south_at_latitude_tilt': (32.2, 180),
                  'flat': (0, 180)}

    for strategy, expected in strategies.items():
        yield run_orientation_strategy, strategy, expected


def run_orientation_strategy(strategy, expected):
    system = PVSystem()
    location = Location(32.2, -111, altitude=700)

    mc = ModelChain(system, location, orientation_strategy=strategy)

    # the || accounts for the coercion of 'None' to None
    assert (mc.orientation_strategy == strategy or
            mc.orientation_strategy is None)
    assert system.surface_tilt == expected[0]
    assert system.surface_azimuth == expected[1]


@raises(NotImplementedError)
def test_run_model_ModelChain():
    system = PVSystem()
    location = Location(32.2, -111, altitude=700)
    mc = ModelChain(system, location)
    mc.run_model()


@incompatible_conda_linux_py3
def test_run_model():
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    expected = {}
    expected[SAPM] = \
        pd.Series(np.array([1.82033564e+02,  -2.00000000e-02]), index=times)
    expected[SingleDiode] = \
        pd.Series(np.array([179.720353871, np.nan]), index=times)

    def run_test(mc):
        ac = mc.run_model(times).ac
        assert_series_equal(ac, expected[type(mc)])

    for mc in (sapm_setup(), singlediode_setup()):
        yield run_test, mc


@incompatible_conda_linux_py3
def test_run_model_with_irradiance():
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    irradiance = pd.DataFrame(
        {'dni': [900, 0], 'ghi': [600, 50], 'dhi': [150, 50]}, index=times)
    expected = {}
    expected[SAPM] = \
        pd.Series(np.array([1.90054749e+02,  -2.00000000e-02]), index=times)
    expected[SingleDiode] = \
        pd.Series(np.array([186.979595403, 7.89417460232]), index=times)

    def run_test(mc):
        ac = mc.run_model(times, irradiance=irradiance).ac
        assert_series_equal(ac, expected[type(mc)])

    for mc in (sapm_setup(), singlediode_setup()):
        yield run_test, mc


@incompatible_conda_linux_py3
def test_run_model_with_weather():
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    weather = pd.DataFrame({'wind_speed': 5, 'temp_air': 10}, index=times)
    expected = {}
    expected[SAPM] = \
        pd.Series(np.array([1.99952400e+02,  -2.00000000e-02]), index=times)
    expected[SingleDiode] = \
        pd.Series(np.array([198.13564009, np.nan]), index=times)

    def run_test(mc):
        ac = mc.run_model(times, weather=weather).ac
        assert_series_equal(ac, expected[type(mc)])

    for mc in (sapm_setup(), singlediode_setup()):
        yield run_test, mc


def test_run_model_tracker():
    system, location = mc_setup()
    system = SingleAxisTracker(module_parameters=system.module_parameters,
                               inverter_parameters=system.inverter_parameters)
    mc = ModelChain(system, location)
    times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
    ac = mc.run_model(times).ac

    expected = pd.Series(np.array([  121.421719,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)

    expected = pd.DataFrame(np.
        array([[ 54.82513187,  90.        ,  11.0039221 ,  11.0039221 ],
               [         nan,   0.        ,   0.        ,          nan]]),
        columns=['aoi', 'surface_azimuth', 'surface_tilt', 'tracker_theta'],
        index=times)
    assert_frame_equal(mc.tracking, expected)


@raises(ValueError)
def test_bad_get_orientation():
    modelchain.get_orientation('bad value')


@raises(ValueError)
def test_basic_chain_required():
    times = pd.DatetimeIndex(start='20160101 1200-0700',
                             end='20160101 1800-0700', freq='6H')
    latitude = 32
    longitude = -111
    altitude = 700

    module_parameters = get_sapm_module_parameters()
    inverter_parameters = get_cec_inverter_parameters()

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    altitude=altitude)


def test_basic_chain_alt_az():
    times = pd.DatetimeIndex(start='20160101 1200-0700',
                             end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    surface_tilt = 0
    surface_azimuth = 0

    module_parameters = get_sapm_module_parameters()
    inverter_parameters = get_cec_inverter_parameters()

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth)

    expected = pd.Series(np.array([1.14490928477e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_basic_chain_strategy():
    times = pd.DatetimeIndex(start='20160101 1200-0700',
                             end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    altitude = 700

    module_parameters = get_sapm_module_parameters()
    inverter_parameters = get_cec_inverter_parameters()

    dc, ac = modelchain.basic_chain(
        times, latitude, longitude, module_parameters, inverter_parameters,
        orientation_strategy='south_at_latitude_tilt', altitude=altitude)

    expected = pd.Series(np.array([1.82033563543e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)


def test_basic_chain_altitude_pressure():
    times = pd.DatetimeIndex(start='20160101 1200-0700',
                             end='20160101 1800-0700', freq='6H')
    latitude = 32.2
    longitude = -111
    altitude = 700
    surface_tilt = 0
    surface_azimuth = 0

    module_parameters = get_sapm_module_parameters()
    inverter_parameters = get_cec_inverter_parameters()

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth,
                                    pressure=93194)

    expected = pd.Series(np.array([1.15771428788e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)

    dc, ac = modelchain.basic_chain(times, latitude, longitude,
                                    module_parameters, inverter_parameters,
                                    surface_tilt=surface_tilt,
                                    surface_azimuth=surface_azimuth,
                                    altitude=altitude)

    expected = pd.Series(np.array([1.15771428788e+02,  -2.00000000e-02]),
                         index=times)
    assert_series_equal(ac, expected)
