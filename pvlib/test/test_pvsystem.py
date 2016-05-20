import inspect
import os
import datetime

import numpy as np
from numpy import nan
import pandas as pd

from nose.tools import assert_equals, assert_almost_equals
from pandas.util.testing import assert_series_equal, assert_frame_equal
from . import incompatible_conda_linux_py3, incompatible_pandas_0180

from pvlib import tmy
from pvlib import pvsystem
from pvlib import clearsky
from pvlib import irradiance
from pvlib import atmosphere
from pvlib import solarposition
from pvlib.location import Location

latitude = 32.2
longitude = -111
tus = Location(latitude, longitude, 'US/Arizona', 700, 'Tucson')
times = pd.date_range(start=datetime.datetime(2014,1,1),
                      end=datetime.datetime(2014,1,2), freq='1Min')
ephem_data = solarposition.get_solarposition(times,
                                             latitude=latitude,
                                             longitude=longitude,
                                             method='nrel_numpy')
irrad_data = clearsky.ineichen(times, latitude=latitude, longitude=longitude,
                               linke_turbidity=3,
                               solarposition_method='nrel_numpy')
aoi = irradiance.aoi(0, 0, ephem_data['apparent_zenith'],
                     ephem_data['azimuth'])
am = atmosphere.relativeairmass(ephem_data.apparent_zenith)

meta = {'latitude': 37.8,
        'longitude': -122.3,
        'altitude': 10,
        'Name': 'Oakland',
        'State': 'CA',
        'TZ': -8}

pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(tmy)))

tmy3_testfile = os.path.join(pvlib_abspath, 'data', '703165TY.csv')
tmy2_testfile = os.path.join(pvlib_abspath, 'data', '12839.tm2')

tmy3_data, tmy3_metadata = tmy.readtmy3(tmy3_testfile)
tmy2_data, tmy2_metadata = tmy.readtmy2(tmy2_testfile)

def test_systemdef_tmy3():
    expected = {'tz': -9.0,
                'albedo': 0.1,
                'altitude': 7.0,
                'latitude': 55.317,
                'longitude': -160.517,
                'name': '"SAND POINT"',
                'parallel_modules': 5,
                'series_modules': 5,
                'surface_azimuth': 0,
                'surface_tilt': 0}
    assert_equals(expected, pvsystem.systemdef(tmy3_metadata, 0, 0, .1, 5, 5))

def test_systemdef_tmy2():
    expected = {'tz': -5,
                'albedo': 0.1,
                'altitude': 2.0,
                'latitude': 25.8,
                'longitude': -80.26666666666667,
                'name': 'MIAMI',
                'parallel_modules': 5,
                'series_modules': 5,
                'surface_azimuth': 0,
                'surface_tilt': 0}
    assert_equals(expected, pvsystem.systemdef(tmy2_metadata, 0, 0, .1, 5, 5))

def test_systemdef_dict():
    expected = {'tz': -8, ## Note that TZ is float, but Location sets tz as string
                'albedo': 0.1,
                'altitude': 10,
                'latitude': 37.8,
                'longitude': -122.3,
                'name': 'Oakland',
                'parallel_modules': 5,
                'series_modules': 5,
                'surface_azimuth': 0,
                'surface_tilt': 5}
    assert_equals(expected, pvsystem.systemdef(meta, 5, 0, .1, 5, 5))


def test_ashraeiam():
    thetas = np.linspace(-90, 90, 9)
    iam = pvsystem.ashraeiam(.05, thetas)
    expected = np.array([        nan,  0.9193437 ,  0.97928932,  0.99588039,  1.        ,
        0.99588039,  0.97928932,  0.9193437 ,         nan])
    assert np.isclose(iam, expected, equal_nan=True).all()


def test_PVSystem_ashraeiam():
    module_parameters = pd.Series({'b': 0.05})
    system = pvsystem.PVSystem(module='blah', inverter='blarg',
                               module_parameters=module_parameters)
    thetas = np.linspace(-90, 90, 9)
    iam = system.ashraeiam(thetas)
    expected = np.array([        nan,  0.9193437 ,  0.97928932,  0.99588039,  1.        ,
        0.99588039,  0.97928932,  0.9193437 ,         nan])
    assert np.isclose(iam, expected, equal_nan=True).all()


def test_physicaliam():
    thetas = np.linspace(-90, 90, 9)
    iam = pvsystem.physicaliam(4, 0.002, 1.526, thetas)
    expected = np.array([        nan,  0.8893998 ,  0.98797788,  0.99926198,         nan,
        0.99926198,  0.98797788,  0.8893998 ,         nan])
    assert np.isclose(iam, expected, equal_nan=True).all()


def test_PVSystem_physicaliam():
    module_parameters = pd.Series({'K': 4, 'L': 0.002, 'n': 1.526})
    system = pvsystem.PVSystem(module='blah', inverter='blarg',
                               module_parameters=module_parameters)
    thetas = np.linspace(-90, 90, 9)
    iam = system.physicaliam(thetas)
    expected = np.array([        nan,  0.8893998 ,  0.98797788,  0.99926198,         nan,
        0.99926198,  0.98797788,  0.8893998 ,         nan])
    assert np.isclose(iam, expected, equal_nan=True).all()


# if this completes successfully we'll be able to do more tests below.
sam_data = {}
def test_retrieve_sam_network():
    sam_data['cecmod'] = pvsystem.retrieve_sam('cecmod')
    sam_data['sandiamod'] = pvsystem.retrieve_sam('sandiamod')
    sam_data['cecinverter'] = pvsystem.retrieve_sam('cecinverter')


def test_sapm():
    modules = sam_data['sandiamod']
    module_parameters = modules['Canadian_Solar_CS5P_220M___2009_']
    times = pd.DatetimeIndex(start='2015-01-01', periods=2, freq='12H')
    irrad_data = pd.DataFrame({'dni':[0,1000], 'ghi':[0,600], 'dhi':[0,100]},
                              index=times)
    am = pd.Series([0, 2.25], index=times)
    aoi = pd.Series([180, 30], index=times)

    sapm = pvsystem.sapm(module_parameters, irrad_data['dni'],
                         irrad_data['dhi'], 25, am, aoi)

    expected = pd.DataFrame(np.array(
    [[   0.        ,    0.        ,    0.        ,    0.        ,
           0.        ,    0.        ,    0.        ,    0.        ],
       [   5.74526799,    5.12194115,   59.67914031,   48.41924255,
         248.00051089,    5.61787615,    3.52581308,    1.12848138]]),
        columns=['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx',
                 'effective_irradiance'],
        index=times)

    assert_frame_equal(sapm, expected)

    # just make sure it works with a dict input
    sapm = pvsystem.sapm(module_parameters.to_dict(), irrad_data['dni'],
                         irrad_data['dhi'], 25, am, aoi)


def test_PVSystem_sapm():
    modules = sam_data['sandiamod']
    module = 'Canadian_Solar_CS5P_220M___2009_'
    module_parameters = modules[module]
    system = pvsystem.PVSystem(module=module,
                               module_parameters=module_parameters)
    times = pd.DatetimeIndex(start='2015-01-01', periods=2, freq='12H')
    irrad_data = pd.DataFrame({'dni':[0,1000], 'ghi':[0,600], 'dhi':[0,100]},
                              index=times)
    am = pd.Series([0, 2.25], index=times)
    aoi = pd.Series([180, 30], index=times)

    sapm = system.sapm(irrad_data['dni'], irrad_data['dhi'], 25, am, aoi)

    expected = pd.DataFrame(np.array(
    [[   0.        ,    0.        ,    0.        ,    0.        ,
           0.        ,    0.        ,    0.        ,    0.        ],
       [   5.74526799,    5.12194115,   59.67914031,   48.41924255,
         248.00051089,    5.61787615,    3.52581308,    1.12848138]]),
        columns=['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx',
                 'effective_irradiance'],
        index=times)

    assert_frame_equal(sapm, expected)


@incompatible_pandas_0180
def test_calcparams_desoto():
    module = 'Example_Module'
    module_parameters = sam_data['cecmod'][module]
    times = pd.DatetimeIndex(start='2015-01-01', periods=2, freq='12H')
    poa_data = pd.Series([0, 800], index=times)

    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                                  poa_data,
                                  temp_cell=25,
                                  alpha_isc=module_parameters['alpha_sc'],
                                  module_parameters=module_parameters,
                                  EgRef=1.121,
                                  dEgdT=-0.0002677)

    assert_series_equal(np.round(IL, 3), pd.Series([0.0, 6.036], index=times))
    assert_almost_equals(I0, 1.943e-9)
    assert_almost_equals(Rs, 0.094)
    assert_series_equal(np.round(Rsh, 3), pd.Series([np.inf, 19.65], index=times))
    assert_almost_equals(nNsVth, 0.473)


@incompatible_pandas_0180
def test_PVSystem_calcparams_desoto():
    module = 'Example_Module'
    module_parameters = sam_data['cecmod'][module].copy()
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    system = pvsystem.PVSystem(module=module,
                               module_parameters=module_parameters)
    times = pd.DatetimeIndex(start='2015-01-01', periods=2, freq='12H')
    poa_data = pd.Series([0, 800], index=times)
    temp_cell = 25

    IL, I0, Rs, Rsh, nNsVth = system.calcparams_desoto(poa_data, temp_cell)

    assert_series_equal(np.round(IL, 3), pd.Series([0.0, 6.036], index=times))
    assert_almost_equals(I0, 1.943e-9)
    assert_almost_equals(Rs, 0.094)
    assert_series_equal(np.round(Rsh, 3), pd.Series([np.inf, 19.65], index=times))
    assert_almost_equals(nNsVth, 0.473)


@incompatible_conda_linux_py3
def test_i_from_v():
    output = pvsystem.i_from_v(20, .1, .5, 40, 6e-7, 7)
    assert_almost_equals(-299.746389916, output, 5)


@incompatible_conda_linux_py3
def test_PVSystem_i_from_v():
    module = 'Example_Module'
    module_parameters = sam_data['cecmod'][module]
    system = pvsystem.PVSystem(module=module,
                               module_parameters=module_parameters)
    output = system.i_from_v(20, .1, .5, 40, 6e-7, 7)
    assert_almost_equals(-299.746389916, output, 5)


def test_singlediode_series():
    module = 'Example_Module'
    module_parameters = sam_data['cecmod'][module]
    times = pd.DatetimeIndex(start='2015-01-01', periods=2, freq='12H')
    poa_data = pd.Series([0, 800], index=times)
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                                         poa_data,
                                         temp_cell=25,
                                         alpha_isc=module_parameters['alpha_sc'],
                                         module_parameters=module_parameters,
                                         EgRef=1.121,
                                         dEgdT=-0.0002677)
    out = pvsystem.singlediode(module_parameters, IL, I0, Rs, Rsh, nNsVth)
    assert isinstance(out, pd.DataFrame)


@incompatible_conda_linux_py3
def test_singlediode_floats():
    module = 'Example_Module'
    module_parameters = sam_data['cecmod'][module]
    out = pvsystem.singlediode(module_parameters, 7, 6e-7, .1, 20, .5)
    expected = {'i_xx': 4.2549732697234193,
                'i_mp': 6.1390251797935704,
                'v_oc': 8.1147298764528042,
                'p_mp': 38.194165464983037,
                'i_x': 6.7556075876880621,
                'i_sc': 6.9646747613963198,
                'v_mp': 6.221535886625464}
    assert isinstance(out, dict)
    for k, v in out.items():
        assert_almost_equals(expected[k], v, 5)


@incompatible_conda_linux_py3
def test_PVSystem_singlediode_floats():
    module = 'Example_Module'
    module_parameters = sam_data['cecmod'][module]
    system = pvsystem.PVSystem(module=module,
                               module_parameters=module_parameters)
    out = system.singlediode(7, 6e-7, .1, 20, .5)
    expected = {'i_xx': 4.2549732697234193,
                'i_mp': 6.1390251797935704,
                'v_oc': 8.1147298764528042,
                'p_mp': 38.194165464983037,
                'i_x': 6.7556075876880621,
                'i_sc': 6.9646747613963198,
                'v_mp': 6.221535886625464}
    assert isinstance(out, dict)
    for k, v in out.items():
        assert_almost_equals(expected[k], v, 5)


def test_scale_voltage_current_power():
    data = pd.DataFrame(
        np.array([[2, 1.5, 10, 8, 12, 0.5, 1.5]]),
        columns=['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx'],
        index=[0])
    expected = pd.DataFrame(
        np.array([[6, 4.5, 20, 16, 72, 1.5, 4.5]]),
        columns=['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx'],
        index=[0])
    out = pvsystem.scale_voltage_current_power(data, voltage=2, current=3)


def test_PVSystem_scale_voltage_current_power():
    data = pd.DataFrame(
        np.array([[2, 1.5, 10, 8, 12, 0.5, 1.5]]),
        columns=['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx'],
        index=[0])
    expected = pd.DataFrame(
        np.array([[6, 4.5, 20, 16, 72, 1.5, 4.5]]),
        columns=['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx'],
        index=[0])
    system = pvsystem.PVSystem(series_modules=2, parallel_modules=3)
    out = system.scale_voltage_current_power(data)


def test_sapm_celltemp():
    default = pvsystem.sapm_celltemp(900, 5, 20)
    assert_almost_equals(43.509, default.ix[0, 'temp_cell'], 3)
    assert_almost_equals(40.809, default.ix[0, 'temp_module'], 3)
    assert_frame_equal(default, pvsystem.sapm_celltemp(900, 5, 20,
                                                       [-3.47, -.0594, 3]))


def test_sapm_celltemp_dict_like():
    default = pvsystem.sapm_celltemp(900, 5, 20)
    assert_almost_equals(43.509, default.ix[0, 'temp_cell'], 3)
    assert_almost_equals(40.809, default.ix[0, 'temp_module'], 3)
    model = {'a':-3.47, 'b':-.0594, 'deltaT':3}
    assert_frame_equal(default, pvsystem.sapm_celltemp(900, 5, 20, model))
    model = pd.Series(model)
    assert_frame_equal(default, pvsystem.sapm_celltemp(900, 5, 20, model))


def test_sapm_celltemp_with_index():
    times = pd.DatetimeIndex(start='2015-01-01', end='2015-01-02', freq='12H')
    temps = pd.Series([0, 10, 5], index=times)
    irrads = pd.Series([0, 500, 0], index=times)
    winds = pd.Series([10, 5, 0], index=times)

    pvtemps = pvsystem.sapm_celltemp(irrads, winds, temps)

    expected = pd.DataFrame({'temp_cell':[0., 23.06066166, 5.],
                             'temp_module':[0., 21.56066166, 5.]},
                            index=times)

    assert_frame_equal(expected, pvtemps)


def test_PVSystem_sapm_celltemp():
    system = pvsystem.PVSystem(racking_model='roof_mount_cell_glassback')
    times = pd.DatetimeIndex(start='2015-01-01', end='2015-01-02', freq='12H')
    temps = pd.Series([0, 10, 5], index=times)
    irrads = pd.Series([0, 500, 0], index=times)
    winds = pd.Series([10, 5, 0], index=times)

    pvtemps = system.sapm_celltemp(irrads, winds, temps)

    expected = pd.DataFrame({'temp_cell':[0., 30.56763059, 5.],
                             'temp_module':[0., 30.06763059, 5.]},
                            index=times)

    assert_frame_equal(expected, pvtemps)


def test_snlinverter():
    inverters = sam_data['cecinverter']
    testinv = 'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'
    vdcs = pd.Series(np.linspace(0,50,3))
    idcs = pd.Series(np.linspace(0,11,3))
    pdcs = idcs * vdcs

    pacs = pvsystem.snlinverter(inverters[testinv], vdcs, pdcs)
    assert_series_equal(pacs, pd.Series([-0.020000, 132.004308, 250.000000]))


def test_PVSystem_snlinverter():
    inverters = sam_data['cecinverter']
    testinv = 'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'
    system = pvsystem.PVSystem(inverter=testinv,
                               inverter_parameters=inverters[testinv])
    vdcs = pd.Series(np.linspace(0,50,3))
    idcs = pd.Series(np.linspace(0,11,3))
    pdcs = idcs * vdcs

    pacs = system.snlinverter(vdcs, pdcs)
    assert_series_equal(pacs, pd.Series([-0.020000, 132.004308, 250.000000]))


def test_snlinverter_float():
    inverters = sam_data['cecinverter']
    testinv = 'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'
    vdcs = 25.
    idcs = 5.5
    pdcs = idcs * vdcs

    pacs = pvsystem.snlinverter(inverters[testinv], vdcs, pdcs)
    assert_almost_equals(pacs, 132.004278, 5)


def test_snlinverter_Pnt_micro():
    inverters = sam_data['cecinverter']
    testinv = 'Enphase_Energy__M250_60_2LL_S2x___ZC____NA__208V_208V__CEC_2013_'
    vdcs = pd.Series(np.linspace(0,50,3))
    idcs = pd.Series(np.linspace(0,11,3))
    pdcs = idcs * vdcs

    pacs = pvsystem.snlinverter(inverters[testinv], vdcs, pdcs)
    assert_series_equal(pacs, pd.Series([-0.043000, 132.545914746, 240.000000]))


def test_PVSystem_creation():
    pv_system = pvsystem.PVSystem(module='blah', inverter='blarg')


def test_PVSystem_get_aoi():
    system = pvsystem.PVSystem(surface_tilt=32, surface_azimuth=135)
    aoi = system.get_aoi(30, 225)
    assert np.round(aoi, 4) == 42.7408


def test_PVSystem_get_irradiance():
    system = pvsystem.PVSystem(surface_tilt=32, surface_azimuth=135)
    times = pd.DatetimeIndex(start='20160101 1200-0700',
                             end='20160101 1800-0700', freq='6H')
    location = Location(latitude=32, longitude=-111)
    solar_position = location.get_solarposition(times)
    irrads = pd.DataFrame({'dni':[900,0], 'ghi':[600,0], 'dhi':[100,0]},
                          index=times)

    irradiance = system.get_irradiance(solar_position['apparent_zenith'],
                                       solar_position['azimuth'],
                                       irrads['dni'],
                                       irrads['ghi'],
                                       irrads['dhi'])

    expected = pd.DataFrame(data=np.array(
        [[ 883.65494055,  745.86141676,  137.79352379,  126.397131  ,
              11.39639279],
           [   0.        ,   -0.        ,    0.        ,    0.        ,    0.        ]]),
                            columns=['poa_global', 'poa_direct',
                                     'poa_diffuse', 'poa_sky_diffuse',
                                     'poa_ground_diffuse'],
                            index=times)

    irradiance = np.round(irradiance, 4)
    expected = np.round(expected, 4)
    assert_frame_equal(irradiance, expected)


def test_PVSystem_localize_with_location():
    system = pvsystem.PVSystem(module='blah', inverter='blarg')
    location = Location(latitude=32, longitude=-111)
    localized_system = system.localize(location=location)

    assert localized_system.module == 'blah'
    assert localized_system.inverter == 'blarg'
    assert localized_system.latitude == 32
    assert localized_system.longitude == -111


def test_PVSystem_localize_with_latlon():
    system = pvsystem.PVSystem(module='blah', inverter='blarg')
    localized_system = system.localize(latitude=32, longitude=-111)

    assert localized_system.module == 'blah'
    assert localized_system.inverter == 'blarg'
    assert localized_system.latitude == 32
    assert localized_system.longitude == -111


# we could retest each of the models tested above
# when they are attached to LocalizedPVSystem, but
# that's probably not necessary at this point.


def test_LocalizedPVSystem_creation():
    localized_system = pvsystem.LocalizedPVSystem(latitude=32,
                                                  longitude=-111,
                                                  module='blah',
                                                  inverter='blarg')

    assert localized_system.module == 'blah'
    assert localized_system.inverter == 'blarg'
    assert localized_system.latitude == 32
    assert localized_system.longitude == -111
