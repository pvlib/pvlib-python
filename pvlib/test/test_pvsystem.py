import logging
pvl_logger = logging.getLogger('pvlib')

import inspect
import os
import datetime

import numpy as np
import pandas as pd

from nose.tools import assert_equals, assert_almost_equals
from pandas.util.testing import assert_series_equal, assert_frame_equal

from pvlib import tmy
from pvlib import pvsystem
from pvlib import clearsky
from pvlib import irradiance
from pvlib import atmosphere
from pvlib import solarposition
from pvlib.location import Location

tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
times = pd.date_range(start=datetime.datetime(2014,1,1),
                      end=datetime.datetime(2014,1,2), freq='1Min')
ephem_data = solarposition.get_solarposition(times, tus, method='pyephem')
irrad_data = clearsky.ineichen(times, tus, linke_turbidity=3,
                               solarposition_method='pyephem')
aoi = irradiance.aoi(0, 0, ephem_data['apparent_zenith'],
                     ephem_data['apparent_azimuth'])
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
    thetas = pd.Series(np.linspace(-180,180,361))
    iam = pvsystem.ashraeiam(.05, thetas)


def test_physicaliam():
    thetas = pd.Series(np.linspace(-180,180,361))
    iam = pvsystem.physicaliam(4, 0.002, 1.526, thetas)


# if this completes successfully we'll be able to do more tests below.
sam_data = {}
def test_retrieve_sam_network():
    sam_data['cecmod'] = pvsystem.retrieve_sam('cecmod')
    sam_data['sandiamod'] = pvsystem.retrieve_sam('sandiamod')
    sam_data['cecinverter'] = pvsystem.retrieve_sam('cecinverter')


def test_sapm():
    modules = sam_data['sandiamod']
    module = modules.Canadian_Solar_CS5P_220M___2009_

    sapm = pvsystem.sapm(module, irrad_data['dni'],
                         irrad_data['dhi'], 25, am, aoi)

    sapm = pvsystem.sapm(module.to_dict(), irrad_data['dni'],
                         irrad_data['dhi'], 25, am, aoi)


def test_calcparams_desoto():
    cecmodule = sam_data['cecmod'].Example_Module 
    pvsystem.calcparams_desoto(irrad_data['ghi'],
                               temp_cell=25,
                               alpha_isc=cecmodule['alpha_sc'],
                               module_parameters=cecmodule,
                               EgRef=1.121,
                               dEgdT=-0.0002677)


def test_i_from_v():
    output = pvsystem.i_from_v(20, .1, .5, 40, 6e-7, 7)
    assert_almost_equals(-299.746389916, output, 5)


def test_singlediode_series():  
    cecmodule = sam_data['cecmod'].Example_Module 
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                                         irrad_data['ghi'],
                                         temp_cell=25,
                                         alpha_isc=cecmodule['alpha_sc'],
                                         module_parameters=cecmodule,
                                         EgRef=1.121,
                                         dEgdT=-0.0002677)                       
    out = pvsystem.singlediode(cecmodule, IL, I0, Rs, Rsh, nNsVth)
    assert isinstance(out, pd.DataFrame)


def test_singlediode_series():  
    cecmodule = sam_data['cecmod'].Example_Module                       
    out = pvsystem.singlediode(cecmodule, 7, 6e-7, .1, 20, .5)
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


def test_sapm_celltemp():
    default = pvsystem.sapm_celltemp(900, 5, 20)
    assert_almost_equals(43.509, default.ix[0, 'temp_cell'], 3)
    assert_almost_equals(40.809, default.ix[0, 'temp_module'], 3)
    assert_frame_equal(default, pvsystem.sapm_celltemp(900, 5, 20,
                                                       [-3.47, -.0594, 3]))


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

    
def test_snlinverter():
    inverters = sam_data['cecinverter']
    testinv = 'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'
    vdcs = pd.Series(np.linspace(0,50,3))
    idcs = pd.Series(np.linspace(0,11,3))
    pdcs = idcs * vdcs

    pacs = pvsystem.snlinverter(inverters[testinv], vdcs, pdcs)
    assert_series_equal(pacs, pd.Series([-0.020000, 132.004308, 250.000000]))


def test_snlinverter_float():
    inverters = sam_data['cecinverter']
    testinv = 'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'
    vdcs = 25.
    idcs = 5.5
    pdcs = idcs * vdcs

    pacs = pvsystem.snlinverter(inverters[testinv], vdcs, pdcs)
    assert_almost_equals(pacs, 132.004278, 5)
    
