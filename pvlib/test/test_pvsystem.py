import inspect
import os
import datetime
from collections import OrderedDict

import numpy as np
from numpy import nan, array
import pandas as pd

import pytest
from pandas.util.testing import assert_series_equal, assert_frame_equal
from numpy.testing import assert_allclose

from pvlib import tmy
from pvlib import pvsystem
from pvlib import clearsky
from pvlib import irradiance
from pvlib import atmosphere
from pvlib import solarposition
from pvlib.location import Location

from conftest import needs_numpy_1_10, requires_scipy

latitude = 32.2
longitude = -111
tus = Location(latitude, longitude, 'US/Arizona', 700, 'Tucson')
times = pd.date_range(start=datetime.datetime(2014,1,1),
                      end=datetime.datetime(2014,1,2), freq='1Min')
ephem_data = solarposition.get_solarposition(times,
                                             latitude=latitude,
                                             longitude=longitude,
                                             method='nrel_numpy')
am = atmosphere.relativeairmass(ephem_data.apparent_zenith)
irrad_data = clearsky.ineichen(ephem_data['apparent_zenith'], am,
                               linke_turbidity=3)
aoi = irradiance.aoi(0, 0, ephem_data['apparent_zenith'],
                     ephem_data['azimuth'])


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
                'strings_per_inverter': 5,
                'modules_per_string': 5,
                'surface_azimuth': 0,
                'surface_tilt': 0}
    assert expected == pvsystem.systemdef(tmy3_metadata, 0, 0, .1, 5, 5)

def test_systemdef_tmy2():
    expected = {'tz': -5,
                'albedo': 0.1,
                'altitude': 2.0,
                'latitude': 25.8,
                'longitude': -80.26666666666667,
                'name': 'MIAMI',
                'strings_per_inverter': 5,
                'modules_per_string': 5,
                'surface_azimuth': 0,
                'surface_tilt': 0}
    assert expected == pvsystem.systemdef(tmy2_metadata, 0, 0, .1, 5, 5)

def test_systemdef_dict():
    expected = {'tz': -8, ## Note that TZ is float, but Location sets tz as string
                'albedo': 0.1,
                'altitude': 10,
                'latitude': 37.8,
                'longitude': -122.3,
                'name': 'Oakland',
                'strings_per_inverter': 5,
                'modules_per_string': 5,
                'surface_azimuth': 0,
                'surface_tilt': 5}
    assert expected == pvsystem.systemdef(meta, 5, 0, .1, 5, 5)


@needs_numpy_1_10
def test_ashraeiam():
    thetas = np.array([-90. , -67.5, -45. , -22.5,   0. ,  22.5,  45. ,  67.5, 89.,  90. , np.nan])
    iam = pvsystem.ashraeiam(thetas, .05)
    expected = np.array([        0,  0.9193437 ,  0.97928932,  0.99588039,  1.        ,
        0.99588039,  0.97928932,  0.9193437 ,         0, 0,  np.nan])
    assert_allclose(iam, expected, equal_nan=True)


@needs_numpy_1_10
def test_ashraeiam_scalar():
    thetas = -45.
    iam = pvsystem.ashraeiam(thetas, .05)
    expected = 0.97928932
    assert_allclose(iam, expected, equal_nan=True)
    thetas = np.nan
    iam = pvsystem.ashraeiam(thetas, .05)
    expected = np.nan
    assert_allclose(iam, expected, equal_nan=True)


@needs_numpy_1_10
def test_PVSystem_ashraeiam():
    module_parameters = pd.Series({'b': 0.05})
    system = pvsystem.PVSystem(module_parameters=module_parameters)
    thetas = np.array([-90. , -67.5, -45. , -22.5,   0. ,  22.5,  45. ,  67.5,  89., 90. , np.nan])
    iam = system.ashraeiam(thetas)
    expected = np.array([        0,  0.9193437 ,  0.97928932,  0.99588039,  1.        ,
        0.99588039,  0.97928932,  0.9193437 ,         0, 0,  np.nan])
    assert_allclose(iam, expected, equal_nan=True)


@needs_numpy_1_10
def test_physicaliam():
    aoi = np.array([-90. , -67.5, -45. , -22.5,   0. ,  22.5,  45. ,  67.5,  90. , np.nan])
    iam = pvsystem.physicaliam(aoi, 1.526, 0.002, 4)
    expected = np.array([        0,  0.8893998,  0.98797788,  0.99926198,         1,
        0.99926198,  0.98797788,  0.8893998,         0, np.nan])
    assert_allclose(iam, expected, equal_nan=True)

    # GitHub issue 397
    aoi = pd.Series(aoi)
    iam = pvsystem.physicaliam(aoi, 1.526, 0.002, 4)
    expected = pd.Series(expected)
    assert_series_equal(iam, expected)


@needs_numpy_1_10
def test_physicaliam_scalar():
    aoi = -45.
    iam = pvsystem.physicaliam(aoi, 1.526, 0.002, 4)
    expected = 0.98797788
    assert_allclose(iam, expected, equal_nan=True)
    aoi = np.nan
    iam = pvsystem.physicaliam(aoi, 1.526, 0.002, 4)
    expected = np.nan
    assert_allclose(iam, expected, equal_nan=True)


@needs_numpy_1_10
def test_PVSystem_physicaliam():
    module_parameters = pd.Series({'K': 4, 'L': 0.002, 'n': 1.526})
    system = pvsystem.PVSystem(module_parameters=module_parameters)
    thetas = np.array([-90. , -67.5, -45. , -22.5,   0. ,  22.5,  45. ,  67.5,  90. , np.nan])
    iam = system.physicaliam(thetas)
    expected = np.array([        0,  0.8893998 ,  0.98797788,  0.99926198,         1,
        0.99926198,  0.98797788,  0.8893998 ,         0, np.nan])
    assert_allclose(iam, expected, equal_nan=True)


# if this completes successfully we'll be able to do more tests below.
@pytest.fixture(scope="session")
def sam_data():
    data = {}
    data['cecmod'] = pvsystem.retrieve_sam('cecmod')
    data['sandiamod'] = pvsystem.retrieve_sam('sandiamod')
    data['cecinverter'] = pvsystem.retrieve_sam('cecinverter')
    data['adrinverter'] = pvsystem.retrieve_sam('adrinverter')
    return data


@pytest.fixture(scope="session")
def sapm_module_params(sam_data):
    modules = sam_data['sandiamod']
    module = 'Canadian_Solar_CS5P_220M___2009_'
    module_parameters = modules[module]
    return module_parameters


@pytest.fixture(scope="session")
def cec_module_params(sam_data):
    modules = sam_data['cecmod']
    module = 'Example_Module'
    module_parameters = modules[module]
    return module_parameters


def test_sapm(sapm_module_params):

    times = pd.DatetimeIndex(start='2015-01-01', periods=5, freq='12H')
    effective_irradiance = pd.Series([-1, 0.5, 1.1, np.nan, 1], index=times)
    temp_cell = pd.Series([10, 25, 50, 25, np.nan], index=times)

    out = pvsystem.sapm(effective_irradiance, temp_cell, sapm_module_params)

    expected = pd.DataFrame(np.array(
      [[  -5.0608322 ,   -4.65037767,           nan,           nan,
                  nan,   -4.91119927,   -4.15367716],
       [   2.545575  ,    2.28773882,   56.86182059,   47.21121608,
         108.00693168,    2.48357383,    1.71782772],
       [   5.65584763,    5.01709903,   54.1943277 ,   42.51861718,
         213.32011294,    5.52987899,    3.48660728],
       [          nan,           nan,           nan,           nan,
                  nan,           nan,           nan],
       [          nan,           nan,           nan,           nan,
                  nan,           nan,           nan]]),
        columns=['i_sc', 'i_mp', 'v_oc', 'v_mp', 'p_mp', 'i_x', 'i_xx'],
        index=times)

    assert_frame_equal(out, expected, check_less_precise=4)

    out = pvsystem.sapm(1, 25, sapm_module_params)

    expected = OrderedDict()
    expected['i_sc'] = 5.09115
    expected['i_mp'] = 4.5462909092579995
    expected['v_oc'] = 59.260800000000003
    expected['v_mp'] = 48.315600000000003
    expected['p_mp'] = 219.65677305534581
    expected['i_x'] = 4.9759899999999995
    expected['i_xx'] = 3.1880204359100004

    for k, v in expected.items():
        assert_allclose(out[k], v, atol=1e-4)

    # just make sure it works with a dict input
    pvsystem.sapm(effective_irradiance, temp_cell,
                  sapm_module_params.to_dict())


def test_PVSystem_sapm(sapm_module_params):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)

    times = pd.DatetimeIndex(start='2015-01-01', periods=5, freq='12H')
    effective_irradiance = pd.Series([-1, 0.5, 1.1, np.nan, 1], index=times)
    temp_cell = pd.Series([10, 25, 50, 25, np.nan], index=times)

    out = system.sapm(effective_irradiance, temp_cell)


@pytest.mark.parametrize('airmass,expected', [
    (1.5, 1.00028714375),
    (np.array([[10, np.nan]]), np.array([[0.999535, 0]])),
    (pd.Series([5]), pd.Series([1.0387675]))
])
def test_sapm_spectral_loss(sapm_module_params, airmass, expected):

    out = pvsystem.sapm_spectral_loss(airmass, sapm_module_params)

    if isinstance(airmass, pd.Series):
        assert_series_equal(out, expected, check_less_precise=4)
    else:
        assert_allclose(out, expected, atol=1e-4)


def test_PVSystem_sapm_spectral_loss(sapm_module_params):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)

    times = pd.DatetimeIndex(start='2015-01-01', periods=2, freq='12H')
    airmass = pd.Series([1, 10], index=times)

    out = system.sapm_spectral_loss(airmass)


@pytest.mark.parametrize('aoi,expected', [
    (45, 0.9975036250000002),
    (np.array([[-30, 30, 100, np.nan]]),
     np.array([[0, 1.007572, 0, np.nan]])),
    (pd.Series([80]), pd.Series([0.597472]))
])
def test_sapm_aoi_loss(sapm_module_params, aoi, expected):

    out = pvsystem.sapm_aoi_loss(aoi, sapm_module_params)

    if isinstance(aoi, pd.Series):
        assert_series_equal(out, expected, check_less_precise=4)
    else:
        assert_allclose(out, expected, atol=1e-4)


def test_sapm_aoi_loss_limits():
    module_parameters = {'B0': 5, 'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0}
    assert pvsystem.sapm_aoi_loss(1, module_parameters) == 5

    module_parameters = {'B0': 5, 'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0}
    assert pvsystem.sapm_aoi_loss(1, module_parameters, upper=1) == 1

    module_parameters = {'B0': -5, 'B1': 0, 'B2': 0, 'B3': 0, 'B4': 0, 'B5': 0}
    assert pvsystem.sapm_aoi_loss(1, module_parameters) == 0


def test_PVSystem_sapm_aoi_loss(sapm_module_params):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)

    times = pd.DatetimeIndex(start='2015-01-01', periods=2, freq='12H')
    aoi = pd.Series([45, 10], index=times)

    out = system.sapm_aoi_loss(aoi)


@pytest.mark.parametrize('test_input,expected', [
    ([1000, 100, 5, 45, 1000], 1.1400510967821877),
    ([np.array([np.nan, 1000, 1000]),
      np.array([100, np.nan, 100]),
      np.array([1.1, 1.1, 1.1]),
      np.array([10, 10, 10]),
      1000],
     np.array([np.nan, np.nan, 1.081157])),
    ([pd.Series([1000]), pd.Series([100]), pd.Series([1.1]),
      pd.Series([10]), 1370],
     pd.Series([0.789166]))
])
def test_sapm_effective_irradiance(sapm_module_params, test_input, expected):

    try:
        kwargs = {'reference_irradiance': test_input[4]}
        test_input = test_input[:-1]
    except IndexError:
        kwargs = {}

    test_input.append(sapm_module_params)

    out = pvsystem.sapm_effective_irradiance(*test_input, **kwargs)

    if isinstance(test_input, pd.Series):
        assert_series_equal(out, expected, check_less_precise=4)
    else:
        assert_allclose(out, expected, atol=1e-4)


def test_PVSystem_sapm_effective_irradiance(sapm_module_params):
    system = pvsystem.PVSystem(module_parameters=sapm_module_params)

    poa_direct = np.array([np.nan, 1000, 1000])
    poa_diffuse = np.array([100, np.nan, 100])
    airmass_absolute = np.array([1.1, 1.1, 1.1])
    aoi = np.array([10, 10, 10])
    reference_irradiance = 1000

    out = system.sapm_effective_irradiance(
        poa_direct, poa_diffuse, airmass_absolute,
        aoi, reference_irradiance=reference_irradiance)


def test_calcparams_desoto(cec_module_params):
    times = pd.DatetimeIndex(start='2015-01-01', periods=2, freq='12H')
    poa_data = pd.Series([0, 800], index=times)

    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                                  poa_data,
                                  temp_cell=25,
                                  alpha_isc=cec_module_params['alpha_sc'],
                                  module_parameters=cec_module_params,
                                  EgRef=1.121,
                                  dEgdT=-0.0002677)

    assert_series_equal(np.round(IL, 3), pd.Series([0.0, 6.036], index=times))
    # changed value in GH 444 for 2017-6-5 module file
    assert_allclose(I0, 1.94e-9)
    assert_allclose(Rs, 0.094)
    assert_series_equal(np.round(Rsh, 3), pd.Series([np.inf, 19.65], index=times))
    assert_allclose(nNsVth, 0.473)


def test_PVSystem_calcparams_desoto(cec_module_params):
    module_parameters = cec_module_params.copy()
    module_parameters['EgRef'] = 1.121
    module_parameters['dEgdT'] = -0.0002677
    system = pvsystem.PVSystem(module_parameters=module_parameters)
    times = pd.DatetimeIndex(start='2015-01-01', periods=2, freq='12H')
    poa_data = pd.Series([0, 800], index=times)
    temp_cell = 25

    IL, I0, Rs, Rsh, nNsVth = system.calcparams_desoto(poa_data, temp_cell)

    assert_series_equal(np.round(IL, 3), pd.Series([0.0, 6.036], index=times))
    # changed value in GH 444 for 2017-6-5 module file
    assert_allclose(I0, 1.94e-9)
    assert_allclose(Rs, 0.094)
    assert_series_equal(np.round(Rsh, 3), pd.Series([np.inf, 19.65], index=times))
    assert_allclose(nNsVth, 0.473)


@pytest.fixture(params=[
    {  # Can handle all python scalar inputs
     'Rsh': 20.,
     'Rs': 0.1,
     'nNsVth': 0.5,
     'I': 3.,
     'I0': 6.e-7,
     'IL': 7.,
     'V_expected': 7.5049875193450521
    },
    {  # Can handle all rank-0 array inputs
     'Rsh': np.array(20.),
     'Rs': np.array(0.1),
     'nNsVth': np.array(0.5),
     'I': np.array(3.),
     'I0': np.array(6.e-7),
     'IL': np.array(7.),
     'V_expected': np.array(7.5049875193450521)
    },
    {  # Can handle all rank-1 singleton array inputs
     'Rsh': np.array([20.]),
     'Rs': np.array([0.1]),
     'nNsVth': np.array([0.5]),
     'I': np.array([3.]),
     'I0': np.array([6.e-7]),
     'IL': np.array([7.]),
     'V_expected': np.array([7.5049875193450521])
    },
    {  # Can handle all rank-1 non-singleton array inputs with infinite shunt
      #  resistance, Rsh=inf gives V=Voc=nNsVth*(np.log(IL + I0) - np.log(I0)
      #  at I=0
      'Rsh': np.array([np.inf, 20.]),
      'Rs': np.array([0.1, 0.1]),
      'nNsVth': np.array([0.5, 0.5]),
      'I': np.array([0., 3.]),
      'I0': np.array([6.e-7, 6.e-7]),
      'IL': np.array([7., 7.]),
      'V_expected': np.array([0.5*(np.log(7. + 6.e-7) - np.log(6.e-7)),
                             7.5049875193450521])
    },
    {  # Can handle mixed inputs with a rank-2 array with infinite shunt
      #  resistance, Rsh=inf gives V=Voc=nNsVth*(np.log(IL + I0) - np.log(I0)
      #  at I=0
      'Rsh': np.array([[np.inf, np.inf], [np.inf, np.inf]]),
      'Rs': np.array([0.1]),
      'nNsVth': np.array(0.5),
      'I': 0.,
      'I0': np.array([6.e-7]),
      'IL': np.array([7.]),
      'V_expected': 0.5*(np.log(7. + 6.e-7) - np.log(6.e-7))*np.ones((2, 2))
    },
    {  # Can handle ideal series and shunt, Rsh=inf and Rs=0 give
      #  V = nNsVth*(np.log(IL - I + I0) - np.log(I0))
      'Rsh': np.inf,
      'Rs': 0.,
      'nNsVth': 0.5,
      'I': np.array([7., 7./2., 0.]),
      'I0': 6.e-7,
      'IL': 7.,
      'V_expected': np.array([0., 0.5*(np.log(7. - 7./2. + 6.e-7) -
                              np.log(6.e-7)), 0.5*(np.log(7. + 6.e-7) -
                              np.log(6.e-7))])
    },
    {  # Can handle only ideal series resistance, no closed form solution
      'Rsh': 20.,
      'Rs': 0.,
      'nNsVth': 0.5,
      'I': 3.,
      'I0': 6.e-7,
      'IL': 7.,
      'V_expected': 7.804987519345062
    },
    {  # Can handle all python scalar inputs with big LambertW arg
      'Rsh': 500.,
      'Rs': 10.,
      'nNsVth': 4.06,
      'I': 0.,
      'I0': 6.e-10,
      'IL': 1.2,
      'V_expected': 86.320000493521079
    },
    {  # Can handle all python scalar inputs with bigger LambertW arg
      #  1000 W/m^2 on a Canadian Solar 220M with 20 C ambient temp
      #  github issue 225 (this appears to be from PR 226 not issue 225)
      'Rsh': 190.,
      'Rs': 1.065,
      'nNsVth': 2.89,
      'I': 0.,
      'I0': 7.05196029e-08,
      'IL': 10.491262,
      'V_expected': 54.303958833791455
    },
    {  # Can handle all python scalar inputs with bigger LambertW arg
      #  1000 W/m^2 on a Canadian Solar 220M with 20 C ambient temp
      #  github issue 225
      'Rsh': 381.68,
      'Rs': 1.065,
      'nNsVth': 2.681527737715915,
      'I': 0.,
      'I0': 1.8739027472625636e-09,
      'IL': 5.1366949999999996,
      'V_expected': 58.19323124611128
    },
    {  # Verify mixed solution type indexing logic
      'Rsh': np.array([np.inf, 190., 381.68]),
      'Rs': 1.065,
      'nNsVth': np.array([2.89, 2.89, 2.681527737715915]),
      'I': 0.,
      'I0': np.array([7.05196029e-08, 7.05196029e-08, 1.8739027472625636e-09]),
      'IL': np.array([10.491262, 10.491262, 5.1366949999999996]),
      'V_expected': np.array([2.89*np.log1p(10.491262/7.05196029e-08),
                              54.303958833791455, 58.19323124611128])
    }])
def fixture_v_from_i(request):
    return request.param


@requires_scipy
def test_v_from_i(fixture_v_from_i):
    # Solution set loaded from fixture
    Rsh = fixture_v_from_i['Rsh']
    Rs = fixture_v_from_i['Rs']
    nNsVth = fixture_v_from_i['nNsVth']
    I = fixture_v_from_i['I']
    I0 = fixture_v_from_i['I0']
    IL = fixture_v_from_i['IL']
    V_expected = fixture_v_from_i['V_expected']

    # Convergence criteria
    atol = 1.e-11

    V = pvsystem.v_from_i(Rsh, Rs, nNsVth, I, I0, IL)
    assert(isinstance(V, type(V_expected)))
    if isinstance(V, type(np.ndarray)):
        assert(isinstance(V.dtype, type(V_expected.dtype)))
        assert(V.shape == V_expected.shape)
    assert_allclose(V, V_expected, atol=atol)


@pytest.fixture(params=[
    {  # Can handle all python scalar inputs
      'Rsh': 20.,
      'Rs': 0.1,
      'nNsVth': 0.5,
      'V': 40.,
      'I0': 6.e-7,
      'IL': 7.,
      'I_expected': -299.746389916
    },
    {  # Can handle all rank-0 array inputs
      'Rsh': np.array(20.),
      'Rs': np.array(0.1),
      'nNsVth': np.array(0.5),
      'V': np.array(40.),
      'I0': np.array(6.e-7),
      'IL': np.array(7.),
      'I_expected': np.array(-299.746389916)
    },
    {  # Can handle all rank-1 singleton array inputs
      'Rsh': np.array([20.]),
      'Rs': np.array([0.1]),
      'nNsVth': np.array([0.5]),
      'V': np.array([40.]),
      'I0': np.array([6.e-7]),
      'IL': np.array([7.]),
      'I_expected': np.array([-299.746389916])
    },
    {  # Can handle all rank-1 non-singleton array inputs with a zero
      #  series resistance, Rs=0 gives I=IL=Isc at V=0
      'Rsh': np.array([20., 20.]),
      'Rs': np.array([0., 0.1]),
      'nNsVth': np.array([0.5, 0.5]),
      'V': np.array([0., 40.]),
      'I0': np.array([6.e-7, 6.e-7]),
      'IL': np.array([7., 7.]),
      'I_expected': np.array([7., -299.746389916])
    },
    {  # Can handle mixed inputs with a rank-2 array with zero series
      #  resistance, Rs=0 gives I=IL=Isc at V=0
      'Rsh': np.array([20.]),
      'Rs': np.array([[0., 0.], [0., 0.]]),
      'nNsVth': np.array(0.5),
      'V': 0.,
      'I0': np.array([6.e-7]),
      'IL': np.array([7.]),
      'I_expected': np.array([[7., 7.], [7., 7.]])
    },
    {  # Can handle ideal series and shunt, Rsh=inf and Rs=0 give
      #  V_oc = nNsVth*(np.log(IL + I0) - np.log(I0))
      'Rsh': np.inf,
      'Rs': 0.,
      'nNsVth': 0.5,
      'V': np.array([0., 0.5*(np.log(7. + 6.e-7) - np.log(6.e-7))/2.,
                     0.5*(np.log(7. + 6.e-7) - np.log(6.e-7))]),
      'I0': 6.e-7,
      'IL': 7.,
      'I_expected': np.array([7., 7. - 6.e-7*np.expm1((np.log(7. + 6.e-7) -
                              np.log(6.e-7))/2.), 0.])
    },
    {  # Can handle only ideal shunt resistance, no closed form solution
      'Rsh': np.inf,
      'Rs': 0.1,
      'nNsVth': 0.5,
      'V': 40.,
      'I0': 6.e-7,
      'IL': 7.,
      'I_expected': -299.7383436645412
    }])
def fixture_i_from_v(request):
    return request.param


@requires_scipy
def test_i_from_v(fixture_i_from_v):
    # Solution set loaded from fixture
    Rsh = fixture_i_from_v['Rsh']
    Rs = fixture_i_from_v['Rs']
    nNsVth = fixture_i_from_v['nNsVth']
    V = fixture_i_from_v['V']
    I0 = fixture_i_from_v['I0']
    IL = fixture_i_from_v['IL']
    I_expected = fixture_i_from_v['I_expected']

    # Convergence criteria
    atol = 1.e-11

    I = pvsystem.i_from_v(Rsh, Rs, nNsVth, V, I0, IL)
    assert(isinstance(I, type(I_expected)))
    if isinstance(I, type(np.ndarray)):
        assert(isinstance(I.dtype, type(I_expected.dtype)))
        assert(I.shape == I_expected.shape)
    assert_allclose(I, I_expected, atol=atol)


@requires_scipy
def test_PVSystem_i_from_v():
    system = pvsystem.PVSystem()
    output = system.i_from_v(20, .1, .5, 40, 6e-7, 7)
    assert_allclose(output, -299.746389916, atol=1e-5)


@requires_scipy
def test_singlediode_series(cec_module_params):
    times = pd.DatetimeIndex(start='2015-01-01', periods=2, freq='12H')
    poa_data = pd.Series([0, 800], index=times)
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                                         poa_data,
                                         temp_cell=25,
                                         alpha_isc=cec_module_params['alpha_sc'],
                                         module_parameters=cec_module_params,
                                         EgRef=1.121,
                                         dEgdT=-0.0002677)
    out = pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth)
    assert isinstance(out, pd.DataFrame)


@requires_scipy
def test_singlediode_array():
    # github issue 221
    photocurrent = np.linspace(0, 10, 11)
    resistance_shunt = 16
    resistance_series = 0.094
    nNsVth = 0.473
    saturation_current = 1.943e-09

    sd = pvsystem.singlediode(photocurrent, saturation_current,
                              resistance_series, resistance_shunt, nNsVth)

    expected = np.array([
        0.        ,  0.54538398,  1.43273966,  2.36328163,  3.29255606,
        4.23101358,  5.16177031,  6.09368251,  7.02197553,  7.96846051,
        8.88220557])

    assert_allclose(sd['i_mp'], expected, atol=0.01)


@requires_scipy
def test_singlediode_floats(sam_data):
    module = 'Example_Module'
    module_parameters = sam_data['cecmod'][module]
    out = pvsystem.singlediode(7, 6e-7, .1, 20, .5)
    expected = {'i_xx': 4.2498,
                'i_mp': 6.1275,
                'v_oc': 8.1063,
                'p_mp': 38.1937,
                'i_x': 6.7558,
                'i_sc': 6.9651,
                'v_mp': 6.2331,
                'i': None,
                'v': None}
    assert isinstance(out, dict)
    for k, v in out.items():
        if k in ['i', 'v']:
            assert v is None
        else:
            assert_allclose(v, expected[k], atol=1e-3)


@requires_scipy
def test_singlediode_floats_ivcurve():
    out = pvsystem.singlediode(7, 6e-7, .1, 20, .5, ivcurve_pnts=3)
    expected = {'i_xx': 4.2498,
                'i_mp': 6.1275,
                'v_oc': 8.1063,
                'p_mp': 38.1937,
                'i_x': 6.7558,
                'i_sc': 6.9651,
                'v_mp': 6.2331,
                'i': np.array([6.965172e+00, 6.755882e+00, 2.575717e-14]),
                'v': np.array([0., 4.05315, 8.1063])}
    assert isinstance(out, dict)
    for k, v in out.items():
        assert_allclose(v, expected[k], atol=1e-3)


@requires_scipy
def test_singlediode_series_ivcurve(cec_module_params):
    times = pd.DatetimeIndex(start='2015-06-01', periods=3, freq='6H')
    poa_data = pd.Series([0, 400, 800], index=times)
    IL, I0, Rs, Rsh, nNsVth = pvsystem.calcparams_desoto(
                                  poa_data, temp_cell=25,
                                  alpha_isc=cec_module_params['alpha_sc'],
                                  module_parameters=cec_module_params,
                                  EgRef=1.121, dEgdT=-0.0002677)

    out = pvsystem.singlediode(IL, I0, Rs, Rsh, nNsVth, ivcurve_pnts=3)

    expected = OrderedDict([('i_sc', array([0., 3.01054475, 6.00675648])),
                            ('v_oc', array([0., 9.96886962, 10.29530483])),
                            ('i_mp', array([0., 2.65191983, 5.28594672])),
                            ('v_mp', array([0., 8.33392491, 8.4159707])),
                            ('p_mp', array([0., 22.10090078, 44.48637274])),
                            ('i_x', array([0., 2.88414114, 5.74622046])),
                            ('i_xx', array([0., 2.04340914, 3.90007956])),
                            ('v', array([[0., 0., 0.],
                                         [0., 4.98443481, 9.96886962],
                                         [0., 5.14765242, 10.29530483]])),
                            ('i', array([[0., 0., 0.],
                                         [3.01079860e+00, 2.88414114e+00,
                                          3.10862447e-14],
                                         [6.00726296e+00, 5.74622046e+00,
                                          0.00000000e+00]]))])

    for k, v in out.items():
        assert_allclose(v, expected[k], atol=1e-2)


def test_scale_voltage_current_power(sam_data):
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
    system = pvsystem.PVSystem(modules_per_string=2, strings_per_inverter=3)
    out = system.scale_voltage_current_power(data)


def test_sapm_celltemp():
    default = pvsystem.sapm_celltemp(900, 5, 20)
    assert_allclose(default['temp_cell'], 43.509, 3)
    assert_allclose(default['temp_module'], 40.809, 3)
    assert_frame_equal(default, pvsystem.sapm_celltemp(900, 5, 20,
                                                       [-3.47, -.0594, 3]))


def test_sapm_celltemp_dict_like():
    default = pvsystem.sapm_celltemp(900, 5, 20)
    assert_allclose(default['temp_cell'], 43.509, 3)
    assert_allclose(default['temp_module'], 40.809, 3)
    model = {'a': -3.47, 'b': -.0594, 'deltaT': 3}
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


def test_adrinverter(sam_data):
    inverters = sam_data['adrinverter']
    testinv = 'Ablerex_Electronics_Co___Ltd___' \
              'ES_2200_US_240__240_Vac__240V__CEC_2011_'
    vdcs = pd.Series([135, 154, 390, 420, 551])
    pdcs = pd.Series([135, 1232, 1170, 420, 551])

    pacs = pvsystem.adrinverter(vdcs, pdcs, inverters[testinv])
    assert_series_equal(pacs, pd.Series([np.nan, 1161.5745, 1116.4459,
                                         382.6679, np.nan]))


def test_adrinverter_vtol(sam_data):
    inverters = sam_data['adrinverter']
    testinv = 'Ablerex_Electronics_Co___Ltd___' \
              'ES_2200_US_240__240_Vac__240V__CEC_2011_'
    vdcs = pd.Series([135, 154, 390, 420, 551])
    pdcs = pd.Series([135, 1232, 1170, 420, 551])

    pacs = pvsystem.adrinverter(vdcs, pdcs, inverters[testinv], vtol=0.20)
    assert_series_equal(pacs, pd.Series([104.8223, 1161.5745, 1116.4459,
                                         382.6679, 513.3385]))


def test_adrinverter_float(sam_data):
    inverters = sam_data['adrinverter']
    testinv = 'Ablerex_Electronics_Co___Ltd___' \
              'ES_2200_US_240__240_Vac__240V__CEC_2011_'
    vdcs = 154.
    pdcs = 1232.

    pacs = pvsystem.adrinverter(vdcs, pdcs, inverters[testinv])
    assert_allclose(pacs, 1161.5745)


def test_adrinverter_invalid_and_night(sam_data):
    inverters = sam_data['adrinverter']
    testinv = 'Zigor__Sunzet_3_TL_US_240V__CEC_2011_'
    vdcs = np.array([39.873036, 0., np.nan, 420])
    pdcs = np.array([188.09182, 0., 420, np.nan])

    pacs = pvsystem.adrinverter(vdcs, pdcs, inverters[testinv])
    assert_allclose(pacs, np.array([np.nan, -0.25, np.nan, np.nan]))


def test_snlinverter(sam_data):
    inverters = sam_data['cecinverter']
    testinv = 'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'
    vdcs = pd.Series(np.linspace(0,50,3))
    idcs = pd.Series(np.linspace(0,11,3))
    pdcs = idcs * vdcs

    pacs = pvsystem.snlinverter(vdcs, pdcs, inverters[testinv])
    assert_series_equal(pacs, pd.Series([-0.020000, 132.004308, 250.000000]))


def test_PVSystem_snlinverter(sam_data):
    inverters = sam_data['cecinverter']
    testinv = 'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'
    system = pvsystem.PVSystem(inverter=testinv,
                               inverter_parameters=inverters[testinv])
    vdcs = pd.Series(np.linspace(0,50,3))
    idcs = pd.Series(np.linspace(0,11,3))
    pdcs = idcs * vdcs

    pacs = system.snlinverter(vdcs, pdcs)
    assert_series_equal(pacs, pd.Series([-0.020000, 132.004308, 250.000000]))


def test_snlinverter_float(sam_data):
    inverters = sam_data['cecinverter']
    testinv = 'ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_'
    vdcs = 25.
    idcs = 5.5
    pdcs = idcs * vdcs

    pacs = pvsystem.snlinverter(vdcs, pdcs, inverters[testinv])
    assert_allclose(pacs, 132.004278, 5)


def test_snlinverter_Pnt_micro(sam_data):
    inverters = sam_data['cecinverter']
    testinv = 'Enphase_Energy__M250_60_2LL_S2x___ZC____NA__208V_208V__CEC_2013_'
    vdcs = pd.Series(np.linspace(0,50,3))
    idcs = pd.Series(np.linspace(0,11,3))
    pdcs = idcs * vdcs

    pacs = pvsystem.snlinverter(vdcs, pdcs, inverters[testinv])
    assert_series_equal(pacs, pd.Series([-0.043000, 132.545914746, 240.000000]))


def test_PVSystem_creation():
    pv_system = pvsystem.PVSystem(module='blah', inverter='blarg')
    # ensure that parameter attributes are dict-like. GH 294
    pv_system.module_parameters['pdc0'] = 1
    pv_system.inverter_parameters['Paco'] = 1


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

    assert_frame_equal(irradiance, expected, check_less_precise=2)


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


def test_PVSystem___repr__():
    system = pvsystem.PVSystem(module='blah', inverter='blarg', name='pv ftw')

    expected = 'PVSystem: \n  name: pv ftw\n  surface_tilt: 0\n  surface_azimuth: 180\n  module: blah\n  inverter: blarg\n  albedo: 0.25\n  racking_model: open_rack_cell_glassback'

    assert system.__repr__() == expected


def test_PVSystem_localize___repr__():
    system = pvsystem.PVSystem(module='blah', inverter='blarg', name='pv ftw')
    localized_system = system.localize(latitude=32, longitude=-111)

    expected = 'LocalizedPVSystem: \n  name: None\n  latitude: 32\n  longitude: -111\n  altitude: 0\n  tz: UTC\n  surface_tilt: 0\n  surface_azimuth: 180\n  module: blah\n  inverter: blarg\n  albedo: 0.25\n  racking_model: open_rack_cell_glassback'

    assert localized_system.__repr__() == expected


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


def test_LocalizedPVSystem___repr__():
    localized_system = pvsystem.LocalizedPVSystem(latitude=32,
                                                  longitude=-111,
                                                  module='blah',
                                                  inverter='blarg',
                                                  name='my name')

    expected = 'LocalizedPVSystem: \n  name: my name\n  latitude: 32\n  longitude: -111\n  altitude: 0\n  tz: UTC\n  surface_tilt: 0\n  surface_azimuth: 180\n  module: blah\n  inverter: blarg\n  albedo: 0.25\n  racking_model: open_rack_cell_glassback'

    assert localized_system.__repr__() == expected


def test_pvwatts_dc_scalars():
    expected = 88.65
    out = pvsystem.pvwatts_dc(900, 30, 100, -0.003)
    assert_allclose(out, expected)


@needs_numpy_1_10
def test_pvwatts_dc_arrays():
    irrad_trans = np.array([np.nan, 900, 900])
    temp_cell = np.array([30, np.nan, 30])
    irrad_trans, temp_cell = np.meshgrid(irrad_trans, temp_cell)
    expected = np.array([[nan,  88.65,  88.65],
                         [nan,    nan,    nan],
                         [nan,  88.65,  88.65]])
    out = pvsystem.pvwatts_dc(irrad_trans, temp_cell, 100, -0.003)
    assert_allclose(out, expected, equal_nan=True)


def test_pvwatts_dc_series():
    irrad_trans = pd.Series([np.nan, 900, 900])
    temp_cell = pd.Series([30, np.nan, 30])
    expected = pd.Series(np.array([   nan,    nan,  88.65]))
    out = pvsystem.pvwatts_dc(irrad_trans, temp_cell, 100, -0.003)
    assert_series_equal(expected, out)


def test_pvwatts_ac_scalars():
    expected = 85.58556604752516
    out = pvsystem.pvwatts_ac(90, 100, 0.95)
    assert_allclose(out, expected)


@needs_numpy_1_10
def test_pvwatts_ac_arrays():
    pdc = np.array([[np.nan], [50], [100]])
    pdc0 = 100
    expected = np.array([[nan],
                         [47.60843624],
                         [95.]])
    out = pvsystem.pvwatts_ac(pdc, pdc0, 0.95)
    assert_allclose(out, expected, equal_nan=True)


def test_pvwatts_ac_series():
    pdc = pd.Series([np.nan, 50, 100])
    pdc0 = 100
    expected = pd.Series(np.array([       nan,  47.608436,  95.      ]))
    out = pvsystem.pvwatts_ac(pdc, pdc0, 0.95)
    assert_series_equal(expected, out)


def test_pvwatts_losses_default():
    expected = 14.075660688264469
    out = pvsystem.pvwatts_losses()
    assert_allclose(out, expected)


@needs_numpy_1_10
def test_pvwatts_losses_arrays():
    expected = np.array([nan, 14.934904])
    age = np.array([nan, 1])
    out = pvsystem.pvwatts_losses(age=age)
    assert_allclose(out, expected)


def test_pvwatts_losses_series():
    expected = pd.Series([nan, 14.934904])
    age = pd.Series([nan, 1])
    out = pvsystem.pvwatts_losses(age=age)
    assert_series_equal(expected, out)


def make_pvwatts_system_defaults():
    module_parameters = {'pdc0': 100, 'gamma_pdc': -0.003}
    inverter_parameters = {}
    system = pvsystem.PVSystem(module_parameters=module_parameters,
                               inverter_parameters=inverter_parameters)
    return system


def make_pvwatts_system_kwargs():
    module_parameters = {'pdc0': 100, 'gamma_pdc': -0.003, 'temp_ref': 20}
    inverter_parameters = {'eta_inv_nom': 0.95, 'eta_inv_ref': 1.0}
    system = pvsystem.PVSystem(module_parameters=module_parameters,
                               inverter_parameters=inverter_parameters)
    return system


def test_PVSystem_pvwatts_dc():
    system = make_pvwatts_system_defaults()
    irrad_trans = pd.Series([np.nan, 900, 900])
    temp_cell = pd.Series([30, np.nan, 30])
    expected = pd.Series(np.array([   nan,    nan,  88.65]))
    out = system.pvwatts_dc(irrad_trans, temp_cell)
    assert_series_equal(expected, out)

    system = make_pvwatts_system_kwargs()
    expected = pd.Series(np.array([   nan,    nan,  87.3]))
    out = system.pvwatts_dc(irrad_trans, temp_cell)
    assert_series_equal(expected, out)


def test_PVSystem_pvwatts_losses():
    system = make_pvwatts_system_defaults()
    expected = pd.Series([nan, 14.934904])
    age = pd.Series([nan, 1])
    out = system.pvwatts_losses(age=age)
    assert_series_equal(expected, out)


def test_PVSystem_pvwatts_ac():
    system = make_pvwatts_system_defaults()
    pdc = pd.Series([np.nan, 50, 100])
    expected = pd.Series(np.array([       nan,  48.1095776694, 96.0]))
    out = system.pvwatts_ac(pdc)
    assert_series_equal(expected, out)

    system = make_pvwatts_system_kwargs()
    expected = pd.Series(np.array([       nan,  45.88025, 91.5515]))
    out = system.pvwatts_ac(pdc)
    assert_series_equal(expected, out)
