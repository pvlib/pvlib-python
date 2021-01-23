import datetime
from collections import OrderedDict
import warnings

import numpy as np
from numpy import array, nan
import pandas as pd

import pytest
from numpy.testing import assert_almost_equal, assert_allclose

from pvlib import irradiance

from conftest import (
    assert_frame_equal,
    assert_series_equal,
    requires_ephem,
    requires_numba
)


# fixtures create realistic test input data
# test input data generated at Location(32.2, -111, 'US/Arizona', 700)
# test input data is hard coded to avoid dependencies on other parts of pvlib


@pytest.fixture
def times():
    # must include night values
    return pd.date_range(start='20140624', freq='6H', periods=4,
                         tz='US/Arizona')


@pytest.fixture
def irrad_data(times):
    return pd.DataFrame(np.array(
        [[   0.        ,    0.        ,    0.        ],
         [  79.73860422,  316.1949056 ,   40.46149818],
         [1042.48031487,  939.95469881,  118.45831879],
         [ 257.20751138,  646.22886049,   62.03376265]]),
        columns=['ghi', 'dni', 'dhi'], index=times)


@pytest.fixture
def ephem_data(times):
    return pd.DataFrame(np.array(
        [[124.0390863 , 124.0390863 , -34.0390863 , -34.0390863 ,
          352.69550699,  -2.36677158],
         [ 82.85457044,  82.97705621,   7.14542956,   7.02294379,
           66.71410338,  -2.42072165],
         [ 10.56413562,  10.56725766,  79.43586438,  79.43274234,
          144.76567754,  -2.47457321],
         [ 72.41687122,  72.46903556,  17.58312878,  17.53096444,
          287.04104128,  -2.52831909]]),
        columns=['apparent_zenith', 'zenith', 'apparent_elevation',
                 'elevation', 'azimuth', 'equation_of_time'],
        index=times)


@pytest.fixture
def dni_et(times):
    return np.array(
        [1321.1655834833093, 1321.1655834833093, 1321.1655834833093,
         1321.1655834833093])


@pytest.fixture
def relative_airmass(times):
    return pd.Series([np.nan, 7.58831596, 1.01688136, 3.27930443], times)


# setup for et rad test. put it here for readability
timestamp = pd.Timestamp('20161026')
dt_index = pd.DatetimeIndex([timestamp])
doy = timestamp.dayofyear
dt_date = timestamp.date()
dt_datetime = datetime.datetime.combine(dt_date, datetime.time(0))
dt_np64 = np.datetime64(dt_datetime)
value = 1383.636203


@pytest.mark.parametrize('testval, expected', [
    (doy, value),
    (np.float64(doy), value),
    (dt_date, value),
    (dt_datetime, value),
    (dt_np64, value),
    (np.array([doy]), np.array([value])),
    (pd.Series([doy]), np.array([value])),
    (dt_index, pd.Series([value], index=dt_index)),
    (timestamp, value)
])
@pytest.mark.parametrize('method', [
    'asce', 'spencer', 'nrel', pytest.param('pyephem', marks=requires_ephem)])
def test_get_extra_radiation(testval, expected, method):
    out = irradiance.get_extra_radiation(testval, method=method)
    assert_allclose(out, expected, atol=10)


def test_get_extra_radiation_epoch_year():
    out = irradiance.get_extra_radiation(doy, method='nrel', epoch_year=2012)
    assert_allclose(out, 1382.4926804890767, atol=0.1)


@requires_numba
def test_get_extra_radiation_nrel_numba(times):
    with warnings.catch_warnings():
        # don't warn on method reload or num threads
        warnings.simplefilter("ignore")
        result = irradiance.get_extra_radiation(
            times, method='nrel', how='numba', numthreads=4)
        # and reset to no-numba state
        irradiance.get_extra_radiation(times, method='nrel')
    assert_allclose(result,
                    [1322.332316, 1322.296282, 1322.261205, 1322.227091])


def test_get_extra_radiation_invalid():
    with pytest.raises(ValueError):
        irradiance.get_extra_radiation(300, method='invalid')


def test_grounddiffuse_simple_float():
    result = irradiance.get_ground_diffuse(40, 900)
    assert_allclose(result, 26.32000014911496)


def test_grounddiffuse_simple_series(irrad_data):
    ground_irrad = irradiance.get_ground_diffuse(40, irrad_data['ghi'])
    assert ground_irrad.name == 'diffuse_ground'


def test_grounddiffuse_albedo_0(irrad_data):
    ground_irrad = irradiance.get_ground_diffuse(
        40, irrad_data['ghi'], albedo=0)
    assert 0 == ground_irrad.all()


def test_grounddiffuse_albedo_invalid_surface(irrad_data):
    with pytest.raises(KeyError):
        irradiance.get_ground_diffuse(
            40, irrad_data['ghi'], surface_type='invalid')


def test_grounddiffuse_albedo_surface(irrad_data):
    result = irradiance.get_ground_diffuse(40, irrad_data['ghi'],
                                           surface_type='sand')
    assert_allclose(result, [0, 3.731058, 48.778813, 12.035025], atol=1e-4)


def test_isotropic_float():
    result = irradiance.isotropic(40, 100)
    assert_allclose(result, 88.30222215594891)


def test_isotropic_series(irrad_data):
    result = irradiance.isotropic(40, irrad_data['dhi'])
    assert_allclose(result, [0, 35.728402, 104.601328, 54.777191], atol=1e-4)


def test_klucher_series_float():
    # klucher inputs
    surface_tilt, surface_azimuth = 40.0, 180.0
    dhi, ghi = 100.0, 900.0
    solar_zenith, solar_azimuth = 20.0, 180.0
    # expect same result for floats and pd.Series
    expected = irradiance.klucher(
        surface_tilt, surface_azimuth,
        pd.Series(dhi), pd.Series(ghi),
        pd.Series(solar_zenith), pd.Series(solar_azimuth)
    )  # 94.99429931664851
    result = irradiance.klucher(
        surface_tilt, surface_azimuth, dhi, ghi, solar_zenith, solar_azimuth
    )
    assert_allclose(result, expected[0])


def test_klucher_series(irrad_data, ephem_data):
    result = irradiance.klucher(40, 180, irrad_data['dhi'], irrad_data['ghi'],
                                ephem_data['apparent_zenith'],
                                ephem_data['azimuth'])
    # pvlib matlab 1.4 does not contain the max(cos_tt, 0) correction
    # so, these values are different
    assert_allclose(result, [0., 36.789794, 109.209347, 56.965916], atol=1e-4)
    # expect same result for np.array and pd.Series
    expected = irradiance.klucher(
        40, 180, irrad_data['dhi'].values, irrad_data['ghi'].values,
        ephem_data['apparent_zenith'].values, ephem_data['azimuth'].values
    )
    assert_allclose(result, expected, atol=1e-4)


def test_haydavies(irrad_data, ephem_data, dni_et):
    result = irradiance.haydavies(
        40, 180, irrad_data['dhi'], irrad_data['dni'], dni_et,
        ephem_data['apparent_zenith'], ephem_data['azimuth'])
    # values from matlab 1.4 code
    assert_allclose(result, [0, 27.1775, 102.9949, 33.1909], atol=1e-4)


def test_reindl(irrad_data, ephem_data, dni_et):
    result = irradiance.reindl(
        40, 180, irrad_data['dhi'], irrad_data['dni'], irrad_data['ghi'],
        dni_et, ephem_data['apparent_zenith'], ephem_data['azimuth'])
    # values from matlab 1.4 code
    assert_allclose(result, [np.nan, 27.9412, 104.1317, 34.1663], atol=1e-4)


def test_king(irrad_data, ephem_data):
    result = irradiance.king(40, irrad_data['dhi'], irrad_data['ghi'],
                             ephem_data['apparent_zenith'])
    assert_allclose(result, [0, 44.629352, 115.182626, 79.719855], atol=1e-4)


def test_perez(irrad_data, ephem_data, dni_et, relative_airmass):
    dni = irrad_data['dni'].copy()
    dni.iloc[2] = np.nan
    out = irradiance.perez(40, 180, irrad_data['dhi'], dni,
                           dni_et, ephem_data['apparent_zenith'],
                           ephem_data['azimuth'], relative_airmass)
    expected = pd.Series(np.array(
        [   0.        ,   31.46046871,  np.nan,   45.45539877]),
        index=irrad_data.index)
    assert_series_equal(out, expected, check_less_precise=2)


def test_perez_components(irrad_data, ephem_data, dni_et, relative_airmass):
    dni = irrad_data['dni'].copy()
    dni.iloc[2] = np.nan
    out = irradiance.perez(40, 180, irrad_data['dhi'], dni,
                           dni_et, ephem_data['apparent_zenith'],
                           ephem_data['azimuth'], relative_airmass,
                           return_components=True)
    expected = pd.DataFrame(np.array(
        [[   0.        ,   31.46046871,  np.nan,   45.45539877],
         [  0.        ,  26.84138589,          np.nan,  31.72696071],
         [ 0.        ,  0.        ,         np.nan,  4.47966439],
         [ 0.        ,  4.62212181,         np.nan,  9.25316454]]).T,
        columns=['sky_diffuse', 'isotropic', 'circumsolar', 'horizon'],
        index=irrad_data.index
    )
    expected_for_sum = expected['sky_diffuse'].copy()
    expected_for_sum.iloc[2] = 0
    sum_components = out.iloc[:, 1:].sum(axis=1)
    sum_components.name = 'sky_diffuse'

    assert_frame_equal(out, expected, check_less_precise=2)
    assert_series_equal(sum_components, expected_for_sum, check_less_precise=2)


def test_perez_arrays(irrad_data, ephem_data, dni_et, relative_airmass):
    dni = irrad_data['dni'].copy()
    dni.iloc[2] = np.nan
    out = irradiance.perez(40, 180, irrad_data['dhi'].values, dni.values,
                           dni_et, ephem_data['apparent_zenith'].values,
                           ephem_data['azimuth'].values,
                           relative_airmass.values)
    expected = np.array(
        [   0.        ,   31.46046871,  np.nan,   45.45539877])
    assert_allclose(out, expected, atol=1e-2)
    assert isinstance(out, np.ndarray)


def test_perez_scalar():
    # copied values from fixtures
    out = irradiance.perez(40, 180, 118.45831879, 939.95469881,
                           1321.1655834833093, 10.56413562, 144.76567754,
                           1.01688136)
    # this will fail. out is ndarry with ndim == 0. fix in future version.
    # assert np.isscalar(out)
    assert_allclose(out, 109.084332)


@pytest.mark.parametrize('model', ['isotropic', 'klucher', 'haydavies',
                                   'reindl', 'king', 'perez'])
def test_sky_diffuse_zenith_close_to_90(model):
    # GH 432
    sky_diffuse = irradiance.get_sky_diffuse(
        30, 180, 89.999, 230,
        dni=10, ghi=51, dhi=50, dni_extra=1360, airmass=12, model=model)
    assert sky_diffuse < 100


def test_get_sky_diffuse_invalid():
    with pytest.raises(ValueError):
        irradiance.get_sky_diffuse(
            30, 180, 0, 180, 1000, 1100, 100, dni_extra=1360, airmass=1,
            model='invalid')


def test_campbell_norman():
    expected = pd.DataFrame(np.array(
        [[863.859736967, 653.123094076, 220.65905025]]),
        columns=['ghi', 'dni', 'dhi'],
        index=[0])
    out = irradiance.campbell_norman(
        pd.Series([10]), pd.Series([0.5]), pd.Series([109764.21013135818]),
        dni_extra=1400)
    assert_frame_equal(out, expected)


def test_get_total_irradiance(irrad_data, ephem_data, dni_et, relative_airmass):
    models = ['isotropic', 'klucher',
              'haydavies', 'reindl', 'king', 'perez']

    for model in models:
        total = irradiance.get_total_irradiance(
            32, 180,
            ephem_data['apparent_zenith'], ephem_data['azimuth'],
            dni=irrad_data['dni'], ghi=irrad_data['ghi'],
            dhi=irrad_data['dhi'],
            dni_extra=dni_et, airmass=relative_airmass,
            model=model,
            surface_type='urban')

        assert total.columns.tolist() == ['poa_global', 'poa_direct',
                                          'poa_diffuse', 'poa_sky_diffuse',
                                          'poa_ground_diffuse']


@pytest.mark.parametrize('model', ['isotropic', 'klucher',
                                   'haydavies', 'reindl', 'king', 'perez'])
def test_get_total_irradiance_scalars(model):
    total = irradiance.get_total_irradiance(
        32, 180,
        10, 180,
        dni=1000, ghi=1100,
        dhi=100,
        dni_extra=1400, airmass=1,
        model=model,
        surface_type='urban')

    assert list(total.keys()) == ['poa_global', 'poa_direct',
                                  'poa_diffuse', 'poa_sky_diffuse',
                                  'poa_ground_diffuse']
    # test that none of the values are nan
    assert np.isnan(np.array(list(total.values()))).sum() == 0


def test_poa_components(irrad_data, ephem_data, dni_et, relative_airmass):
    aoi = irradiance.aoi(40, 180, ephem_data['apparent_zenith'],
                         ephem_data['azimuth'])
    gr_sand = irradiance.get_ground_diffuse(40, irrad_data['ghi'],
                                            surface_type='sand')
    diff_perez = irradiance.perez(
        40, 180, irrad_data['dhi'], irrad_data['dni'], dni_et,
        ephem_data['apparent_zenith'], ephem_data['azimuth'], relative_airmass)
    out = irradiance.poa_components(
        aoi, irrad_data['dni'], diff_perez, gr_sand)
    expected = pd.DataFrame(np.array(
        [[  0.        ,  -0.        ,   0.        ,   0.        ,
            0.        ],
         [ 35.19456561,   0.        ,  35.19456561,  31.4635077 ,
            3.73105791],
         [956.18253696, 798.31939281, 157.86314414, 109.08433162,
           48.77881252],
         [ 90.99624896,  33.50143401,  57.49481495,  45.45978964,
           12.03502531]]),
        columns=['poa_global', 'poa_direct', 'poa_diffuse', 'poa_sky_diffuse',
                 'poa_ground_diffuse'],
        index=irrad_data.index)
    assert_frame_equal(out, expected)


@pytest.mark.parametrize('pressure,expected', [
    (93193,  [[830.46567,   0.79742,   0.93505],
              [676.09497,   0.63776,   3.02102]]),
    (None,   [[868.72425,   0.79742,   1.01664],
              [680.66679,   0.63776,   3.28463]]),
    (101325, [[868.72425,   0.79742,   1.01664],
              [680.66679,   0.63776,   3.28463]])
])
def test_disc_value(pressure, expected):
    # see GH 449 for pressure=None vs. 101325.
    columns = ['dni', 'kt', 'airmass']
    times = pd.DatetimeIndex(['2014-06-24T1200', '2014-06-24T1800'],
                             tz='America/Phoenix')
    ghi = pd.Series([1038.62, 254.53], index=times)
    zenith = pd.Series([10.567, 72.469], index=times)
    out = irradiance.disc(ghi, zenith, times, pressure=pressure)
    expected_values = np.array(expected)
    expected = pd.DataFrame(expected_values, columns=columns, index=times)
    # check the pandas dataframe. check_less_precise is weird
    assert_frame_equal(out, expected, check_less_precise=True)
    # use np.assert_allclose to check values more clearly
    assert_allclose(out.values, expected_values, atol=1e-5)


def test_disc_overirradiance():
    columns = ['dni', 'kt', 'airmass']
    ghi = np.array([3000])
    solar_zenith = np.full_like(ghi, 0)
    times = pd.date_range(start='2016-07-19 12:00:00', freq='1s',
                          periods=len(ghi), tz='America/Phoenix')
    out = irradiance.disc(ghi=ghi, solar_zenith=solar_zenith,
                          datetime_or_doy=times)
    expected = pd.DataFrame(np.array(
        [[8.72544336e+02, 1.00000000e+00, 9.99493933e-01]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)


def test_disc_min_cos_zenith_max_zenith():
    # map out behavior under difficult conditions with various
    # limiting kwargs settings
    columns = ['dni', 'kt', 'airmass']
    times = pd.DatetimeIndex(['2016-07-19 06:11:00'], tz='America/Phoenix')
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times)
    expected = pd.DataFrame(np.array(
        [[0.00000000e+00, 1.16046346e-02, 12.0]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # max_zenith and/or max_airmass keep these results reasonable
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          min_cos_zenith=0)
    expected = pd.DataFrame(np.array(
        [[0.00000000e+00, 1.0, 12.0]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # still get reasonable values because of max_airmass=12 limit
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          max_zenith=100)
    expected = pd.DataFrame(np.array(
        [[0., 1.16046346e-02, 12.0]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # still get reasonable values because of max_airmass=12 limit
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          min_cos_zenith=0, max_zenith=100)
    expected = pd.DataFrame(np.array(
        [[277.50185968, 1.0, 12.0]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # max_zenith keeps this result reasonable
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          min_cos_zenith=0, max_airmass=100)
    expected = pd.DataFrame(np.array(
        [[0.00000000e+00, 1.0, 36.39544757]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # allow zenith to be close to 90 and airmass to be infinite
    # and we get crazy values
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          max_zenith=100, max_airmass=100)
    expected = pd.DataFrame(np.array(
        [[6.68577449e+03, 1.16046346e-02, 3.63954476e+01]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # allow min cos zenith to be 0, zenith to be close to 90,
    # and airmass to be very big and we get even higher DNI values
    out = irradiance.disc(ghi=1.0, solar_zenith=89.99, datetime_or_doy=times,
                          min_cos_zenith=0, max_zenith=100, max_airmass=100)
    expected = pd.DataFrame(np.array(
        [[7.21238390e+03, 1., 3.63954476e+01]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)


def test_dirint_value():
    times = pd.DatetimeIndex(['2014-06-24T12-0700', '2014-06-24T18-0700'])
    ghi = pd.Series([1038.62, 254.53], index=times)
    zenith = pd.Series([10.567, 72.469], index=times)
    pressure = 93193.
    dirint_data = irradiance.dirint(ghi, zenith, times, pressure=pressure)
    assert_almost_equal(dirint_data.values,
                        np.array([868.8,  699.7]), 1)


def test_dirint_nans():
    times = pd.date_range(start='2014-06-24T12-0700', periods=5, freq='6H')
    ghi = pd.Series([np.nan, 1038.62, 1038.62, 1038.62, 1038.62], index=times)
    zenith = pd.Series([10.567, np.nan, 10.567, 10.567, 10.567], index=times)
    pressure = pd.Series([93193., 93193., np.nan, 93193., 93193.], index=times)
    temp_dew = pd.Series([10, 10, 10, np.nan, 10], index=times)
    dirint_data = irradiance.dirint(ghi, zenith, times, pressure=pressure,
                                    temp_dew=temp_dew)
    assert_almost_equal(dirint_data.values,
                        np.array([np.nan, np.nan, np.nan, np.nan, 893.1]), 1)


def test_dirint_tdew():
    times = pd.DatetimeIndex(['2014-06-24T12-0700', '2014-06-24T18-0700'])
    ghi = pd.Series([1038.62, 254.53], index=times)
    zenith = pd.Series([10.567, 72.469], index=times)
    pressure = 93193.
    dirint_data = irradiance.dirint(ghi, zenith, times, pressure=pressure,
                                    temp_dew=10)
    assert_almost_equal(dirint_data.values,
                        np.array([882.1,  672.6]), 1)


def test_dirint_no_delta_kt():
    times = pd.DatetimeIndex(['2014-06-24T12-0700', '2014-06-24T18-0700'])
    ghi = pd.Series([1038.62, 254.53], index=times)
    zenith = pd.Series([10.567, 72.469], index=times)
    pressure = 93193.
    dirint_data = irradiance.dirint(ghi, zenith, times, pressure=pressure,
                                    use_delta_kt_prime=False)
    assert_almost_equal(dirint_data.values,
                        np.array([861.9,  670.4]), 1)


def test_dirint_coeffs():
    coeffs = irradiance._get_dirint_coeffs()
    assert coeffs[0, 0, 0, 0] == 0.385230
    assert coeffs[0, 1, 2, 1] == 0.229970
    assert coeffs[3, 2, 6, 3] == 1.032260


def test_dirint_min_cos_zenith_max_zenith():
    # map out behavior under difficult conditions with various
    # limiting kwargs settings
    # times don't have any physical relevance
    times = pd.DatetimeIndex(['2014-06-24T12-0700', '2014-06-24T18-0700'])
    ghi = pd.Series([0, 1], index=times)
    solar_zenith = pd.Series([90, 89.99], index=times)

    out = irradiance.dirint(ghi, solar_zenith, times)
    expected = pd.Series([0.0, 0.0], index=times, name='dni')
    assert_series_equal(out, expected)

    out = irradiance.dirint(ghi, solar_zenith, times, min_cos_zenith=0)
    expected = pd.Series([0.0, 0.0], index=times, name='dni')
    assert_series_equal(out, expected)

    out = irradiance.dirint(ghi, solar_zenith, times, max_zenith=90)
    expected = pd.Series([0.0, 0.0], index=times, name='dni')
    assert_series_equal(out, expected, check_less_precise=True)

    out = irradiance.dirint(ghi, solar_zenith, times, min_cos_zenith=0,
                            max_zenith=90)
    expected = pd.Series([0.0, 144.264507], index=times, name='dni')
    assert_series_equal(out, expected, check_less_precise=True)

    out = irradiance.dirint(ghi, solar_zenith, times, min_cos_zenith=0,
                            max_zenith=100)
    expected = pd.Series([0.0, 144.264507], index=times, name='dni')
    assert_series_equal(out, expected, check_less_precise=True)


def test_gti_dirint():
    times = pd.DatetimeIndex(
        ['2014-06-24T06-0700', '2014-06-24T09-0700', '2014-06-24T12-0700'])
    poa_global = np.array([20, 300, 1000])
    aoi = np.array([100, 70, 10])
    zenith = np.array([80, 45, 20])
    azimuth = np.array([90, 135, 180])
    surface_tilt = 30
    surface_azimuth = 180

    # test defaults
    output = irradiance.gti_dirint(
        poa_global, aoi, zenith, azimuth, times, surface_tilt, surface_azimuth)

    expected_col_order = ['ghi', 'dni', 'dhi']
    expected = pd.DataFrame(array(
        [[  21.05796198,    0.        ,   21.05796198],
         [ 288.22574368,   60.59964218,  245.37532576],
         [ 931.04078010,  695.94965324,  277.06172442]]),
        columns=expected_col_order, index=times)

    assert_frame_equal(output, expected)

    # test ignore calculate_gt_90
    output = irradiance.gti_dirint(
        poa_global, aoi, zenith, azimuth, times, surface_tilt, surface_azimuth,
        calculate_gt_90=False)

    expected_no_90 = expected.copy()
    expected_no_90.iloc[0, :] = np.nan

    assert_frame_equal(output, expected_no_90)

    # test pressure input
    pressure = 93193.
    output = irradiance.gti_dirint(
        poa_global, aoi, zenith, azimuth, times, surface_tilt, surface_azimuth,
        pressure=pressure)

    expected = pd.DataFrame(array(
        [[  21.05796198,    0.        ,   21.05796198],
         [ 289.81109139,   60.52460392,  247.01373353],
         [ 932.46756378,  648.05001357,  323.49974813]]),
        columns=expected_col_order, index=times)

    assert_frame_equal(output, expected)

    # test albedo input
    albedo = 0.05
    output = irradiance.gti_dirint(
        poa_global, aoi, zenith, azimuth, times, surface_tilt, surface_azimuth,
        albedo=albedo)

    expected = pd.DataFrame(array(
        [[  21.3592591,    0.        ,   21.3592591 ],
         [ 292.5162373,   64.42628826,  246.95997198],
         [ 941.6753031,  727.16311901,  258.36548605]]),
        columns=expected_col_order, index=times)

    assert_frame_equal(output, expected)

    # test temp_dew input
    temp_dew = np.array([70, 80, 20])
    output = irradiance.gti_dirint(
        poa_global, aoi, zenith, azimuth, times, surface_tilt, surface_azimuth,
        temp_dew=temp_dew)

    expected = pd.DataFrame(array(
        [[  21.05796198,    0.        ,   21.05796198],
         [ 292.40468994,   36.79559287,  266.3862767 ],
         [ 931.79627208,  689.81549269,  283.5817439]]),
        columns=expected_col_order, index=times)

    assert_frame_equal(output, expected)


def test_erbs():
    index = pd.DatetimeIndex(['20190101']*3 + ['20190620'])
    ghi = pd.Series([0, 50, 1000, 1000], index=index)
    zenith = pd.Series([120, 85, 10, 10], index=index)
    expected = pd.DataFrame(np.array(
        [[0.00000000e+00, 0.00000000e+00, 0.00000000e+00],
         [9.67192672e+01, 4.15703604e+01, 4.05723511e-01],
         [7.94205651e+02, 2.17860117e+02, 7.18132729e-01],
         [8.42001578e+02, 1.70790318e+02, 7.68214312e-01]]),
        columns=['dni', 'dhi', 'kt'], index=index)

    out = irradiance.erbs(ghi, zenith, index)

    assert_frame_equal(np.round(out, 0), np.round(expected, 0))


def test_erbs_min_cos_zenith_max_zenith():
    # map out behavior under difficult conditions with various
    # limiting kwargs settings
    columns = ['dni', 'dhi', 'kt']
    times = pd.DatetimeIndex(['2016-07-19 06:11:00'], tz='America/Phoenix')

    # max_zenith keeps these results reasonable
    out = irradiance.erbs(ghi=1.0, zenith=89.99999,
                          datetime_or_doy=times, min_cos_zenith=0)
    expected = pd.DataFrame(np.array(
        [[0., 1., 1.]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # 4-5 9s will produce bad behavior without max_zenith limit
    out = irradiance.erbs(ghi=1.0, zenith=89.99999,
                          datetime_or_doy=times, max_zenith=100)
    expected = pd.DataFrame(np.array(
        [[6.00115286e+03, 9.98952601e-01, 1.16377640e-02]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # 1-2 9s will produce bad behavior without either limit
    out = irradiance.erbs(ghi=1.0, zenith=89.99, datetime_or_doy=times,
                          min_cos_zenith=0, max_zenith=100)
    expected = pd.DataFrame(np.array(
        [[4.78419761e+03, 1.65000000e-01, 1.00000000e+00]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)

    # check default behavior under hardest condition
    out = irradiance.erbs(ghi=1.0, zenith=90, datetime_or_doy=times)
    expected = pd.DataFrame(np.array(
        [[0., 1., 0.01163776]]),
        columns=columns, index=times)
    assert_frame_equal(out, expected)


def test_erbs_all_scalar():
    ghi = 1000
    zenith = 10
    doy = 180

    expected = OrderedDict()
    expected['dni'] = 8.42358014e+02
    expected['dhi'] = 1.70439297e+02
    expected['kt'] = 7.68919470e-01

    out = irradiance.erbs(ghi, zenith, doy)

    for k, v in out.items():
        assert_allclose(v, expected[k], 5)


def test_dirindex(times):
    ghi = pd.Series([0, 0, 1038.62, 254.53], index=times)
    ghi_clearsky = pd.Series(
        np.array([0., 79.73860422, 1042.48031487, 257.20751138]),
        index=times
    )
    dni_clearsky = pd.Series(
        np.array([0., 316.1949056, 939.95469881, 646.22886049]),
        index=times
    )
    zenith = pd.Series(
        np.array([124.0390863, 82.85457044, 10.56413562, 72.41687122]),
        index=times
    )
    pressure = 93193.
    tdew = 10.
    out = irradiance.dirindex(ghi, ghi_clearsky, dni_clearsky,
                              zenith, times, pressure=pressure,
                              temp_dew=tdew)
    dirint_close_values = irradiance.dirint(ghi, zenith, times,
                                            pressure=pressure,
                                            use_delta_kt_prime=True,
                                            temp_dew=tdew).values
    expected_out = np.array([np.nan, 0., 748.31562753, 630.72592644])

    tolerance = 1e-8
    assert np.allclose(out, expected_out, rtol=tolerance, atol=0,
                       equal_nan=True)
    tol_dirint = 0.2
    assert np.allclose(out.values, dirint_close_values, rtol=tol_dirint, atol=0,
                       equal_nan=True)


def test_dirindex_min_cos_zenith_max_zenith():
    # map out behavior under difficult conditions with various
    # limiting kwargs settings
    # times don't have any physical relevance
    times = pd.DatetimeIndex(['2014-06-24T12-0700', '2014-06-24T18-0700'])
    ghi = pd.Series([0, 1], index=times)
    ghi_clearsky = pd.Series([0, 1], index=times)
    dni_clearsky = pd.Series([0, 5], index=times)
    solar_zenith = pd.Series([90, 89.99], index=times)

    out = irradiance.dirindex(ghi, ghi_clearsky, dni_clearsky, solar_zenith,
                              times)
    expected = pd.Series([nan, nan], index=times)
    assert_series_equal(out, expected)

    out = irradiance.dirindex(ghi, ghi_clearsky, dni_clearsky, solar_zenith,
                              times, min_cos_zenith=0)
    expected = pd.Series([nan, nan], index=times)
    assert_series_equal(out, expected)

    out = irradiance.dirindex(ghi, ghi_clearsky, dni_clearsky, solar_zenith,
                              times, max_zenith=90)
    expected = pd.Series([nan, nan], index=times)
    assert_series_equal(out, expected)

    out = irradiance.dirindex(ghi, ghi_clearsky, dni_clearsky, solar_zenith,
                              times, min_cos_zenith=0, max_zenith=100)
    expected = pd.Series([nan, 5.], index=times)
    assert_series_equal(out, expected)


def test_dni():
    ghi = pd.Series([90, 100, 100, 100, 100])
    dhi = pd.Series([100, 90, 50, 50, 50])
    zenith = pd.Series([80, 100, 85, 70, 85])
    clearsky_dni = pd.Series([50, 50, 200, 50, 300])

    dni = irradiance.dni(ghi, dhi, zenith,
                         clearsky_dni=clearsky_dni, clearsky_tolerance=2)
    assert_series_equal(dni,
                        pd.Series([float('nan'), float('nan'), 400,
                                   146.190220008, 573.685662283]))

    dni = irradiance.dni(ghi, dhi, zenith)
    assert_series_equal(dni,
                        pd.Series([float('nan'), float('nan'), 573.685662283,
                                   146.190220008, 573.685662283]))


@pytest.mark.parametrize(
    'surface_tilt,surface_azimuth,solar_zenith,' +
    'solar_azimuth,aoi_expected,aoi_proj_expected',
    [(0, 0, 0, 0, 0, 1),
     (30, 180, 30, 180, 0, 1),
     (30, 180, 150, 0, 180, -1),
     (90, 0, 30, 60, 75.5224878, 0.25),
     (90, 0, 30, 170, 119.4987042, -0.4924038)])
def test_aoi_and_aoi_projection(surface_tilt, surface_azimuth, solar_zenith,
                                solar_azimuth, aoi_expected,
                                aoi_proj_expected):
    aoi = irradiance.aoi(surface_tilt, surface_azimuth, solar_zenith,
                         solar_azimuth)
    assert_allclose(aoi, aoi_expected, atol=1e-6)

    aoi_projection = irradiance.aoi_projection(
        surface_tilt, surface_azimuth, solar_zenith, solar_azimuth)
    assert_allclose(aoi_projection, aoi_proj_expected, atol=1e-6)


@pytest.fixture
def airmass_kt():
    # disc algorithm stopped at am=12. test am > 12 for out of range behavior
    return np.array([1, 5, 12, 20])


def test_kt_kt_prime_factor(airmass_kt):
    out = irradiance._kt_kt_prime_factor(airmass_kt)
    expected = np.array([ 0.999971,  0.723088,  0.548811,  0.471068])
    assert_allclose(out, expected, atol=1e-5)


def test_clearsky_index():
    ghi = np.array([-1., 0., 1., 500., 1000., np.nan])
    ghi_measured, ghi_modeled = np.meshgrid(ghi, ghi)
    # default max_clearsky_index
    with np.errstate(invalid='ignore', divide='ignore'):
        out = irradiance.clearsky_index(ghi_measured, ghi_modeled)
    expected = np.array(
        [[1.    , 0.    , 0.    , 0.    , 0.    , np.nan],
         [0.    , 0.    , 0.    , 0.    , 0.    , np.nan],
         [0.    , 0.    , 1.    , 2.    , 2.    , np.nan],
         [0.    , 0.    , 0.002 , 1.    , 2.    , np.nan],
         [0.    , 0.    , 0.001 , 0.5   , 1.    , np.nan],
         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    assert_allclose(out, expected, atol=0.001)
    # specify max_clearsky_index
    with np.errstate(invalid='ignore', divide='ignore'):
        out = irradiance.clearsky_index(ghi_measured, ghi_modeled,
                                        max_clearsky_index=1.5)
    expected = np.array(
        [[1.    , 0.    , 0.    , 0.    , 0.    , np.nan],
         [0.    , 0.    , 0.    , 0.    , 0.    , np.nan],
         [0.    , 0.    , 1.    , 1.5   , 1.5   , np.nan],
         [0.    , 0.    , 0.002 , 1.    , 1.5   , np.nan],
         [0.    , 0.    , 0.001 , 0.5   , 1.    , np.nan],
         [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan]])
    assert_allclose(out, expected, atol=0.001)
    # scalars
    out = irradiance.clearsky_index(10, 1000)
    expected = 0.01
    assert_allclose(out, expected, atol=0.001)
    # series
    times = pd.date_range(start='20180601', periods=2, freq='12H')
    ghi_measured = pd.Series([100,  500], index=times)
    ghi_modeled = pd.Series([500, 1000], index=times)
    out = irradiance.clearsky_index(ghi_measured, ghi_modeled)
    expected = pd.Series([0.2, 0.5], index=times)
    assert_series_equal(out, expected)


def test_clearness_index():
    ghi = np.array([-1, 0, 1, 1000])
    solar_zenith = np.array([180, 90, 89.999, 0])
    ghi, solar_zenith = np.meshgrid(ghi, solar_zenith)
    # default min_cos_zenith
    out = irradiance.clearness_index(ghi, solar_zenith, 1370)
    # np.set_printoptions(precision=3, floatmode='maxprec', suppress=True)
    expected = np.array(
        [[0.   , 0.   , 0.011, 2.   ],
         [0.   , 0.   , 0.011, 2.   ],
         [0.   , 0.   , 0.011, 2.   ],
         [0.   , 0.   , 0.001, 0.73 ]])
    assert_allclose(out, expected, atol=0.001)
    # specify min_cos_zenith
    with np.errstate(invalid='ignore', divide='ignore'):
        out = irradiance.clearness_index(ghi, solar_zenith, 1400,
                                         min_cos_zenith=0)
    expected = np.array(
        [[0.   ,   nan, 2.   , 2.   ],
         [0.   , 0.   , 2.   , 2.   ],
         [0.   , 0.   , 2.   , 2.   ],
         [0.   , 0.   , 0.001, 0.714]])
    assert_allclose(out, expected, atol=0.001)
    # specify max_clearness_index
    out = irradiance.clearness_index(ghi, solar_zenith, 1370,
                                     max_clearness_index=0.82)
    expected = np.array(
        [[ 0.   ,  0.   ,  0.011,  0.82 ],
         [ 0.   ,  0.   ,  0.011,  0.82 ],
         [ 0.   ,  0.   ,  0.011,  0.82 ],
         [ 0.   ,  0.   ,  0.001,  0.73 ]])
    assert_allclose(out, expected, atol=0.001)
    # specify min_cos_zenith and max_clearness_index
    with np.errstate(invalid='ignore', divide='ignore'):
        out = irradiance.clearness_index(ghi, solar_zenith, 1400,
                                         min_cos_zenith=0,
                                         max_clearness_index=0.82)
    expected = np.array(
        [[ 0.   ,    nan,  0.82 ,  0.82 ],
         [ 0.   ,  0.   ,  0.82 ,  0.82 ],
         [ 0.   ,  0.   ,  0.82 ,  0.82 ],
         [ 0.   ,  0.   ,  0.001,  0.714]])
    assert_allclose(out, expected, atol=0.001)
    # scalars
    out = irradiance.clearness_index(1000, 10, 1400)
    expected = 0.725
    assert_allclose(out, expected, atol=0.001)
    # series
    times = pd.date_range(start='20180601', periods=2, freq='12H')
    ghi = pd.Series([0, 1000], index=times)
    solar_zenith = pd.Series([90, 0], index=times)
    extra_radiation = pd.Series([1360, 1400], index=times)
    out = irradiance.clearness_index(ghi, solar_zenith, extra_radiation)
    expected = pd.Series([0, 0.714285714286], index=times)
    assert_series_equal(out, expected)


def test_clearness_index_zenith_independent(airmass_kt):
    clearness_index = np.array([-1, 0, .1, 1])
    clearness_index, airmass_kt = np.meshgrid(clearness_index, airmass_kt)
    out = irradiance.clearness_index_zenith_independent(clearness_index,
                                                        airmass_kt)
    expected = np.array(
        [[0.   , 0.   , 0.1  , 1.   ],
         [0.   , 0.   , 0.138, 1.383],
         [0.   , 0.   , 0.182, 1.822],
         [0.   , 0.   , 0.212, 2.   ]])
    assert_allclose(out, expected, atol=0.001)
    # test max_clearness_index
    out = irradiance.clearness_index_zenith_independent(
        clearness_index, airmass_kt, max_clearness_index=0.82)
    expected = np.array(
        [[ 0.   ,  0.   ,  0.1  ,  0.82 ],
         [ 0.   ,  0.   ,  0.138,  0.82 ],
         [ 0.   ,  0.   ,  0.182,  0.82 ],
         [ 0.   ,  0.   ,  0.212,  0.82 ]])
    assert_allclose(out, expected, atol=0.001)
    # scalars
    out = irradiance.clearness_index_zenith_independent(.4, 2)
    expected = 0.443
    assert_allclose(out, expected, atol=0.001)
    # series
    times = pd.date_range(start='20180601', periods=2, freq='12H')
    clearness_index = pd.Series([0, .5], index=times)
    airmass = pd.Series([np.nan, 2], index=times)
    out = irradiance.clearness_index_zenith_independent(clearness_index,
                                                        airmass)
    expected = pd.Series([np.nan, 0.553744437562], index=times)
    assert_series_equal(out, expected)
