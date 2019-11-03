import inspect
import os
from collections import OrderedDict

import numpy as np
from numpy import nan
import pandas as pd
import pytz

import pytest
from numpy.testing import assert_allclose
from pandas.util.testing import assert_frame_equal, assert_series_equal

from pvlib.location import Location
from pvlib import clearsky
from pvlib import solarposition
from pvlib import atmosphere
from pvlib import irradiance

from conftest import requires_scipy, requires_tables


def test_ineichen_series():
    times = pd.date_range(start='2014-06-24', end='2014-06-25', freq='3h',
                          tz='America/Phoenix')
    apparent_zenith = pd.Series(np.array(
        [124.0390863, 113.38779941, 82.85457044, 46.0467599, 10.56413562,
         34.86074109, 72.41687122, 105.69538659, 124.05614124]),
        index=times)
    am = pd.Series(np.array(
        [nan, nan, 6.97935524, 1.32355476, 0.93527685,
         1.12008114, 3.01614096, nan, nan]),
        index=times)
    expected = pd.DataFrame(np.
        array([[   0.        ,    0.        ,    0.        ],
               [   0.        ,    0.        ,    0.        ],
               [  65.49426624,  321.03665146,   25.56107792],
               [ 704.6968125 ,  888.55751839,   87.97474001],
               [1044.1230677 ,  952.88040806,  107.39369589],
               [ 853.02065704,  921.70446345,   96.72185108],
               [ 251.99427693,  655.19563211,   54.0667509 ],
               [   0.        ,    0.        ,    0.        ],
               [   0.        ,    0.        ,    0.        ]]),
                            columns=['ghi', 'dni', 'dhi'],
                            index=times)

    out = clearsky.ineichen(apparent_zenith, am, 3)
    assert_frame_equal(expected, out)


def test_ineichen_series_perez_enhancement():
    times = pd.date_range(start='2014-06-24', end='2014-06-25', freq='3h',
                          tz='America/Phoenix')
    apparent_zenith = pd.Series(np.array(
        [124.0390863, 113.38779941, 82.85457044, 46.0467599, 10.56413562,
         34.86074109, 72.41687122, 105.69538659, 124.05614124]),
        index=times)
    am = pd.Series(np.array(
        [nan, nan, 6.97935524, 1.32355476, 0.93527685,
         1.12008114, 3.01614096, nan, nan]),
        index=times)
    expected = pd.DataFrame(np.
        array([[   0.        ,    0.        ,    0.        ],
               [   0.        ,    0.        ,    0.        ],
               [  91.1249279 ,  321.03665146,   51.1917396 ],
               [ 716.46580547,  888.55751839,   99.7437328 ],
               [1053.42066073,  952.88040806,  116.69128857],
               [ 863.54692748,  921.70446345,  107.24812191],
               [ 271.06382275,  655.1956321 ,   73.13629663],
               [   0.        ,    0.        ,    0.        ],
               [   0.        ,    0.        ,    0.        ]]),
                            columns=['ghi', 'dni', 'dhi'],
                            index=times)

    out = clearsky.ineichen(apparent_zenith, am, 3, perez_enhancement=True)
    assert_frame_equal(expected, out)


def test_ineichen_scalar_input():
    expected = OrderedDict()
    expected['ghi'] = 1038.159219
    expected['dni'] = 941.843607
    expected['dhi'] = 110.624333

    out = clearsky.ineichen(10., 1., 3.)
    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_ineichen_nans():
    length = 4

    apparent_zenith = np.full(length, 10.)
    apparent_zenith[0] = np.nan

    linke_turbidity = np.full(length, 3.)
    linke_turbidity[1] = np.nan

    dni_extra = np.full(length, 1370.)
    dni_extra[2] = np.nan

    airmass_absolute = np.full(length, 1.)

    expected = OrderedDict()
    expected['ghi'] = np.full(length, np.nan)
    expected['dni'] = np.full(length, np.nan)
    expected['dhi'] = np.full(length, np.nan)

    expected['ghi'][length-1] = 1042.72590228
    expected['dni'][length-1] = 945.986614
    expected['dhi'][length-1] = 111.11095

    out = clearsky.ineichen(apparent_zenith, airmass_absolute,
                            linke_turbidity, dni_extra=dni_extra)

    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_ineichen_arrays():
    expected = OrderedDict()

    expected['ghi'] = (np.
        array([[[1095.77074798, 1054.17449885, 1014.15727338],
                [ 839.40909243,  807.54451692,  776.88954373],
                [ 190.27859353,  183.05548067,  176.10656239]],

               [[ 773.49041181,  625.19479557,  505.33080493],
                [ 592.52803177,  478.92699901,  387.10585505],
                [ 134.31520045,  108.56393694,   87.74977339]],

               [[ 545.9968869 ,  370.78162375,  251.79449885],
                [ 418.25788117,  284.03520249,  192.88577665],
                [  94.81136442,   64.38555328,   43.72365587]]]))

    expected['dni'] = (np.
        array([[[1014.38807396,  941.8436073 ,  860.78024436],
                [1014.38807396,  941.8436073 ,  860.78024436],
                [1014.38807396,  941.8436073 ,  860.78024436]],

               [[ 687.34698591,  418.98672583,  255.40211861],
                [ 687.34698591,  418.98672583,  255.40211861],
                [ 687.34698591,  418.98672583,  255.40211861]],

               [[ 458.44450061,  186.38962462,   75.78036626],
                [ 458.44450061,  186.38962462,   75.78036626],
                [ 458.44450061,  186.38962462,   75.78036626]]]))

    expected['dhi'] = (np.
        array([[[ 81.38267402, 112.33089156, 153.37702903],
                [ 62.3427452 ,  86.05045527, 117.49362079],
                [ 14.13195304,  19.50605461,  26.63364159]],

               [[ 86.1434259 , 206.20806974, 249.92868632],
                [ 65.98969272, 157.96454595, 191.45648133],
                [ 14.95864893,  35.80765553,  43.39966093]],

               [[ 87.55238629, 184.39199913, 176.01413259],
                [ 67.069019  , 141.25246629, 134.83464818],
                [ 15.20331233,  32.01933462,  30.56453337]]]))

    apparent_zenith = np.linspace(0, 80, 3)
    airmass_absolute = np.linspace(1, 10, 3)
    linke_turbidity = np.linspace(2, 4, 3)

    apparent_zenith, airmass_absolute, linke_turbidity = \
        np.meshgrid(apparent_zenith, airmass_absolute, linke_turbidity)

    out = clearsky.ineichen(apparent_zenith, airmass_absolute, linke_turbidity)

    for k, v in expected.items():
        print(expected[k])
        print(out[k])
        assert_allclose(expected[k], out[k])


def test_ineichen_dni_extra():
    expected = pd.DataFrame(
        np.array([[1042.72590228,  945.98661437, 111.11095021]]),
        columns=['ghi', 'dni', 'dhi'])

    out = clearsky.ineichen(10, 1, 3, dni_extra=pd.Series(1370))
    assert_frame_equal(expected, out)


def test_ineichen_altitude():
    expected = pd.DataFrame(
        np.array([[1134.24312405,  994.48564998,  154.86594569]]),
        columns=['ghi', 'dni', 'dhi'])

    out = clearsky.ineichen(10, 1, 3, altitude=pd.Series(2000))
    assert_frame_equal(expected, out)


@requires_tables
def test_lookup_linke_turbidity():
    times = pd.date_range(start='2014-06-24', end='2014-06-25',
                          freq='12h', tz='America/Phoenix')
    # expect same value on 2014-06-24 0000 and 1200, and
    # diff value on 2014-06-25
    expected = pd.Series(
        np.array([3.11803278689, 3.11803278689, 3.13114754098]), index=times
    )
    out = clearsky.lookup_linke_turbidity(times, 32.125, -110.875)
    assert_series_equal(expected, out)


@requires_tables
def test_lookup_linke_turbidity_leapyear():
    times = pd.date_range(start='2016-06-24', end='2016-06-25',
                          freq='12h', tz='America/Phoenix')
    # expect same value on 2016-06-24 0000 and 1200, and
    # diff value on 2016-06-25
    expected = pd.Series(
        np.array([3.11803278689, 3.11803278689, 3.13114754098]), index=times
    )
    out = clearsky.lookup_linke_turbidity(times, 32.125, -110.875)
    assert_series_equal(expected, out)


@requires_tables
def test_lookup_linke_turbidity_nointerp():
    times = pd.date_range(start='2014-06-24', end='2014-06-25',
                          freq='12h', tz='America/Phoenix')
    # expect same value for all days
    expected = pd.Series(np.array([3., 3., 3.]), index=times)
    out = clearsky.lookup_linke_turbidity(times, 32.125, -110.875,
                                          interp_turbidity=False)
    assert_series_equal(expected, out)


@requires_tables
def test_lookup_linke_turbidity_months():
    times = pd.date_range(start='2014-04-01', end='2014-07-01',
                          freq='1M', tz='America/Phoenix')
    expected = pd.Series(
        np.array([2.89918032787, 2.97540983607, 3.19672131148]), index=times
    )
    out = clearsky.lookup_linke_turbidity(times, 32.125, -110.875)
    assert_series_equal(expected, out)


@requires_tables
def test_lookup_linke_turbidity_months_leapyear():
    times = pd.date_range(start='2016-04-01', end='2016-07-01',
                          freq='1M', tz='America/Phoenix')
    expected = pd.Series(
        np.array([2.89918032787, 2.97540983607, 3.19672131148]), index=times
    )
    out = clearsky.lookup_linke_turbidity(times, 32.125, -110.875)
    assert_series_equal(expected, out)


@requires_tables
def test_lookup_linke_turbidity_nointerp_months():
    times = pd.date_range(start='2014-04-10', end='2014-07-10',
                          freq='1M', tz='America/Phoenix')
    expected = pd.Series(np.array([2.85, 2.95, 3.]), index=times)
    out = clearsky.lookup_linke_turbidity(times, 32.125, -110.875,
                                          interp_turbidity=False)
    assert_series_equal(expected, out)
    # changing the dates shouldn't matter if interp=False
    times = pd.date_range(start='2014-04-05', end='2014-07-05',
                          freq='1M', tz='America/Phoenix')
    out = clearsky.lookup_linke_turbidity(times, 32.125, -110.875,
                                          interp_turbidity=False)
    assert_series_equal(expected, out)


def test_haurwitz():
    apparent_solar_elevation = np.array([-20, -0.05, -0.001, 5, 10, 30, 50, 90])
    apparent_solar_zenith = 90 - apparent_solar_elevation
    data_in = pd.DataFrame(data=apparent_solar_zenith,
                           index=apparent_solar_zenith,
                           columns=['apparent_zenith'])
    expected = pd.DataFrame(np.array([0.,
                                      0.,
                                      0.,
                                      48.6298687941956,
                                      135.741748091813,
                                      487.894132885425,
                                      778.766689344363,
                                      1035.09203253450]),
                            columns=['ghi'],
                            index=apparent_solar_zenith)
    out = clearsky.haurwitz(data_in['apparent_zenith'])
    assert_frame_equal(expected, out)


def test_simplified_solis_scalar_elevation():
    expected = OrderedDict()
    expected['ghi'] = 1064.653145
    expected['dni'] = 959.335463
    expected['dhi'] = 129.125602

    out = clearsky.simplified_solis(80)
    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_simplified_solis_scalar_neg_elevation():
    expected = OrderedDict()
    expected['ghi'] = 0
    expected['dni'] = 0
    expected['dhi'] = 0

    out = clearsky.simplified_solis(-10)
    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_simplified_solis_series_elevation():
    expected = pd.DataFrame(
        np.array([[959.335463,  1064.653145,  129.125602]]),
        columns=['dni', 'ghi', 'dhi'])
    expected = expected[['ghi', 'dni', 'dhi']]

    out = clearsky.simplified_solis(pd.Series(80))
    assert_frame_equal(expected, out)


def test_simplified_solis_dni_extra():
    expected = pd.DataFrame(np.array([[963.555414,  1069.33637,  129.693603]]),
                            columns=['dni', 'ghi', 'dhi'])
    expected = expected[['ghi', 'dni', 'dhi']]

    out = clearsky.simplified_solis(80, dni_extra=pd.Series(1370))
    assert_frame_equal(expected, out)


def test_simplified_solis_pressure():
    expected = pd.DataFrame(np.
        array([[  964.26930718,  1067.96543669,   127.22841797],
               [  961.88811874,  1066.36847963,   128.1402539 ],
               [  959.58112234,  1064.81837558,   129.0304193 ]]),
                            columns=['dni', 'ghi', 'dhi'])
    expected = expected[['ghi', 'dni', 'dhi']]

    out = clearsky.simplified_solis(
        80, pressure=pd.Series([95000, 98000, 101000]))
    assert_frame_equal(expected, out)


def test_simplified_solis_aod700():
    expected = pd.DataFrame(np.
        array([[ 1056.61710493,  1105.7229086 ,    64.41747323],
               [ 1007.50558875,  1085.74139063,   102.96233698],
               [  959.3354628 ,  1064.65314509,   129.12560167],
               [  342.45810926,   638.63409683,    77.71786575],
               [   55.24140911,     7.5413313 ,     0.        ]]),
                            columns=['dni', 'ghi', 'dhi'])
    expected = expected[['ghi', 'dni', 'dhi']]

    aod700 = pd.Series([0.0, 0.05, 0.1, 1, 10])
    out = clearsky.simplified_solis(80, aod700=aod700)
    assert_frame_equal(expected, out)


def test_simplified_solis_precipitable_water():
    expected = pd.DataFrame(np.
        array([[ 1001.15353307,  1107.84678941,   128.58887606],
               [ 1001.15353307,  1107.84678941,   128.58887606],
               [  983.51027357,  1089.62306672,   129.08755996],
               [  959.3354628 ,  1064.65314509,   129.12560167],
               [  872.02335029,   974.18046717,   125.63581346]]),
                            columns=['dni', 'ghi', 'dhi'])
    expected = expected[['ghi', 'dni', 'dhi']]

    out = clearsky.simplified_solis(
        80, precipitable_water=pd.Series([0.0, 0.2, 0.5, 1.0, 5.0]))
    assert_frame_equal(expected, out)


def test_simplified_solis_small_scalar_pw():

    expected = OrderedDict()
    expected['ghi'] = 1107.84678941
    expected['dni'] = 1001.15353307
    expected['dhi'] = 128.58887606

    out = clearsky.simplified_solis(80, precipitable_water=0.1)
    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_simplified_solis_return_arrays():
    expected = OrderedDict()

    expected['ghi'] = np.array([[ 1148.40081325,   913.42330823],
                                [  965.48550828,   760.04527609]])

    expected['dni'] = np.array([[ 1099.25706525,   656.24601381],
                                [  915.31689149,   530.31697378]])

    expected['dhi'] = np.array([[   64.1063074 ,   254.6186615 ],
                                [   62.75642216,   232.21931597]])

    aod700 = np.linspace(0, 0.5, 2)
    precipitable_water = np.linspace(0, 10, 2)

    aod700, precipitable_water = np.meshgrid(aod700, precipitable_water)

    out = clearsky.simplified_solis(80, aod700, precipitable_water)

    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_simplified_solis_nans_arrays():

    # construct input arrays that each have 1 nan offset from each other,
    # the last point is valid for all arrays

    length = 6

    apparent_elevation = np.full(length, 80.)
    apparent_elevation[0] = np.nan

    aod700 = np.full(length, 0.1)
    aod700[1] = np.nan

    precipitable_water = np.full(length, 0.5)
    precipitable_water[2] = np.nan

    pressure = np.full(length, 98000.)
    pressure[3] = np.nan

    dni_extra = np.full(length, 1370.)
    dni_extra[4] = np.nan

    expected = OrderedDict()
    expected['ghi'] = np.full(length, np.nan)
    expected['dni'] = np.full(length, np.nan)
    expected['dhi'] = np.full(length, np.nan)

    expected['ghi'][length-1] = 1096.022736
    expected['dni'][length-1] = 990.306854
    expected['dhi'][length-1] = 128.664594

    out = clearsky.simplified_solis(apparent_elevation, aod700,
                                    precipitable_water, pressure, dni_extra)

    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_simplified_solis_nans_series():

    # construct input arrays that each have 1 nan offset from each other,
    # the last point is valid for all arrays

    length = 6

    apparent_elevation = pd.Series(np.full(length, 80.))
    apparent_elevation[0] = np.nan

    aod700 = np.full(length, 0.1)
    aod700[1] = np.nan

    precipitable_water = np.full(length, 0.5)
    precipitable_water[2] = np.nan

    pressure = np.full(length, 98000.)
    pressure[3] = np.nan

    dni_extra = np.full(length, 1370.)
    dni_extra[4] = np.nan

    expected = OrderedDict()
    expected['ghi'] = np.full(length, np.nan)
    expected['dni'] = np.full(length, np.nan)
    expected['dhi'] = np.full(length, np.nan)

    expected['ghi'][length-1] = 1096.022736
    expected['dni'][length-1] = 990.306854
    expected['dhi'][length-1] = 128.664594

    expected = pd.DataFrame.from_dict(expected)

    out = clearsky.simplified_solis(apparent_elevation, aod700,
                                    precipitable_water, pressure, dni_extra)

    assert_frame_equal(expected, out)


@requires_tables
def test_linke_turbidity_corners():
    """Test Linke turbidity corners out of bounds."""
    months = pd.DatetimeIndex('%d/1/2016' % (m + 1) for m in range(12))

    def monthly_lt_nointerp(lat, lon, time=months):
        """monthly Linke turbidity factor without time interpolation"""
        return clearsky.lookup_linke_turbidity(
            time, lat, lon, interp_turbidity=False
        )

    # Northwest
    assert np.allclose(
        monthly_lt_nointerp(90, -180),
        [1.9, 1.9, 1.9, 2.0, 2.05, 2.05, 2.1, 2.1, 2.0, 1.95, 1.9, 1.9])
    # Southwest
    assert np.allclose(
        monthly_lt_nointerp(-90, -180),
        [1.35, 1.3, 1.45, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.4, 1.4, 1.3])
    # Northeast
    assert np.allclose(
        monthly_lt_nointerp(90, 180),
        [1.9, 1.9, 1.9, 2.0, 2.05, 2.05, 2.1, 2.1, 2.0, 1.95, 1.9, 1.9])
    # Southeast
    assert np.allclose(
        monthly_lt_nointerp(-90, 180),
        [1.35, 1.7, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.35, 1.7])
    # test out of range exceptions at corners
    with pytest.raises(IndexError):
        monthly_lt_nointerp(91, -122)  # exceeds max latitude
    with pytest.raises(IndexError):
        monthly_lt_nointerp(38.2, 181)  # exceeds max longitude
    with pytest.raises(IndexError):
        monthly_lt_nointerp(-91, -122)  # exceeds min latitude
    with pytest.raises(IndexError):
        monthly_lt_nointerp(38.2, -181)  # exceeds min longitude


def test_degrees_to_index_1():
    """Test that _degrees_to_index raises an error when something other than
    'latitude' or 'longitude' is passed."""
    with pytest.raises(IndexError):  # invalid value for coordinate argument
        clearsky._degrees_to_index(degrees=22.0, coordinate='width')


@pytest.fixture
def detect_clearsky_data():
    test_dir = os.path.dirname(os.path.abspath(
        inspect.getfile(inspect.currentframe())))
    file = os.path.join(test_dir, '..', 'data', 'detect_clearsky_data.csv')
    expected = pd.read_csv(file, index_col=0, parse_dates=True, comment='#')
    expected = expected.tz_localize('UTC').tz_convert('Etc/GMT+7')
    metadata = {}
    with open(file) as f:
        for line in f:
            if line.startswith('#'):
                key, value = line.strip('# \n').split(':')
                metadata[key] = float(value)
            else:
                break
    metadata['window_length'] = int(metadata['window_length'])
    loc = Location(metadata['latitude'], metadata['longitude'],
                   altitude=metadata['elevation'])
    # specify turbidity to guard against future lookup changes
    cs = loc.get_clearsky(expected.index, linke_turbidity=2.658197)
    return expected, cs


@requires_scipy
def test_detect_clearsky(detect_clearsky_data):
    expected, cs = detect_clearsky_data
    clear_samples = clearsky.detect_clearsky(
        expected['GHI'], cs['ghi'], cs.index, 10)
    assert_series_equal(expected['Clear or not'], clear_samples,
                        check_dtype=False, check_names=False)


@requires_scipy
def test_detect_clearsky_components(detect_clearsky_data):
    expected, cs = detect_clearsky_data
    clear_samples, components, alpha = clearsky.detect_clearsky(
        expected['GHI'], cs['ghi'], cs.index, 10, return_components=True)
    assert_series_equal(expected['Clear or not'], clear_samples,
                        check_dtype=False, check_names=False)
    assert isinstance(components, OrderedDict)
    assert np.allclose(alpha, 0.9633903181941296)


@requires_scipy
def test_detect_clearsky_iterations(detect_clearsky_data):
    expected, cs = detect_clearsky_data
    alpha = 1.0448
    with pytest.warns(RuntimeWarning):
        clear_samples = clearsky.detect_clearsky(
            expected['GHI'], cs['ghi']*alpha, cs.index, 10, max_iterations=1)
    assert (clear_samples[:'2012-04-01 10:41:00'] == True).all()
    assert (clear_samples['2012-04-01 10:42:00':] == False).all()
    clear_samples = clearsky.detect_clearsky(
            expected['GHI'], cs['ghi']*alpha, cs.index, 10, max_iterations=20)
    assert_series_equal(expected['Clear or not'], clear_samples,
                        check_dtype=False, check_names=False)


@requires_scipy
def test_detect_clearsky_kwargs(detect_clearsky_data):
    expected, cs = detect_clearsky_data
    clear_samples = clearsky.detect_clearsky(
        expected['GHI'], cs['ghi'], cs.index, 10,
        mean_diff=1000, max_diff=1000, lower_line_length=-1000,
        upper_line_length=1000, var_diff=10, slope_dev=1000)
    assert clear_samples.all()


@requires_scipy
def test_detect_clearsky_window(detect_clearsky_data):
    expected, cs = detect_clearsky_data
    clear_samples = clearsky.detect_clearsky(
        expected['GHI'], cs['ghi'], cs.index, 3)
    expected = expected['Clear or not'].copy()
    expected.iloc[-3:] = True
    assert_series_equal(expected, clear_samples,
                        check_dtype=False, check_names=False)


@requires_scipy
def test_detect_clearsky_arrays(detect_clearsky_data):
    expected, cs = detect_clearsky_data
    clear_samples = clearsky.detect_clearsky(
        expected['GHI'].values, cs['ghi'].values, cs.index, 10)
    assert isinstance(clear_samples, np.ndarray)
    assert (clear_samples == expected['Clear or not'].values).all()


@requires_scipy
def test_detect_clearsky_irregular_times(detect_clearsky_data):
    expected, cs = detect_clearsky_data
    times = cs.index.values.copy()
    times[0] += 10**9
    times = pd.DatetimeIndex(times)
    with pytest.raises(NotImplementedError):
        clear_samples = clearsky.detect_clearsky(
            expected['GHI'].values, cs['ghi'].values, times, 10)


def test_bird():
    """Test Bird/Hulstrom Clearsky Model"""
    times = pd.date_range(start='1/1/2015 0:00', end='12/31/2015 23:00',
                          freq='H')
    tz = -7  # test timezone
    gmt_tz = pytz.timezone('Etc/GMT%+d' % -(tz))
    times = times.tz_localize(gmt_tz)  # set timezone
    # match test data from BIRD_08_16_2012.xls
    latitude = 40.
    longitude = -105.
    press_mB = 840.
    o3_cm = 0.3
    h2o_cm = 1.5
    aod_500nm = 0.1
    aod_380nm = 0.15
    b_a = 0.85
    alb = 0.2
    eot = solarposition.equation_of_time_spencer71(times.dayofyear)
    hour_angle = solarposition.hour_angle(times, longitude, eot) - 0.5 * 15.
    declination = solarposition.declination_spencer71(times.dayofyear)
    zenith = solarposition.solar_zenith_analytical(
        np.deg2rad(latitude), np.deg2rad(hour_angle), declination
    )
    zenith = np.rad2deg(zenith)
    airmass = atmosphere.get_relative_airmass(zenith, model='kasten1966')
    etr = irradiance.get_extra_radiation(times)
    # test Bird with time series data
    field_names = ('dni', 'direct_horizontal', 'ghi', 'dhi')
    irrads = clearsky.bird(
        zenith, airmass, aod_380nm, aod_500nm, h2o_cm, o3_cm, press_mB * 100.,
        etr, b_a, alb
    )
    Eb, Ebh, Gh, Dh = (irrads[_] for _ in field_names)
    clearsky_path = os.path.dirname(os.path.abspath(__file__))
    pvlib_path = os.path.dirname(clearsky_path)
    data_path = os.path.join(pvlib_path, 'data', 'BIRD_08_16_2012.csv')
    testdata = pd.read_csv(data_path, usecols=range(1, 26), header=1).dropna()
    testdata.index = times[1:48]
    assert np.allclose(testdata['DEC'], np.rad2deg(declination[1:48]))
    assert np.allclose(testdata['EQT'], eot[1:48], rtol=1e-4)
    assert np.allclose(testdata['Hour Angle'], hour_angle[1:48])
    assert np.allclose(testdata['Zenith Ang'], zenith[1:48])
    dawn = zenith < 88.
    dusk = testdata['Zenith Ang'] < 88.
    am = pd.Series(np.where(dawn, airmass, 0.), index=times).fillna(0.0)
    assert np.allclose(
        testdata['Air Mass'].where(dusk, 0.), am[1:48], rtol=1e-3
    )
    direct_beam = pd.Series(np.where(dawn, Eb, 0.), index=times).fillna(0.)
    assert np.allclose(
        testdata['Direct Beam'].where(dusk, 0.), direct_beam[1:48], rtol=1e-3
    )
    direct_horz = pd.Series(np.where(dawn, Ebh, 0.), index=times).fillna(0.)
    assert np.allclose(
        testdata['Direct Hz'].where(dusk, 0.), direct_horz[1:48], rtol=1e-3
    )
    global_horz = pd.Series(np.where(dawn, Gh, 0.), index=times).fillna(0.)
    assert np.allclose(
        testdata['Global Hz'].where(dusk, 0.), global_horz[1:48], rtol=1e-3
    )
    diffuse_horz = pd.Series(np.where(dawn, Dh, 0.), index=times).fillna(0.)
    assert np.allclose(
        testdata['Dif Hz'].where(dusk, 0.), diffuse_horz[1:48], rtol=1e-3
    )
    # test keyword parameters
    irrads2 = clearsky.bird(
        zenith, airmass, aod_380nm, aod_500nm, h2o_cm, dni_extra=etr
    )
    Eb2, Ebh2, Gh2, Dh2 = (irrads2[_] for _ in field_names)
    clearsky_path = os.path.dirname(os.path.abspath(__file__))
    pvlib_path = os.path.dirname(clearsky_path)
    data_path = os.path.join(pvlib_path, 'data', 'BIRD_08_16_2012_patm.csv')
    testdata2 = pd.read_csv(data_path, usecols=range(1, 26), header=1).dropna()
    testdata2.index = times[1:48]
    direct_beam2 = pd.Series(np.where(dawn, Eb2, 0.), index=times).fillna(0.)
    assert np.allclose(
        testdata2['Direct Beam'].where(dusk, 0.), direct_beam2[1:48], rtol=1e-3
    )
    direct_horz2 = pd.Series(np.where(dawn, Ebh2, 0.), index=times).fillna(0.)
    assert np.allclose(
        testdata2['Direct Hz'].where(dusk, 0.), direct_horz2[1:48], rtol=1e-3
    )
    global_horz2 = pd.Series(np.where(dawn, Gh2, 0.), index=times).fillna(0.)
    assert np.allclose(
        testdata2['Global Hz'].where(dusk, 0.), global_horz2[1:48], rtol=1e-3
    )
    diffuse_horz2 = pd.Series(np.where(dawn, Dh2, 0.), index=times).fillna(0.)
    assert np.allclose(
        testdata2['Dif Hz'].where(dusk, 0.), diffuse_horz2[1:48], rtol=1e-3
    )
    # test scalars just at noon
    # XXX: calculations start at 12am so noon is at index = 12
    irrads3 = clearsky.bird(
        zenith[12], airmass[12], aod_380nm, aod_500nm, h2o_cm, dni_extra=etr[12]
    )
    Eb3, Ebh3, Gh3, Dh3 = (irrads3[_] for _ in field_names)
    # XXX: testdata starts at 1am so noon is at index = 11
    np.allclose(
        [Eb3, Ebh3, Gh3, Dh3],
        testdata2[['Direct Beam', 'Direct Hz', 'Global Hz', 'Dif Hz']].iloc[11],
        rtol=1e-3)
    return pd.DataFrame({'Eb': Eb, 'Ebh': Ebh, 'Gh': Gh, 'Dh': Dh}, index=times)
