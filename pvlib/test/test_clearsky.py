from collections import OrderedDict

import numpy as np
import pandas as pd

import xlrd
import os

import pytest
from numpy.testing import assert_almost_equal, assert_allclose
from pandas.util.testing import assert_frame_equal, assert_series_equal

from pvlib.location import Location
from pvlib import clearsky
from pvlib import solarposition
from pvlib import atmosphere

from conftest import requires_scipy


def test_ineichen_series():
    tus = Location(32.2, -111, 'US/Arizona', 700)
    times = pd.date_range(start='2014-06-24', end='2014-06-25', freq='3h')
    times_localized = times.tz_localize(tus.tz)
    ephem_data = solarposition.get_solarposition(times_localized, tus.latitude,
                                                 tus.longitude)
    am = atmosphere.relativeairmass(ephem_data['apparent_zenith'])
    am = atmosphere.absoluteairmass(am, atmosphere.alt2pres(tus.altitude))
    expected = pd.DataFrame(np.
        array([[    0.        ,     0.        ,     0.        ],
               [    0.        ,     0.        ,     0.        ],
               [   91.12492792,   321.16092181,    51.17628184],
               [  716.46580533,   888.90147035,    99.5050056 ],
               [ 1053.42066043,   953.24925854,   116.32868969],
               [  863.54692781,   922.06124712,   106.95536561],
               [  271.06382274,   655.44925241,    73.05968071],
               [    0.        ,     0.        ,     0.        ],
               [    0.        ,     0.        ,     0.        ]]),
                            columns=['ghi', 'dni', 'dhi'],
                            index=times_localized)

    out = clearsky.ineichen(ephem_data['apparent_zenith'], am, 3)
    assert_frame_equal(expected, out)


def test_ineichen_scalar_input():
    expected = OrderedDict()
    expected['ghi'] = 1048.592893113678
    expected['dni'] = 942.2081860378344
    expected['dhi'] = 120.6989665520498

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

    expected['ghi'][length-1] = 1053.205472
    expected['dni'][length-1] = 946.352797
    expected['dhi'][length-1] = 121.2299

    out = clearsky.ineichen(apparent_zenith, airmass_absolute,
                            linke_turbidity, dni_extra=dni_extra)

    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_ineichen_arrays():
    expected = OrderedDict()

    expected['ghi'] = (np.
        array([[[ 1106.78342709,  1064.7691287 ,  1024.34972343],
                [  847.84529406,   815.66047425,   784.69741345],
                [  192.19092519,   184.89521884,   177.87646277]],

               [[  959.12310134,   775.2374976 ,   626.60692548],
                [  734.73092205,   593.86637713,   480.00875328],
                [  166.54997871,   134.61857872,   108.80915072]],

               [[ 1026.15144142,   696.85030591,   473.22483724],
                [  786.0776095 ,   533.81830453,   362.51125692],
                [  178.18932781,   121.00678573,    82.17463061]]]))

    expected['dni'] = (np.
        array([[[ 1024.58284359,   942.20818604,   861.11344424],
                [ 1024.58284359,   942.20818604,   861.11344424],
                [ 1024.58284359,   942.20818604,   861.11344424]],

               [[  687.61305142,   419.14891162,   255.50098235],
                [  687.61305142,   419.14891162,   255.50098235],
                [  687.61305142,   419.14891162,   255.50098235]],

               [[  458.62196014,   186.46177428,    75.80970012],
                [  458.62196014,   186.46177428,    75.80970012],
                [  458.62196014,   186.46177428,    75.80970012]]]))

    expected['dhi'] = (np.
            array([[[  82.20058349,  122.56094266,  163.23627919],
                    [  62.96930021,   93.88712907,  125.04624459],
                    [  14.27398153,   21.28248435,   28.34568241]],

                   [[ 271.51004993,  356.08858598,  371.10594313],
                    [ 207.988765  ,  272.77968255,  284.28364554],
                    [  47.14722539,   61.83413404,   64.44187075]],

                   [[ 567.52948128,  510.38853163,  397.41513712],
                    [ 434.75280544,  390.98029849,  304.4376574 ],
                    [  98.5504602 ,   88.62803842,   69.01041434]]]))

    apparent_zenith = np.linspace(0, 80, 3)
    airmass_absolute = np.linspace(1, 10, 3)
    linke_turbidity = np.linspace(2, 4, 3)

    apparent_zenith, airmass_absolute, linke_turbidity = \
        np.meshgrid(apparent_zenith, airmass_absolute, linke_turbidity)

    out = clearsky.ineichen(apparent_zenith, airmass_absolute, linke_turbidity)

    for k, v in expected.items():
        assert_allclose(expected[k], out[k])


def test_ineichen_dni_extra():
    expected = pd.DataFrame(
        np.array([[ 1053.20547182,   946.35279683,   121.22990042]]),
        columns=['ghi', 'dni', 'dhi'])

    out = clearsky.ineichen(10, 1, 3, dni_extra=pd.Series(1370))
    assert_frame_equal(expected, out)


def test_ineichen_altitude():
    expected = pd.DataFrame(
        np.array([[ 1145.64245696,   994.95377835,   165.80426215]]),
        columns=['ghi', 'dni', 'dhi'])

    out = clearsky.ineichen(10, 1, 3, altitude=pd.Series(2000))
    assert_frame_equal(expected, out)


@requires_scipy
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


@requires_scipy
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


@requires_scipy
def test_lookup_linke_turbidity_nointerp():
    times = pd.date_range(start='2014-06-24', end='2014-06-25',
                          freq='12h', tz='America/Phoenix')
    # expect same value for all days
    expected = pd.Series(np.array([3., 3., 3.]), index=times)
    out = clearsky.lookup_linke_turbidity(times, 32.125, -110.875,
                                          interp_turbidity=False)
    assert_series_equal(expected, out)


@requires_scipy
def test_lookup_linke_turbidity_months():
    times = pd.date_range(start='2014-04-01', end='2014-07-01',
                          freq='1M', tz='America/Phoenix')
    expected = pd.Series(
        np.array([2.89918032787, 2.97540983607, 3.19672131148]), index=times
    )
    out = clearsky.lookup_linke_turbidity(times, 32.125, -110.875)
    assert_series_equal(expected, out)


@requires_scipy
def test_lookup_linke_turbidity_months_leapyear():
    times = pd.date_range(start='2016-04-01', end='2016-07-01',
                          freq='1M', tz='America/Phoenix')
    expected = pd.Series(
        np.array([2.89918032787, 2.97540983607, 3.19672131148]), index=times
    )
    out = clearsky.lookup_linke_turbidity(times, 32.125, -110.875)
    assert_series_equal(expected, out)


@requires_scipy
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
    tus = Location(32.2, -111, 'US/Arizona', 700)
    times = pd.date_range(start='2014-06-24', end='2014-06-25', freq='3h')
    times_localized = times.tz_localize(tus.tz)
    ephem_data = solarposition.get_solarposition(times_localized, tus.latitude,
                                                 tus.longitude)
    expected = pd.DataFrame(np.array([[0.],
                                      [0.],
                                      [82.85934048],
                                      [699.74514735],
                                      [1016.50198354],
                                      [838.32103769],
                                      [271.90853863],
                                      [0.],
                                      [0.]]),
                             columns=['ghi'], index=times_localized)
    out = clearsky.haurwitz(ephem_data['zenith'])
    assert_frame_equal(expected, out)


def test_simplified_solis_series_elevation():
    tus = Location(32.2, -111, 'US/Arizona', 700)
    times = pd.date_range(start='2014-06-24', end='2014-06-25', freq='3h')
    times_localized = times.tz_localize(tus.tz)
    ephem_data = solarposition.get_solarposition(times_localized, tus.latitude,
                                                 tus.longitude)
    expected = pd.DataFrame(
        np.array([[    0.        ,     0.        ,     0.        ],
                  [    0.        ,     0.        ,     0.        ],
                  [  377.80060035,    79.91931339,    42.77453223],
                  [  869.47538184,   706.37903999,   110.05635962],
                  [  958.89448856,  1062.44821373,   129.02349236],
                  [  913.3209839 ,   860.48978599,   118.94598678],
                  [  634.01066762,   256.00505836,    72.18396705],
                  [    0.        ,     0.        ,     0.        ],
                  [    0.        ,     0.        ,     0.        ]]),
                            columns=['dni', 'ghi', 'dhi'],
                            index=times_localized)
    expected = expected[['dhi', 'dni', 'ghi']]

    out = clearsky.simplified_solis(ephem_data['apparent_elevation'])
    assert_frame_equal(expected, out)


def test_simplified_solis_scalar_elevation():
    expected = OrderedDict()
    expected['ghi'] = 1064.653145
    expected['dni'] = 959.335463
    expected['dhi'] = 129.125602

    out = clearsky.simplified_solis(80)
    for k, v in expected.items():
        yield assert_allclose, expected[k], out[k]


def test_simplified_solis_scalar_neg_elevation():
    expected = OrderedDict()
    expected['ghi'] = 0
    expected['dni'] = 0
    expected['dhi'] = 0

    out = clearsky.simplified_solis(-10)
    for k, v in expected.items():
        yield assert_allclose, expected[k], out[k]


def test_simplified_solis_series_elevation():
    expected = pd.DataFrame(np.array([[959.335463,  1064.653145,  129.125602]]),
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


@requires_scipy
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


"""
Richard E. Bird 
        Clear Sky Broadband
        Solar Radiation Model 
From the publication "A Simplified Clear Sky model for Direct and Diffuse
Insolation on Horizontal Surfaces" by R.E. Bird and R.L Hulstrom, SERI Technical
Report SERI/TR-642-761, Feb 1991. Solar Energy Research Institute, Golden, CO.

The model is based on comparisons with results from rigourous radiative transfer
codes. It is composed of simple algebraic expressions with 10 User defined
inputs (green cells to left). Model results should be expected to agree within
+/-10% with rigourous radiative transfer codes. The model computes solar
radiation for every hour of the year, based on the 10 user input parameters.

The graphical presentation includes diurnal clear sky radiation patterns for
every day of the year. The user may copy cells of interest or the graph and
paste it to an unprotected worksheet for manipulation.

The workbook is PROTECTED using the password BIRD (all caps). To generate data
for the entire year, choose TOOLS, PROTECTION, UNPROTECT and enter the password.
Copy row 49 and paste it from row 50 all the way down to row 8761. 

NOTE: Columns L to U contain intermediate calculations and have been collapsed
down for convenient pressentation of model results.

Contact:
Daryl R. Myers,
National Renewable Energy Laboratory,
1617 Cole Blvd. MS 3411, Golden CO 80401
(303)-384-6768 e-mail daryl_myers@nrel.gov

http://rredc.nrel.gov/solar/models/clearsky/
http://rredc.nrel.gov/solar/pubs/pdfs/tr-642-761.pdf
http://rredc.nrel.gov/solar/models/clearsky/error_reports.html
"""


def test_bird():
    dt = pd.DatetimeIndex(start='1/1/2015 0:00', end='12/31/2015 23:00', freq='H')
    kwargs = {
        'lat': 40, 'lon': -105, 'tz': -7,
        'press_mB': 840,
        'o3_cm': 0.3, 'h2o_cm': 1.5,
        'aod_500nm':  0.1, 'aod_380nm':  0.15,
        'b_a': 0.85,
        'alb': 0.2
    }
    Eb, Ebh, Gh, Dh, tv = bird(dt.dayofyear, np.array(range(24)*365), **kwargs)
    day_angle, declination, eqt, hour_angle, zenith, airmass = tv
    clearsky_path = os.path.dirname(os.path.abspath(__file__))
    pvlib_path = os.path.dirname(clearsky_path)
    wb = xlrd.open_workbook(
        os.path.join(pvlib_path, 'data', 'BIRD_08_16_2012.xls')
    )
    sheet = wb.sheets()[0]
    headers = [h.value for h in sheet.row(1)][4:]
    testdata = pd.DataFrame({h: [c.value for c in sheet.col(n + 4, 2, 49)]
                            for n, h in enumerate(headers)},
                            index=dt[1:48])
    assert np.allclose(testdata['Dangle'], day_angle[1:48])
    assert np.allclose(testdata['DEC'], declination[1:48])
    assert np.allclose(testdata['EQT'], eqt[1:48])
    assert np.allclose(testdata['Hour Angle'], hour_angle[1:48])
    assert np.allclose(testdata['Zenith Ang'], zenith[1:48])
    assert np.allclose(testdata['Air Mass'], airmass[1:48])
    assert np.allclose(testdata['Direct Beam'], Eb[1:48])
    assert np.allclose(testdata['Direct Hz'], Ebh[1:48])
    assert np.allclose(testdata['Global Hz'], Gh[1:48])
    assert np.allclose(testdata['Dif Hz'], Dh[1:48])
    return pd.DataFrame({'Eb': Eb, 'Ebh': Ebh, 'Gh': Gh, 'Dh': Dh}, index=dt)
