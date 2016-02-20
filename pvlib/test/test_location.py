import datetime

import numpy as np
from numpy import nan
import pandas as pd
import pytz

from nose.tools import raises
from pytz.exceptions import UnknownTimeZoneError
from pandas.util.testing import assert_series_equal, assert_frame_equal

from ..location import Location

aztz = pytz.timezone('US/Arizona')

def test_location_required():
    Location(32.2, -111)

def test_location_all():
    Location(32.2, -111, 'US/Arizona', 700, 'Tucson')

@raises(UnknownTimeZoneError)
def test_location_invalid_tz():
    Location(32.2, -111, 'invalid')

@raises(TypeError)
def test_location_invalid_tz_type():
    Location(32.2, -111, [5])

def test_location_pytz_tz():
    Location(32.2, -111, aztz)

def test_location_int_float_tz():
    Location(32.2, -111, -7)
    Location(32.2, -111, -7.0)

def test_location_print_all():
    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    expected_str = 'Tucson: latitude=32.2, longitude=-111, tz=US/Arizona, altitude=700'
    assert tus.__str__() == expected_str

def test_location_print_pytz():
    tus = Location(32.2, -111, aztz, 700, 'Tucson')
    expected_str = 'Tucson: latitude=32.2, longitude=-111, tz=US/Arizona, altitude=700'
    assert tus.__str__() == expected_str


def test_get_clearsky():
    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    times = pd.DatetimeIndex(start='20160101T0600-0700',
                             end='20160101T1800-0700',
                             freq='3H')
    clearsky = tus.get_clearsky(times)
    expected = pd.DataFrame(data=np.array(
        [[   0.        ,    0.        ,    0.        ],
         [  49.99257714,  762.92663984,  258.84368467],
         [  70.79757257,  957.14396999,  612.04545874],
         [  59.01570645,  879.06844381,  415.26616693],
         [   0.        ,    0.        ,    0.        ]]),
                            columns=['dhi', 'dni', 'ghi'],
                            index=times)
    assert_frame_equal(expected, clearsky)


def test_get_clearsky_haurwitz():
    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    times = pd.DatetimeIndex(start='20160101T0600-0700',
                             end='20160101T1800-0700',
                             freq='3H')
    clearsky = tus.get_clearsky(times, model='haurwitz')
    expected = pd.DataFrame(data=np.array(
                            [[   0.        ],
                             [ 242.30085588],
                             [ 559.38247117],
                             [ 384.6873791 ],
                             [   0.        ]]),
                            columns=['ghi'],
                            index=times)
    assert_frame_equal(expected, clearsky)


@raises(ValueError)
def test_get_clearsky_valueerror():
    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    times = pd.DatetimeIndex(start='20160101T0600-0700',
                             end='20160101T1800-0700',
                             freq='3H')
    clearsky = tus.get_clearsky(times, model='invalid_model')


def test_from_tmy_3():
    from .test_tmy import tmy3_testfile
    from ..tmy import readtmy3
    data, meta = readtmy3(tmy3_testfile)
    print(meta)
    loc = Location.from_tmy(meta, data)
    assert loc.name is not None
    assert loc.altitude != 0
    assert loc.tz != 'UTC'
    assert_frame_equal(loc.tmy_data, data)


def test_from_tmy_2():
    from .test_tmy import tmy2_testfile
    from ..tmy import readtmy2
    data, meta = readtmy2(tmy2_testfile)
    print(meta)
    loc = Location.from_tmy(meta, data)
    assert loc.name is not None
    assert loc.altitude != 0
    assert loc.tz != 'UTC'
    assert_frame_equal(loc.tmy_data, data)


def test_get_solarposition():
    from .test_solarposition import expected, golden_mst
    times = pd.date_range(datetime.datetime(2003,10,17,12,30,30),
                          periods=1, freq='D', tz=golden_mst.tz)
    ephem_data = golden_mst.get_solarposition(times, temperature=11)
    ephem_data = np.round(ephem_data, 3)
    this_expected = expected.copy()
    this_expected.index = times
    this_expected = np.round(this_expected, 3)
    print(this_expected, ephem_data[expected.columns])
    assert_frame_equal(this_expected, ephem_data[expected.columns])


def test_get_airmass():
    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    times = pd.DatetimeIndex(start='20160101T0600-0700',
                             end='20160101T1800-0700',
                             freq='3H')
    airmass = tus.get_airmass(times)
    expected = pd.DataFrame(data=np.array(
                            [[        nan,         nan],
                             [ 3.61046506,  3.32072602],
                             [ 1.76470864,  1.62309115],
                             [ 2.45582153,  2.25874238],
                             [        nan,         nan]]),
                            columns=['airmass_relative', 'airmass_absolute'],
                            index=times)
    assert_frame_equal(expected, airmass)

    airmass = tus.get_airmass(times, model='young1994')
    expected = pd.DataFrame(data=np.array(
                            [[        nan,         nan],
                             [ 3.6075018 ,  3.31800056],
                             [ 1.7641033 ,  1.62253439],
                             [ 2.45413091,  2.25718744],
                             [        nan,         nan]]),
                            columns=['airmass_relative', 'airmass_absolute'],
                            index=times)
    assert_frame_equal(expected, airmass)


@raises(ValueError)
def test_get_airmass_valueerror():
    tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
    times = pd.DatetimeIndex(start='20160101T0600-0700',
                             end='20160101T1800-0700',
                             freq='3H')
    clearsky = tus.get_airmass(times, model='invalid_model')
