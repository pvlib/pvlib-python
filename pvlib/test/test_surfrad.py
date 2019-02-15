import inspect
import os

import pandas as pd
from pandas.util.testing import network

from pvlib.iotools import surfrad

test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
testfile = os.path.join(test_dir, '../data/surfrad-slv16001.dat')
network_testfile = ('ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/'
                    'Alamosa_CO/2016/slv16001.dat')


@network
def test_read_surfrad_network():
    # If this test begins failing, SURFRAD's data structure or data
    # archive may have changed.
    local_data, _ = surfrad.read_surfrad(testfile)
    network_data, _ = surfrad.read_surfrad(network_testfile)
    assert local_data.equals(network_data)


def test_read_surfrad_columns_no_map():
    data, _ = surfrad.read_surfrad(testfile, map_variables=False)
    assert 'zen' in data.columns
    assert 'temp' in data.columns
    assert 'par' in data.columns
    assert 'pressure' in data.columns


def test_read_surfrad_columns_map():
    data, _ = surfrad.read_surfrad(testfile)
    assert 'solar_zenith' in data.columns
    assert 'ghi' in data.columns
    assert 'ghi_flag' in data.columns
    assert 'dni' in data.columns
    assert 'dni_flag' in data.columns
    assert 'dhi' in data.columns
    assert 'dhi_flag' in data.columns
    assert 'wind_direction' in data.columns
    assert 'wind_direction_flag' in data.columns
    assert 'wind_speed' in data.columns
    assert 'wind_speed_flag' in data.columns
    assert 'temp_air' in data.columns
    assert 'temp_air_flag' in data.columns


def test_format_index():
    start = pd.Timestamp('20160101 00:00')
    expected = pd.date_range(start=start, periods=1440, freq='1min', tz='UTC')
    actual, _ = surfrad.read_surfrad(testfile)
    assert actual.index.equals(expected)


def test_read_surfrad_metadata():
    expected = {'name': 'Alamosa',
                'latitude': 37.70,
                'longitude': 105.92,
                'elevation': 2317,
                'surfrad_version': 1,
                'tz': 'UTC'}
    _, metadata = surfrad.read_surfrad(testfile)
    assert metadata == expected
