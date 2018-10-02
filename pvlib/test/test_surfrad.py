import inspect
import os

from numpy import isnan
import pandas as pd
from pandas.util.testing import network
import pytest

from pvlib.iotools import surfrad


test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
testfile = os.path.join(test_dir, '../data/surfrad-slv16001.dat')
network_testfile ='ftp://aftp.cmdl.noaa.gov/data/radiation/surfrad/Alamosa_CO/2016/slv16001.dat'


@pytest.mark.parametrize('filename', [
    testfile,
    network_testfile ])
def test_read_surfrad(filename):
    surfrad.read_surfrad(testfile)

def test_read_surfrad_columns_exist():
    data, _ = surfrad.read_surfrad(testfile)
    assert 'zen' in data.columns
    assert 'temp' in data.columns
    assert 'par' in data.columns
    assert 'pressure' in data.columns


def test_format_index():
    start = pd.Timestamp('20160101 00:00')
    expected_index = pd.DatetimeIndex(start = start,
                                      periods = 1440,
                                      freq='1min')
    expected_index.tz_localize('UTC')
    actual, _ = surfrad.read_surfrad(testfile)
    assert actual.index.equals(expected_index)

def test_read_surfrad_metadata():
    expected = { 'name': 'Alamosa',
                 'latitude': '37.70',
                 'longitude': '105.92',
                 'elevation': '2317',
                 'surfrad_version': '1' }
    _, metadata = surfrad.read_surfrad(testfile)
    assert metadata == expected
