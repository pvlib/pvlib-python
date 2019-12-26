"""
test iotools for PSM3
"""

import os
from pvlib.iotools import psm3
from conftest import needs_pandas_0_22
import numpy as np
import pandas as pd
import pytest
from requests import HTTPError
from io import StringIO

BASEDIR = os.path.abspath(os.path.dirname(__file__))
PROJDIR = os.path.dirname(BASEDIR)
DATADIR = os.path.join(PROJDIR, 'data')
TMY_TEST_DATA = os.path.join(DATADIR, 'test_psm3_tmy-2017.csv')
YEAR_TEST_DATA = os.path.join(DATADIR, 'test_psm3_2017.csv')
MANUAL_TEST_DATA = os.path.join(DATADIR, 'test_read_psm3.csv')
LATITUDE, LONGITUDE = 40.5137, -108.5449
HEADER_FIELDS = [
    'Source', 'Location ID', 'City', 'State', 'Country', 'Latitude',
    'Longitude', 'Time Zone', 'Elevation', 'Local Time Zone',
    'Dew Point Units', 'DHI Units', 'DNI Units', 'GHI Units',
    'Temperature Units', 'Pressure Units', 'Wind Direction Units',
    'Wind Speed', 'Surface Albedo Units', 'Version']
PVLIB_EMAIL = 'pvlib-admin@googlegroups.com'
DEMO_KEY = 'DEMO_KEY'


@needs_pandas_0_22
def test_get_psm3_tmy():
    """test get_psm3 with a TMY"""
    header, data = psm3.get_psm3(LATITUDE, LONGITUDE, DEMO_KEY, PVLIB_EMAIL,
                                 names='tmy-2017')
    expected = pd.read_csv(TMY_TEST_DATA)
    # check datevec columns
    assert np.allclose(data.Year, expected.Year)
    assert np.allclose(data.Month, expected.Month)
    assert np.allclose(data.Day, expected.Day)
    assert np.allclose(data.Hour, expected.Hour)
    assert np.allclose(data.Minute, expected.Minute)
    # check data columns
    assert np.allclose(data.GHI, expected.GHI)
    assert np.allclose(data.DNI, expected.DNI)
    assert np.allclose(data.DHI, expected.DHI)
    assert np.allclose(data.Temperature, expected.Temperature)
    assert np.allclose(data.Pressure, expected.Pressure)
    assert np.allclose(data['Dew Point'], expected['Dew Point'])
    assert np.allclose(data['Surface Albedo'], expected['Surface Albedo'])
    assert np.allclose(data['Wind Speed'], expected['Wind Speed'])
    assert np.allclose(data['Wind Direction'], expected['Wind Direction'])
    # check header
    for hf in HEADER_FIELDS:
        assert hf in header
    # check timezone
    assert (data.index.tzinfo.zone == 'Etc/GMT%+d' % -header['Time Zone'])
    # check errors
    with pytest.raises(HTTPError):
        # HTTP 403 forbidden because api_key is rejected
        psm3.get_psm3(LATITUDE, LONGITUDE, api_key='BAD', email=PVLIB_EMAIL)
    with pytest.raises(HTTPError):
        # coordinates were not found in the NSRDB
        psm3.get_psm3(51, -5, DEMO_KEY, PVLIB_EMAIL)
    with pytest.raises(HTTPError):
        # names is not one of the available options
        psm3.get_psm3(LATITUDE, LONGITUDE, DEMO_KEY, PVLIB_EMAIL, names='bad')


@needs_pandas_0_22
def test_get_psm3_singleyear():
    """test get_psm3 with a single year"""
    header, data = psm3.get_psm3(LATITUDE, LONGITUDE, DEMO_KEY, PVLIB_EMAIL,
                                 names='2017', interval=30)
    expected = pd.read_csv(YEAR_TEST_DATA)
    # check datevec columns
    assert np.allclose(data.Year, expected.Year)
    assert np.allclose(data.Month, expected.Month)
    assert np.allclose(data.Day, expected.Day)
    assert np.allclose(data.Hour, expected.Hour)
    assert np.allclose(data.Minute, expected.Minute)
    # check data columns
    assert np.allclose(data.GHI, expected.GHI)
    assert np.allclose(data.DNI, expected.DNI)
    assert np.allclose(data.DHI, expected.DHI)
    assert np.allclose(data.Temperature, expected.Temperature)
    assert np.allclose(data.Pressure, expected.Pressure)
    assert np.allclose(data['Dew Point'], expected['Dew Point'])
    assert np.allclose(data['Surface Albedo'], expected['Surface Albedo'])
    assert np.allclose(data['Wind Speed'], expected['Wind Speed'])
    assert np.allclose(data['Wind Direction'], expected['Wind Direction'])
    # check header
    for hf in HEADER_FIELDS:
        assert hf in header
    # check timezone
    assert (data.index.tzinfo.zone == 'Etc/GMT%+d' % -header['Time Zone'])
    # check leap day
    _, data_2012 = psm3.get_psm3(LATITUDE, LONGITUDE, DEMO_KEY, PVLIB_EMAIL,
                                 names='2012', interval=60, leap_day=True)
    assert len(data_2012) == (8760+24)
    # check errors
    with pytest.raises(HTTPError):
        # HTTP 403 forbidden because api_key is rejected
        psm3.get_psm3(LATITUDE, LONGITUDE, api_key='BAD', email=PVLIB_EMAIL,
                      names='2017')
    with pytest.raises(HTTPError):
        # coordinates were not found in the NSRDB
        psm3.get_psm3(51, -5, DEMO_KEY, PVLIB_EMAIL, names='2017')
    with pytest.raises(HTTPError):
        # intervals can only be 30 or 60 minutes
        psm3.get_psm3(LATITUDE, LONGITUDE, DEMO_KEY, PVLIB_EMAIL, names='2017',
                      interval=15)


@pytest.fixture(scope='module',
                params=[True, False])
def io_input(request):
    """Switch between filename and file-like object for read_psm3"""
    if request.param:
        with open(MANUAL_TEST_DATA, 'r') as f:
            data = f.read()
        obj = StringIO(data)
    else:
        obj = MANUAL_TEST_DATA
    return obj


@needs_pandas_0_22
def test_read_psm3(io_input):
    """test read_psm3"""
    header, data = psm3.read_psm3(io_input)
    expected = pd.read_csv(YEAR_TEST_DATA)
    # check datevec columns
    assert np.allclose(data.Year, expected.Year)
    assert np.allclose(data.Month, expected.Month)
    assert np.allclose(data.Day, expected.Day)
    assert np.allclose(data.Hour, expected.Hour)
    assert np.allclose(data.Minute, expected.Minute)
    # check data columns
    assert np.allclose(data.GHI, expected.GHI)
    assert np.allclose(data.DNI, expected.DNI)
    assert np.allclose(data.DHI, expected.DHI)
    assert np.allclose(data.Temperature, expected.Temperature)
    assert np.allclose(data.Pressure, expected.Pressure)
    assert np.allclose(data['Dew Point'], expected['Dew Point'])
    assert np.allclose(data['Surface Albedo'], expected['Surface Albedo'])
    assert np.allclose(data['Wind Speed'], expected['Wind Speed'])
    assert np.allclose(data['Wind Direction'], expected['Wind Direction'])
    # check header
    for hf in HEADER_FIELDS:
        assert hf in header
    # check timezone
    assert (data.index.tzinfo.zone == 'Etc/GMT%+d' % -header['Time Zone'])
