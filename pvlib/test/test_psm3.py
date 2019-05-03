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

BASEDIR = os.path.abspath(os.path.dirname(__file__))
PROJDIR = os.path.dirname(BASEDIR)
DATADIR = os.path.join(PROJDIR, 'data')
TEST_DATA = os.path.join(DATADIR, 'test_psm3.csv')
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
def test_get_psm3():
    """test get_psm3"""
    header, data = psm3.get_psm3(LATITUDE, LONGITUDE, DEMO_KEY, PVLIB_EMAIL)
    expected = pd.read_csv(TEST_DATA)
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
    with pytest.raises(HTTPError):
        # intervals can only be 30 or 60 minutes
        psm3.get_psm3(LATITUDE, LONGITUDE, DEMO_KEY, PVLIB_EMAIL, interval=15)
