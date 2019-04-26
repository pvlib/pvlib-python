"""
test iotools for PSM3
"""

import os
from pvlib.iotools import psm3
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


def test_psm3():
    """test get_psm3"""
    header, data = psm3.get_psm3(LATITUDE, LONGITUDE)
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
    # check error
    with pytest.raises(HTTPError):
        psm3.get_psm3(LATITUDE, LONGITUDE, 'bad')
