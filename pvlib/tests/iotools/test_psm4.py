"""
test iotools for PSM4
"""

import os
from pvlib.iotools import psm4
from ..conftest import DATA_DIR, RERUNS, RERUNS_DELAY, assert_index_equal
import numpy as np
import pandas as pd
import pytest
from requests import HTTPError
from io import StringIO
import warnings

TMY_TEST_DATA = DATA_DIR / 'test_psm4_tmy-2023.csv'
YEAR_TEST_DATA = DATA_DIR / 'test_psm4_2023.csv'
YEAR_TEST_DATA_5MIN = DATA_DIR / 'test_psm4_2023_5min.csv'
MANUAL_TEST_DATA = DATA_DIR / 'test_read_psm4.csv'
LATITUDE, LONGITUDE = 40.5137, -108.5449
METADATA_FIELDS = [
    'Source', 'Location ID', 'City', 'State', 'Country', 'Latitude',
    'Longitude', 'Time Zone', 'Elevation', 'Local Time Zone',
    'Dew Point Units', 'DHI Units', 'DNI Units', 'GHI Units',
    'Temperature Units', 'Pressure Units', 'Wind Direction Units',
    'Wind Speed Units', 'Surface Albedo Units', 'Version']
PVLIB_EMAIL = 'pvlib-admin@googlegroups.com'


@pytest.fixture(scope="module")
def nrel_api_key():
    """Supplies pvlib-python's NREL Developer Network API key.

    Azure Pipelines CI utilizes a secret variable set to NREL_API_KEY
    to mitigate failures associated with using the default key of
    "DEMO_KEY". A user is capable of using their own key this way if
    desired however the default key should suffice for testing purposes.
    """
    try:
        demo_key = os.environ["NREL_API_KEY"]
    except KeyError:
        warnings.warn(
            "WARNING: NREL API KEY environment variable not set! "
            "Using DEMO_KEY instead. Unexpected failures may occur."
        )
        demo_key = 'DEMO_KEY'
    return demo_key


def assert_psm4_equal(data, metadata, expected):
    """check consistency of PSM4 data"""
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
    for mf in METADATA_FIELDS:
        assert mf in metadata
    # check timezone
    assert (data.index.tzinfo.zone == 'Etc/GMT%+d' % -metadata['Time Zone'])


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_psm4_tmy(nrel_api_key):
    """test get_psm4 with a TMY"""
    data, metadata = psm4.get_psm4(LATITUDE, LONGITUDE, nrel_api_key,
                                   PVLIB_EMAIL, names='tmy-2023',
                                   leap_day=False, map_variables=False)
    expected = pd.read_csv(TMY_TEST_DATA)
    assert_psm4_equal(data, metadata, expected)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_psm4_singleyear(nrel_api_key):
    """test get_psm4 with a single year"""
    data, metadata = psm4.get_psm4(LATITUDE, LONGITUDE, nrel_api_key,
                                   PVLIB_EMAIL, names='2023',
                                   leap_day=False,  map_variables=False,
                                   interval=30)
    expected = pd.read_csv(YEAR_TEST_DATA)
    assert_psm4_equal(data, metadata, expected)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_psm4_5min(nrel_api_key):
    """test get_psm4 for 5-minute data"""
    data, metadata = psm4.get_psm4(LATITUDE, LONGITUDE, nrel_api_key,
                                   PVLIB_EMAIL, names='2023', interval=5,
                                   leap_day=False, map_variables=False)
    assert len(data) == 525600/5
    first_day = data.loc['2023-01-01']
    expected = pd.read_csv(YEAR_TEST_DATA_5MIN)
    assert_psm4_equal(first_day, metadata, expected)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_psm4_check_leap_day(nrel_api_key):
    data_2012, _ = psm4.get_psm4(LATITUDE, LONGITUDE, nrel_api_key,
                                 PVLIB_EMAIL, names="2012", interval=60,
                                 leap_day=True, map_variables=False)
    assert len(data_2012) == (8760 + 24)


@pytest.mark.parametrize('latitude, longitude, api_key, names, interval',
                         [(LATITUDE, LONGITUDE, 'BAD', 'tmy-2023', 60),
                          (51, -5, nrel_api_key, 'tmy-2023', 60),
                          (LATITUDE, LONGITUDE, nrel_api_key, 'bad', 60),
                          (LATITUDE, LONGITUDE, nrel_api_key, '2023', 15),
                          ])
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_psm4_tmy_errors(
    latitude, longitude, api_key, names, interval
):
    """Test get_psm4() for multiple erroneous input scenarios.

    These scenarios include:
    * Bad API key -> HTTP 403 forbidden because api_key is rejected
    * Bad latitude/longitude -> Coordinates were not found in the NSRDB.
    * Bad name -> Name is not one of the available options.
    * Bad interval, single year -> Intervals can only be 30 or 60 minutes.
    """
    with pytest.raises(HTTPError) as excinfo:
        psm4.get_psm4(latitude, longitude, api_key, PVLIB_EMAIL,
                      names=names, interval=interval, leap_day=False,
                      map_variables=False)
    # ensure the HTTPError caught isn't due to overuse of the API key
    assert "OVER_RATE_LIMIT" not in str(excinfo.value)


@pytest.fixture
def io_input(request):
    """file-like object for parse_psm4"""
    with MANUAL_TEST_DATA.open() as f:
        data = f.read()
    obj = StringIO(data)
    return obj


def test_parse_psm4(io_input):
    """test parse_psm4"""
    data, metadata = psm4.parse_psm4(io_input, map_variables=False)
    expected = pd.read_csv(YEAR_TEST_DATA)
    assert_psm4_equal(data, metadata, expected)


def test_read_psm4():
    """test read_psm4"""
    data, metadata = psm4.read_psm4(MANUAL_TEST_DATA, map_variables=False)
    expected = pd.read_csv(YEAR_TEST_DATA)
    assert_psm4_equal(data, metadata, expected)


def test_read_psm4_map_variables():
    """test read_psm4 map_variables=True"""
    data, metadata = psm4.read_psm4(MANUAL_TEST_DATA, map_variables=True)
    columns_mapped = ['Year', 'Month', 'Day', 'Hour', 'Minute', 'temp_air',
                      'alpha', 'aod', 'asymmetry', 'dhi_clear', 'dni_clear',
                      'ghi_clear', 'Cloud Fill Flag', 'Cloud Type',
                      'temp_dew', 'dhi', 'dni', 'Fill Flag', 'ghi', 'Ozone',
                      'relative_humidity', 'solar_zenith', 'SSA', 'albedo',
                      'pressure', 'precipitable_water', 'wind_direction',
                      'wind_speed']
    assert_index_equal(data.columns, pd.Index(columns_mapped))


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_psm4_attribute_mapping(nrel_api_key):
    """Test that pvlib names can be passed in as attributes and get correctly
    reverse mapped to psm4 names"""
    data, meta = psm4.get_psm4(LATITUDE, LONGITUDE, nrel_api_key,
                               PVLIB_EMAIL, names=2019, interval=60,
                               attributes=['ghi', 'wind_speed'],
                               leap_day=False, map_variables=True)
    # Check that columns are in the correct order (GH1647)
    expected_columns = [
        'Year', 'Month', 'Day', 'Hour', 'Minute', 'ghi', 'wind_speed']
    pd.testing.assert_index_equal(pd.Index(expected_columns), data.columns)
    assert 'latitude' in meta.keys()
    assert 'longitude' in meta.keys()
    assert 'altitude' in meta.keys()
