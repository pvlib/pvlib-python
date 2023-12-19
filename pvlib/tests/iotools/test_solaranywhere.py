import pandas as pd
import pytest
import pvlib
import os
from ..conftest import (DATA_DIR, RERUNS, RERUNS_DELAY,
                        requires_solaranywhere_credentials)

# High spatial resolution and 5-min data, true dynamics enabled
TESTFILE_HIGH_RESOLUTION = DATA_DIR / 'Burlington, United States SolarAnywhere Time Series 20210101 to 20210103 Lat_44_4675 Lon_-73_2075 SA format.csv'  # noqa: E501
# TGY test file (v3.6) containing GHI/DHI and temperature.
# Note, the test file only contains the first three days.
TESTFILE_TMY = DATA_DIR / 'Burlington, United States SolarAnywhere Typical GHI Year Lat_44_465 Lon_-73_205 SA format.csv'  # noqa: E501


@pytest.fixture(scope="module")
def solaranywhere_api_key():
    """Supplies the pvlib's SolarAnywhere API key for testing purposes.
    Users can freely registre for an API key."""
    solaranywhere_api_key = os.environ["SOLARANYWHERE_API_KEY"]
    return solaranywhere_api_key


@pytest.fixture
def high_resolution_index():
    index = pd.date_range(start='2021-01-01 05:05', end='2021-01-03 05:00',
                          freq='5min', tz='UTC')
    index.name = 'ObservationTime(GMT)'
    return index


@pytest.fixture
def tmy_index():
    index = pd.date_range(start='2000-01-01 06:00', periods=3*24, freq='1h',
                          tz='UTC')
    index.name = 'ObservationTime(GMT)'
    index.freq = None
    return index


@pytest.fixture
def tmy_ghi_series(tmy_index):
    ghi = [
        0, 0, 0, 0, 0, 0, 0, 3, 50, 171, 234, 220, 202, 122, 141, 65, 2, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 48, 105, 161, 135, 108, 72, 58,
        33, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 47, 124, 99, 116,
        130, 165, 110, 36, 1, 0, 0, 0, 0, 0, 0, 0
    ]
    return pd.Series(data=ghi, index=tmy_index, name='ghi')


def test_read_solaranywhere_high_resolution(high_resolution_index):
    data, meta = pvlib.iotools.read_solaranywhere(TESTFILE_HIGH_RESOLUTION,
                                                  map_variables=False)
    # Check that metadata is parsed correctly
    assert meta['latitude'] == 44.4675
    assert meta['longitude'] == -73.2075
    assert meta['altitude'] == 41.0
    assert meta['name'] == 'Burlington  United States'
    assert meta['TZ'] == -5.0
    assert meta['Data Version'] == '3.6'
    assert meta['LatLon Resolution'] == '0.005'
    # Check that columns are parsed correctly
    assert 'Albedo' in data.columns
    assert 'Global Horizontal Irradiance (GHI) W/m2' in data.columns
    assert 'Direct Normal Irradiance (DNI) W/m2' in data.columns
    assert 'WindSpeed (m/s)' in data.columns
    assert 'WindSpeedObservationType' in data.columns
    assert 'Particulate Matter 10 (µg/m3)' in data.columns
    # Check that data is parsed correctly
    assert data.loc['2021-01-01 12:00:00+0000', 'Albedo'] == 0.6
    assert data.loc['2021-01-01 12:00:00+0000', 'WindSpeed (m/s)'] == 0
    # Assert that the index is parsed correctly
    pd.testing.assert_index_equal(data.index, high_resolution_index)


def test_read_solaranywhere_map_variables(high_resolution_index):
    # Check that variables are mapped by default to pvlib names
    data, meta = pvlib.iotools.read_solaranywhere(TESTFILE_HIGH_RESOLUTION)
    mapped_column_names = ['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed',
                           'relative_humidity', 'ghi_clear', 'dni_clear',
                           'dhi_clear', 'albedo']
    for c in mapped_column_names:
        assert c in data.columns
    assert meta['latitude'] == 44.4675
    assert meta['longitude'] == -73.2075
    assert meta['altitude'] == 41.0


def test_read_solaranywhere_tmy(tmy_index, tmy_ghi_series):
    # Check that TMY files are correctly parsed
    data, meta = pvlib.iotools.read_solaranywhere(TESTFILE_TMY)
    # Check that columns names are correct and mapped to pvlib names
    assert 'ghi' in data.columns
    assert 'dni' in data.columns
    assert 'dhi' in data.columns
    assert 'temp_air' in data.columns
    # Check that metadata is parsed correctly
    assert meta['latitude'] == 44.465
    assert meta['longitude'] == -73.205
    assert meta['altitude'] == 41.0
    assert meta['name'] == 'Burlington  United States'
    assert meta['TZ'] == -5.0
    assert meta['Data Version'] == '3.6'
    assert meta['LatLon Resolution'] == '0.010'
    assert meta['Time Resolution'] == '60 minutes'
    # Assert that the index is parsed correctly
    pd.testing.assert_index_equal(data.index, tmy_index)
    # Test one column
    pd.testing.assert_series_equal(data['ghi'], tmy_ghi_series)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_solaranywhere_bad_probability_of_exceedance():
    # Test if ValueError is raised if probability_of_exceedance is not integer
    with pytest.raises(ValueError, match="must be an integer"):
        pvlib.iotools.get_solaranywhere(
            latitude=44, longitude=-73, api_key='empty',
            source='SolarAnywherePOELatest', probability_of_exceedance=0.5)


@pytest.mark.remote_data
@requires_solaranywhere_credentials
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_solaranywhere_missing_start_end(solaranywhere_api_key):
    # Test if ValueError is raised if start/end is missing for non-TMY request
    with pytest.raises(ValueError, match="`end` is required."):
        pvlib.iotools.get_solaranywhere(
            latitude=44, longitude=-73, api_key=solaranywhere_api_key,
            source='SolarAnywhereLatest')


@pytest.fixture
def time_series_index():
    index = pd.date_range(start='2020-01-01 00:02:30', periods=288,
                          freq='5min', tz='UTC')
    index.name = 'ObservationTime'
    index.freq = None
    return index


@pytest.fixture
def timeseries_temp_air(time_series_index):
    temp_air = [
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2,
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2
    ]
    return pd.Series(data=temp_air, index=time_series_index, name='temp_air')


@requires_solaranywhere_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_solaranywhere_no_timezone_information(
        high_resolution_index, solaranywhere_api_key, time_series_index,
        timeseries_temp_air):
    # Test if data can be retrieved. This test only retrieves one day of data
    # to minimize the request time.
    data, meta = pvlib.iotools.get_solaranywhere(
        latitude=44.4675, longitude=-73.2075, api_key=solaranywhere_api_key,
        # test specific version of SolarAnywhere
        source='SolarAnywhere3_6',
        # specify start/end without timezone information
        start=pd.Timestamp(2020, 1, 1), end=pd.Timestamp(2020, 1, 2),
        spatial_resolution=0.005, time_resolution=5, true_dynamics=True)

    # Check metadata, including that true-dynamics is set
    assert meta['WeatherSiteName'] == 'SolarAnywhere3_6'
    assert meta['ApplyTrueDynamics'] is True
    assert meta['time_resolution'] == 5
    assert meta['latitude'] == 44.4675
    assert meta['longitude'] == -73.2075
    assert meta['altitude'] == 41.0

    # Check that variables have been mapped (default convention)
    assert 'StartTime' in data.columns
    assert 'ObservationTime' in data.columns
    assert 'EndTime' in data.columns
    assert 'ghi' in data.columns
    assert 'dni' in data.columns
    assert 'dhi' in data.columns
    assert 'temp_air' in data.columns
    assert 'wind_speed' in data.columns
    assert 'albedo' in data.columns
    assert 'DataVersion' in data.columns

    # Assert index (checks that time resolution is 5 min)
    pd.testing.assert_index_equal(data.index, time_series_index)
    # Test one column
    pd.testing.assert_series_equal(data['temp_air'], timeseries_temp_air)


@requires_solaranywhere_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_solaranywhere_timeout(solaranywhere_api_key):
    # Test if the service times out when the timeout parameter is close to zero
    with pytest.raises(TimeoutError, match="Time exceeded"):
        pvlib.iotools.get_solaranywhere(
            latitude=44.4675, longitude=-73.2075,
            api_key=solaranywhere_api_key,
            start=pd.Timestamp('2020-01-01 00:00:00+0000'),
            end=pd.Timestamp('2020-01-05 12:00:00+0000'),
            timeout=0.00001)


@requires_solaranywhere_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_solaranywhere_not_available(solaranywhere_api_key):
    # Test if RuntimeError is raised if location in the ocean is requested
    with pytest.raises(RuntimeError, match="Tile is outside of our coverage"):
        pvlib.iotools.get_solaranywhere(
            latitude=40, longitude=-70,
            api_key=solaranywhere_api_key,
            start=pd.Timestamp('2020-01-01 00:00:00+0000'),
            end=pd.Timestamp('2020-01-05 12:00:00+0000'))


# Second live test
# True dynamics is False
# Different variables

# Mock tets:
# TGY
