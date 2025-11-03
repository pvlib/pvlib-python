"""
tests for pvlib/iotools/era5.py
"""

import pandas as pd
import pytest
import pvlib
import requests
import os
from tests.conftest import RERUNS, RERUNS_DELAY, requires_ecmwf_credentials


@pytest.fixture
def params():
    api_key = os.environ["ECMWF_API_KEY"]

    return {
        'latitude': 40.01, 'longitude': -80.01,
        'start': '2020-06-01', 'end': '2020-06-01',
        'variables': ['ghi', 'temp_air'],
        'api_key': api_key,
    }


@pytest.fixture
def expected():
    index = pd.date_range("2020-06-01 00:00", "2020-06-01 23:59", freq="h",
                          tz="UTC")
    index.name = 'valid_time'
    temp_air = [16.6, 15.2, 13.5, 11.2, 10.8, 9.1, 7.3, 6.8, 7.6, 7.4, 8.5,
                8.1, 9.8, 11.5, 14.1, 17.4, 18.3, 20., 20.7, 20.9, 21.5,
                21.6, 21., 20.7]
    ghi = [153., 18.4, 0., 0., 0., 0., 0., 0., 0., 0., 0., 60., 229.5,
           427.8, 620.1, 785.5, 910.1, 984.2, 1005.9, 962.4, 844.1, 685.2,
           526.9, 331.4]
    df = pd.DataFrame({'temp_air': temp_air, 'ghi': ghi}, index=index)
    return df


@requires_ecmwf_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_era5(params, expected):
    df, meta = pvlib.iotools.get_era5(**params)
    pd.testing.assert_frame_equal(df, expected, check_freq=False, atol=0.1)
    assert meta['longitude'] == -80.0
    assert meta['latitude'] == 40.0
    assert isinstance(meta['jobID'], str)


@requires_ecmwf_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_era5_timezone(params, expected):
    params['start'] = pd.to_datetime(params['start']).tz_localize('Etc/GMT+8')
    params['end'] = pd.to_datetime(params['end']).tz_localize('Etc/GMT+8')
    df, meta = pvlib.iotools.get_era5(**params)
    pd.testing.assert_frame_equal(df, expected, check_freq=False, atol=0.1)
    assert meta['longitude'] == -80.0
    assert meta['latitude'] == 40.0
    assert isinstance(meta['jobID'], str)


@requires_ecmwf_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_era5_map_variables(params, expected):
    df, meta = pvlib.iotools.get_era5(**params, map_variables=False)
    expected = expected.rename(columns={'temp_air': 't2m', 'ghi': 'ssrd'})
    df['t2m'] -= 273.15  # apply unit conversions manually
    df['ssrd'] /= 3600
    pd.testing.assert_frame_equal(df, expected, check_freq=False, atol=0.1)
    assert meta['longitude'] == -80.0
    assert meta['latitude'] == 40.0
    assert isinstance(meta['jobID'], str)


@requires_ecmwf_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_era5_error(params):
    params['variables'] = ['nonexistent']
    match = 'Request failed. Please check the ECMWF website'
    with pytest.raises(Exception, match=match):
        df, meta = pvlib.iotools.get_era5(**params)


@requires_ecmwf_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_era5_timeout(params):
    match = 'Request timed out. Try increasing'
    with pytest.raises(requests.exceptions.Timeout, match=match):
        df, meta = pvlib.iotools.get_era5(**params, timeout=1)
