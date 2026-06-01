"""
tests for pvlib/iotools/merra2.py
"""

import pandas as pd
import pytest
import pvlib
import os
import requests
from tests.conftest import RERUNS, RERUNS_DELAY, requires_earthdata_credentials


@pytest.fixture
def params():
    earthdata_username = os.environ["EARTHDATA_USERNAME"]
    earthdata_password = os.environ["EARTHDATA_PASSWORD"]

    return {
        'latitude': 40.01, 'longitude': -80.01,
        'start': '2020-06-01 15:00', 'end': '2020-06-01 20:00',
        'dataset': 'M2T1NXRAD.5.12.4', 'variables': ['ALBEDO', 'SWGDN'],
        'username': earthdata_username, 'password': earthdata_password,
    }


@pytest.fixture
def expected():
    index = pd.date_range("2020-06-01 15:30", "2020-06-01 20:30", freq="h",
                          tz="UTC")
    index.name = 'time'
    albedo = [0.163931, 0.1609407, 0.1601474, 0.1612476, 0.164664, 0.1711341]
    ghi = [ 930., 1002.75, 1020.25, 981.25, 886.5, 743.5]
    df = pd.DataFrame({'albedo': albedo, 'ghi': ghi}, index=index)
    return df


@pytest.fixture
def expected_meta():
    return {
        'dataset': 'M2T1NXRAD.5.12.4',
        'station': 'GridPointRequestedAt[40.010N_80.010W]',
        'latitude': 40.0,
        'longitude': -80.0,
        'units': {'ALBEDO': '1', 'SWGDN': 'W m-2'}
    }


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2(params, expected, expected_meta):
    df, meta = pvlib.iotools.get_merra2(**params)
    pd.testing.assert_frame_equal(df, expected, check_freq=False)
    assert meta == expected_meta


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_map_variables(params, expected, expected_meta):
    df, meta = pvlib.iotools.get_merra2(**params, map_variables=False)
    expected = expected.rename(columns={'albedo': 'ALBEDO', 'ghi': 'SWGDN'})
    pd.testing.assert_frame_equal(df, expected, check_freq=False)
    assert meta == expected_meta


def test_get_merra2_error():
    with pytest.raises(ValueError, match='must be in the same year'):
        pvlib.iotools.get_merra2(40, -80, '2019-12-31', '2020-01-02',
                                 username='anything', password='anything',
                                 dataset='anything', variables=[])


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_timezones(params, expected, expected_meta):
    # check with tz-aware start/end inputs
    for key in ['start', 'end']:
        dt = pd.to_datetime(params[key])
        params[key] = dt.tz_localize('UTC').tz_convert('Etc/GMT+5')
    df, meta = pvlib.iotools.get_merra2(**params)
    pd.testing.assert_frame_equal(df, expected, check_freq=False)
    assert meta == expected_meta


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_bad_credentials(params, expected, expected_meta):
    params['username'] = 'nonexistent'
    with pytest.raises(requests.exceptions.HTTPError, match='Unauthorized'):
        pvlib.iotools.get_merra2(**params)


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_bad_dataset(params, expected, expected_meta):
    params['dataset'] = 'nonexistent'
    with pytest.raises(requests.exceptions.HTTPError, match='404'):
        pvlib.iotools.get_merra2(**params)


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_bad_variables(params, expected, expected_meta):
    params['variables'] = ['nonexistent']
    with pytest.raises(requests.exceptions.HTTPError, match='400'):
        pvlib.iotools.get_merra2(**params)
