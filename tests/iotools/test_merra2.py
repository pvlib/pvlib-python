"""
tests for pvlib/iotools/merra2.py
"""

import pandas as pd
import pytest
import pvlib
import os
from requests.exceptions import HTTPError
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
    index = pd.date_range("2020-06-01 15:30", "2020-06-01 19:30", freq="h",
                          tz="UTC")
    index.name = 'Timestamp (UTC)'
    albedo = [0.163931, 0.1609407, 0.1601474, 0.1612476, 0.164664]
    ghi = [ 930., 1002.75, 1020.25, 981.25, 886.5]
    df = pd.DataFrame({'albedo': albedo, 'ghi': ghi}, index=index)
    return df


@pytest.fixture
def expected_meta():
    return {
        'dataset': 'M2T1NXRAD.5.12.4',
        'latitude': 40.0,
        'longitude': -80.0,
        'ALBEDO': {
            'prod_name': 'M2T1NXRAD.5.12.4',
            'doi': '10.5067/Q9QMY5PBNV1T',
            'param_short_name': 'ALBEDO',
            'param_name': 'Surface albedo, time average',
            'unit': '1',
            'undef': '1e+15',
            'begin_time': '2020-06-01 15:30:00',
            'end_time': '2020-06-01 19:30:00',
            'lat': '40.0',
            'lon': '-80.0',
            'lat_resolution': '0.5',
            'lon_resolution': '0.625',
            'mean': '1.6219e-01',
            # 'Request_time': '2026-08-05 12:35:06'
        },
        'SWGDN': {
            'prod_name': 'M2T1NXRAD.5.12.4',
            'doi': '10.5067/Q9QMY5PBNV1T',
            'param_short_name': 'SWGDN',
            'param_name': 'Surface incoming shortwave flux, time average',
            'unit': 'W m-2',
            'undef': '1e+15',
            'begin_time': '2020-06-01 15:30:00',
            'end_time': '2020-06-01 19:30:00',
            'lat': '40.0',
            'lon': '-80.0',
            'lat_resolution': '0.5',
            'lon_resolution': '0.625',
            'mean': '9.6415e+02',
            # 'Request_time': '2026-08-05 12:35:09'
        }
    }


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2(params, expected, expected_meta):
    df, meta = pvlib.iotools.get_merra2(**params)
    pd.testing.assert_frame_equal(df, expected, check_freq=False)
    meta["SWGDN"].pop("Request_time")  # this changes from run to run,
    meta["ALBEDO"].pop("Request_time")  # so don't check it
    assert meta == expected_meta


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_map_variables(params, expected, expected_meta):
    df, meta = pvlib.iotools.get_merra2(**params, map_variables=False)
    expected = expected.rename(columns={'albedo': 'ALBEDO', 'ghi': 'SWGDN'})
    pd.testing.assert_frame_equal(df, expected, check_freq=False)
    meta["SWGDN"].pop("Request_time")  # this changes from run to run,
    meta["ALBEDO"].pop("Request_time")  # so don't check it
    assert meta == expected_meta


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_error():
    with pytest.raises(HTTPError, match='Unauthorized for url'):
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
    meta["SWGDN"].pop("Request_time")  # this changes from run to run,
    meta["ALBEDO"].pop("Request_time")  # so don't check it
    assert meta == expected_meta


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_bad_credentials(params, expected, expected_meta):
    params['username'] = 'nonexistent'
    with pytest.raises(HTTPError, match='Unauthorized'):
        pvlib.iotools.get_merra2(**params)


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_bad_dataset(params, expected, expected_meta):
    params['dataset'] = 'nonexistent'
    with pytest.raises(HTTPError, match='Forbidden for url'):
        pvlib.iotools.get_merra2(**params)


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_bad_variables(params, expected, expected_meta):
    params['variables'] = ['nonexistent']
    with pytest.raises(HTTPError, match='Forbidden for url'):
        pvlib.iotools.get_merra2(**params)


@requires_earthdata_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_merra2_multiple_datasets(params):
    params['variables'] = ["SWGDN", "T2M"]
    params["dataset"] = ["M2T1NXRAD.5.12.4", "M2T1NXSLV.5.12.4"]
    df, meta = pvlib.iotools.get_merra2(**params)
    assert set(df.columns) == {"ghi", "temp_air"}
