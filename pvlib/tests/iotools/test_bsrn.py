"""
tests for :mod:`pvlib.iotools.bsrn`
"""

import pandas as pd
import pytest
import os
from pvlib.iotools import read_bsrn, get_bsrn
from ..conftest import (DATA_DIR, RERUNS, RERUNS_DELAY, assert_index_equal,
                        requires_bsrn_credentials)


@pytest.fixture(scope="module")
def bsrn_credentials():
    """Supplies the BSRN FTP credentials for testing purposes.

    Users should obtain there own credentials as described in the `read_bsrn`
    documentation."""
    bsrn_username = os.environ["BSRN_FTP_USERNAME"]
    bsrn_password = os.environ["BSRN_FTP_PASSWORD"]
    return bsrn_username, bsrn_password


@pytest.fixture
def expected_index():
    return pd.date_range(start='20160601', periods=43200, freq='1min',
                         tz='UTC')


@pytest.mark.parametrize('testfile', [
    ('bsrn-pay0616.dat.gz'),
    ('bsrn-lr0100-pay0616.dat'),
])
def test_read_bsrn(testfile, expected_index):
    data, metadata = read_bsrn(DATA_DIR / testfile)
    assert_index_equal(expected_index, data.index)
    assert 'ghi' in data.columns
    assert 'dni_std' in data.columns
    assert 'dhi_min' in data.columns
    assert 'lwd_max' in data.columns
    assert 'relative_humidity' in data.columns


@requires_bsrn_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_bsrn(expected_index, bsrn_credentials):
    # Retrieve irradiance data from the BSRN FTP server
    # the TAM station is chosen due to its small file sizes
    username, password = bsrn_credentials
    data, metadata = get_bsrn(
        start=pd.Timestamp(2016, 6, 1),
        end=pd.Timestamp(2016, 6, 29),
        station='tam',
        username=username,
        password=password,
        local_path='')
    assert_index_equal(expected_index, data.index)
    assert 'ghi' in data.columns
    assert 'dni_std' in data.columns
    assert 'dhi_min' in data.columns
    assert 'lwd_max' in data.columns
    assert 'relative_humidity' in data.columns


@requires_bsrn_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_bsrn_bad_station(bsrn_credentials):
    # Test if ValueError is raised if a bad station name is passed
    username, password = bsrn_credentials
    with pytest.raises(KeyError, match='sub-directory does not exist'):
        get_bsrn(
            start=pd.Timestamp(2016, 6, 1),
            end=pd.Timestamp(2016, 6, 29),
            station='not_a_station_name',
            username='bsrnftp',
            password='bsrn1')


@requires_bsrn_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_bsrn_no_files(bsrn_credentials):
    username, password = bsrn_credentials
    # Test if Warning is given if no files are found for the entire time frame
    with pytest.warns(UserWarning, match='No files'):
        get_bsrn(
            start=pd.Timestamp(1990, 6, 1),
            end=pd.Timestamp(1990, 6, 29),
            station='tam',
            username='bsrnftp',
            password='bsrn1')
