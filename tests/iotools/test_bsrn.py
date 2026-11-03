"""
tests for :mod:`pvlib.iotools.bsrn`
"""

import pandas as pd
import pytest
import io
import os
import tempfile
from pvlib.iotools import read_bsrn, get_bsrn
from tests.conftest import (
    TESTS_DATA_DIR,
    RERUNS,
    RERUNS_DELAY,
    assert_index_equal,
    requires_bsrn_credentials,
)


@pytest.fixture(scope="module")
def bsrn_credentials():
    """Supplies the BSRN FTP credentials for testing purposes.

    Users should obtain their own credentials as described in the `read_bsrn`
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
    data, metadata = read_bsrn(TESTS_DATA_DIR / testfile)
    assert_index_equal(expected_index, data.index)
    assert 'ghi' in data.columns
    assert 'dni_std' in data.columns
    assert 'dhi_min' in data.columns
    assert 'lwd_max' in data.columns
    assert 'relative_humidity' in data.columns


def test_read_bsrn_buffer(expected_index):
    with open(TESTS_DATA_DIR / 'bsrn-lr0100-pay0616.dat') as fbuf:
        data, metadata = read_bsrn(fbuf)
    assert_index_equal(expected_index, data.index)
    assert 'ghi' in data.columns
    assert 'dni_std' in data.columns
    assert 'dhi_min' in data.columns
    assert 'lwd_max' in data.columns
    assert 'relative_humidity' in data.columns


def test_read_bsrn_logical_records(expected_index):
    # Test if logical records 0300 and 0500 are correct parsed
    # and that 0100 is not passed when not specified
    data, metadata = read_bsrn(TESTS_DATA_DIR / 'bsrn-pay0616.dat.gz',
                               logical_records=['0300', '0500'])
    assert_index_equal(expected_index, data.index)
    assert 'lwu' in data.columns
    assert 'uva_global' in data.columns
    assert 'uvb_reflected_std' in data.columns
    assert 'ghi' not in data.columns
    # Check that there are no duplicate columns when parsing
    # LR0500 data (see #2841)
    assert not data.columns.duplicated().any()


def test_read_bsrn_bad_logical_record():
    # Test if ValueError is raised if an unsupported logical record is passed
    with pytest.raises(ValueError, match='not in'):
        read_bsrn(TESTS_DATA_DIR / 'bsrn-lr0100-pay0616.dat',
                  logical_records=['dummy'])


def test_read_bsrn_logical_records_not_found():
    # Test if an empty dataframe is returned if specified LRs are not present
    data, metadata = read_bsrn(TESTS_DATA_DIR / 'bsrn-lr0100-pay0616.dat',
                               logical_records=['0300', '0500'])
    assert data.empty  # assert that the dataframe is empty
    assert 'uva_global' in data.columns
    assert 'uvb_reflected_std' in data.columns
    assert 'uva_global_max' in data.columns
    assert 'dni' not in data.columns
    assert 'day' not in data.columns


@requires_bsrn_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_bsrn(expected_index, bsrn_credentials):
    # Retrieve irradiance data from the BSRN FTP server
    # the TAM station is chosen due to its small file sizes
    temp_dir = tempfile.TemporaryDirectory()  # create temporary directory
    username, password = bsrn_credentials
    data, metadata = get_bsrn(
        start=pd.Timestamp(2016, 6, 1),
        end=pd.Timestamp(2016, 6, 29),
        station='tam',
        username=username,
        password=password,
        save_path=temp_dir.name)
    assert_index_equal(expected_index, data.index)
    assert 'ghi' in data.columns
    assert 'dni_std' in data.columns
    assert 'dhi_min' in data.columns
    assert 'lwd_max' in data.columns
    assert 'relative_humidity' in data.columns
    # test that a local file was saved and is read correctly
    data2, metadata2 = read_bsrn(os.path.join(temp_dir.name, 'tam0616.dat.gz'))
    assert_index_equal(expected_index, data2.index)
    assert 'ghi' in data2.columns
    temp_dir.cleanup()  # explicitly remove temporary directory


@requires_bsrn_credentials
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_bsrn_bad_station(bsrn_credentials):
    # Test if KeyError is raised if a bad station name is passed
    username, password = bsrn_credentials
    with pytest.raises(KeyError, match='sub-directory does not exist'):
        get_bsrn(
            start=pd.Timestamp(2016, 6, 1),
            end=pd.Timestamp(2016, 6, 29),
            station='not_a_station_name',
            username=username,
            password=password)


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
            username=username,
            password=password)


def _mock_bsrn_ftp(mocker, payload):
    """Patch ftplib.FTP so that get_bsrn 'downloads' the given bytes."""
    def retrbinary(cmd, callback):
        callback(payload)
        return '226 Transfer complete.'

    mock_ftp_class = mocker.patch('pvlib.iotools.bsrn.ftplib.FTP')
    mock_ftp = mock_ftp_class.return_value.__enter__.return_value
    mock_ftp.retrbinary.side_effect = retrbinary
    return mock_ftp


def test_get_bsrn_mocked_ftp(expected_index, mocker):
    # Serve a local test file through a mocked FTP connection; the
    # result must match read_bsrn on the same file exactly.
    with open(TESTS_DATA_DIR / 'bsrn-pay0616.dat.gz', 'rb') as f:
        payload = f.read()
    mock_ftp = _mock_bsrn_ftp(mocker, payload)

    data, metadata = get_bsrn(station='pay',
                              start=pd.Timestamp(2016, 6, 1),
                              end=pd.Timestamp(2016, 6, 29),
                              username='user', password='pass')

    mock_ftp.cwd.assert_called_once_with('/pay')
    assert mock_ftp.retrbinary.call_count == 1
    assert (mock_ftp.retrbinary.call_args.args[0]
            == 'RETR pay0616.dat.gz')
    expected_data, expected_metadata = read_bsrn(
        TESTS_DATA_DIR / 'bsrn-pay0616.dat.gz')
    pd.testing.assert_frame_equal(data, expected_data)
    assert metadata.keys() == expected_metadata.keys()
    for key, expected_value in expected_metadata.items():
        if isinstance(expected_value, pd.Series):
            pd.testing.assert_series_equal(metadata[key], expected_value)
        else:
            assert metadata[key] == expected_value
    assert_index_equal(expected_index, data.index)


def test_get_bsrn_mocked_ftp_logical_records(mocker):
    # Non-default logical records must be forwarded to read_bsrn, which
    # get_bsrn calls with the downloaded gzip TextIOWrapper.
    with open(TESTS_DATA_DIR / 'bsrn-pay0616.dat.gz', 'rb') as f:
        payload = f.read()
    logical_records = ('0100', '0300', '0500')
    expected_data, _ = read_bsrn(TESTS_DATA_DIR / 'bsrn-pay0616.dat.gz',
                                 logical_records=logical_records)
    _mock_bsrn_ftp(mocker, payload)
    spy = mocker.patch('pvlib.iotools.bsrn.read_bsrn', wraps=read_bsrn)

    data, metadata = get_bsrn(station='pay',
                              start=pd.Timestamp(2016, 6, 1),
                              end=pd.Timestamp(2016, 6, 29),
                              username='user', password='pass',
                              logical_records=logical_records)

    spy.assert_called_once()
    buffer_arg, lr_arg = spy.call_args.args
    assert isinstance(buffer_arg, io.TextIOWrapper)
    assert lr_arg == logical_records
    # LR0300 content proves the records were actually parsed
    assert 'lwu' in data.columns
    pd.testing.assert_frame_equal(data, expected_data)
