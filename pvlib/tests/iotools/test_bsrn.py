"""
tests for :mod:`pvlib.iotools.bsrn`
"""


import pandas as pd
import pytest

from pvlib.iotools import read_bsrn, get_bsrn
from ..conftest import DATA_DIR, RERUNS, RERUNS_DELAY, assert_index_equal

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


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_bsrn(expected_index):
    # Retrieve irradiance data from the BSRN FTP server
    # the TAM station is chosen due to its small file sizes
    data, metadata = get_bsrn(
        start=pd.Timestamp(2016,6,1),
        end=pd.Timestamp(2016,6,29),
        station='tam',
        username='bsrnftp',
        password='bsrn1')
    assert_index_equal(expected_index, data.index)
    assert 'ghi' in data.columns
    assert 'dni_std' in data.columns
    assert 'dhi_min' in data.columns
    assert 'lwd_max' in data.columns
    assert 'relative_humidity' in data.columns
