"""
tests for :mod:`pvlib.iotools.bsrn`
"""


import pandas as pd
import pytest

from pvlib.iotools import bsrn
from conftest import DATA_DIR


testfile = DATA_DIR / 'bsrn-pay0616.dat.gz'


def test_read_bsrn_columns():
    data = bsrn.read_bsrn(testfile)
    assert 'ghi' in data.columns
    assert 'dni_std' in data.columns
    assert 'dhi_min' in data.columns
    assert 'lwd_max' in data.columns
    assert 'relative_humidity' in data.columns


@pytest.fixture
def expected_index():
    start = pd.Timestamp(2016, 6, 1, 0, 0)
    return pd.date_range(start=start, periods=43200, freq='1min', tz='UTC')


def test_format_index(expected_index):
    actual = bsrn.read_bsrn(testfile)
    assert actual.index.equals(expected_index)
