"""
tests for :mod:`pvlib.iotools.bsrn`
"""


import pandas as pd
import pytest

from pvlib.iotools import bsrn
from conftest import DATA_DIR, assert_index_equal


@pytest.mark.parametrize('testfile,expected_index', [
    ('bsrn-pay0616.dat.gz',
     pd.date_range(start='20160601', periods=43200, freq='1min', tz='UTC')),
    ('bsrn-lr0100-pay0616.dat',
     pd.date_range(start='20160601', periods=43200, freq='1min', tz='UTC')),
])
def test_read_bsrn(testfile, expected_index):
    data = bsrn.read_bsrn(DATA_DIR / testfile)
    assert_index_equal(expected_index, data.index)
    assert 'ghi' in data.columns
    assert 'dni_std' in data.columns
    assert 'dhi_min' in data.columns
    assert 'lwd_max' in data.columns
    assert 'relative_humidity' in data.columns
