import inspect
import os

from pandas.util.testing import network

from pvlib.iotools import tmy

test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
tmy3_testfile = os.path.join(test_dir, '../data/703165TY.csv')
tmy2_testfile = os.path.join(test_dir, '../data/12839.tm2')


def test_read_tmy3():
    tmy.read_tmy3(tmy3_testfile)


@network
def test_read_tmy3_remote():
    url = 'http://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/703165TYA.CSV'
    tmy.read_tmy3(url)


def test_read_tmy3_recolumn():
    data, meta = tmy.read_tmy3(tmy3_testfile)
    assert 'GHISource' in data.columns


def test_read_tmy3_norecolumn():
    data, meta = tmy.read_tmy3(tmy3_testfile, recolumn=False)
    assert 'GHI source' in data.columns


def test_read_tmy3_coerce_year():
    coerce_year = 1987
    data, meta = tmy.read_tmy3(tmy3_testfile, coerce_year=coerce_year)
    assert (data.index.year == 1987).all()


def test_read_tmy3_no_coerce_year():
    coerce_year = None
    data, meta = tmy.read_tmy3(tmy3_testfile, coerce_year=coerce_year)
    assert 1997 and 1999 in data.index.year


def test_read_tmy2():
    tmy.read_tmy2(tmy2_testfile)
