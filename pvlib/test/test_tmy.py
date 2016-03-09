import inspect
import os

from pandas.util.testing import network

test_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
tmy3_testfile = os.path.join(test_dir, '../data/703165TY.csv')
tmy2_testfile = os.path.join(test_dir, '../data/12839.tm2')

from pvlib import tmy


def test_readtmy3():
    tmy.readtmy3(tmy3_testfile)

@network
def test_readtmy3_remote():
    url = 'http://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/703165TYA.CSV'
    tmy.readtmy3(url)
    
def test_readtmy3_recolumn():
    data, meta = tmy.readtmy3(tmy3_testfile)
    assert 'GHISource' in data.columns
    
def test_readtmy3_norecolumn():
    data, meta = tmy.readtmy3(tmy3_testfile, recolumn=False)
    assert 'GHI source' in data.columns
    
def test_readtmy3_coerce_year():
    coerce_year = 1987
    data, meta = tmy.readtmy3(tmy3_testfile, coerce_year=coerce_year)
    assert (data.index.year == 1987).all()
    
def test_readtmy3_no_coerce_year():
    coerce_year = None
    data, meta = tmy.readtmy3(tmy3_testfile, coerce_year=coerce_year)
    assert 1997 and 1999 in data.index.year
    
def test_readtmy2():
    tmy.readtmy2(tmy2_testfile)
    
