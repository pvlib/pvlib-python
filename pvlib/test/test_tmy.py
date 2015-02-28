import logging
pvl_logger = logging.getLogger('pvlib')

import inspect
import os

test_dir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
tmy3_testfile = os.path.join(test_dir, '../data/703165TY.csv')
tmy2_testfile = os.path.join(test_dir, '../data/12839.tm2')

from pvlib import tmy


def test_readtmy3():
    tmy.readtmy3(tmy3_testfile)
    
def test_readtmy3_remote():
    url = 'http://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/703165TYA.CSV'
    tmy.readtmy3(url)
    
def test_readtmy3_recolumn():
    data, meta = tmy.readtmy3(tmy3_testfile)
    assert 'GHISource' in data.columns
    
def test_readtmy3_norecolumn():
    data, meta = tmy.readtmy3(tmy3_testfile, recolumn=False)
    assert 'GHI source' in data.columns
    
def test_readtmy2():
    tmy.readtmy2(tmy2_testfile)
    
