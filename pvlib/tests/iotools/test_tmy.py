import numpy as np
import pandas as pd
from pvlib.iotools import tmy
from conftest import DATA_DIR

# test the API works
from pvlib.iotools import read_tmy3

TMY3_TESTFILE = DATA_DIR / '703165TY.csv'
TMY2_TESTFILE = DATA_DIR / '12839.tm2'
TMY3_FEB_LEAPYEAR = DATA_DIR / '723170TYA.CSV'


def test_read_tmy3():
    tmy.read_tmy3(TMY3_TESTFILE)


def test_read_tmy3_recolumn():
    data, meta = tmy.read_tmy3(TMY3_TESTFILE)
    assert 'GHISource' in data.columns


def test_read_tmy3_norecolumn():
    data, _ = tmy.read_tmy3(TMY3_TESTFILE, recolumn=False)
    assert 'GHI source' in data.columns


def test_read_tmy3_coerce_year():
    coerce_year = 1987
    data, _ = tmy.read_tmy3(TMY3_TESTFILE, coerce_year=coerce_year)
    assert (data.index[:-1].year == 1987).all()
    assert data.index[-1].year == 1988


def test_read_tmy3_no_coerce_year():
    coerce_year = None
    data, _ = tmy.read_tmy3(TMY3_TESTFILE, coerce_year=coerce_year)
    assert 1997 and 1999 in data.index.year
    assert data.index[-2] == pd.Timestamp('1998-12-31 23:00:00-09:00')
    assert data.index[-1] == pd.Timestamp('1999-01-01 00:00:00-09:00')


def test_read_tmy2():
    tmy.read_tmy2(TMY2_TESTFILE)


def test_gh865_read_tmy3_feb_leapyear_hr24():
    """correctly parse the 24th hour if the tmy3 file has a leap year in feb"""
    data, meta = read_tmy3(TMY3_FEB_LEAPYEAR)
    # just to be safe, make sure this _IS_ the Greensboro file
    greensboro = {
        'USAF': 723170,
        'Name': '"GREENSBORO PIEDMONT TRIAD INT"',
        'State': 'NC',
        'TZ': -5.0,
        'latitude': 36.1,
        'longitude': -79.95,
        'altitude': 273.0}
    assert meta == greensboro
    # February for Greensboro is 1996, a leap year, so check to make sure there
    # aren't any rows in the output that contain Feb 29th
    assert data.index[1414] == pd.Timestamp('1996-02-28 23:00:00-0500')
    assert data.index[1415] == pd.Timestamp('1996-03-01 00:00:00-0500')
    # now check if it parses correctly when we try to coerce the year
    data, _ = read_tmy3(TMY3_FEB_LEAPYEAR, coerce_year=1990)
    # if get's here w/o an error, then gh865 is fixed, but let's check anyway
    assert all(data.index[:-1].year == 1990)
    assert data.index[-1].year == 1991
    # let's do a quick sanity check, are the indices monotonically increasing?
    assert all(np.diff(data.index.astype(int)) == 3600000000000)
    # according to the TMY3 manual, each record corresponds to the previous
    # hour so check that the 1st hour is 1AM and the last hour is midnite
    assert data.index[0].hour == 1
    assert data.index[-1].hour == 0
