import datetime
from pandas.util.testing import network
import numpy as np
import pandas as pd
import pytest
import pytz
from pvlib.iotools import tmy
from pvlib.iotools import read_tmy3, tmy3_monotonic_index
from conftest import DATA_DIR

# test the API works
from pvlib.iotools import read_tmy3

TMY3_TESTFILE = DATA_DIR / '703165TY.csv'
TMY2_TESTFILE = DATA_DIR / '12839.tm2'
TMY3_FEB_LEAPYEAR = DATA_DIR / '723170TYA.CSV'


def test_read_tmy3():
    tmy.read_tmy3(TMY3_TESTFILE)


@network
@pytest.mark.remote_data
def test_read_tmy3_remote():
    url = 'http://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/703165TYA.CSV'
    tmy.read_tmy3(url)


def test_read_tmy3_recolumn():
    data, meta = tmy.read_tmy3(TMY3_TESTFILE)
    assert 'GHISource' in data.columns


def test_read_tmy3_norecolumn():
    data, meta = tmy.read_tmy3(TMY3_TESTFILE, recolumn=False)
    assert 'GHI source' in data.columns


def test_read_tmy3_coerce_year():
    coerce_year = 1987
    data, meta = tmy.read_tmy3(TMY3_TESTFILE, coerce_year=coerce_year)
    assert (data.index.year == 1987).all()


def test_read_tmy3_no_coerce_year():
    coerce_year = None
    data, meta = tmy.read_tmy3(TMY3_TESTFILE, coerce_year=coerce_year)
    assert 1997 and 1999 in data.index.year


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
    assert all(data.index.year == 1990)
    # let's do a quick sanity check, are the indices monotonically increasing?
    assert all(np.diff(data.index[:-1].astype(int)) == 3600000000000)
    # according to the TMY3 manual, each record corresponds to the previous
    # hour so check that the 1st hour is 1AM and the last hour is midnite
    assert data.index[0].hour == 1
    assert data.index[-1].hour == 0


def test_fix_tmy_coerce_year_monotonicity():
    # greensboro timezone is UTC-5 or Eastern time
    gmt_5 = pytz.timezone('Etc/GMT+5')

    # tmy3 coerced to year is not monotonically increasing
    greensboro, _ = read_tmy3(DATA_DIR / '723170TYA.CSV', coerce_year=1990)

    # check first hour was coerced to 1990
    firsthour = gmt_5.localize(datetime.datetime(1990, 1, 1, 1, 0, 0))
    assert firsthour == greensboro.index[0]

    # check last hour was coerced to 1990
    lasthour = gmt_5.localize(datetime.datetime(1990, 12, 31, 23, 0, 0))
    assert lasthour == greensboro.index[-2]

    # check last day, was coerced to 1990
    lastday1990 = gmt_5.localize(datetime.datetime(1990, 1, 1, 0, 0, 0))
    assert lastday1990 == greensboro.index[-1]

    # fix the index to be monotonically increasing
    greensboro = tmy3_monotonic_index(greensboro)

    # check first and last hours are still 1990
    assert firsthour == greensboro.index[0]
    assert lasthour == greensboro.index[-2]

    # check last day, should be 1991 now
    lastday1991 = lastday1990.replace(year=1991)
    assert lastday1991 == greensboro.index[-1]
