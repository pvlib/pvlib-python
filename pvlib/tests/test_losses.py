# -*- coding: utf-8 -*-
"""Test losses"""

import datetime
import numpy as np
import pandas as pd
from pandas.util.testing import assert_series_equal
from pvlib.losses import soiling_hsu, soiling_kimber
from pvlib.iotools import read_tmy3
from conftest import (
    requires_scipy, needs_pandas_0_22, DATA_DIR)
import pytest


@pytest.fixture
def expected_output():
    # Sample output (calculated manually)
    dt = pd.date_range(start=pd.datetime(2019, 1, 1, 0, 0, 0),
                       end=pd.datetime(2019, 1, 1, 23, 59, 0), freq='1h')

    expected_no_cleaning = pd.Series(
        data=[0.884980357535360, 0.806308930084762, 0.749974647038078,
              0.711804155175089, 0.687489866078621, 0.672927554408964,
              0.664714899337491, 0.660345851212099, 0.658149551658860,
              0.657104593968981, 0.656633344364056, 0.656431630729954,
              0.656349579062171, 0.656317825078228, 0.656306121502393,
              0.656302009396500, 0.656300630853678, 0.656300189543417,
              0.656300054532516, 0.656300015031680, 0.656300003971846,
              0.656300001006533, 0.656300000244750, 0.656300000057132],
        index=dt)

    return expected_no_cleaning


@pytest.fixture
def expected_output_2(expected_output):
    # Sample output (calculated manually)
    dt = pd.date_range(start=pd.datetime(2019, 1, 1, 0, 0, 0),
                       end=pd.datetime(2019, 1, 1, 23, 59, 0), freq='1h')

    expected_no_cleaning = expected_output

    expected = pd.Series(index=dt)
    expected[dt[:4]] = expected_no_cleaning[dt[:4]]
    expected[dt[4:7]] = 1.
    expected[dt[7]] = expected_no_cleaning[dt[0]]
    expected[dt[8:12]] = 1.
    expected[dt[12:17]] = expected_no_cleaning[dt[:5]]
    expected[dt[17:21]] = 1.
    expected[dt[21:]] = expected_no_cleaning[:3]

    return expected


@pytest.fixture
def rainfall_input():

    dt = pd.date_range(start=pd.datetime(2019, 1, 1, 0, 0, 0),
                       end=pd.datetime(2019, 1, 1, 23, 59, 0), freq='1h')
    rainfall = pd.Series(
        data=[0., 0., 0., 0., 1., 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0., 0.,
              0., 0.3, 0.3, 0.3, 0.3, 0., 0., 0., 0.], index=dt)
    return rainfall


@requires_scipy
def test_soiling_hsu_no_cleaning(rainfall_input, expected_output):
    """Test Soiling HSU function"""

    rainfall = rainfall_input
    pm2_5 = 1.0
    pm10 = 2.0
    depo_veloc = {'2_5': 1.0, '10': 1.0}
    tilt = 0.
    expected_no_cleaning = expected_output

    result = soiling_hsu(rainfall=rainfall, cleaning_threshold=10., tilt=tilt,
                         pm2_5=pm2_5, pm10=pm10, depo_veloc=depo_veloc,
                         rain_accum_period=pd.Timedelta('1h'))
    assert_series_equal(result, expected_no_cleaning)


@requires_scipy
def test_soiling_hsu(rainfall_input, expected_output_2):
    """Test Soiling HSU function"""

    rainfall = rainfall_input
    pm2_5 = 1.0
    pm10 = 2.0
    depo_veloc = {'2_5': 1.0, '10': 1.0}
    tilt = 0.
    expected = expected_output_2

    # three cleaning events at 4:00-6:00, 8:00-11:00, and 17:00-20:00
    result = soiling_hsu(rainfall=rainfall, cleaning_threshold=0.5, tilt=tilt,
                         pm2_5=pm2_5, pm10=pm10, depo_veloc=depo_veloc,
                         rain_accum_period=pd.Timedelta('3h'))

    assert_series_equal(result, expected)


@pytest.fixture
def greensboro_rain():
    # get TMY3 data with rain
    greensboro = read_tmy3(DATA_DIR / '723170TYA.CSV', coerce_year=1990)
    # NOTE: can't use Sand Point, AK b/c Lprecipdepth is -9900, ie: missing
    return greensboro[0].Lprecipdepth


@pytest.fixture
def expected_kimber_soiling_nowash():
    return pd.read_csv(
        DATA_DIR / 'greensboro_kimber_soil_nowash.dat',
        parse_dates=True, index_col='timestamp')


@needs_pandas_0_22
def test_kimber_soiling_nowash(greensboro_rain,
                               expected_kimber_soiling_nowash):
    """Test Kimber soiling model with no manual washes"""
    # Greensboro typical expected annual rainfall is 8345mm
    assert greensboro_rain.sum() == 8345
    # calculate soiling with no wash dates
    soiling_nowash = soiling_kimber(greensboro_rain, istmy=True)
    # test no washes
    assert np.allclose(
        soiling_nowash.values,
        expected_kimber_soiling_nowash['soiling'].values)


@pytest.fixture
def expected_kimber_soiling_manwash():
    return pd.read_csv(
        DATA_DIR / 'greensboro_kimber_soil_manwash.dat',
        parse_dates=True, index_col='timestamp')


@needs_pandas_0_22
def test_kimber_soiling_manwash(greensboro_rain,
                                expected_kimber_soiling_manwash):
    """Test Kimber soiling model with a manual wash"""
    # a manual wash date
    manwash = [datetime.date(1990, 2, 15), ]
    # calculate soiling with manual wash
    soiling_manwash = soiling_kimber(
        greensboro_rain, manual_wash_dates=manwash, istmy=True)
    # test manual wash
    assert np.allclose(
        soiling_manwash.values,
        expected_kimber_soiling_manwash['soiling'].values)


@pytest.fixture
def expected_kimber_soiling_norain():
    # expected soiling reaches maximum
    soiling_loss_rate = 0.0015
    max_loss_rate = 0.3
    norain = np.ones(8760) * soiling_loss_rate/24
    norain[0] = 0.0
    norain = np.cumsum(norain)
    return np.where(norain > max_loss_rate, max_loss_rate, norain)


@needs_pandas_0_22
def test_kimber_soiling_norain(greensboro_rain,
                               expected_kimber_soiling_norain):
    """Test Kimber soiling model with no rain"""
    # a year with no rain
    norain = pd.Series(0, index=greensboro_rain.index)
    # calculate soiling with no rain
    soiling_norain = soiling_kimber(norain, istmy=True)
    # test no rain, soiling reaches maximum
    assert np.allclose(soiling_norain.values, expected_kimber_soiling_norain)


@pytest.fixture
def expected_kimber_soiling_initial_soil():
    # expected soiling reaches maximum
    soiling_loss_rate = 0.0015
    max_loss_rate = 0.3
    norain = np.ones(8760) * soiling_loss_rate/24
    norain[0] = 0.1
    norain = np.cumsum(norain)
    return np.where(norain > max_loss_rate, max_loss_rate, norain)


@needs_pandas_0_22
def test_kimber_soiling_initial_soil(greensboro_rain,
                                     expected_kimber_soiling_initial_soil):
    """Test Kimber soiling model with initial soiling"""
    # a year with no rain
    norain = pd.Series(0, index=greensboro_rain.index)
    # calculate soiling with no rain
    soiling_norain = soiling_kimber(norain, initial_soiling=0.1, istmy=True)
    # test no rain, soiling reaches maximum
    assert np.allclose(
        soiling_norain.values, expected_kimber_soiling_initial_soil)
