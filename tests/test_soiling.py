"""Test losses"""

import datetime
import numpy as np
import pandas as pd
from .conftest import assert_series_equal
from pvlib.soiling import hsu, kimber
from pvlib.iotools import read_tmy3
from .conftest import TESTS_DATA_DIR, PVLIB_DATA_DIR
import pytest


@pytest.fixture
def expected_output():
    # Sample output (calculated manually)
    dt = pd.date_range(start=pd.Timestamp(2019, 1, 1, 0, 0, 0),
                       end=pd.Timestamp(2019, 1, 1, 23, 59, 0), freq='1h')

    expected_no_cleaning = pd.Series(
        data=[0.96998483, 0.94623958, 0.92468139, 0.90465654, 0.88589707,
              0.86826366, 0.85167258, 0.83606715, 0.82140458, 0.80764919,
              0.79476875, 0.78273241, 0.77150951, 0.76106905, 0.75137932,
              0.74240789, 0.73412165, 0.72648695, 0.71946981, 0.7130361,
              0.70715176, 0.70178307, 0.69689677, 0.69246034],
        index=dt)
    return expected_no_cleaning


@pytest.fixture
def expected_output_1():
    dt = pd.date_range(start=pd.Timestamp(2019, 1, 1, 0, 0, 0),
                       end=pd.Timestamp(2019, 1, 1, 23, 59, 0), freq='1h')
    expected_output_1 = pd.Series(
        data=[0.98484972, 0.97277367, 0.96167471, 0.95119603, 1.,
              0.98484972, 0.97277367, 0.96167471, 1., 1.,
              0.98484972, 0.97277367, 0.96167471, 0.95119603, 0.94118234,
              0.93154854, 0.922242, 0.91322759, 0.90448058, 0.89598283,
              0.88772062, 0.87968325, 0.8718622, 0.86425049],
        index=dt)
    return expected_output_1


@pytest.fixture
def expected_output_2():
    dt = pd.date_range(start=pd.Timestamp(2019, 1, 1, 0, 0, 0),
                       end=pd.Timestamp(2019, 1, 1, 23, 59, 0), freq='1h')
    expected_output_2 = pd.Series(
        data=[0.95036261, 0.91178179, 0.87774818, 0.84732079, 1.,
              1., 1., 0.95036261, 1., 1.,
              1., 1., 0.95036261, 0.91178179, 0.87774818,
              0.84732079, 0.8201171, 1., 1., 1.,
              1., 0.95036261, 0.91178179, 0.87774818],
        index=dt)
    return expected_output_2


@pytest.fixture
def expected_output_3():
    dt = pd.date_range(start=pd.Timestamp(2019, 1, 1, 0, 0, 0),
                       end=pd.Timestamp(2019, 1, 1, 23, 59, 0), freq='1h')
    timedelta = [0, 0, 0, 0, 0, 30, 0, 30, 0, 30, 0, -30,
                 -30, -30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    dt_new = dt + pd.to_timedelta(timedelta, 'm')
    expected_output_3 = pd.Series(
        data=[0.96576705, 0.9387675, 0.91437615, 0.89186852, 1.,
              1., 0.98093819, 0.9387675, 1., 1.,
              1., 1., 0.96576705, 0.9387675, 0.90291005,
              0.88122293, 0.86104089, 1., 1., 1.,
              0.96576705, 0.9387675, 0.91437615, 0.89186852],
        index=dt_new)
    return expected_output_3


@pytest.fixture
def rainfall_input():

    dt = pd.date_range(start=pd.Timestamp(2019, 1, 1, 0, 0, 0),
                       end=pd.Timestamp(2019, 1, 1, 23, 59, 0), freq='1h')
    rainfall = pd.Series(
        data=[0., 0., 0., 0., 1., 0., 0., 0., 0.5, 0.5, 0., 0., 0., 0., 0.,
              0., 0.3, 0.3, 0.3, 0.3, 0., 0., 0., 0.], index=dt)
    return rainfall


def test_hsu_no_cleaning(rainfall_input, expected_output):
    """Test Soiling HSU function"""

    rainfall = rainfall_input
    pm2_5 = 1.0
    pm10 = 2.0
    depo_veloc = {'2_5': 1.0e-5, '10': 1.0e-4}
    tilt = 0.
    expected_no_cleaning = expected_output

    result = hsu(rainfall=rainfall, cleaning_threshold=10., surface_tilt=tilt,
                 pm2_5=pm2_5, pm10=pm10, depo_veloc=depo_veloc,
                 rain_accum_period=pd.Timedelta('1h'))
    assert_series_equal(result, expected_no_cleaning)


def test_hsu(rainfall_input, expected_output_2):
    """Test Soiling HSU function with cleanings"""

    rainfall = rainfall_input
    pm2_5 = 1.0
    pm10 = 2.0
    depo_veloc = {'2_5': 1.0e-4, '10': 1.0e-4}
    tilt = 0.

    # three cleaning events at 4:00-6:00, 8:00-11:00, and 17:00-20:00
    result = hsu(rainfall=rainfall, cleaning_threshold=0.5, surface_tilt=tilt,
                 pm2_5=pm2_5, pm10=pm10, depo_veloc=depo_veloc,
                 rain_accum_period=pd.Timedelta('3h'))

    assert_series_equal(result, expected_output_2)


def test_hsu_defaults(rainfall_input, expected_output_1):
    """
    Test Soiling HSU function with default deposition velocity and default rain
    accumulation period.
    """
    result = hsu(rainfall=rainfall_input, cleaning_threshold=0.5,
                 surface_tilt=0.0, pm2_5=1.0e-2, pm10=2.0e-2)
    assert np.allclose(result.values, expected_output_1)


def test_hsu_variable_time_intervals(rainfall_input, expected_output_3):
    """
    Test Soiling HSU function with variable time intervals.
    """
    depo_veloc = {'2_5': 1.0e-4, '10': 1.0e-4}
    rain = pd.DataFrame(data=rainfall_input)
    # define time deltas in minutes
    timedelta = [0, 0, 0, 0, 0, 30, 0, 30, 0, 30, 0, -30,
                 -30, -30, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    rain['mins_added'] = pd.to_timedelta(timedelta, 'm')
    rain['new_time'] = rain.index + rain['mins_added']
    rain_var_times = rain.set_index('new_time').iloc[:, 0]
    result = hsu(
        rainfall=rain_var_times, cleaning_threshold=0.5, surface_tilt=50.0,
        pm2_5=1, pm10=2, depo_veloc=depo_veloc,
        rain_accum_period=pd.Timedelta('2h'))
    assert np.allclose(result, expected_output_3)


@pytest.fixture
def greensboro_rain():
    # get TMY3 data with rain
    greensboro, _ = read_tmy3(
        PVLIB_DATA_DIR / "723170TYA.CSV", coerce_year=1990, map_variables=True
    )
    return greensboro["Lprecip depth (mm)"]


@pytest.fixture
def expected_kimber_nowash():
    return pd.read_csv(
        TESTS_DATA_DIR / 'greensboro_kimber_soil_nowash.dat',
        parse_dates=True, index_col='timestamp')


def test_kimber_nowash(greensboro_rain, expected_kimber_nowash):
    """Test Kimber soiling model with no manual washes"""
    # Greensboro typical expected annual rainfall is 8345mm
    assert greensboro_rain.sum() == 8345
    # calculate soiling with no wash dates
    nowash = kimber(greensboro_rain)
    # test no washes
    assert np.allclose(nowash.values, expected_kimber_nowash['soiling'].values)


@pytest.fixture
def expected_kimber_manwash():
    return pd.read_csv(
        TESTS_DATA_DIR / 'greensboro_kimber_soil_manwash.dat',
        parse_dates=True, index_col='timestamp')


def test_kimber_manwash(greensboro_rain, expected_kimber_manwash):
    """Test Kimber soiling model with a manual wash"""
    # a manual wash date
    manwash = [datetime.date(1990, 2, 15), ]
    # calculate soiling with manual wash
    manwash = kimber(greensboro_rain, manual_wash_dates=manwash)
    # test manual wash
    assert np.allclose(
        manwash.values,
        expected_kimber_manwash['soiling'].values)


@pytest.fixture
def expected_kimber_norain():
    # expected soiling reaches maximum
    soiling_loss_rate = 0.0015
    max_loss_rate = 0.3
    norain = np.ones(8760) * soiling_loss_rate/24
    norain[0] = 0.0
    norain = np.cumsum(norain)
    return np.where(norain > max_loss_rate, max_loss_rate, norain)


def test_kimber_norain(greensboro_rain, expected_kimber_norain):
    """Test Kimber soiling model with no rain"""
    # a year with no rain
    norain = pd.Series(0, index=greensboro_rain.index)
    # calculate soiling with no rain
    norain = kimber(norain)
    # test no rain, soiling reaches maximum
    assert np.allclose(norain.values, expected_kimber_norain)


@pytest.fixture
def expected_kimber_initial_soil():
    # expected soiling reaches maximum
    soiling_loss_rate = 0.0015
    max_loss_rate = 0.3
    norain = np.ones(8760) * soiling_loss_rate/24
    norain[0] = 0.1
    norain = np.cumsum(norain)
    return np.where(norain > max_loss_rate, max_loss_rate, norain)


def test_kimber_initial_soil(greensboro_rain, expected_kimber_initial_soil):
    """Test Kimber soiling model with initial soiling"""
    # a year with no rain
    norain = pd.Series(0, index=greensboro_rain.index)
    # calculate soiling with no rain
    norain = kimber(norain, initial_soiling=0.1)
    # test no rain, soiling reaches maximum
    assert np.allclose(norain.values, expected_kimber_initial_soil)


def test_kimber_threshold_equal_cleans():
    """Rainfall exactly equal to cleaning_threshold should trigger cleaning."""
    dt = pd.date_range(start='1990-01-01', periods=48, freq='h')
    rainfall = pd.Series(0.0, index=dt)
    cleaning_threshold = 10
    # exactly at the threshold, not above it
    rainfall.iloc[0] = cleaning_threshold
    soiling = kimber(rainfall, cleaning_threshold=cleaning_threshold,
                     grace_period=1)
    # soiling should be reset (near zero) right at the rain event
    assert soiling.iloc[0] < 1e-9


def test_kimber_grace_period_one_day():
    """grace_period=1 should protect exactly 1 day after a rain event."""
    dt = pd.date_range(start='1990-01-01', periods=72, freq='h')
    rainfall = pd.Series(0.0, index=dt)
    cleaning_threshold = 10
    rainfall.iloc[0] = cleaning_threshold + 1  # comfortably above threshold
    soiling = kimber(rainfall, cleaning_threshold=cleaning_threshold,
                      grace_period=1)
    # 24 hours after the rain event: still within the grace day, should be
    # protected (soiling reset to ~0)
    assert soiling.iloc[24] < 1e-9
    # 48 hours after: grace period has ended, soiling should have built up
    assert soiling.iloc[48] > 0
