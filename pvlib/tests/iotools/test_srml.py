from numpy import isnan
import pandas as pd
import pytest

from pvlib.iotools import srml
from conftest import DATA_DIR, RERUNS, RERUNS_DELAY

srml_testfile = DATA_DIR / 'SRML-day-EUPO1801.txt'


def test_read_srml():
    srml.read_srml(srml_testfile)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_read_srml_remote():
    srml.read_srml('http://solardat.uoregon.edu/download/Archive/EUPO1801.txt')


def test_read_srml_columns_exist():
    data = srml.read_srml(srml_testfile)
    assert 'ghi_0' in data.columns
    assert 'ghi_0_flag' in data.columns
    assert 'dni_1' in data.columns
    assert 'dni_1_flag' in data.columns
    assert '7008' in data.columns
    assert '7008_flag' in data.columns


def test_read_srml_nans_exist():
    data = srml.read_srml(srml_testfile)
    assert isnan(data['dni_0'][1119])
    assert data['dni_0_flag'][1119] == 99


@pytest.mark.parametrize('url,year,month', [
    ('http://solardat.uoregon.edu/download/Archive/EUPO1801.txt',
     2018, 1),
    ('http://solardat.uoregon.edu/download/Archive/EUPO1612.txt',
     2016, 12),
])
@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_read_srml_dt_index(url, year, month):
    data = srml.read_srml(url)
    start = pd.Timestamp(f'{year:04d}{month:02d}01 00:00')
    start = start.tz_localize('Etc/GMT+8')
    end = pd.Timestamp(f'{year:04d}{month:02d}31 23:59')
    end = end.tz_localize('Etc/GMT+8')
    assert data.index[0] == start
    assert data.index[-1] == end
    assert (data.index[59::60].minute == 59).all()
    assert str(year) not in data.columns


@pytest.mark.parametrize('column,expected', [
    ('1001', 'ghi_1'),
    ('7324', '7324'),
    ('2001', '2001'),
    ('2017', 'dni_7')
])
def test_map_columns(column, expected):
    assert srml.map_columns(column) == expected


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_read_srml_month_from_solardat():
    url = 'http://solardat.uoregon.edu/download/Archive/EUPO1801.txt'
    file_data = srml.read_srml(url)
    requested = srml.read_srml_month_from_solardat('EU', 2018, 1)
    assert file_data.equals(requested)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_15_minute_dt_index():
    data = srml.read_srml_month_from_solardat('TW', 2019, 4, 'RQ')
    start = pd.Timestamp('20190401 00:00')
    start = start.tz_localize('Etc/GMT+8')
    end = pd.Timestamp('20190430 23:45')
    end = end.tz_localize('Etc/GMT+8')
    assert data.index[0] == start
    assert data.index[-1] == end
    assert (data.index[3::4].minute == 45).all()


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_hourly_dt_index():
    data = srml.read_srml_month_from_solardat('CD', 1986, 4, 'PH')
    start = pd.Timestamp('19860401 00:00')
    start = start.tz_localize('Etc/GMT+8')
    end = pd.Timestamp('19860430 23:00')
    end = end.tz_localize('Etc/GMT+8')
    assert data.index[0] == start
    assert data.index[-1] == end
    assert (data.index.minute == 0).all()
