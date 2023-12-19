from numpy import isnan
import pandas as pd
import pytest
import os

from pvlib.iotools import srml
from ..conftest import (DATA_DIR, RERUNS, RERUNS_DELAY, assert_index_equal,
                        assert_frame_equal)

srml_testfile = DATA_DIR / 'SRML-day-EUPO1801.txt'

BASE_URL = 'http://is-solardata01.uoregon.edu/Step3B_Original_Format/'


def test_read_srml():
    srml.read_srml(srml_testfile)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_read_srml_remote():
    srml.read_srml(os.path.join(BASE_URL, 'EUO/EUO_2018/EUPO1801.txt'))


def test_read_srml_columns_exist():
    data = srml.read_srml(srml_testfile)
    assert 'ghi_0' in data.columns
    assert 'ghi_0_flag' in data.columns
    assert 'dni_1' in data.columns
    assert 'dni_1_flag' in data.columns
    assert '7008' in data.columns
    assert '7008_flag' in data.columns


def test_read_srml_map_variables_false():
    data = srml.read_srml(srml_testfile, map_variables=False)
    assert '1000' in data.columns
    assert '1000_flag' in data.columns
    assert '2010' in data.columns
    assert '2010_flag' in data.columns
    assert '7008' in data.columns
    assert '7008_flag' in data.columns


def test_read_srml_nans_exist():
    data = srml.read_srml(srml_testfile)
    assert isnan(data['dni_0'].iloc[1119])
    assert data['dni_0_flag'].iloc[1119] == 99


@pytest.mark.parametrize('url,year,month', [
    (os.path.join(BASE_URL, 'EUO/EUO_2018/EUPO1801.txt'),
     2018, 1),
    (os.path.join(BASE_URL, 'EUO/EUO_2016/EUPO1612.txt'),
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
def test__map_columns(column, expected):
    assert srml._map_columns(column) == expected


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_srml():
    url = os.path.join(BASE_URL, 'EUO/EUO_2018/EUPO1801.txt')
    file_data = srml.read_srml(url)
    requested, _ = srml.get_srml(station='EUO', start='2018-01-01',
                                 end='2018-01-31')
    assert_frame_equal(file_data, requested)


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_srml_minute():
    data_read = srml.read_srml(srml_testfile)
    data_get, meta = srml.get_srml(station='EUO', start='2018-01-01',
                                   end='2018-01-31', filetype='PO')
    expected_index = pd.date_range(start='2018-01-01', end='2018-01-31 23:59',
                                   freq='1min', tz='Etc/GMT+8')
    assert_index_equal(data_get.index, expected_index)
    assert all(c in data_get.columns for c in data_read.columns)
    # Check that all indices in example file are present in remote file
    assert data_read.index.isin(data_get.index).all()
    assert meta['station'] == 'EUO'
    assert meta['filetype'] == 'PO'
    assert meta['filenames'] == ['EUPO1801.txt']


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_srml_nonexisting_month_warning():
    with pytest.warns(UserWarning, match='file was not found: EUO/EUO_2009/EUPO0912.txt'):  # noqa: E501
        # Request data for a period where not all files exist
        # Eugene (EUO) station started reporting 1-minute data in January 2010
        data, meta = data, meta = srml.get_srml(
            station='EUO', start='2009-12-01', end='2010-01-31', filetype='PO')


@pytest.mark.remote_data
@pytest.mark.flaky(reruns=RERUNS, reruns_delay=RERUNS_DELAY)
def test_get_srml_station_two_letter_error():
    with pytest.raises(ValueError, match='should be a 3 letter station abbreviation'):  # noqa: E501
        # Test that an error is raised when specifying a station using the old
        # two-letter station code. E.g., Eugene, Oregon station used to be EU
        # and now goes by EUO
        data, meta = data, meta = srml.get_srml(
            station='EU', start='2022-01-01', end='2022-01-31', filetype='PO')
