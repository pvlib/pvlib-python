import inspect
import os

import pandas as pd
from pandas.util.testing import network
import pytest
import pytz

from pvlib.iotools import midc


@pytest.fixture
def test_mapping():
    return {
        'Direct Normal [W/m^2]': 'dni',
        'Global PSP [W/m^2]': 'ghi',
        'Rel Humidity [%]': 'relative_humidity',
        'Temperature @ 2m [deg C]': 'temp_air',
        'Non Existant': 'variable',
    }


test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
midc_testfile = os.path.join(test_dir, '../data/midc_20181014.txt')
midc_raw_testfile = os.path.join(test_dir, '../data/midc_raw_20181018.txt')
midc_network_testfile = ('https://midcdmz.nrel.gov/apps/data_api.pl'
                         '?site=UAT&begin=20181018&end=20181019')


def test_midc_format_index():
    data = pd.read_csv(midc_testfile)
    data = midc.format_index(data)
    start = pd.Timestamp("20181014 00:00")
    start = start.tz_localize("MST")
    end = pd.Timestamp("20181014 23:59")
    end = end.tz_localize("MST")
    assert type(data.index) == pd.DatetimeIndex
    assert data.index[0] == start
    assert data.index[-1] == end


def test_midc_format_index_tz_conversion():
    data = pd.read_csv(midc_testfile)
    data = data.rename(columns={'MST': 'PST'})
    data = midc.format_index(data)
    assert data.index[0].tz == pytz.timezone('Etc/GMT+8')


def test_midc_format_index_raw():
    data = pd.read_csv(midc_raw_testfile)
    data = midc.format_index_raw(data)
    start = pd.Timestamp('20181018 00:00')
    start = start.tz_localize('MST')
    end = pd.Timestamp('20181018 23:59')
    end = end.tz_localize('MST')
    assert data.index[0] == start
    assert data.index[-1] == end


def test_read_midc_var_mapping_as_arg(test_mapping):
    data = midc.read_midc(midc_testfile, variable_map=test_mapping)
    assert 'ghi' in data.columns
    assert 'temp_air' in data.columns


@network
def test_read_midc_raw_data_from_nrel():
    start_ts = pd.Timestamp('20181018')
    end_ts = pd.Timestamp('20181019')
    var_map = midc.MIDC_VARIABLE_MAP['UAT']
    data = midc.read_midc_raw_data_from_nrel('UAT', start_ts, end_ts, var_map)
    for k, v in var_map.items():
        assert v in data.columns
    assert data.index.size == 2880
