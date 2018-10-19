import inspect
import os

import pandas as pd
import pytest

from pvlib.iotools import midc


test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
midc_testfile = os.path.join(test_dir, '../data/midc_20181014.txt')


@pytest.mark.parametrize('field_name,expected', [
    ('Temperature @ 2m [deg C]', 'temp_air_@_2m'),
    ('Global PSP [W/m^2]', 'ghi_PSP'),
    ('Temperature @ 50m [deg C]', 'temp_air_@_50m'),
])
def test_read_midc_mapper_function(field_name, expected):
    assert midc.map_midc_to_pvlib(midc.VARIABLE_MAP, field_name) == expected


def test_read_midc_format_index():
    data = pd.read_csv(midc_testfile)
    data = midc.format_index(data)
    start = pd.Timestamp("20181014 00:00")
    start = start.tz_localize("MST")
    end = pd.Timestamp("20181014 23:59")
    end = end.tz_localize("MST")
    assert type(data.index) == pd.DatetimeIndex
    assert data.index[0] == start
    assert data.index[-1] == end


def test_read_midc_var_mapping_as_arg():
    data = midc.read_midc(midc_testfile, variable_map=midc.VARIABLE_MAP)
    assert 'ghi_PSP' in data.columns
    assert 'temp_air_@_2m' in data.columns
    assert 'temp_air_@_50m' in data.columns
