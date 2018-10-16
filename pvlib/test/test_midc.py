import inspect
import os

import pandas as pd
import pytest

from pvlib.iotools import midc


test_dir = os.path.dirname(
    os.path.abspath(inspect.getfile(inspect.currentframe())))
midc_testfile = os.path.join(test_dir, '../data/midc_20181014.txt')

VARIABLE_MAP = {'Global PSP [W/m^2]': 'ghi',
                'Diffuse Horiz [W/m^2]': 'dhi',
                'Temperature @ 2m [deg C]': 'temp_air',
                'Avg Wind Speed @ 3m [m/s]': 'wind_speed'}


@pytest.mark.paramatrize('field_name,expected', [
    ('Temperature @ 2m [deg C]', 'temp_air'),
    ('Global PSP [W/m^2]', 'ghi'),
    ('Temperature @ 50m [deg C]', 'Temperature @ 50m [deg C]')
])
def test_read_midc_mapper_function(field_name, expected):
    assert midc.map_midc_to_pvlib(VARIABLE_MAP, field_name) == expected


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
    data = midc.read_midc(midc_testfile, variable_map=VARIABLE_MAP)
    assert 'ghi' in data.columns
    assert 'temp_air' in data.columns
    assert 'Temperature @ 50m [deg C]' in data.columns
