"""
test the pvgis IO tools
"""
import json
import numpy as np
import pandas as pd
import pytest
import requests
from pvlib.iotools import get_pvgis_tmy
from conftest import DATA_DIR


@pytest.fixture
def expected():
    return pd.read_csv(DATA_DIR / 'pvgis_tmy_test.dat', index_col='time(UTC)')


@pytest.fixture
def userhorizon_expected():
    return pd.read_json(DATA_DIR / 'tmy_45.000_8.000_userhorizon.json')


@pytest.fixture
def month_year_expected():
    return [
        2009, 2012, 2014, 2010, 2011, 2013, 2011, 2011, 2013, 2013, 2013, 2011]


@pytest.fixture
def inputs_expected():
    return {
        'location': {'latitude': 45.0, 'longitude': 8.0, 'elevation': 250.0},
        'meteo_data': {
            'radiation_db': 'PVGIS-SARAH',
            'meteo_db': 'ERA-Interim',
            'year_min': 2005,
            'year_max': 2016,
            'use_horizon': True,
            'horizon_db': 'DEM-calculated'}}


@pytest.fixture
def epw_meta():
    return {
        'loc': 'LOCATION',
        'city': 'unknown',
        'state-prov': '-',
        'country': 'unknown',
        'data_type': 'ECMWF/ERA',
        'WMO_code': 'unknown',
        'latitude': 45.0,
        'longitude': 8.0,
        'TZ': 1.0,
        'altitude': 250.0}


@pytest.fixture
def meta_expected():
    with (DATA_DIR / 'pvgis_tmy_meta.json').open() as f:
        return json.load(f)


@pytest.fixture
def csv_meta(meta_expected):
    return [
        '%s: %s (%s)' % (k, v['description'], v['units']) for k, v
        in meta_expected['outputs']['tmy_hourly']['variables'].items()]


def test_get_pvgis_tmy(expected, month_year_expected, inputs_expected,
                       meta_expected):
    data, months_selected, inputs, meta = get_pvgis_tmy(45, 8)
    # check each column of output separately
    for outvar in meta_expected['outputs']['tmy_hourly']['variables'].keys():
        assert np.allclose(data[outvar], expected[outvar])
    assert np.allclose(
        [_['month'] for _ in months_selected], np.arange(1, 13, 1))
    assert np.allclose(
        [_['year'] for _ in months_selected], month_year_expected)
    inputs_loc = inputs['location']
    assert inputs_loc['latitude'] == inputs_expected['location']['latitude']
    assert inputs_loc['longitude'] == inputs_expected['location']['longitude']
    assert inputs_loc['elevation'] == inputs_expected['location']['elevation']
    inputs_met_data = inputs['meteo_data']
    expected_met_data = inputs_expected['meteo_data']
    assert (
        inputs_met_data['radiation_db'] == expected_met_data['radiation_db'])
    assert inputs_met_data['year_min'] == expected_met_data['year_min']
    assert inputs_met_data['year_max'] == expected_met_data['year_max']
    assert inputs_met_data['use_horizon'] == expected_met_data['use_horizon']
    assert inputs_met_data['horizon_db'] == expected_met_data['horizon_db']
    assert meta == meta_expected


def test_get_pvgis_tmy_kwargs(userhorizon_expected):
    _, _, inputs, _ = get_pvgis_tmy(45, 8, usehorizon=False)
    assert inputs['meteo_data']['use_horizon'] is False
    data, _, _, _ = get_pvgis_tmy(
        45, 8, userhorizon=[0, 10, 20, 30, 40, 15, 25, 5])
    assert np.allclose(
        data['G(h)'], userhorizon_expected['G(h)'].values)
    assert np.allclose(
        data['Gb(n)'], userhorizon_expected['Gb(n)'].values)
    assert np.allclose(
        data['Gd(h)'], userhorizon_expected['Gd(h)'].values)
    _, _, inputs, _ = get_pvgis_tmy(45, 8, startyear=2005)
    assert inputs['meteo_data']['year_min'] == 2005
    _, _, inputs, _ = get_pvgis_tmy(45, 8, endyear=2016)
    assert inputs['meteo_data']['year_max'] == 2016


def test_get_pvgis_tmy_basic(expected, meta_expected):
    data, _, _, _ = get_pvgis_tmy(45, 8, outputformat='basic')
    # check each column of output separately
    for outvar in meta_expected['outputs']['tmy_hourly']['variables'].keys():
        assert np.allclose(data[outvar], expected[outvar])


def test_get_pvgis_tmy_csv(expected, month_year_expected, inputs_expected,
                           meta_expected, csv_meta):
    data, months_selected, inputs, meta = get_pvgis_tmy(
        45, 8, outputformat='csv')
    # check each column of output separately
    for outvar in meta_expected['outputs']['tmy_hourly']['variables'].keys():
        assert np.allclose(data[outvar], expected[outvar])
    assert np.allclose(
        [_['month'] for _ in months_selected], np.arange(1, 13, 1))
    assert np.allclose(
        [_['year'] for _ in months_selected], month_year_expected)
    assert inputs['latitude'] == inputs_expected['location']['latitude']
    assert inputs['longitude'] == inputs_expected['location']['longitude']
    assert inputs['elevation'] == inputs_expected['location']['elevation']
    for meta_value in meta:
        if not meta_value:
            continue
        if meta_value == 'PVGIS (c) European Communities, 2001-2020':
            continue
        assert meta_value in csv_meta


def test_get_pvgis_tmy_epw(expected, epw_meta):
    data, _, _, meta = get_pvgis_tmy(
        45, 8, outputformat='epw')
    assert np.allclose(data.ghi, expected['G(h)'])
    assert np.allclose(data.dni, expected['Gb(n)'])
    assert np.allclose(data.dhi, expected['Gd(h)'])
    assert np.allclose(data.temp_air, expected['T2m'])
    assert meta == epw_meta


def test_get_pvgis_tmy_error():
    err_msg = 'outputformat: Incorrect value.'
    with pytest.raises(requests.HTTPError, match=err_msg):
        get_pvgis_tmy(45, 8, outputformat='bad')
    with pytest.raises(requests.HTTPError, match='404 Client Error'):
        get_pvgis_tmy(45, 8, url='https://re.jrc.ec.europa.eu/')
