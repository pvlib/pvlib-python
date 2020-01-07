"""
test the pvgis IO tools
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import requests
from pvlib.iotools import get_pvgis_tmy

TESTS = Path(__file__).parent
PROJECT = TESTS.parent
DATA = PROJECT / 'data'
EXPECTED = pd.read_csv(DATA / 'pvgis_tmy_test.dat', index_col='time(UTC)')
MONTHS_SELECTED = [
    2009, 2012, 2014, 2010, 2011, 2013, 2011, 2011, 2013, 2013, 2013, 2011]
INPUTS = {
    'location': {'latitude': 45.0, 'longitude': 8.0, 'elevation': 250.0},
    'meteo_data': {
        'radiation_db': 'PVGIS-SARAH',
        'meteo_db': 'ERA-Interim',
        'year_min': 2005,
        'year_max': 2016,
        'use_horizon': True,
        'horizon_db': 'DEM-calculated'}}


def test_get_pvgis_tmy():
    data, months_selected, inputs, meta = get_pvgis_tmy(45, 8)
    # check each column of output separately
    for outvar in META['outputs']['tmy_hourly']['variables'].keys():
        assert np.allclose(data[outvar], EXPECTED[outvar], equal_nan=True)
    assert np.allclose(
        [_['month'] for _ in months_selected], np.arange(1, 13, 1))
    assert np.allclose([_['year'] for _ in months_selected], MONTHS_SELECTED)
    assert inputs['location']['latitude'] == INPUTS['location']['latitude']
    assert inputs['location']['longitude'] == INPUTS['location']['longitude']
    assert inputs['location']['elevation'] == INPUTS['location']['elevation']
    inputs_met_data = inputs['meteo_data']
    expected_met_data = INPUTS['meteo_data']
    assert (
        inputs_met_data['radiation_db'] == expected_met_data['radiation_db'])
    assert inputs_met_data['year_min'] == expected_met_data['year_min']
    assert inputs_met_data['year_max'] == expected_met_data['year_max']
    assert inputs_met_data['use_horizon'] == expected_met_data['use_horizon']
    assert inputs_met_data['horizon_db'] == expected_met_data['horizon_db']
    assert meta == META


def test_get_pvgis_tmy_basic():
    data, _, _, _ = get_pvgis_tmy(45, 8, outputformat='basic')
    # check each column of output separately
    for outvar in META['outputs']['tmy_hourly']['variables'].keys():
        assert np.allclose(data[outvar], EXPECTED[outvar], equal_nan=True)


def test_get_pvgis_tmy_csv():
    data, months_selected, inputs, meta = get_pvgis_tmy(
        45, 8, outputformat='csv')
    # check each column of output separately
    for outvar in META['outputs']['tmy_hourly']['variables'].keys():
        assert np.allclose(data[outvar], EXPECTED[outvar], equal_nan=True)
    assert np.allclose(
        [_['month'] for _ in months_selected], np.arange(1, 13, 1))
    assert np.allclose([_['year'] for _ in months_selected], MONTHS_SELECTED)
    assert inputs['latitude'] == INPUTS['location']['latitude']
    assert inputs['longitude'] == INPUTS['location']['longitude']
    assert inputs['elevation'] == INPUTS['location']['elevation']
    for meta_value in meta:
        if not meta_value:
            continue
        if meta_value == 'PVGIS (c) European Communities, 2001-2020':
            continue
        assert meta_value in META_VALUES


def test_get_pvgis_tmy_outputformat_error():
    err_msg = 'outputformat: Incorrect value.'
    with pytest.raises(requests.HTTPError, match=err_msg):
        get_pvgis_tmy(45, 8, outputformat='bad')


META = {
    'inputs': {
        'location': {
            'description': 'Selected location',
            'variables': {
                'latitude': {
                    'description': 'Latitude', 'units': 'decimal degree'},
                'longitude': {
                    'description': 'Longitude', 'units': 'decimal degree'},
                'elevation': {'description': 'Elevation', 'units': 'm'}}},
        'meteo_data': {
            'description': 'Sources of meteorological data',
            'variables': {
                'radiation_db': {'description': 'Solar radiation database'},
                'meteo_db': {
                    'description': 'Database used for meteorological variables'
                                   ' other than solar radiation'},
                'year_min': {'description': 'First year of the calculations'},
                'year_max': {'description': 'Last year of the calculations'},
                'use_horizon': {'description': 'Include horizon shadows'},
                'horizon_db': {'description': 'Source of horizon data'}}}},
    'outputs': {
        'months_selected': {
            'type': 'time series',
            'timestamp': 'monthly',
            'description': 'months selected for the TMY'},
        'tmy_hourly': {
            'type': 'time series',
            'timestamp': 'hourly',
            'variables': {
                'T2m': {
                    'description': '2-m air temperature',
                    'units': 'degree Celsius'},
                'RH': {'description': 'relative humidity', 'units': '%'},
                'G(h)': {
                    'description': 'Global irradiance on the horizontal plane',
                    'units': 'W/m2'},
                'Gb(n)': {
                    'description': 'Beam/direct irradiance on a plane always'
                                   ' normal to sun rays',
                    'units': 'W/m2'},
                'Gd(h)': {
                    'description': 'Diffuse irradiance on the horizontal'
                                   ' plane',
                    'units': 'W/m2'},
                'IR(h)': {
                    'description': 'Surface infrared (thermal) irradiance on a'
                                   ' horizontal plane',
                    'units': 'W/m2'},
                'WS10m': {
                    'description': '10-m total wind speed', 'units': 'm/s'},
                'WD10m': {
                    'description': '10-m wind direction (0 = N, 90 = E)',
                    'units': 'degree'},
                'SP': {
                    'description': 'Surface (air) pressure', 'units': 'Pa'}}}}}
META_VALUES = [
    '%s: %s (%s)' % (k, v['description'], v['units'])
    for k, v in META['outputs']['tmy_hourly']['variables'].items()]
