from datetime import datetime, timedelta
import inspect
from math import isnan
from pytz import timezone

import numpy as np
import pandas as pd

from nose.tools import raises, assert_almost_equals
from nose.plugins.skip import SkipTest
from numpy.testing import assert_almost_equal

from . import requires_siphon, has_siphon

if has_siphon:
    import requests
    from requests.exceptions import HTTPError
    from xml.etree.ElementTree import ParseError

    from pvlib.forecast import GFS, HRRR_ESRL, HRRR, NAM, NDFD, RAP
    from pvlib.location import Location

    # setup times and location to be tested. Tucson, AZ
    _latitude = 32.2
    _longitude = -110.9
    _tz = 'US/Arizona'
    _start = pd.Timestamp.now(tz=_tz)
    _end = _start + pd.Timedelta(days=1)
    _models = [GFS, NAM, HRRR, RAP, NDFD, HRRR_ESRL]
    _working_models = []
    _variables = ['temp_air', 'wind_speed', 'total_clouds', 'low_clouds',
                  'mid_clouds', 'high_clouds', 'dni', 'dhi', 'ghi',]
    _nonnan_variables = ['temp_air', 'wind_speed', 'total_clouds', 'dni',
                         'dhi', 'ghi',]

@requires_siphon
def test_model_creation():
    for model in _models:
        if model.__name__ != 'HRRR_ESRL':
            try:
                resolutions = model._resolutions
            except AttributeError:
                amodel = model()
                _working_models.append(amodel)
            else:
                for resolution in resolutions:
                    amodel = model(resolution=resolution)
                    _working_models.append(amodel)

@requires_siphon
def test_data_query():
    for model in _working_models:
        yield run_query, model

def run_query(model):
    model.data = model.get_processed_data(_latitude, _longitude, _start, _end)

@requires_siphon
def test_dataframe_variables():
    for amodel in _working_models:
        yield run_variables, amodel

def run_variables(amodel):
#     for variable in _variables:
#         assert variable in amodel.data.columns
    for variable in _nonnan_variables:
        assert not amodel.data[variable].isnull().values.any()

@requires_siphon
def test_vert_level():
    amodel = _working_models[0]
    vert_level = 5000
    data = amodel.get_processed_data(_latitude, _longitude, _start, _end,
                                     vert_level=vert_level)

@requires_siphon
def test_datetime():
    amodel = _working_models[0]
    start = datetime.now()
    end = start + timedelta(days=1)
    data = amodel.get_processed_data(_latitude, _longitude , start, end)

@requires_siphon
def test_queryvariables():
    amodel = _working_models[0]
    old_variables = amodel.variables
    new_variables = ['u-component_of_wind_height_above_ground']
    data = amodel.get_data(_latitude, _longitude, _start, _end,
                           query_variables=new_variables)
    data['u-component_of_wind_height_above_ground']

@requires_siphon
def test_latest():
    GFS(set_type='latest')

@requires_siphon
def test_full():
    GFS(set_type='full')

@requires_siphon
def test_temp_convert():
    amodel = _working_models[0]
    data = pd.DataFrame({'temp_air': [273.15]})
    data['temp_air'] = amodel.kelvin_to_celsius(data['temp_air'])

    assert data['temp_air'].values == 0.0

# @requires_siphon
# def test_bounding_box():
#     amodel = GFS()
#     latitude = [31.2,32.2]
#     longitude = [-111.9,-110.9]
#     new_variables = {'temperature':'Temperature_surface'}
#     data = amodel.get_query_data(latitude, longitude, _start, _end,
#                                  variables=new_variables)

@requires_siphon
def test_set_location():
    amodel = _working_models[0]
    time = datetime.now(timezone('UTC'))
    amodel.set_location(time)

