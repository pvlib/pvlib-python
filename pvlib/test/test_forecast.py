from datetime import datetime,timedelta
import inspect
from math import isnan

import numpy as np
import pandas as pd
import requests

from nose.tools import raises, assert_almost_equals
from nose.plugins.skip import SkipTest
from numpy.testing import assert_almost_equal

import pvlib.forecast as forecast
import pvlib.solarposition as solarposition
from pvlib.location import Location

# setup times and location to be tested. Tucson, AZ
_latitude = 32.2
_longitude = -110.9
_tz = 'US/Arizona'
_time = pd.DatetimeIndex([datetime.now()], tz=_tz)
_model_list = ['GFS', 'HRRR_ESRL', 'NAM', 'HRRR', 'NDFD', 'RAP']
_models = {}
_variables = np.array(['temperature',
                        'wind_speed',
                        'pressure',
                        'total_clouds',
                        'low_clouds',
                        'mid_clouds',
                        'high_clouds',
                        'dni',
                        'dhi',
                        'ghi',])


def test_fmcreation():
    members = inspect.getmembers(forecast,inspect.isclass)
    [members.remove(cls) for cls in members if cls[0] in ['ForecastModel', 
        'NCSS', 'TDSCatalog']]

    for model in members:
        if model[0] in _model_list:
            amodel = model[1]()
            if model[0] != _model_list[1]:
                _models[model[0]] = amodel


def test_data_query():
    exclude_models = []
    for name, model in _models.items():
        data = model.get_query_data(_latitude, _longitude, _time)


def test_dataframe():
    for name, model in _models.items():
        for variable in _variables:
            assert variable in model.data.columns


def test_vert_level():
    amodel = _models['GFS']
    vert_level = 5000
    data = amodel.get_query_data(_latitude, _longitude, _time, vert_level=vert_level)


def test_timerange():
    amodel = _models['GFS']
    start = datetime.now() # today's date
    end = start + timedelta(days=7) # 7 days from today
    timerange = pd.date_range(start, end, tz=_tz)
    data = amodel.get_query_data(_latitude, _longitude , timerange)


def test_variables():
    amodel = _models['GFS']
    old_variables = amodel.variables
    new_variables = {'u':'u-component_of_wind_height_above_ground'}
    data = amodel.get_query_data(_latitude, _longitude, _time,
        variables=new_variables)
    amodel.variables = old_variables


def test_latest():
    forecast.GFS(set_type='latest')


def test_full():
    forecast.GFS(set_type='full')


def test_gfs():
    forecast.GFS(res='quarter')


def test_temp_convert():
    amodel = _models['GFS']
    amodel.data = pd.DataFrame({'temperature':[273.15]})
    amodel.convert_temperature()

    assert amodel.data['temperature'].values == 0.0


def test_bounding_box():
    amodel = forecast.GFS()
    latitude = [31.2,32.2]
    longitude = [-111.9,-110.9]
    new_variables = {'temperature':'Temperature_surface'}
    data = amodel.get_query_data(latitude, longitude, _time, 
        variables=new_variables)
