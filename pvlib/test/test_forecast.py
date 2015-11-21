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

# setup times and location to be tested.
_location = [32.2,-110.9] # Tucson, AZ
_tz = 'US/Arizona'
_start = datetime.now() # today's date
_end = _start + timedelta(days=7) # 7 days from today
_timerange = pd.date_range(_start, _end, tz=_tz)

_models = {}
_variables = np.array(['temperature',
                       'temperature_iso',
                       'wind_speed',
                       'total_clouds',
                       'low_clouds',
                       'mid_clouds',
                       'high_clouds',
                       'boundary_clouds',
                       'convect_clouds',
                       'downward_shortwave_radflux',
                       'downward_shortwave_radflux_avg',])

def test_fmcreation():
    members = inspect.getmembers(forecast,inspect.isclass)
    [members.remove(cls) for cls in members if cls[0] in ['ForecastModel','NCSS','TDSCatalog'] ]

    for model in members:
        amodel = model[1]()
        _models[model[0]] = amodel

def test_data_query():
    for name, model in _models.items():
        try:
            data = model.get_query_data(_location,_timerange)
        except requests.exceptions.HTTPError:
            raise SkipTest

        for variable in model.variables:
            try:
                assert isnan(data[variable][0]) == False
            except AssertionError:
                pass

def test_cloudy_day():
    amodel = _models['GFS']
    a_point = Location(_location[0], _location[1], altitude=7.0, name='Tucson',tz=_tz)
    solpos = solarposition.get_solarposition(amodel.time, a_point)

    amodel.addRadiation(solpos.zenith)
    amodel.cloudy_day_check()

def test_vert_level():
    amodel = _models['GFS']
    vert_level = 5000
    data = amodel.get_query_data(_location,_timerange,vert_level=vert_level)

def single_day():
    amodel = _models['GFS']
    timerange = pd.DatetimeIndex([datetime.now()], tz=_tz)
    data = amodel.get_query_data(_location,timerange)

def test_variables():
    amodel = _models['GFS']
    label_dict = {'u':'u-component_of_wind_height_above_ground'}
    data = amodel.get_query_data(_location,_timerange,labels=label_dict)

def test_latest():
    forecast.GFS(set_type='latest')

def test_full():
    forecast.GFS(set_type='full')

def test_bounding_box():
    amodel = forecast.GFS()
    location = [-111.9,-110.9,31.2,32.2] 
    data = amodel.get_query_data(location,_timerange,labels={'pressure':'Pressure_surface'})

