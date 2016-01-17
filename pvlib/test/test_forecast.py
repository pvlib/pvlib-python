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

    from pvlib.forecast import GFS,HRRR_ESRL,HRRR,NAM,NDFD,RAP
    import pvlib.solarposition as solarposition
    from pvlib.location import Location

    # setup times and location to be tested. Tucson, AZ
    _latitude = 32.2
    _longitude = -110.9
    _tz = 'US/Arizona'
    _time = pd.DatetimeIndex([datetime.now()], tz=_tz)
    _models = [GFS, NAM, HRRR, RAP, NDFD, HRRR_ESRL]
    _working_models = []
    _variables = np.array(['temperature',
                           'wind_speed',
                           'total_clouds',
                           'low_clouds',
                           'mid_clouds',
                           'high_clouds',
                           'dni',
                           'dhi',
                           'ghi',])
    _nonnan_variables = np.array(['temperature',
                'wind_speed',
                'total_clouds',
                'dni',
                'dhi',
                'ghi',])

@requires_siphon
def test_fmcreation():
    for model in _models:
        if model.__name__ is not 'HRRR_ESRL':
            amodel = model()
            _working_models.append(amodel)
        else:
            try:
                amodel = model()
            except (ParseError, HTTPError):
                pass

@requires_siphon
def test_data_query():
    for amodel in _working_models:
        data = amodel.get_query_data(_latitude, _longitude, _time)

@requires_siphon
def test_dataframe_variables():
    for amodel in _working_models:
        for variable in _variables:
            assert variable in amodel.data.columns
        for variable in _nonnan_variables:
            assert not amodel.data[variable].isnull().values.any()

@requires_siphon
def test_vert_level():
    amodel = _working_models[0]
    vert_level = 5000
    data = amodel.get_query_data(_latitude, _longitude, _time,
        vert_level=vert_level)

@requires_siphon
def test_timerange():
    amodel = _working_models[0]
    start = datetime.now() # today's date
    end = start + timedelta(days=7) # 7 days from today
    timerange = pd.date_range(start, end, tz=_tz)
    data = amodel.get_query_data(_latitude, _longitude , timerange)

@requires_siphon
def test_queryvariables():
    amodel = _working_models[0]
    old_variables = amodel.variables
    new_variables = {'u':'u-component_of_wind_height_above_ground'}
    data = amodel.get_query_data(_latitude, _longitude, _time,
        variables=new_variables)
    amodel.variables = old_variables

@requires_siphon
def test_latest():
    GFS(set_type='latest')

@requires_siphon
def test_full():
    GFS(set_type='full')

@requires_siphon
def test_gfs():
    GFS(res='quarter')

@requires_siphon
def test_temp_convert():
    amodel = _working_models[0]
    amodel.queryvariables = ['Temperature_surface']
    amodel.data = pd.DataFrame({'temperature':[273.15]})
    amodel.convert_temperature()

    assert amodel.data['temperature'].values == 0.0

@requires_siphon
def test_bounding_box():
    amodel = GFS()
    latitude = [31.2,32.2]
    longitude = [-111.9,-110.9]
    new_variables = {'temperature':'Temperature_surface'}
    data = amodel.get_query_data(latitude, longitude, _time, 
        variables=new_variables)

@requires_siphon
def test_set_location():
    amodel = _working_models[0]
    time = datetime.now(timezone('UTC'))
    amodel.set_location(time)

