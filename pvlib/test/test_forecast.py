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

# setup times and location to be tested.
start = datetime.utcnow() # today's date
end = start + timedelta(days=7) # 7 days from today
timerange = [start, end]
tus = [-110.9, 32.2] # Tucson, AZ

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
            data = model.get_query_data(tus,timerange)
        except requests.exceptions.HTTPError:
            pass
            # raise SkipTest

        for variable in model.variables:
            try:
                assert isnan(data[variable][0]) == False
            except AssertionError:
                pass