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


def test_GFSdict():

    gfs = _models['GFS']

    varlist = ['Temperature_isobaric',
               'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
               'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
               'Total_cloud_cover_middle_cloud_Mixed_intervals_Average',
               'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
               'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
               'Total_cloud_cover_convective_cloud']

    idx = [1,3,4,5,6,7,8]
    label_dict = dict(zip(_variables[idx],varlist))

    for key, _ in gfs.data_labels.iteritems():
        assert label_dict[key] == gfs.data_labels[key]


def test_GSDdict():

    gsd = _models['GFS']

    varlist = ['Temperature_surface',
               'Total_cloud_cover_entire_atmosphere',
               'Low_cloud_cover_UnknownLevelType-214',
               'Medium_cloud_cover_UnknownLevelType-224',
               'High_cloud_cover_UnknownLevelType-234',
               'Downward_short-wave_radiation_flux_surface']

    idx = [0,3,4,5,6,9]
    label_dict = dict(zip(_variables[idx],varlist))

    for key, _ in gsd.data_labels.iteritems():
        assert label_dict[key] == gsd.data_labels[key]


def test_NAMdict():

    nam = _models['GFS']

    varlist = ['Temperature_surface',
                 'Temperature_isobaric',
                 'Total_cloud_cover_entire_atmosphere_single_layer',
                 'Low_cloud_cover_low_cloud',
                 'Medium_cloud_cover_middle_cloud',
                 'High_cloud_cover_high_cloud',
                 'Downward_Short-Wave_Radiation_Flux_surface',
                 'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average']
    idx = [0,1,3,4,5,6,9,10]
    label_dict = dict(zip(_variables[idx],varlist))

    for key, _ in nam.data_labels.iteritems():
        assert label_dict[key] == nam.data_labels[key]


def test_NCEPdict():

    ncep = _models['GFS']

    varlist = ['Total_cloud_cover_entire_atmosphere',
               'Low_cloud_cover_low_cloud',
               'Medium_cloud_cover_middle_cloud',
               'High_cloud_cover_high_cloud',]

    idx = [3,4,5,6]
    label_dict = dict(zip(_variables[idx],varlist))

    for key, _ in ncep.data_labels.iteritems():
        assert label_dict[key] == ncep.data_labels[key]


def test_NDFDdict():

    ndfd = _models['GFS']

    varlist = ['Temperature_surface',
               'Wind_speed_surface',
               'Total_cloud_cover_surface']

    idx = [0,2,3]
    label_dict = dict(zip(_variables[idx],varlist))

    for key, _ in ndfd.data_labels.iteritems():
        assert label_dict[key] == ndfd.data_labels[key]


def test_RAPdict():

    rap = _models['GFS']

    varlist = ['Temperature_surface',
               'Total_cloud_cover_entire_atmosphere_single_layer',
               'Low_cloud_cover_low_cloud',
               'Medium_cloud_cover_middle_cloud',
               'High_cloud_cover_high_cloud']

    idx = [0,3,4,5,6]
    label_dict = dict(zip(_variables[idx],varlist))

    for key, _ in rap.data_labels.iteritems():
        assert label_dict[key] == rap.data_labels[key]


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