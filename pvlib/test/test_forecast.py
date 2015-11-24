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
_timerange = pd.DatetimeIndex([datetime.now()], tz=_tz)
_model_list = ['GFS','HRRR_ESRL','NAM','HRRR','NDFD','RAP']
_models = {}
_variables = np.array(['temperature',
                       'temperature_iso',
                       'wind_speed',
                       'wind_speed_gust',
                       'pressure',
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
        if model[0] in _model_list:
            amodel = model[1]()
            _models[model[0]] = amodel

def test_data_query():
    names = []
    for name, model in _models.items():
        try:
            data = model.get_query_data(_location,_timerange) 
        except requests.exceptions.HTTPError:
            names.append(name)

    for name in names:
        _models.pop(name)

def test_dataframe():
    for name, model in _models.items():
        for variable in _variables:
            assert variable in model.data.columns

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

def test_single_day():
    amodel = _models['GFS']
    start = datetime.now() # today's date
    end = start + timedelta(days=7) # 7 days from today
    timerange = pd.date_range(start, end, tz=_tz)
    data = amodel.get_query_data(_location,timerange)

def test_variables():
    amodel = _models['GFS']
    data_labels = amodel.data_labels
    label_dict = {'u':'u-component_of_wind_height_above_ground'}
    data = amodel.get_query_data(_location,_timerange,labels=label_dict)
    amodel.data_labels = data_labels

def test_latest():
    forecast.GFS(set_type='latest')

def test_full():
    forecast.GFS(set_type='full')

def test_bounding_box():
    amodel = forecast.GFS()
    location = [-111.9,-110.9,31.2,32.2] 
    data = amodel.get_query_data(location,_timerange,labels={'pressure':'Pressure_surface'})

def test_tao():
    amodel = _models['GFS']
    amodel.data = pd.DataFrame({'total_clouds':[0.0,100.0]})
    tao = amodel.tao()

    assert tao[0] == 0.4
    assert tao[1] == 0.75

def test_airmass():
    amodel = _models['GFS']
    amodel.zenith = np.array([90,60,30])
    m = amodel.air_mass()

    assert_almost_equal(m,[37.919608,
                            1.9942928,
                            1.1539922], 1)

def test_temp_convert():
    amodel = _models['GFS']
    amodel.data = pd.DataFrame({'temperature':[273.15],'temperature_iso':[273.15]})
    amodel.convert_temperature()

    assert amodel.data['temperature'].values == 0.0
    assert amodel.data['temperature_iso'].values == 0.0

def test_dni():
    pass

def test_dhi():
    pass

def test_ghi():
    pass

def test_varmap():
    varmap_dict = \
    {'GFS':['Temperature_isobaric',
            'Wind_speed_gust_surface',
            'Pressure_surface',
            'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
            'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
            'Total_cloud_cover_middle_cloud_Mixed_intervals_Average',
            'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
            'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
            'Total_cloud_cover_convective_cloud',
            'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average',],
    'HRRR_ESRL':['Temperature_surface',
            'Wind_speed_gust_surface',
            'Pressure_surface',
            'Total_cloud_cover_entire_atmosphere',
            'Low_cloud_cover_UnknownLevelType-214',
            'Medium_cloud_cover_UnknownLevelType-224',
            'High_cloud_cover_UnknownLevelType-234',
            'Downward_short-wave_radiation_flux_surface',],
    'NAM':['Temperature_surface',
            'Temperature_isobaric',
            'Wind_speed_gust_surface',
            'Pressure_surface',
            'Total_cloud_cover_entire_atmosphere_single_layer',
            'Low_cloud_cover_low_cloud',
            'Medium_cloud_cover_middle_cloud',
            'High_cloud_cover_high_cloud',
            'Downward_Short-Wave_Radiation_Flux_surface',
            'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average',],
   'HRRR':['Temperature_height_above_ground',
            'Temperature_isobaric',
            'Wind_speed_gust_surface',
            'Pressure_surface',
            'Total_cloud_cover_entire_atmosphere',
            'Low_cloud_cover_low_cloud',
            'Medium_cloud_cover_middle_cloud',
            'High_cloud_cover_high_cloud',],
    'NDFD':['Temperature_surface',
            'Wind_speed_surface',
            'Wind_speed_gust_surface',
            'Total_cloud_cover_surface',],
    'RAP':['Temperature_surface',
            'Wind_speed_gust_surface',
            'Pressure_surface',
            'Total_cloud_cover_entire_atmosphere_single_layer',
            'Low_cloud_cover_low_cloud',
            'Medium_cloud_cover_middle_cloud',
            'High_cloud_cover_high_cloud',]}

    idx_dict = \
    {'GFS':[1,3,4,5,6,7,8,9,10],
    'HRRR_ESRL':[0,3,4,5,6,7,8,11],
    'NAM':[0,1,3,4,5,6,7,8,11,12],
    'HRRR':[0,1,3,4,5,6,7,8],
    'NDFD':[0,2,3,5],
    'RAP':[0,3,4,5,6,7,8]}

    for name,model in _models.items():
        label_dict = dict(zip(_variables[idx_dict[name]],varmap_dict[name]))
        for key, _ in model.data_labels.items():
            assert label_dict[key] == model.data_labels[key]
