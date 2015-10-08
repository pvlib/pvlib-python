
# # Testing Unidata/Siphon for solar power forecasting applications

# This notebook is meant to test the usefulness of [Unidata's Siphon project](https://github.com/Unidata/siphon) for solar forecasting applications. Siphon has the potential to make accessing subsets of model data lot easier than using the traditional NOMADS services.
# 
# Sections:
# 1. [GFS](#GFS) and [GFS 0.25 deg](#GFS-0.25-deg)
# 2. [NAM](#NAM)
# 2. [RAP](#RAP)
# 2. HRRR [NCEP](#NCEP) and [ESRL](#ESRL)
# 2. [NDFD](#NDFD)
# 
# Requirements:
# * All siphon requirements as described [here](http://siphon.readthedocs.org/en/latest/developerguide.html).
# * Matplotlib, Seaborn, IPython notebook.
# 
# Authors: 
# * Will Holmgren, University of Arizona, July 2015

# Standard scientific python imports

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes()

import pandas as pd
import numpy as np

from datetime import datetime, timedelta
import os

# Imports for weather data.
import netCDF4
from netCDF4 import num2date

from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS

from pvlibfm import PvlibFM

def testGFS(start,end):

    model_list = ['GFS Half Degree Forecast',
                  'NAM CONUS 12km from CONDUIT']

    fm = PvlibFM(model_type='Forecast Model Data')
    # fm.get_fm_models()
    fm.set_model(model_list[0])
    fm.set_dataset()
    # print(fm.get_model_datasets())
    # print(fm.get_model_vars())

    fm.set_latlon([-110.9,32.2])
    fm.set_time([start,end])
    fm.set_vertical_level(100000)
    fm.set_query_vars(['Temperature_isobaric'])

    data = fm.get_query_data()

    temp = fm.get_temp()
    time_vals = fm.get_time()

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.plot(time_vals, temp[:].squeeze(), 'r', linewidth=2)
    ax.set_ylabel(temp.standard_name + ' (%s)' % temp.units)
    ax.set_xlabel('Forecast Time (UTC)')

    fm.close()
 
    plt.show()

    fm = PvlibFM(model_type='Forecast Model Data')
    fm.set_model(model_list[0])
    fm.set_dataset()

    fm.set_latlon([-110.9,32.2])
    fm.set_time([start,end])
    fm.set_vertical_level(100000)

    cloud_vars = ['Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
                  'Total_cloud_cover_convective_cloud',
                  'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
                  'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
                  'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
                  'Total_cloud_cover_middle_cloud_Mixed_intervals_Average']

    fm.set_query_vars(cloud_vars)

    data = fm.get_query_data()

    time_vals = fm.get_time()

    total_cloud_cover = fm.get_cloud_cover()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in cloud_vars:
        ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
        
    ax.set_ylabel('Cloud cover' + ' (%s)' % total_cloud_cover.units)
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend(bbox_to_anchor=(1.4,1.1))
    ax.set_title('GFS 0.5 deg')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_vals, total_cloud_cover[:].squeeze(), 'r', linewidth=2)
    ax.set_ylabel('Total cloud cover' + ' (%s)' % total_cloud_cover.units)
    ax.set_xlabel('Forecast Time (UTC)')
    ax.set_title('GFS 0.5 deg')

    fm.close()

    plt.show()    

def testNAM(start,end):

    model_list = ['GFS Half Degree Forecast',
                  'NAM CONUS 12km from CONDUIT']

    fm = PvlibFM(model_type='Forecast Model Data')
    # fm.get_fm_models()
    fm.set_model(model_list[1])
    fm.set_dataset()
    # print(fm.get_model_datasets())
    # print(fm.get_model_vars())

    fm.set_latlon([-110.9,32.2])
    fm.set_time([start,end])

    variables = ['Downward_Short-Wave_Radiation_Flux_surface',
                 'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average',
                 'Temperature_surface',
                 'High_cloud_cover_high_cloud',
                 'Low_cloud_cover_low_cloud',
                 'Medium_cloud_cover_middle_cloud',
                 'Total_cloud_cover_entire_atmosphere_single_layer']

    fm.set_query_vars(variables)

    data = fm.get_query_data()
 
    time_vals = fm.get_time()

    # The NAM has cloud cover and GHI data. I'm guessing that GHI is not very good.

    cloud_vars = variables[3:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in cloud_vars:
        ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
        
    ax.set_ylabel('Cloud cover (%)')
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend(bbox_to_anchor=(1.4,1.1))
    ax.set_title('NAM')

    ghis = variables[:2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in ghis:
        ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
    ax.set_ylabel('GHI W/m**2')
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend()

    data.close()

    plt.show()


def testNDFD(start,end):

    # The times are messed up, but I'm sure they can be fixed with a little effort.

    # ## NDFD

    model_list = ['National Weather Service CONUS Forecast Grids (CONDUIT)']

    fm = PvlibFM(model_type='Forecast Products and Analyses')
    # print(fm.get_fm_models())
    fm.set_model(model_list[2])
    fm.set_dataset()
    # print(fm.get_model_datasets())
    # print(fm.get_model_vars())

    fm.set_latlon([-110.9,32.2])
    fm.set_time([start,end])

    ndfd_vars = ['Total_cloud_cover_surface',
                 'Temperature_surface',
                 'Wind_speed_surface']

    fm.set_query_vars(ndfd_vars)

    data = fm.get_query_data()

    total_cloud_cover = data['Total_cloud_cover_surface']
    temp = data['Temperature_surface']
    wind = data['Wind_speed_surface']
    time_vals = fm.get_time()

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_vals, total_cloud_cover[:].squeeze(), 'r', linewidth=2)
    ax.set_ylabel('Cloud cover (%)')
    ax.set_xlabel('Forecast Time (UTC)')
    plt.ylim(0,100)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_vals, temp[:].squeeze(), 'r', linewidth=2)
    ax.set_ylabel('temp {}'.format(temp.units))
    ax.set_xlabel('Forecast Time (UTC)')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_vals, wind[:].squeeze(), 'r', linewidth=2)
    ax.set_ylabel('wind {}'.format(wind.units))
    ax.set_xlabel('Forecast Time (UTC)')

    data.close()

    plt.show()


def exploreTDS():

    cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog.xml')

    print(list(cat.catalog_refs.keys()))
    print(cat.catalog_refs['Forecast Model Data'].href)
    print(cat.services)

    cat = TDSCatalog(cat.catalog_refs['Forecast Model Data'].href)
    catList = sorted(list(cat.catalog_refs.keys()))
    for item in catList:
        print(item)

    # print(cat.catalog_refs['GFS Half Degree Forecast'].href)
    # cat = TDSCatalog(cat.catalog_refs['GFS Half Degree Forecast'].href)
    cat = TDSCatalog(cat.catalog_refs['NAM CONUS 12km from CONDUIT'].href)
    print(cat.datasets.keys())
    # print(cat.__dict__['datasets'])
    # print(cat.__dict__.keys())
    # print(cat.datasets['Best GFS Half Degree Forecast Time Series'].__dict__)
    catKey = list(cat.datasets.keys())[2]
    print(cat.datasets[catKey].__dict__)
    
    # catList = sorted(list(cat.catalog_refs.keys()))
    # for item in catList:
    #     print(item)


def main():

    start = datetime.utcnow()
    end = start + timedelta(days=7)

    # testGFS(start,end) # 3 plots
    # testNAM(start,end) # 2 plots
    # testNDFD(start,end) # 3 plots

    # exploreTDS()

if __name__=='__main__':
    main()