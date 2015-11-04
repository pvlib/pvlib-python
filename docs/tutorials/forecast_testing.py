
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

from pvlib.forecast import *

import pickle as pkl

def testGFS(start,end):

    fm = GFS()
    data = fm.get_query_data([-110.9,32.2],[start,end])

    time_vals = fm.time

    var_name = fm.variables[0]

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.plot(time_vals, data[var_name], 'r', linewidth=2)
    ax.set_ylabel(fm.var_stdnames[var_name] + ' (%s)' % fm.var_units[var_name])
    ax.set_xlabel('Forecast Time (UTC)')

    var_name = fm.variables[3]

    total_cloud_cover = fm.data[var_name]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in fm.variables[1:]:
        ax.plot(time_vals, fm.data[varname], linewidth=2, label=varname)
        
    ax.set_ylabel('Cloud cover' + ' (%s)' % fm.var_units[var_name])
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend(bbox_to_anchor=(1.4,1.1))
    ax.set_title('GFS 0.5 deg')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_vals, total_cloud_cover, 'r', linewidth=2)
    ax.set_ylabel('Total cloud cover' + ' (%s)' % fm.var_units[var_name])
    ax.set_xlabel('Forecast Time (UTC)')
    ax.set_title('GFS 0.5 deg')

    plt.show()    

def testNAM(start,end):

    fm = GFS()
    data = fm.get_query_data([-110.9,32.2],[start,end])

    time_vals = fm.time

    cloud_vars = fm.variables[3:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in cloud_vars:
        ax.plot(time_vals, fm.data[varname], linewidth=2, label=varname)
        
    ax.set_ylabel('Cloud cover (%)')
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend(bbox_to_anchor=(1.4,1.1))
    ax.set_title('NAM')

    ghis = fm.variables[:2]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in ghis:
        ax.plot(time_vals, fm.data[varname], linewidth=2, label=varname)
    ax.set_ylabel('GHI W/m**2')
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend()

    plt.show()


def testNDFD(start,end):

    # The times are messed up, but I'm sure they can be fixed with a little effort.

    # ## NDFD

    fm = NDFD()
    data = fm.get_query_data([-110.9,32.2],[start,end])

    time_vals = fm.time

    total_cloud_cover = fm.data['Total_cloud_cover_surface']
    temp = fm.data['Temperature_surface']
    wind = fm.data['Wind_speed_surface']

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_vals, total_cloud_cover, 'r', linewidth=2)
    ax.set_ylabel('Cloud cover (%)')
    ax.set_xlabel('Forecast Time (UTC)')
    plt.ylim(0,100)

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_vals, temp, 'r', linewidth=2)
    ax.set_ylabel('temp {}'.format(fm.var_units['Temperature_surface']))
    ax.set_xlabel('Forecast Time (UTC)')

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_vals, wind, 'r', linewidth=2)
    ax.set_ylabel('wind {}'.format(fm.var_units['Wind_speed_surface']))
    ax.set_xlabel('Forecast Time (UTC)')

    plt.show()


def exploreTDS():

    cat = TDSCatalog('http://thredds.ucar.edu/thredds/catalog.xml')

    # print(list(cat.catalog_refs.keys()))
    # print(cat.catalog_refs['Forecast Model Data'].href)
    # print(cat.services)

    cat = TDSCatalog(cat.catalog_refs['Forecast Model Data'].href)
    # cat = TDSCatalog(cat.catalog_refs['Forecast Products and Analyses'].href)
    catList = sorted(list(cat.catalog_refs.keys()))
    for item in catList:
        print(item)

    # print(cat.catalog_refs['GFS Half Degree Forecast'].href)
    cat = TDSCatalog(cat.catalog_refs['GFS Half Degree Forecast'].href)
    # cat = TDSCatalog(cat.catalog_refs['NAM CONUS 12km from CONDUIT'].href)
    print(cat.datasets.keys())
    print(cat.__dict__['datasets'])
    # print(cat.__dict__.keys())
    print(cat.datasets['Best GFS Half Degree Forecast Time Series'].__dict__)
    catKey = list(cat.datasets.keys())[2]
    print(cat.datasets[catKey].__dict__)
    
    catList = sorted(list(cat.catalog_refs.keys()))
    for item in catList:
        print(item)

    # best_gfs_nc_sub = NCSS('http://thredds-jumbo.unidata.ucar.edu/thredds/ncss/grib/NCEP/GFS/Global_0p25deg/Best')
    # best_gfs_nc_sub.variables
    # variables = [
    #              'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average',
    #              'Temperature_surface',
    #              'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
    #              'Total_cloud_cover_convective_cloud',
    #              'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
    #              'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
    #              'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
    #              'Total_cloud_cover_middle_cloud_Mixed_intervals_Average']
    # query = best_gfs_nc_sub.query()
    # query.lonlat_point(-110.9, 32.2).time_range(start, end)
    # query.variables(*variables)
    # #query.vertical_level(100000)
    # #query.add_query_parameter(height_above_ground=1)
    # query.accept('netcdf')
    # data = best_gfs_nc_sub.get_data(query)


def testFM(start,end):

    # gfs = GFS()
    # gfshires = GFS_HIRES()
    # nam = NAM()
    # rap = RAP()
    hrrr = HRRR('esrl')
    # noaa = NOAA()
    # ndfd = NDFD()

    # models = [gfs,gfshires,nam,rap,hrrr,noaa,ndfd]
    models = [hrrr]

    varlists = []

    for model in models:
        model.get_query_data([-110.9,32.2],[start,end])
        print(model.data_labels)

    #     # print(model.access_url)
    #     print(len(model.ncss.variables))
    #     # print(model.ncss.variables)
    #     varlists.append(sorted(model.ncss.variables))

    # pkl_file = open('varlists.pkl', 'wb')
    # pkl.dump(varlists,pkl_file)
    # pkl_file.close()

    # pkl_file = open('varlists.pkl', 'rb')
    # lists = pkl.load(pkl_file)
    # pkl_file.close()

    # outlines = []

    # for alist in lists:
    #     outlines.append(','.join(alist)+'\n')

    # outfile = open('varlist.csv','w')
    # outfile.writelines(outlines)
    # outfile.close()

    # data = gfs.get_query_data([-110.9,32.2],[start,end])
    # print(data)


    # print(gfs.model_name,nam.model_name)

    # gfs.model_name = 'test'

    # print(gfs.model_name,nam.model_name)


def main():

    start = datetime.utcnow()
    end = start + timedelta(days=7)

    # testGFS(start,end) # 3 plots
    # testNAM(start,end) # 2 plots
    # testNDFD(start,end) # 3 plots

    # exploreTDS()

    testFM(start,end)

if __name__=='__main__':
    main()