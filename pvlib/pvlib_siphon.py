
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


def testGFS(start,end):

    ## GFS

    # To get started, I'll try to follow Siphon's [timeseries example docs](http://siphon.readthedocs.org/en/latest/examples/generated/NCSS_Timeseries_Examples.html) fairly closely.

    best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p5deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p5deg/Best')

    best_ds = list(best_gfs.datasets.values())[0]

    ncss = NCSS(best_ds.access_urls['NetcdfSubset']) #NCSSDataset
    # print(ncss.variables)

    query = ncss.query()

    query.lonlat_point(-110.9, 32.2).vertical_level(100000).time_range(start, end)
    query.variables('Temperature_isobaric').accept('netcdf')

    data = ncss.get_data(query)

    temp = data.variables['Temperature_isobaric']
    time = data.variables['time']

    # print(data.filepath())
    # tmp_path = data.filepath()

    time_vals = num2date(time[:].squeeze(), time.units)
    print(time_vals[::5])

    fig, ax = plt.subplots(1, 1, figsize=(9, 8))
    ax.plot(time_vals, temp[:].squeeze(), 'r', linewidth=2)
    ax.set_ylabel(temp.standard_name + ' (%s)' % temp.units)
    ax.set_xlabel('Forecast Time (UTC)')

    data.close()
    # os.remove(tmp_path)

    plt.show()

    cloud_vars = ['Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
                    'Total_cloud_cover_convective_cloud',
                    'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
                    'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
                    'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
                    'Total_cloud_cover_middle_cloud_Mixed_intervals_Average']

    query = ncss.query()
    query.lonlat_point(-110.9, 32.2).time_range(start, end)
    query.variables(*cloud_vars)
    query.accept('netcdf')

    data = ncss.get_data(query)

    total_cloud_cover = data.variables['Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average']

    time = data.variables['time']
    time_vals = num2date(time[:].squeeze(), time.units)

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

    data.close()

    # plt.show()
    
    ## GFS 0.25 deg

    # Now with the new, higher resolution GFS...

    best_gfs_nc_sub = NCSS('http://thredds-jumbo.unidata.ucar.edu/thredds/ncss/grib/NCEP/GFS/Global_0p25deg/Best')

    variables = [
                 'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average',
                 'Temperature_surface',
                 'Total_cloud_cover_boundary_layer_cloud_Mixed_intervals_Average',
                 'Total_cloud_cover_convective_cloud',
                 'Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average',
                 'Total_cloud_cover_high_cloud_Mixed_intervals_Average',
                 'Total_cloud_cover_low_cloud_Mixed_intervals_Average',
                 'Total_cloud_cover_middle_cloud_Mixed_intervals_Average',
                 ]
    
    query = best_gfs_nc_sub.query()
    query.lonlat_point(-110.9, 32.2).time_range(start, end)
    query.variables(*variables)
    #query.vertical_level(100000)
    #query.add_query_parameter(height_above_ground=1)
    query.accept('netcdf')

    data = best_gfs_nc_sub.get_data(query)

    total_cloud_cover = data.variables['Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average']
    total_cloud_cover_05_deg = total_cloud_cover

    time = data.variables['time']
    time_vals = num2date(time[:].squeeze(), time.units)

    cloud_vars = variables[2:]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in cloud_vars:
        ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
        
    ax.set_ylabel('Cloud cover' + ' (%s)' % total_cloud_cover.units)
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend(bbox_to_anchor=(1.4,1.1))
    ax.set_title('GFS 0.25 deg')

    ghis = variables[:1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in ghis:
        ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
    ax.set_ylabel('GHI W/m**2')
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend()

    # I don't know anything about the GHI forecasts from the GFS, but I don't think they'll be very good for solar power forecasting.

    # Compare the 0.5 and 0.25 deg forecasts...

    total_cloud_cover_025_deg = data.variables['Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average']

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_vals, total_cloud_cover_05_deg, label='0.5 deg')
    ax.plot(time_vals, total_cloud_cover_025_deg, label='0.25 deg')
    ax.legend()
    ax.set_ylabel('Cloud cover (%)')
    ax.set_xlabel('Forecast Time (UTC)')
    ax.set_title('GFS 0.5 and 0.25 deg comparison')

    data.close()

    plt.show()

    # print('Cloud cover' + ' (%s)' % total_cloud_cover.units)

def testNAM(start,end):
    # ## NAM

    # UCAR thredds has 12 km NAM from both the NOAAPORT and CONDUIT services.

    best_nam_nc_sub = NCSS('http://thredds.ucar.edu/thredds/ncss/grib/NCEP/NAM/CONUS_12km/conduit/Best')

    # The NAM has cloud cover and GHI data. I'm guessing that GHI is not very good.

    variables = ['Downward_Short-Wave_Radiation_Flux_surface',
                 'Downward_Short-Wave_Radiation_Flux_surface_Mixed_intervals_Average',
                 'Temperature_surface',
                 'High_cloud_cover_high_cloud',
                 'Low_cloud_cover_low_cloud',
                 'Medium_cloud_cover_middle_cloud',
                 'Total_cloud_cover_entire_atmosphere_single_layer',
                 ]

    query = best_nam_nc_sub.query()
    query.lonlat_point(-110.9, 32.2).time_range(start, end)
    query.variables(*variables)
    #query.vertical_level(100000)
    #query.add_query_parameter(height_above_ground=1)
    query.accept('netcdf')

    data = best_nam_nc_sub.get_data(query)

    time = data.variables['time']
    time_vals = num2date(time[:].squeeze(), time.units)

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


def testRAP(start,end):
    # ## RAP

    # The UCAR THREDDS server does not appear to pull cloud cover or irradiance data from the 13 km NCEP RAP, so we will use the 20 km data.

    best_rr_nc_sub = NCSS('http://thredds.ucar.edu/thredds/ncss/grib/NCEP/RAP/CONUS_20km/Best')

    # There is low, mid, high, and total cloud cover data.

    variables = ['Total_cloud_cover_entire_atmosphere_single_layer',
                 'Low_cloud_cover_low_cloud',
                 'Medium_cloud_cover_middle_cloud',
                 'High_cloud_cover_high_cloud',
                 'Temperature_surface']

    query = best_rr_nc_sub.query()
    query.lonlat_point(-110.9, 32.2).time_range(start, end)
    query.variables(*variables)
    #query.vertical_level(100000)
    #query.add_query_parameter(height_above_ground=1)
    query.accept('netcdf')

    data = best_rr_nc_sub.get_data(query)

    time = data.variables['time']
    time_vals = num2date(time[:].squeeze(), time.units)

    cloud_vars = variables[:-1]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in cloud_vars:
        ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
        
    ax.set_ylabel('Cloud cover (%)')
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend(bbox_to_anchor=(1.4,1.1))

    data.close()

    plt.show()

    

def testNCEP(start,end):
    ### NCEP

    best_hrrr_nc_sub = NCSS('http://thredds-jumbo.unidata.ucar.edu/thredds/ncss/grib/NCEP/HRRR/CONUS_2p5km/Best')

    # Like the RAP, there is low, mid, high, and total cloud cover. NCEP HRRR 2014-2015 does have GHI in the original grib files, but it's not very good (see Holmgren et al AMS 2015).

    variables = ['Total_cloud_cover_entire_atmosphere',
                 'Low_cloud_cover_low_cloud',
                 'Medium_cloud_cover_middle_cloud',
                 'High_cloud_cover_high_cloud',]

    query = best_hrrr_nc_sub.query()
    query.lonlat_point(-110.9, 32.2).time_range(start, end)
    query.variables(*variables)
    #query.vertical_level(100000)
    #query.add_query_parameter(height_above_ground=1)
    query.accept('netcdf')

    data = best_hrrr_nc_sub.get_data(query)

    time = data.variables['time']
    time_vals = num2date(time[:].squeeze(), time.units)

    cloud_vars = variables

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in cloud_vars:
        ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
        
    ax.set_ylabel('Cloud cover (%)')
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend(bbox_to_anchor=(1.4,1.1))

    data.close()

    plt.show()


def testNOAA():

    # ### NOAA
    # 
    # NOAA lists a bunch of different HRRR runs at 
    # 
    # http://rapidrefresh.noaa.gov/HRRR/
    # 
    # The HRRRX run at NOAA has a much improved irradiance forecast, but I'm not sure if that's what the Unidata thredds server pulls in.
    # 
    # Unidata's "Best" link is for an archive instead of the most recent forecasts. So, we'll just look at one forecast.

    best_hrrr_nc_sub = NCSS('http://thredds-jumbo.unidata.ucar.edu/thredds/ncss/grib/HRRR/CONUS_3km/surface/HRRR_CONUS_3km_surface_201510021200.grib2')

    start = datetime(2015,10,2,12) #utc
    end = datetime(2015,10,2,23) #utc

    variables = ['Total_cloud_cover_entire_atmosphere',
                 'Low_cloud_cover_UnknownLevelType-214',
                 'Medium_cloud_cover_UnknownLevelType-224',
                 'High_cloud_cover_UnknownLevelType-234',
                 'Temperature_surface',
                 'Downward_short-wave_radiation_flux_surface']

    query = best_hrrr_nc_sub.query()
    query.lonlat_point(-110.9, 32.2).time_range(start, end)
    query.variables(*variables)
    query.accept('netcdf')

    data = best_hrrr_nc_sub.get_data(query)

    # ESRL HRRR has low, mid, high, and total cloud cover variables. 
    # 
    # It also has a GHI field.

    time = data.variables['time']
    time_vals = num2date(time[:].squeeze(), time.units) # this isn't working right for the hrrr

    ghi = data['Downward_short-wave_radiation_flux_surface']

    cloud_vars = variables[:4]

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    for varname in cloud_vars:
        ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
        
    ax.set_ylabel('Cloud cover (%)')
    ax.set_xlabel('Forecast Time (UTC)')
    ax.legend(bbox_to_anchor=(1.4,1.1))

    fig, ax = plt.subplots(1, 1, figsize=(10, 6))
    ax.plot(time_vals, ghi[:].squeeze(), linewidth=2)
    ax.set_ylabel('GHI' + ' (%s)' % ghi.units)
    ax.set_xlabel('Forecast Time (UTC)')

    data.close()

    plt.show()


def testNDFD(start,end):

    # The times are messed up, but I'm sure they can be fixed with a little effort.

    # ## NDFD

    best_ndfd_nc_sub = NCSS('http://thredds.ucar.edu/thredds/ncss/grib/NCEP/NDFD/NWS/CONUS/CONDUIT/Best')

    ndfd_vars = ['Total_cloud_cover_surface',
                 'Temperature_surface',
                 'Wind_speed_surface']

    query = best_ndfd_nc_sub.query()
    query.lonlat_point(-110.9, 32.2).time_range(start, end)
    query.variables(*ndfd_vars)
    #query.vertical_level(100000)
    #query.add_query_parameter(height_above_ground=1)
    query.accept('netcdf')

    data = best_ndfd_nc_sub.get_data(query)

    time = data.variables['time']
    time_vals = num2date(time[:].squeeze(), time.units)

    total_cloud_cover = data.variables['Total_cloud_cover_surface']
    temp = data.variables['Temperature_surface']
    wind = data.variables['Wind_speed_surface']

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

    

def main():

    # Define start and end of forecast period.

    start = datetime.utcnow()
    end = start + timedelta(days=7)
    
    # testGFS(start,end) # 5 plots

    # testNAM(start,end) # 2 plots
    
    # testRAP(start,end) # 1 plot

    # # ## HRRR
    # # There are two different UCAR HRRR datasets: NOAA/GSD and NCEP.

    # testNCEP(start,end) # 1 plot

    # testNOAA() # 1 plot

    # testNDFD(start,netestd) # 3 plots

if __name__=='__main__':
    main()