
# coding: utf-8

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

# In[1]:

get_ipython().magic('matplotlib inline')

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
sns.set_color_codes()

import pandas as pd
import numpy as np

import datetime


# Imports for weather data.

# In[2]:

import netCDF4
from netCDF4 import num2date

from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS


# Define start and end of forecast period.

# In[3]:

start = datetime.datetime(2015,7,29,12) #utc
end = start + datetime.timedelta(days=7)


# ## GFS

# To get started, I'll try to follow Siphon's [timeseries example docs](http://siphon.readthedocs.org/en/latest/examples/generated/NCSS_Timeseries_Examples.html) fairly closely.

# In[4]:

best_gfs = TDSCatalog('http://thredds.ucar.edu/thredds/catalog/grib/NCEP/GFS/Global_0p5deg/catalog.xml?dataset=grib/NCEP/GFS/Global_0p5deg/Best')
best_gfs.datasets


# In[5]:

best_ds = list(best_gfs.datasets.values())[0]
best_ds.access_urls


# In[6]:

ncss = NCSS(best_ds.access_urls['NetcdfSubset'])


# In[7]:

ncss.variables


# In[8]:

query = ncss.query()
query.lonlat_point(-110.9, 32.2).vertical_level(100000).time_range(start, end)
query.variables('Temperature_isobaric').accept('netcdf')


# In[9]:

data = ncss.get_data(query)


# In[10]:

data


# In[11]:

temp = data.variables['Temperature_isobaric']
time = data.variables['time']


# In[12]:

time


# In[13]:

time_vals = num2date(time[:].squeeze(), time.units)
time_vals[::5]


# In[14]:

fig, ax = plt.subplots(1, 1, figsize=(9, 8))
ax.plot(time_vals, temp[:].squeeze(), 'r', linewidth=2)
ax.set_ylabel(temp.standard_name + ' (%s)' % temp.units)
ax.set_xlabel('Forecast Time (UTC)')


# In[15]:

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
data


# In[16]:

total_cloud_cover = data.variables['Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average']


# In[17]:

time = data.variables['time']
time_vals = num2date(time[:].squeeze(), time.units)


# In[18]:

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in cloud_vars:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
    
ax.set_ylabel('Cloud cover' + ' (%s)' % total_cloud_cover.units)
ax.set_xlabel('Forecast Time (UTC)')
ax.legend(bbox_to_anchor=(1.4,1.1))
ax.set_title('GFS 0.5 deg')


# In[19]:

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, total_cloud_cover[:].squeeze(), 'r', linewidth=2)
ax.set_ylabel('Total cloud cover' + ' (%s)' % total_cloud_cover.units)
ax.set_xlabel('Forecast Time (UTC)')
ax.set_title('GFS 0.5 deg')


# In[20]:

total_cloud_cover_05_deg = total_cloud_cover


# ## GFS 0.25 deg

# Now with the new, higher resolution GFS...

# In[21]:

best_gfs_nc_sub = NCSS('http://thredds-jumbo.unidata.ucar.edu/thredds/ncss/grib/NCEP/GFS/Global_0p25deg/Best')


# In[22]:

best_gfs_nc_sub.variables


# In[23]:

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
data


# In[24]:

time = data.variables['time']
time_vals = num2date(time[:].squeeze(), time.units)


# In[25]:

cloud_vars = variables[2:]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in cloud_vars:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
    
ax.set_ylabel('Cloud cover' + ' (%s)' % total_cloud_cover.units)
ax.set_xlabel('Forecast Time (UTC)')
ax.legend(bbox_to_anchor=(1.4,1.1))
ax.set_title('GFS 0.25 deg')


# In[26]:

ghis = variables[:1]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in ghis:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
ax.set_ylabel('GHI W/m**2')
ax.set_xlabel('Forecast Time (UTC)')
ax.legend()


# I don't know anything about the GHI forecasts from the GFS, but I don't think they'll be very good for solar power forecasting.

# Compare the 0.5 and 0.25 deg forecasts...

# In[27]:

total_cloud_cover_025_deg = data.variables['Total_cloud_cover_entire_atmosphere_Mixed_intervals_Average']


# In[28]:

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, total_cloud_cover_05_deg, label='0.5 deg')
ax.plot(time_vals, total_cloud_cover_025_deg, label='0.25 deg')
ax.legend()
ax.set_ylabel('Cloud cover (%)')
ax.set_xlabel('Forecast Time (UTC)')
ax.set_title('GFS 0.5 and 0.25 deg comparison')


# ## NAM

# UCAR thredds has 12 km NAM from both the NOAAPORT and CONDUIT services.

# In[29]:

best_nam_nc_sub = NCSS('http://thredds.ucar.edu/thredds/ncss/grib/NCEP/NAM/CONUS_12km/conduit/Best')


# In[30]:

best_nam_nc_sub.variables


# The NAM has cloud cover and GHI data. I'm guessing that GHI is not very good.

# In[31]:

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
data


# In[32]:

time = data.variables['time']
time_vals = num2date(time[:].squeeze(), time.units)


# In[33]:

cloud_vars = variables[3:]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in cloud_vars:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
    
ax.set_ylabel('Cloud cover' + ' (%s)' % total_cloud_cover.units)
ax.set_xlabel('Forecast Time (UTC)')
ax.legend(bbox_to_anchor=(1.4,1.1))
ax.set_title('NAM')


# In[34]:

ghis = variables[:2]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in ghis:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
ax.set_ylabel('GHI W/m**2')
ax.set_xlabel('Forecast Time (UTC)')
ax.legend()


# ## RAP

# The UCAR THREDDS server does not appear to pull cloud cover or irradiance data from the 13 km NCEP RAP, so we will use the 20 km data.

# In[35]:

best_rr_nc_sub = NCSS('http://thredds.ucar.edu/thredds/ncss/grib/NCEP/RAP/CONUS_20km/Best')


# In[36]:

best_rr_nc_sub.variables


# There is low, mid, high, and total cloud cover data.

# In[37]:

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
data


# In[38]:

time = data.variables['time']
time_vals = num2date(time[:].squeeze(), time.units)


# In[39]:

cloud_vars = variables[:-1]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in cloud_vars:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
    
ax.set_ylabel('Cloud cover' + ' (%s)' % total_cloud_cover.units)
ax.set_xlabel('Forecast Time (UTC)')
ax.legend(bbox_to_anchor=(1.4,1.1))


# ## HRRR

# There are two different UCAR HRRR datasets: NOAA/GSD and NCEP.

# ### NCEP

# In[40]:

best_hrrr_nc_sub = NCSS('http://thredds-jumbo.unidata.ucar.edu/thredds/ncss/grib/NCEP/HRRR/CONUS_2p5km/Best')


# In[41]:

best_hrrr_nc_sub.variables


# Like the RAP, there is low, mid, high, and total cloud cover. NCEP HRRR 2014-2015 does have GHI in the original grib files, but it's not very good (see Holmgren et al AMS 2015).

# In[42]:

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
data


# In[43]:

time = data.variables['time']
time_vals = num2date(time[:].squeeze(), time.units)


# In[44]:

cloud_vars = variables

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in cloud_vars:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
    
ax.set_ylabel('Cloud cover' + ' (%s)' % total_cloud_cover.units)
ax.set_xlabel('Forecast Time (UTC)')
ax.legend(bbox_to_anchor=(1.4,1.1))


# ### NOAA
# 
# NOAA lists a bunch of different HRRR runs at 
# 
# http://rapidrefresh.noaa.gov/HRRR/
# 
# The HRRRX run at NOAA has a much improved irradiance forecast, but I'm not sure if that's what the Unidata thredds server pulls in.
# 
# Unidata's "Best" link is for an archive instead of the most recent forecasts. So, we'll just look at one forecast.

# In[45]:

best_hrrr_nc_sub = NCSS('http://thredds-jumbo.unidata.ucar.edu/thredds/ncss/grib/HRRR/CONUS_3km/surface/HRRR_CONUS_3km_surface_201507291400.grib2')


# In[46]:

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
data


# ESRL HRRR has low, mid, high, and total cloud cover variables. 
# 
# It also has a GHI field.

# In[47]:

time = data.variables['time']
time_vals = num2date(time[:].squeeze(), time.units) # this isn't working right for the hrrr


# In[48]:

ghi = data['Downward_short-wave_radiation_flux_surface']


# In[49]:

cloud_vars = variables[:4]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in cloud_vars:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
    
ax.set_ylabel('Cloud cover' + ' (%s)' % total_cloud_cover.units)
ax.set_xlabel('Forecast Time (UTC)')
ax.legend(bbox_to_anchor=(1.4,1.1))


# In[50]:

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, ghi[:].squeeze(), linewidth=2)
ax.set_ylabel('GHI' + ' (%s)' % ghi.units)
ax.set_xlabel('Forecast Time (UTC)')


# The times are messed up, but I'm sure they can be fixed with a little effort.

# ## NDFD

# In[51]:

best_ndfd_nc_sub = NCSS('http://thredds.ucar.edu/thredds/ncss/grib/NCEP/NDFD/NWS/CONUS/CONDUIT/Best')


# In[52]:

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
data


# In[53]:

time = data.variables['time']
time_vals = num2date(time[:].squeeze(), time.units)


# In[54]:

total_cloud_cover = data.variables['Total_cloud_cover_surface']
temp = data.variables['Temperature_surface']
wind = data.variables['Wind_speed_surface']


# In[55]:

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, total_cloud_cover[:].squeeze(), 'r', linewidth=2)
ax.set_ylabel('Total cloud cover' + ' (%s)' % total_cloud_cover.units)
ax.set_xlabel('Forecast Time (UTC)')
plt.ylim(0,100)


# In[56]:

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, temp[:].squeeze(), 'r', linewidth=2)
ax.set_ylabel('temp {}'.format(temp.units))
ax.set_xlabel('Forecast Time (UTC)')


# In[57]:

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, wind[:].squeeze(), 'r', linewidth=2)
ax.set_ylabel('wind {}'.format(wind.units))
ax.set_xlabel('Forecast Time (UTC)')


# In[ ]:



