# coding: utf-8
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set(rc={"figure.figsize": (12, 6)})
except ImportError:
    print('We suggest you install seaborn using conda or pip and rerun this cell')

# built in python modules
from datetime import datetime, timedelta
import os

# python add-ons
import numpy as np
import pandas as pd
try:
    import netCDF4
    from netCDF4 import num2date
except ImportError:
    print('We suggest you install netCDF4 using conda rerun this cell')

# for accessing UNIDATA THREDD servers
from siphon.catalog import TDSCatalog
from siphon.ncss import NCSS

from pvlib.forecast import *

start = datetime.utcnow() # today's date
end = start + timedelta(days=7) # 7 days from today
timerange = [start, end]

coordinates = [-110.9, 32.2] # Tucson, AZ

fm = GFS()

data = fm.get_query_data(coordinates,timerange)

time_vals = fm.time

var_name = 'temperature_iso'

fig, ax = plt.subplots(1, 1, figsize=(9, 8))
ax.plot(time_vals, data[var_name], 'r', linewidth=2)
ax.set_ylabel(fm.var_stdnames[var_name] + ' (%s)' % fm.var_units[var_name])
ax.set_xlabel('Forecast Time (UTC)')

var_name = 'total_clouds'

total_cloud_cover = data[var_name]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.set_ylabel('Cloud cover' + ' (%s)' % fm.var_units[var_name])
ax.set_xlabel('Forecast Time (UTC)')
ax.set_title('GFS 0.5 deg')
for varname in fm.variables[1:]:
    ax.plot(time_vals, data[varname], linewidth=2, label=varname)
ax.legend(bbox_to_anchor=(1.4,1.1))    

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, total_cloud_cover, 'r', linewidth=2)
ax.set_ylabel('Total cloud cover' + ' (%s)' % fm.var_units[var_name])
ax.set_xlabel('Forecast Time (UTC)')
ax.set_title('GFS 0.5 deg')

fm = NAM()

data = fm.get_query_data(coordinates,timerange)

time_vals = fm.time

cloud_vars = fm.variables[3:]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in cloud_vars:
    ax.plot(time_vals, data[varname], linewidth=2, label=varname)
ax.set_ylabel('Cloud cover (%)')
ax.set_xlabel('Forecast Time (UTC)')
ax.legend(bbox_to_anchor=(1.4,1.1))
ax.set_title('NAM')

ghis = fm.variables[:2]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in ghis:
    ax.plot(time_vals, data[varname], linewidth=2, label=varname)
ax.set_ylabel('GHI W/m**2')
ax.set_xlabel('Forecast Time (UTC)')
ax.legend()

fm = NDFD()

data = fm.get_query_data([-110.9,32.2],[start,end])

time_vals = fm.time

total_cloud_cover = data['total_clouds']
temp = data['temperature']
wind = data['wind_speed']

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, total_cloud_cover, 'r', linewidth=2)
ax.set_ylabel('Cloud cover (%)')
ax.set_xlabel('Forecast Time (UTC)')
plt.ylim(0,100)

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, temp, 'r', linewidth=2)
ax.set_ylabel('temp {}'.format(fm.var_units['temperature']))
ax.set_xlabel('Forecast Time (UTC)')

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, wind, 'r', linewidth=2)
ax.set_ylabel('wind {}'.format(fm.var_units['wind_speed']))
ax.set_xlabel('Forecast Time (UTC)')

fm = RAP()

data = fm.get_query_data([-110.9,32.2],[start,end])

time_vals = fm.time

cloud_vars = ['total_clouds','high_clouds','mid_clouds','low_clouds']

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in cloud_vars:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
ax.set_ylabel('Cloud cover (%)')
ax.set_xlabel('Forecast Time (UTC)')
ax.legend(bbox_to_anchor=(1.4,1.1))

fm = GSD()

data = fm.get_query_data([-110.9,32.2],[start,end])

time_vals = fm.time

cloud_vars = ['total_clouds','high_clouds','mid_clouds','low_clouds']

var_name = 'downward_shortwave_radflux'

ghi = data[var_name]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in cloud_vars:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
ax.set_ylabel('Cloud cover (%)')
ax.set_xlabel('Forecast Time (UTC)')
ax.legend(bbox_to_anchor=(1.4,1.1))

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
ax.plot(time_vals, ghi[:].squeeze(), linewidth=2)
ax.set_ylabel('GHI' + ' (%s)' % fm.var_units[var_name])
ax.set_xlabel('Forecast Time (UTC)')

fm = NCEP()

data = fm.get_query_data([-110.9,32.2],[start,end])

time_vals = fm.time

cloud_vars = fm.variables[3:]

fig, ax = plt.subplots(1, 1, figsize=(10, 6))
for varname in cloud_vars:
    ax.plot(time_vals, data[varname][:].squeeze(), linewidth=2, label=varname)
ax.set_ylabel('Cloud cover (%)')
ax.set_xlabel('Forecast Time (UTC)')
ax.legend(bbox_to_anchor=(1.4,1.1))

