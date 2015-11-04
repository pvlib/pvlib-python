# coding: utf-8

# # TMY data and diffuse irradiance models
# 
# This tutorial explores using TMY data as inputs to different plane of array diffuse irradiance models.
# 
# This tutorial has been tested against the following package versions:
# * pvlib 0.2.0
# * Python 2.7.10
# * IPython 3.2
# * pandas 0.16.2
# 
# It should work with other Python and Pandas versions. It requires pvlib > 0.2.0 and IPython > 3.0.
# 
# Authors:
# * Rob Andrews (@Calama-Consulting), Heliolytics, June 2014
# * Will Holmgren (@wholmgren), University of Arizona, July 2015

# ## Setup

# See the ``tmy_to_power`` tutorial for more detailed explanations for the initial setup
# built-in python modules
import os
import inspect

# scientific python add-ons
import numpy as np
import pandas as pd

# plotting stuff
# first line makes the plots appear in the notebook
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
# seaborn makes your plots look better
try:
    import seaborn as sns
    sns.set(rc={"figure.figsize": (12, 6)})
except ImportError:
    print('We suggest you install seaborn using conda or pip and rerun this cell')

# finally, we import the pvlib library
import pvlib

# Find the absolute file path to your pvlib installation
pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))

# absolute path to a data file
datapath = os.path.join(pvlib_abspath, 'data', '703165TY.csv')

# read tmy data with year values coerced to a single year
tmy_data, meta = pvlib.tmy.readtmy3(datapath, coerce_year=2015)
tmy_data.index.name = 'Time'

# TMY data seems to be given as hourly data with time stamp at the end
# shift the index 30 Minutes back for calculation of sun positions
tmy_data = tmy_data.shift(freq='-30Min')

tmy_data.GHI.plot()
plt.ylabel('Irradiance (W/m**2)')

tmy_data.DHI.plot()
plt.ylabel('Irradiance (W/m**2)')

surface_tilt = 30
surface_azimuth = 180 # pvlib uses 0=North, 90=East, 180=South, 270=West convention
albedo = 0.2

# create pvlib Location object based on meta data
sand_point = pvlib.location.Location(meta['latitude'], meta['longitude'], tz='US/Alaska', 
                                     altitude=meta['altitude'], name=meta['Name'].replace('"',''))
print(sand_point)

solpos = pvlib.solarposition.get_solarposition(tmy_data.index, sand_point)

solpos.plot()

# the extraradiation function returns a simple numpy array
# instead of a nice pandas series. We will change this
# in a future version
dni_extra = pvlib.irradiance.extraradiation(tmy_data.index)
dni_extra = pd.Series(dni_extra, index=tmy_data.index)

dni_extra.plot()
plt.ylabel('Extra terrestrial radiation (W/m**2)')

airmass = pvlib.atmosphere.relativeairmass(solpos['apparent_zenith'])

airmass.plot()
plt.ylabel('Airmass')


# ## Diffuse irradiance models

# Make an empty pandas DataFrame for the results.
diffuse_irrad = pd.DataFrame(index=tmy_data.index)

models = ['Perez', 'Hay-Davies', 'Isotropic', 'King', 'Klucher', 'Reindl']


# ### Perez
diffuse_irrad['Perez'] = pvlib.irradiance.perez(surface_tilt,
                               surface_azimuth,
                               dhi=tmy_data.DHI,
                               dni=tmy_data.DNI,
                               dni_extra=dni_extra,
                               solar_zenith=solpos.apparent_zenith,
                               solar_azimuth=solpos.azimuth,
                               airmass=airmass)


# ### HayDavies
diffuse_irrad['Hay-Davies'] = pvlib.irradiance.haydavies(surface_tilt,
                               surface_azimuth,
                               dhi=tmy_data.DHI,
                               dni=tmy_data.DNI,
                               dni_extra=dni_extra,
                               solar_zenith=solpos.apparent_zenith,
                               solar_azimuth=solpos.azimuth)


# ### Isotropic
diffuse_irrad['Isotropic'] = pvlib.irradiance.isotropic(surface_tilt,
                               dhi=tmy_data.DHI)


# ### King Diffuse model
diffuse_irrad['King'] = pvlib.irradiance.king(surface_tilt,
                               dhi=tmy_data.DHI,
                               ghi=tmy_data.GHI,
                               solar_zenith=solpos.apparent_zenith)


# ### Klucher Model
diffuse_irrad['Klucher'] = pvlib.irradiance.klucher(surface_tilt, surface_azimuth,
                                                    dhi=tmy_data.DHI,
                                                    ghi=tmy_data.GHI,
                                                    solar_zenith=solpos.apparent_zenith,
                                                    solar_azimuth=solpos.azimuth)


# ### Reindl
diffuse_irrad['Reindl'] = pvlib.irradiance.reindl(surface_tilt,
                               surface_azimuth,
                               dhi=tmy_data.DHI,
                               dni=tmy_data.DNI,
                               ghi=tmy_data.GHI,
                               dni_extra=dni_extra,
                               solar_zenith=solpos.apparent_zenith,
                               solar_azimuth=solpos.azimuth)


# Calculate yearly, monthly, daily sums.
yearly = diffuse_irrad.resample('A', how='sum').dropna().squeeze() / 1000.0  # kWh
monthly = diffuse_irrad.resample('M', how='sum', kind='period') / 1000.0
daily = diffuse_irrad.resample('D', how='sum') / 1000.0


# ## Plot Results
ax = diffuse_irrad.plot(title='In-plane diffuse irradiance', alpha=.75, lw=1)
ax.set_ylim(0, 800)
ylabel = ax.set_ylabel('Diffuse Irradiance [W]')
plt.legend()

diffuse_irrad.describe()

diffuse_irrad.dropna().plot(kind='density')


# Daily
ax_daily = daily.tz_convert('UTC').plot(title='Daily diffuse irradiation')
ylabel = ax_daily.set_ylabel('Irradiation [kWh]')


# Monthly
ax_monthly = monthly.plot(title='Monthly average diffuse irradiation', kind='bar')
ylabel = ax_monthly.set_ylabel('Irradiation [kWh]')


# Yearly
yearly.plot(kind='barh')


# Compute the mean deviation from measured for each model and display as a function of the model
mean_yearly = yearly.mean()
yearly_mean_deviation = (yearly - mean_yearly) / yearly * 100.0
yearly_mean_deviation.plot(kind='bar')



