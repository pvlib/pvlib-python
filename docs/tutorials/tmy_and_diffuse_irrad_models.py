"""
TMY data and diffuse irradiance models
======================================
This tutorial explores using TMY data as inputs to different plane of array
diffuse irradiance models.
"""

# %%
# This tutorial requires pvlib > 0.6.0.
#
# Authors:
#
# - Rob Andrews (@Calama-Consulting), Heliolytics, June 2014
# - Will Holmgren (@wholmgren), University of Arizona, July 2015, March 2016,
#   August 2018
#
# Setup
# -----
# See the :ref:`sphx_glr_gallery_tutorials_tmy_to_power.py` tutorial for more
# detailed explanations for the initial setup

import os
import inspect

import pandas as pd
import matplotlib.pyplot as plt

import pvlib


# %%
# TODO: :pull:`1763`

# Find the absolute file path to your pvlib installation
pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))

# absolute path to a data file
datapath = os.path.join(pvlib_abspath, "data", "703165TY.csv")

# read tmy data with year values coerced to a single year
tmy_data, meta = pvlib.iotools.read_tmy3(datapath, coerce_year=2015)
tmy_data.index.name = "Time"

# TMY data seems to be given as hourly data with time stamp at the end
# shift the index 30 Minutes back for calculation of sun positions
tmy_data = tmy_data.shift(freq="-30Min")["2015"]


# %%
tmy_data.GHI.plot()
plt.ylabel("Irradiance (W/m**2)")


# %%
tmy_data.DHI.plot()
plt.ylabel("Irradiance (W/m**2)")


# %%
surface_tilt = 30
surface_azimuth = (
    180  # pvlib uses 0=North, 90=East, 180=South, 270=West convention
)
albedo = 0.2

# create pvlib Location object based on meta data
sand_point = pvlib.location.Location(
    meta["latitude"],
    meta["longitude"],
    tz="US/Alaska",
    altitude=meta["altitude"],
    name=meta["Name"].replace('"', ""),
)
print(sand_point)


# %%
solpos = pvlib.solarposition.get_solarposition(
    tmy_data.index, sand_point.latitude, sand_point.longitude
)

solpos.plot()


# %%
# the extraradiation function returns a simple numpy array
# instead of a nice pandas series. We will change this
# in a future version
dni_extra = pvlib.irradiance.get_extra_radiation(tmy_data.index)
dni_extra = pd.Series(dni_extra, index=tmy_data.index)

dni_extra.plot()
plt.ylabel("Extra terrestrial radiation (W/m**2)")


# %%
airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])

airmass.plot()
plt.ylabel("Airmass")

# %%
# Diffuse irradiance models
# -------------------------

# Make an empty pandas DataFrame for the results.
diffuse_irrad = pd.DataFrame(index=tmy_data.index)


# %%
models = ["Perez", "Hay-Davies", "Isotropic", "King", "Klucher", "Reindl"]


# %%
# Perez
# ^^^^^

diffuse_irrad["Perez"] = pvlib.irradiance.perez(
    surface_tilt,
    surface_azimuth,
    dhi=tmy_data.DHI,
    dni=tmy_data.DNI,
    dni_extra=dni_extra,
    solar_zenith=solpos.apparent_zenith,
    solar_azimuth=solpos.azimuth,
    airmass=airmass,
)


# %%
# HayDavies
# ^^^^^^^^^

diffuse_irrad["Hay-Davies"] = pvlib.irradiance.haydavies(
    surface_tilt,
    surface_azimuth,
    dhi=tmy_data.DHI,
    dni=tmy_data.DNI,
    dni_extra=dni_extra,
    solar_zenith=solpos.apparent_zenith,
    solar_azimuth=solpos.azimuth,
)


# %%
# Isotropic
# ^^^^^^^^^

diffuse_irrad["Isotropic"] = pvlib.irradiance.isotropic(
    surface_tilt, dhi=tmy_data.DHI
)


# %%
# King Diffuse model
# ^^^^^^^^^^^^^^^^^^

diffuse_irrad["King"] = pvlib.irradiance.king(
    surface_tilt,
    dhi=tmy_data.DHI,
    ghi=tmy_data.GHI,
    solar_zenith=solpos.apparent_zenith,
)


# %%
# Klucher Model
# ^^^^^^^^^^^^^

diffuse_irrad["Klucher"] = pvlib.irradiance.klucher(
    surface_tilt,
    surface_azimuth,
    dhi=tmy_data.DHI,
    ghi=tmy_data.GHI,
    solar_zenith=solpos.apparent_zenith,
    solar_azimuth=solpos.azimuth,
)


# %%
# Reindl
# ^^^^^^

diffuse_irrad["Reindl"] = pvlib.irradiance.reindl(
    surface_tilt,
    surface_azimuth,
    dhi=tmy_data.DHI,
    dni=tmy_data.DNI,
    ghi=tmy_data.GHI,
    dni_extra=dni_extra,
    solar_zenith=solpos.apparent_zenith,
    solar_azimuth=solpos.azimuth,
)

# %%
# Calculate yearly, monthly, daily sums.

yearly = diffuse_irrad.resample("A").sum().dropna().squeeze() / 1000.0  # kWh
monthly = diffuse_irrad.resample("M", kind="period").sum() / 1000.0
daily = diffuse_irrad.resample("D").sum() / 1000.0


# %%
# Plot Results
# ------------

ax = diffuse_irrad.plot(title="In-plane diffuse irradiance", alpha=0.75, lw=1)
ax.set_ylim(0, 800)
ylabel = ax.set_ylabel("Diffuse Irradiance [W]")
plt.legend()


# %%
diffuse_irrad.describe()


# %%
diffuse_irrad.dropna().plot(kind="density")

# %%
# Daily

ax_daily = daily.tz_convert("UTC").plot(title="Daily diffuse irradiation")
ylabel = ax_daily.set_ylabel("Irradiation [kWh]")


# %%
# Monthly

ax_monthly = monthly.plot(
    title="Monthly average diffuse irradiation", kind="bar"
)
ylabel = ax_monthly.set_ylabel("Irradiation [kWh]")


# %%
# Yearly

yearly.plot(kind="barh")


# %%
# Compute the mean deviation from measured for each model and display as a
# function of the model

mean_yearly = yearly.mean()
yearly_mean_deviation = (yearly - mean_yearly) / yearly * 100.0
yearly_mean_deviation.plot(kind="bar")
