"""
Comparing PV Module Temperature Models
======================================

Examples of calculating PV cell temperature using different temperature models.

Temperature models estimate PV module operating temperature based on
environmental conditions such as irradiance, air temperature, and wind speed.

This example compares the SAPM, Faiman, and PVsyst temperature models
using identical environmental inputs. Differences between models illustrate
how temperature model selection can influence predicted module temperature.
"""

# %%
# Overview
# --------
#
# Temperature models estimate PV cell temperature using environmental inputs.
# Different models use different empirical relationships and assumptions.
#
# This example compares the SAPM, Faiman, and PVsyst temperature models
# using identical weather conditions to illustrate differences in predicted
# cell temperature.

# %%
# Import required libraries
# -------------------------

import pvlib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# %%
# Generate example environmental data
# -----------------------------------
#
# We simulate one week of hourly irradiance and temperature conditions.
# Clear-sky irradiance is generated using pvlib's Location object.

times = pd.date_range(
    "2019-06-01 00:00",
    "2019-06-07 23:00",
    freq="1h",
    tz="Etc/GMT+7"
)

location = pvlib.location.Location(32.2, -110.9)

clearsky = location.get_clearsky(times)

poa_global = clearsky["ghi"]

# Create air temperature profile with daily variation
temp_air = 20 + 10 * np.sin(
    2 * np.pi * (times.hour - 6) / 24
)

# Constant wind speed
wind_speed = np.full(len(times), 1.0)

weather = pd.DataFrame({
    "poa_global": poa_global,
    "temp_air": temp_air,
    "wind_speed": wind_speed,
}, index=times)

# %%
# Calculate cell temperature using SAPM model
# --------------------------------------------
#
# The SAPM model uses empirical coefficients based on mounting configuration.

sapm_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS[
    "sapm"
]["open_rack_glass_glass"]

sapm_temp = pvlib.temperature.sapm_cell(
    poa_global=weather["poa_global"],
    temp_air=weather["temp_air"],
    wind_speed=weather["wind_speed"],
    a=sapm_parameters["a"],
    b=sapm_parameters["b"],
    deltaT=sapm_parameters["deltaT"],
)

# %%
# Calculate cell temperature using Faiman model
# ----------------------------------------------
#
# The Faiman model uses heat transfer coefficients to estimate module
# temperature.

faiman_temp = pvlib.temperature.faiman(
    poa_global=weather["poa_global"],
    temp_air=weather["temp_air"],
    wind_speed=weather["wind_speed"],
    u0=25,
    u1=6.84,
)

# %%
# Calculate cell temperature using PVsyst model
# ----------------------------------------------
#
# The PVsyst model uses empirical heat loss coefficients.

pvsyst_temp = pvlib.temperature.pvsyst_cell(
    poa_global=weather["poa_global"],
    temp_air=weather["temp_air"],
    wind_speed=weather["wind_speed"],
    u_c=29,
    u_v=0,
)

# %%
# Combine results into a DataFrame
# --------------------------------

temperature_models = pd.DataFrame({
    "SAPM": sapm_temp,
    "Faiman": faiman_temp,
    "PVsyst": pvsyst_temp,
})

# %%
# Plot temperature model comparison
# ---------------------------------
#
# This plot shows how predicted cell temperature varies across models.

plt.figure(figsize=(10, 5))

temperature_models.plot(ax=plt.gca())

plt.xlabel("Time")
plt.ylabel("Cell Temperature (°C)")
plt.title("Comparison of PV Temperature Models")

plt.legend()
plt.tight_layout()
plt.show()

# %%
# Plot temperature differences relative to SAPM
# ---------------------------------------------
#
# This highlights differences between models more clearly.

temperature_difference = temperature_models.subtract(
    temperature_models["SAPM"], axis=0
)

plt.figure(figsize=(10, 5))

temperature_difference.plot(ax=plt.gca())

plt.xlabel("Time")
plt.ylabel("Temperature Difference (°C)")
plt.title("Temperature Difference Relative to SAPM Model")

plt.legend()
plt.tight_layout()
plt.show()

# %%
# Plot daily average temperature
# ------------------------------
#
# This shows daily trends in temperature prediction.

daily_average = temperature_models.resample("D").mean()

fig, ax = plt.subplots(figsize=(8, 5))

daily_average.plot(kind="bar", ax=ax)

ax.set_ylabel("Average Cell Temperature (°C)")
ax.set_title("Daily Average Temperature by Model")

plt.tight_layout()
plt.show()
