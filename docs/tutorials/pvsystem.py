"""
``pvsystem.py`` tutorial
========================
This tutorial explores the :py:mod:`pvlib.pvsystem` module.
The module has functions for importing PV module and inverter data and
functions for modeling module and inverter performance.
"""

# %%
# 1. `Angle of Incidence Modifiers`_
# 2. `Sandia Cell Temp correction`_
# 3. `Sandia Inverter Model`_
# 4. `Sandia Array Performance Model`_
#     1. `SAPM IV curves`_
# 5. `DeSoto Model`_
# 6. `Single Diode Model`_
#
# This notebook requires pvlib >= 0.8.
# See output of ``pd.show_versions()`` to see the other packages that this
# notebook was last used with. It should work with other Python and Pandas
# versions.
#
# Authors:
#
# - Will Holmgren (@wholmgren), University of Arizona. 2015, March 2016,
#   November 2016, May 2017.

import datetime
import warnings

# scientific python add-ons
import numpy as np
import pandas as pd

# plotting stuff
# first line makes the plots appear in the notebook
import matplotlib.pyplot as plt

# seaborn makes your plots look better
try:
    import seaborn as sns

    sns.set(rc={"figure.figsize": (12, 6)})
except ImportError:
    print(
        "We suggest you install seaborn using conda or pip and rerun this cell"
    )

# finally, we import the pvlib library
import pvlib
from pvlib import pvsystem, inverter
from pvlib.location import Location

# %%
# Angle of Incidence Modifiers
# ----------------------------

angles = np.linspace(-180, 180, 3601)
ashraeiam = pd.Series(pvsystem.iam.ashrae(angles, 0.05), index=angles)

ashraeiam.plot()
plt.ylabel("ASHRAE modifier")
plt.xlabel("input angle (deg)")

# %%
angles = np.linspace(-180, 180, 3601)
physicaliam = pd.Series(pvsystem.iam.ashrae(angles), index=angles)

physicaliam.plot()
plt.ylabel("physical modifier")
plt.xlabel("input index")

# %%
plt.figure()
ashraeiam.plot(label="ASHRAE")
physicaliam.plot(label="physical")
plt.ylabel("modifier")
plt.xlabel("input angle (deg)")
plt.legend()

# %%
# Sandia Cell Temp correction
# ---------------------------
# PV system efficiency can vary by up to 0.5% per degree C, so it's important
# to accurately model cell temperature.
# The :py:func:`pvlib.temperature.sapm_cell` function uses plane of array
# irradiance, ambient temperature, wind speed, and module and racking type to
# calculate cell temperature. From King et. al. (2004):
#
# . math::
#   T_m = E e^{a+b*WS} + T_a
#   T_c = T_m + \frac{E}{E_0} \Delta T
#
# The :math:`a`, :math:`b`, and :math:`\Delta T` parameters depend on the
# module and racking type.
# Here we use the ``open_rack_glass_glass`` parameters.
#
# :py:func:`~pvlib.temperature.sapm_cell` works with either scalar or vector
# inputs.

# scalar inputs
thermal_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
    "open_rack_glass_glass"
]
pvlib.temperature.sapm_cell(900, 20, 5, **thermal_params)  # irrad, temp, wind

# %%

# vector inputs
times = pd.date_range(start="2015-01-01", end="2015-01-02", freq="12H")
temps = pd.Series([0, 10, 5], index=times)
irrads = pd.Series([0, 500, 0], index=times)
winds = pd.Series([10, 5, 0], index=times)

pvtemps = pvlib.temperature.sapm_cell(irrads, temps, winds, **thermal_params)
pvtemps.plot()

# %%
# Cell temperature as a function of wind speed.

wind = np.linspace(0, 20, 21)
temps = pd.Series(
    pvlib.temperature.sapm_cell(900, 20, wind, **thermal_params), index=wind
)

temps.plot()
plt.xlabel("wind speed (m/s)")
plt.ylabel("temperature (deg C)")

# %%
# Cell temperature as a function of ambient temperature.

atemp = np.linspace(-20, 50, 71)
temps = pd.Series(
    pvlib.temperature.sapm_cell(900, atemp, 2, **thermal_params), index=atemp
)

temps.plot()
plt.xlabel("ambient temperature (deg C)")
plt.ylabel("temperature (deg C)")

# %%
# Cell temperature as a function of incident irradiance.

irrad = np.linspace(0, 1000, 101)
temps = pd.Series(
    pvlib.temperature.sapm_cell(irrad, 20, 2, **thermal_params), index=irrad
)

temps.plot()
plt.xlabel("incident irradiance (W/m**2)")
plt.ylabel("temperature (deg C)")

# %%
# Cell temperature for different module and racking types.

models = [
    "open_rack_glass_glass",
    "close_mount_glass_glass",
    "open_rack_glass_polymer",
    "insulated_back_glass_polymer",
]

temps = pd.Series(dtype=float)

for model in models:
    params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][model]
    temps[model] = pvlib.temperature.sapm_cell(1000, 20, 5, **params)

temps.plot(kind="bar")
plt.ylabel("temperature (deg C)")

# %%
# Sandia Inverter Model
# ---------------------

inverters = pvsystem.retrieve_sam("sandiainverter")
inverters

# %%
vdcs = pd.Series(np.linspace(0, 50, 51))
idcs = pd.Series(np.linspace(0, 11, 110))
pdcs = idcs * vdcs

pacs = inverter.sandia(
    vdcs, pdcs, inverters["ABB__MICRO_0_25_I_OUTD_US_208__208V_"]
)
# pacs.plot()
plt.plot(pdcs, pacs)
plt.ylabel("ac power")
plt.xlabel("dc power")

# %%
# Need to put more effort into describing this function.

# %%
# Sandia Array Performance Model
# ------------------------------
# This example shows use of the Desoto module performance model and the Sandia
# Array Performance Model (SAPM). Both models reuire a set of parameter values
# which can be read from SAM databases for modules.
#
# For the Desoto model, the database content is returned by supplying the
# keyword `cecmod` to `pvsystem.retrievesam`.

cec_modules = pvsystem.retrieve_sam("cecmod")
cec_modules

cecmodule = cec_modules.Canadian_Solar_Inc__CS5P_220M
cecmodule

# %%
# The Sandia module database is read by the same function with the keyword
# `SandiaMod`.

sandia_modules = pvsystem.retrieve_sam(name="SandiaMod")
sandia_modules

sandia_module = sandia_modules.Canadian_Solar_CS5P_220M___2009_
sandia_module

# %%
# Generate some irradiance data for modeling.

tus = Location(32.2, -111, "US/Arizona", 700, "Tucson")

times_loc = pd.date_range(
    start=datetime.datetime(2014, 4, 1),
    end=datetime.datetime(2014, 4, 2),
    freq="30s",
    tz=tus.tz,
)
solpos = pvlib.solarposition.get_solarposition(
    times_loc, tus.latitude, tus.longitude
)
dni_extra = pvlib.irradiance.get_extra_radiation(times_loc)
airmass = pvlib.atmosphere.get_relative_airmass(solpos["apparent_zenith"])
pressure = pvlib.atmosphere.alt2pres(tus.altitude)
am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
cs = tus.get_clearsky(times_loc)

surface_tilt = tus.latitude
surface_azimuth = 180  # pointing south

aoi = pvlib.irradiance.aoi(
    surface_tilt, surface_azimuth, solpos["apparent_zenith"], solpos["azimuth"]
)
total_irrad = pvlib.irradiance.get_total_irradiance(
    surface_tilt,
    surface_azimuth,
    solpos["apparent_zenith"],
    solpos["azimuth"],
    cs["dni"],
    cs["ghi"],
    cs["dhi"],
    dni_extra=dni_extra,
    model="haydavies",
)

# %%
# Now we can run the module parameters and the irradiance data through the SAPM
# functions.

module = sandia_module

# a sunny, calm, and hot day in the desert
thermal_params = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS["sapm"][
    "open_rack_glass_glass"
]
temps = pvlib.temperature.sapm_cell(
    total_irrad["poa_global"], 30, 0, **thermal_params
)

effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
    total_irrad["poa_direct"], total_irrad["poa_diffuse"], am_abs, aoi, module
)

sapm_1 = pvlib.pvsystem.sapm(effective_irradiance, temps, module)

sapm_1.plot()


# %%
def plot_sapm(sapm_data, effective_irradiance):
    """
    Makes a nice figure with the SAPM data.

    Parameters
    ----------
    sapm_data : DataFrame
        The output of ``pvsystem.sapm``
    """
    fig, axes = plt.subplots(
        2, 3, figsize=(16, 10), sharex=False, sharey=False, squeeze=False
    )
    plt.subplots_adjust(wspace=0.2, hspace=0.3)

    ax = axes[0, 0]
    sapm_data.filter(like="i_").plot(ax=ax)
    ax.set_ylabel("Current (A)")

    ax = axes[0, 1]
    sapm_data.filter(like="v_").plot(ax=ax)
    ax.set_ylabel("Voltage (V)")

    ax = axes[0, 2]
    sapm_data.filter(like="p_").plot(ax=ax)
    ax.set_ylabel("Power (W)")

    ax = axes[1, 0]
    [
        ax.plot(effective_irradiance, current, label=name)
        for name, current in sapm_data.filter(like="i_").iteritems()
    ]
    ax.set_ylabel("Current (A)")
    ax.set_xlabel("Effective Irradiance")
    ax.legend(loc=2)

    ax = axes[1, 1]
    [
        ax.plot(effective_irradiance, voltage, label=name)
        for name, voltage in sapm_data.filter(like="v_").iteritems()
    ]
    ax.set_ylabel("Voltage (V)")
    ax.set_xlabel("Effective Irradiance")
    ax.legend(loc=4)

    ax = axes[1, 2]
    ax.plot(effective_irradiance, sapm_data["p_mp"], label="p_mp")
    ax.set_ylabel("Power (W)")
    ax.set_xlabel("Effective Irradiance")
    ax.legend(loc=2)

    # needed to show the time ticks
    for ax in axes.flatten():
        for tk in ax.get_xticklabels():
            tk.set_visible(True)


# %%
plot_sapm(sapm_1, effective_irradiance)

# %%
# For comparison, here's the SAPM for a sunny, windy, cold version of
# the same day.

temps = pvlib.temperature.sapm_cell(
    total_irrad["poa_global"], 5, 10, **thermal_params
)

sapm_2 = pvlib.pvsystem.sapm(effective_irradiance, temps, module)

plot_sapm(sapm_2, effective_irradiance)

# %%
sapm_1["p_mp"].plot(label="30 C,  0 m/s")
sapm_2["p_mp"].plot(label=" 5 C, 10 m/s")
plt.legend()
plt.ylabel("Pmp")
plt.title("Comparison of a hot, calm day and a cold, windy day")

# %%
# SAPM IV curves
# ^^^^^^^^^^^^^^
# The IV curve function only calculates the 5 points of the SAPM.
# We will add arbitrary points in a future release, but for now we just
# interpolate between the 5 SAPM points.

warnings.simplefilter("ignore", np.RankWarning)


# %%
def sapm_to_ivframe(sapm_row):
    pnt = sapm_row

    ivframe = {
        "Isc": (pnt["i_sc"], 0),
        "Pmp": (pnt["i_mp"], pnt["v_mp"]),
        "Ix": (pnt["i_x"], 0.5 * pnt["v_oc"]),
        "Ixx": (pnt["i_xx"], 0.5 * (pnt["v_oc"] + pnt["v_mp"])),
        "Voc": (0, pnt["v_oc"]),
    }
    ivframe = pd.DataFrame(ivframe, index=["current", "voltage"]).T
    ivframe = ivframe.sort_values(by="voltage")

    return ivframe


def ivframe_to_ivcurve(ivframe, points=100):
    ivfit_coefs = np.polyfit(ivframe["voltage"], ivframe["current"], 30)
    fit_voltages = np.linspace(0, ivframe.loc["Voc", "voltage"], points)
    fit_currents = np.polyval(ivfit_coefs, fit_voltages)

    return fit_voltages, fit_currents


# %%
times = [
    "2014-04-01 07:00:00",
    "2014-04-01 08:00:00",
    "2014-04-01 09:00:00",
    "2014-04-01 10:00:00",
    "2014-04-01 11:00:00",
    "2014-04-01 12:00:00",
]
times.reverse()

fig, ax = plt.subplots(1, 1, figsize=(12, 8))

for time in times:
    ivframe = sapm_to_ivframe(sapm_1.loc[time])

    fit_voltages, fit_currents = ivframe_to_ivcurve(ivframe)

    ax.plot(fit_voltages, fit_currents, label=time)
    ax.plot(ivframe["voltage"], ivframe["current"], "ko")

ax.set_xlabel("Voltage (V)")
ax.set_ylabel("Current (A)")
ax.set_ylim(0, None)
ax.set_title("IV curves at multiple times")
ax.legend()

# %%
# DeSoto Model
# ------------
# The same weather data run through the Desoto model.

(
    photocurrent,
    saturation_current,
    resistance_series,
    resistance_shunt,
    nNsVth,
) = pvsystem.calcparams_desoto(
    total_irrad["poa_global"],
    temp_cell=temps,
    alpha_sc=cecmodule["alpha_sc"],
    a_ref=cecmodule["a_ref"],
    I_L_ref=cecmodule["I_L_ref"],
    I_o_ref=cecmodule["I_o_ref"],
    R_sh_ref=cecmodule["R_sh_ref"],
    R_s=cecmodule["R_s"],
)

# %%
photocurrent.plot()
plt.ylabel("Light current I_L (A)")

# %%
saturation_current.plot()
plt.ylabel("Saturation current I_0 (A)")

# %%
print(resistance_series)

# %%
resistance_shunt.plot()
plt.ylabel("Shunt resistance (ohms)")
plt.ylim(0, 5000)

# %%
nNsVth.plot()
plt.ylabel("nNsVth")

# %%
# Single diode model
# ------------------

single_diode_out = pvsystem.singlediode(
    photocurrent,
    saturation_current,
    resistance_series,
    resistance_shunt,
    nNsVth,
)

# %%
single_diode_out["i_sc"].plot()

# %%
single_diode_out["v_oc"].plot()

# %%
single_diode_out["p_mp"].plot()
