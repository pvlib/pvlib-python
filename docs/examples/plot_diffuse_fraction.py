"""
Diffuse Fraction Estimation
===========================

Comparison of diffuse fraction estimation methods used to derive direct and
diffuse components from measured global horizontal irradiance.
"""

# %%
# This example demonstrates how to use diffuse fraction estimation methods to
# obtain direct and diffuse components from measured global horizontal
# irradiance (GHI). Irradiance sensors such as pyranometers typically only
# measure GHI. pvlib provides several functions that can be used to separate
# GHI into the diffuse and direct components. The separate components are
# needed to estimate the total irradiance on a tilted surface.

import pathlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pvlib.iotools import read_tmy3
from pvlib.solarposition import get_solarposition
from pvlib import irradiance
import pvlib

# For this example we use the Greensboro, North Carolina, TMY3 file which is
# in the pvlib data directory. TMY3 are made from the median months from years
# of data measured from 1990 to 2010. Therefore we change the timestamps to a
# common year, 1990.
DATA_DIR = pathlib.Path(pvlib.__file__).parent / 'data'
greensboro, metadata = read_tmy3(DATA_DIR / '723170TYA.CSV', coerce_year=1990)

# Many of the diffuse fraction estimation methods require the "true" zenith, so
# we calculate the solar positions for the 1990 at Greensboro, NC.
solpos = get_solarposition(
    greensboro.index, latitude=metadata['latitude'],
    longitude=metadata['longitude'], altitude=metadata['altitude'],
    pressure=greensboro.Pressure*100,  # convert from millibar to Pa
    temperature=greensboro.DryBulb)

# %%
# Methods for separating DHI into diffuse and direct components include:
# **DISC**, **DIRINT**, **Erbs** and **Boland**.

# %%
# DISC
# ----
#
# DISC :py:func:`~pvlib.irradiance.disc` is an empirical correlation developed
# at SERI (now NREL) in 1987. The direct normal irradiance (DNI) is related to
# clearness index (kt) by two polynomials split at kt = 0.6, then combined with
# an exponential relation with airmass.

out_disc = irradiance.disc(
    greensboro.GHI, solpos.zenith, greensboro.index, greensboro.Pressure*100)
out_disc = out_disc.rename(columns={'dni': 'dni_disc'})
out_disc['dhi_disc'] = (
    greensboro.GHI
    - out_disc.dni_disc*np.cos(np.radians(solpos.apparent_zenith)))

# %%
# DIRINT
# ------
#
# DIRINT :py:func:`~pvlib.irradiance.dirint` is a modification of DISC
# developed by Richard Perez and Pierre Ineichen in 1992.

dni_dirint = irradiance.dirint(
    greensboro.GHI, solpos.zenith, greensboro.index, greensboro.Pressure*100,
    temp_dew=greensboro.DewPoint)
dhi_dirint = (
    greensboro.GHI
    - dni_dirint*np.cos(np.radians(solpos.apparent_zenith)))
out_dirint = pd.DataFrame(
    {'dni_dirint': dni_dirint, 'dhi_dirint': dhi_dirint},
    index=greensboro.index)

# %%
# Erbs
# ----
#
# The Erbs method, :py:func:`~pvlib.irradiance.erbs` developed by Daryl Gregory
# Erbs at the University of Wisconsin in 1982 is a piecewise correlation that
# splits kt into 3 regions: linear for kt <= 0.22, a 4th order polynomial
# between 0.22 < kt <= 0.8, and a horizontal line for kt > 0.8.

out_erbs = irradiance.erbs(greensboro.GHI, solpos.zenith, greensboro.index)
out_erbs = out_erbs.rename(columns={'dni': 'dni_erbs', 'dhi': 'dhi_erbs'})

# %%
# Boland
# ----
#
# The Boland method, :py:func:`~pvlib.irradiance.boland` is a single logistic
# exponential correlation that is continuously differentiable and bounded
# between zero and one.

out_boland = irradiance.boland(greensboro.GHI, solpos.zenith, greensboro.index)
out_boland = out_boland.rename(
    columns={'dni': 'dni_boland', 'dhi': 'dhi_boland'})

# %%
# Combine everything together.

dni_renames = {
    'DNI': 'TMY', 'dni_disc': 'DISC', 'dni_dirint': 'DIRINT',
    'dni_erbs': 'Erbs', 'dni_boland': 'Boland'}
dni = [
    greensboro.DNI, out_disc.dni_disc, out_dirint.dni_dirint,
    out_erbs.dni_erbs, out_boland.dni_boland]
dni = pd.concat(dni, axis=1).rename(columns=dni_renames)
dhi_renames = {
    'DHI': 'TMY', 'dhi_disc': 'DISC', 'dhi_dirint': 'DIRINT',
    'dhi_erbs': 'Erbs', 'dhi_boland': 'Boland'}
dhi = [
    greensboro.DHI, out_disc.dhi_disc, out_dirint.dhi_dirint,
    out_erbs.dhi_erbs, out_boland.dhi_boland]
dhi = pd.concat(dhi, axis=1).rename(columns=dhi_renames)
ghi_kt = pd.concat([greensboro.GHI/1366.1, out_erbs.kt], axis=1)

# %%
# Finally, let's plot them for a few winter days and compare

JAN6AM, JAN6PM = '1990-01-04 00:00:00-05:00', '1990-01-07 23:59:59-05:00'
f, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
dni[JAN6AM:JAN6PM].plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].set_ylabel('DNI $[W/m^2]$')
ax[0].set_title('Comparison of Diffuse Fraction Estimation Methods')
dhi[JAN6AM:JAN6PM].plot(ax=ax[1])
ax[1].grid(which="both")
ax[1].set_ylabel('DHI $[W/m^2]$')
ghi_kt[JAN6AM:JAN6PM].plot(ax=ax[2])
ax[2].grid(which='both')
ax[2].set_ylabel(r'$\frac{GHI}{G_{SC}}, k_t$')
f.tight_layout()

# %%
# And a few spring days ...

APR6AM, APR6PM = '1990-04-04 00:00:00-05:00', '1990-04-07 23:59:59-05:00'
f, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
dni[APR6AM:APR6PM].plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].set_ylabel('DNI $[W/m^2]$')
ax[0].set_title('Comparison of Diffuse Fraction Estimation Methods')
dhi[APR6AM:APR6PM].plot(ax=ax[1])
ax[1].grid(which="both")
ax[1].set_ylabel('DHI $[W/m^2]$')
ghi_kt[APR6AM:APR6PM].plot(ax=ax[2])
ax[2].grid(which='both')
ax[2].set_ylabel(r'$\frac{GHI}{E0}, k_t$')
f.tight_layout()

# %%
# And few summer days to finish off the seasons.

JUL6AM, JUL6PM = '1990-07-04 00:00:00-05:00', '1990-07-07 23:59:59-05:00'
f, ax = plt.subplots(3, 1, figsize=(8, 10), sharex=True)
dni[JUL6AM:JUL6PM].plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].set_ylabel('DNI $[W/m^2]$')
ax[0].set_title('Comparison of Diffuse Fraction Estimation Methods')
dhi[JUL6AM:JUL6PM].plot(ax=ax[1])
ax[1].grid(which="both")
ax[1].set_ylabel('DHI $[W/m^2]$')
ghi_kt[JUL6AM:JUL6PM].plot(ax=ax[2])
ax[2].grid(which='both')
ax[2].set_ylabel(r'$\frac{GHI}{E0}, k_t$')
f.tight_layout()

# %%
# Conclusion
# ----------
# The Erbs and Boland are correlations with only kt, which is derived from the
# horizontal component of the extra-terrestrial irradiance. Therefore at low
# sun elevation (zenith ~ 90-deg), especially near sunset, this causes kt to
# explode as the denominator approaches zero. This is controlled in pvlib by
# setting ``min_cos_zenith`` and ``max_clearness_index`` which each have
# reasonable defaults, but there are still concerning spikes at sunset for Jan.
# 5th & 7th, April 4th, 5th, & 7th, and July 6th & 7th. The DISC & DIRINT
# methods differ from Erbs and Boland be including airmass, which seems to
# reduce DNI spikes over 1000[W/m^2], but still have errors at other times.
#
# Another difference is that DISC & DIRINT return DNI whereas Erbs & Boland
# calculate the diffuse fraction which is then used to derive DNI from GHI and
# the solar zenith, which exacerbates errors at low sun elevation due to the
# relation: DNI = GHI*(1 - DF)/cos(zenith).
