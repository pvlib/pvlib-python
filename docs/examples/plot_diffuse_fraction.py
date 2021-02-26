"""
Diffuse Fraction Estimation
===========================

Comparison of diffuse fraction estimation methods used to derive direct and
diffuse components from measured global horizontal irradiance.
"""
# %%
# PV systems are often tilted to optimize performance. Determining the total
# irradiance incident on the plane of the array requires transposing the
# diffuse component, because the entire isotropic sky dome is not visible to
# the surface of the PV. However, irradiance sensors typically only measure
# global horizontal irradiance, GHI, therefore correlations to estimate the
# diffuse fraction the GHI can be used to resolve the diffuse and beam
# components.

# This example demonstrates how to use :py:meth:`pvlib.irradiance.erbs`,
# :py:meth:`pvlib.irradiance.boland`, and several other methods of varying
# complexity


from datetime import datetime
import pathlib
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from pvlib.iotools import read_tmy3
from pvlib.solarposition import get_solarposition
from pvlib import irradiance, solarposition
import pvlib

# get full path to the data directory
DATA_DIR = pathlib.Path(pvlib.__file__).parent / 'data'

# get TMY3 data with rain
greensboro, metadata = read_tmy3(DATA_DIR / '723170TYA.CSV', coerce_year=1990)
solpos = get_solarposition(
    greensboro.index, latitude=metadata['latitude'],
    longitude=metadata['longitude'], altitude=metadata['altitude'],
    pressure=greensboro.Pressure*100, temperature=greensboro.DryBulb)

# %%
# DISC is an NREL function from 1987 that uses an empirical relation between
# GHI and clearness index
out_disc = irradiance.disc(
    greensboro.GHI, solpos.zenith, greensboro.index, greensboro.Pressure*100)
out_disc = out_disc.rename(columns={'dni': 'dni_disc'})
out_disc['dhi_disc'] = (
    greensboro.GHI
    - out_disc.dni_disc*np.cos(np.radians(solpos.apparent_zenith)))

# %%
# next is erbs
out_erbs = irradiance.erbs(greensboro.GHI, solpos.zenith, greensboro.index)
out_erbs = out_erbs.rename(columns={'dni': 'dni_erbs', 'dhi': 'dhi_erbs'})
# %%
JAN6AM,JAN6PM = '1990-01-04 00:00:00-05:00', '1990-01-07 23:59:59-05:00'
f, ax = plt.subplots(2, 1, figsize=(8, 6), sharex=True)
dni_renames = {'DNI': 'TMY', 'dni_disc': 'DISC', 'dni_erbs': 'Erbs'}
dni = pd.concat(
    [greensboro.DNI, out_disc.dni_disc, out_erbs.dni_erbs], axis=1)
dni = dni.rename(columns=dni_renames)
dni[JAN6AM:JAN6PM].plot(ax=ax[0])
ax[0].grid(which="both")
ax[0].set_ylabel('DNI $[W/m^2]$')
dhi_renames = {'DHI': 'TMY', 'dhi_disc': 'DISC', 'dhi_erbs': 'Erbs'}
dhi = pd.concat(
    [greensboro.DHI, out_disc.dhi_disc, out_erbs.dhi_erbs], axis=1)
dhi = dhi.rename(columns=dhi_renames)
dhi[JAN6AM:JAN6PM].plot(ax=ax[1])
ax[1].grid(which="both")
ax[1].set_ylabel('DHI $[W/m^2]$')
f.tight_layout()
# %%
