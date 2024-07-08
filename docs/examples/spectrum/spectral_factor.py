"""
Spectral Mismatch Estimation
============================

Comparison of methods to estimate the spectral mismatch factor
from atmospheric variable inputs.
"""

# %%
# Introduction
# ------------
# This example demonstrates how to use different ``spectrum.spectral_factor_*``
# models in pvlib-python to calculate the spectral mismatch factor, :math:`M`.
# While :math:`M` for a photovoltaic (PV) module can be calculated exactly
# using spectral irradiance and module spectral response data, these data are
# often not available. pvlib-python provides several functions to estimate the
# spectral mismatch factor, :math:`M`, using proxies of the prevailing spectral
# irradiance conditions, such as air mass and clearsky index, which are easily
# derived from common ground-based measurements such as broadband irradiance.
# More information on a range of spectral factor models, as well as the
# variables upon which they are based, can be found in [1]_.
#
# To demonstrate this functionality, first we need to import some data.
# This example uses a Typical Meteorological Year 3
# (TMY3) file for the location of Greensboro, North Carolina, from the
# pvlib-python data directory. This TMY3 file is constructed using the median
# month from each year between 1980 and 2003, from which we extract the first
# week of August 2001 to analyse.

# %%
import pathlib
from matplotlib import pyplot as plt
import pandas as pd
import pvlib
from pvlib import location

DATA_DIR = pathlib.Path(pvlib.__file__).parent / 'data'
meteo, metadata = pvlib.iotools.read_tmy3(DATA_DIR / '723170TYA.CSV',
                                          coerce_year=2001, map_variables=True)
meteo = meteo.loc['2001-08-01':'2001-08-07']

# %%
# Spectral Factor Functions
# -------------------------
# This example demonstrates the application of three pvlib-python
# spectral factor functions:
#
# - :py:func:`~pvlib.spectrum.spectral_factor_sapm`, which requires only
#   the absolute airmass, :math:`AM_a`
# - :py:func:`~pvlib.spectrum.spectral_factor_pvspec`, which requires
#   :math:`AM_a` and the clearsky index, :math:`k_c`
# - :py:func:`~pvlib.spectrum.spectral_factor_firstsolar`, which requires
#   :math:`AM_a` and the atmospheric precipitable water content, :math:`W`

# %%
# Calculation of inputs
# ---------------------
# Let's calculate the absolute air mass, which is required for all three
# models. We use the Kasten and Young [2]_ model, which requires the apparent
# sun zenith. Note: TMY3 files timestamps indicate the end of the hour, so we
# shift the indices back 30-minutes to calculate solar position at the centre
# of the interval.

# Create a location object
lat, lon = metadata['latitude'], metadata['longitude']
alt = altitude = metadata['altitude']
tz = 'Etc/GMT+5'
loc = location.Location(lat, lon, tz=tz, name='Greensboro, NC')

# Calculate solar position parameters
solpos = loc.get_solarposition(
    meteo.index.shift(freq="-30min"),
    pressure=meteo.pressure*100,  # convert from millibar to Pa
    temperature=meteo.temp_air)
solpos.index = meteo.index  # reset index to end of the hour

airmass_relative = pvlib.atmosphere.get_relative_airmass(
    solpos.apparent_zenith).dropna()
airmass_absolute = pvlib.atmosphere.get_absolute_airmass(airmass_relative,
                                                         meteo.pressure*100)
# %%
# Now we calculate the clearsky index, :math:`k_c`, which is the ratio of GHI
# to clearsky GHI.

cs = loc.get_clearsky(meteo.index.shift(freq="-30min"))
cs.index = meteo.index  # reset index to end of the hour
kc = pvlib.irradiance.clearsky_index(meteo.ghi, cs.ghi)
# %%
# :math:`W` is provided in the TMY3 file but in other cases can be calculated
# from temperature and relative humidity
# (see :py:func:`pvlib.atmosphere.gueymard94_pw`).

w = meteo.precipitable_water

# %%
# Calculation of Spectral Mismatch
# --------------------------------
# Let's calculate the spectral mismatch factor using the three spectral factor
# functions. First, we need to import some model coefficients for the SAPM
# spectral factor function, which, unlike the other two functions, lacks
# built-in coefficients.

# Import some for a mc-Si module from the SAPM module database.
module = pvlib.pvsystem.retrieve_sam('SandiaMod')['LG_LG290N1C_G3__2013_']
#
# Calculate M using the three models for an mc-Si PV module.
m_sapm = pvlib.spectrum.spectral_factor_sapm(airmass_absolute, module)
m_pvspec = pvlib.spectrum.spectral_factor_pvspec(airmass_absolute, kc,
                                                 'multisi')
m_fs = pvlib.spectrum.spectral_factor_firstsolar(w, airmass_absolute,
                                                 'multisi')

df_results = pd.concat([m_sapm, m_pvspec, m_fs], axis=1)
df_results.columns = ['SAPM', 'PVSPEC', 'FS']
# %%
# Comparison Plots
# ----------------
# We can plot the results to visualise variability between the models. Note
# that this is not an exact comparison since the exact same PV modules has
# not been modelled in each case, only the same module technology.

# Plot M
fig1, (ax1, ax2) = plt.subplots(2, 1)
df_results.plot(ax=ax1, legend=False)
ax1.set_xlabel('Day')
ax1.set_ylabel('Spectral mismatch (-)')
ax1.set_ylim(0.85, 1.15)
ax1.legend(loc='upper center', frameon=False, ncols=3,
           bbox_to_anchor=(0.5, 1.3))

# We can also zoom in one one day, for example August 2nd.
df_results.loc['2001-08-02'].plot(ax=ax2, legend=False)
ax2.set_xlabel('Time')
ax2.set_ylabel('Spectral mismatch (-)')
ax2.set_ylim(0.85, 1.15)

plt.tight_layout()

plt.show()

# %%
# References
# ----------
# .. [1] Daxini, R., and Wu, Y., 2023. "Review of methods to account
#        for the solar spectral influence on photovoltaic device performance."
#        Energy 286
#        :doi:`10.1016/j.energy.2023.129461`
# .. [2] Kasten, F. and Young, A.T., 1989. Revised optical air mass tables
#        and approximation formula. Applied Optics, 28(22), pp.4735-4738.
#        :doi:`10.1364/AO.28.004735`
