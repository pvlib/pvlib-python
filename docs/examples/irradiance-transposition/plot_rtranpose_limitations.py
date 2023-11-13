"""
Explore the limitations of reverse transposition
================================================

Unfortunately, sometimes there is more than one solution.

Author: Anton Driesse

"""

# %%
#
# In this example we look at a single point in time and consider a full range
# of possible GHI and POA global values as shown in figures 3 and 4 of [1]_.
# Then we use :py:func:`pvlib.irradiance.rtranspose_driesse_2023` to estimate
# the original GHI from POA global.
#
# References
# ----------
# .. [1] A. Driesse, A. Jensen, R. Perez, A Continuous Form of the Perez
#     Diffuse Sky Model for Forward and Reverse Transposition, accepted
#     for publication in the Solar Energy Journal.
#

import numpy as np
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['axes.grid'] = True

import pvlib
from pvlib import iotools, location
from pvlib.irradiance import (erbs_driesse,
                              get_total_irradiance,
                              rtranspose_driesse_2023,
                              )

# %% define conditions for figure 3 in [1]_

dni_extra = 1366.1
albedo = 0.25
surface_tilt = 40
surface_azimuth = 180

solar_azimuth = 82
solar_zenith = 75

ghi = np.linspace(0, 500, 100+1)

# %% transpose forward

erbsout = erbs_driesse(ghi, solar_zenith, dni_extra=dni_extra)

dni = erbsout['dni']
dhi = erbsout['dhi']

irrads = get_total_irradiance(surface_tilt, surface_azimuth,
                              solar_zenith, solar_azimuth,
                              dni, ghi, dhi,
                              dni_extra, #airmass, albedo,
                              model='perez-driesse')

gti = irrads['poa_global']

# %% reverse transpose a single GTI value of 200 W/m²

gti_test = 200

ghi_hat = rtranspose_driesse_2023(surface_tilt, surface_azimuth,
                                  solar_zenith, solar_azimuth,
                                  gti_test,
                                  dni_extra,
                                  full_output=False,
                                  )

plt.figure()
plt.plot(ghi, gti, 'g-')
plt.axvline(ghi_hat, color='b')
plt.axhline(gti_test, color='b')
plt.plot(ghi_hat, gti_test, 'bs')
plt.annotate('(%.2f, %.0f)' % (ghi_hat, gti_test),
             xy=(ghi_hat-2, 200+2),
             xytext=(ghi_hat-20, 200+20),
             ha='right',
             arrowprops={'arrowstyle':'simple'})
plt.xlim(0, 500)
plt.ylim(0, 250)
plt.xlabel('GHI [W/m²]')
plt.ylabel('GTI or POA [W/m²]')

# %% change to the conditions for figure 4 in [1]_

solar_azimuth = 76

# %% transpose forward

erbsout = erbs_driesse(ghi, solar_zenith, dni_extra=dni_extra)

dni = erbsout['dni']
dhi = erbsout['dhi']

irrads = get_total_irradiance(surface_tilt, surface_azimuth,
                              solar_zenith, solar_azimuth,
                              dni, ghi, dhi,
                              dni_extra, #airmass, albedo,
                              model='perez-driesse')

gti = irrads['poa_global']

# %% reverse transpose the full range of possible POA values

result = rtranspose_driesse_2023(surface_tilt, surface_azimuth,
                              solar_zenith, solar_azimuth,
                              gti,
                              dni_extra,
                              full_output=True,
                              )

ghi_hat, conv, niter = result
correct = np.isclose(ghi, ghi_hat, atol=0.01)

plt.figure()

plt.plot(np.where(correct, ghi, np.nan),
         np.where(correct, gti, np.nan), 'g-', label='GHI correct')

plt.plot(ghi[~correct], gti[~correct], 'r.', label='GHI incorrect')
plt.plot(ghi[~conv], gti[~conv], 'm.', label='out of range (kt > 1.25)')
plt.xlim(0, 500)
plt.ylim(0, 250)
plt.xlabel('GHI [W/m²]')
plt.ylabel('GTI or POA [W/m²]')
plt.legend()
plt.show()
