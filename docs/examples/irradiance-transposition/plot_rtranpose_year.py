"""
Reverse transposition using one year of hourly data
===================================================

With a brief look at accuracy and speed.

Author: Anton Driesse
"""

# %%
#
# In this example we start with a TMY file and calculate POA global irradiance.
# Then we use :py:func:`pvlib.irradiance.rtranspose_driesse_2023` to estimate
# the original GHI from POA global.  Details of the method are more fully
# described in [1]_.
#
# References
# ----------
# .. [1] A. Driesse, A. Jensen, R. Perez, A Continuous Form of the Perez
#     Diffuse Sky Model for Forward and Reverse Transposition, accepted
#     for publication in the Solar Energy Journal.
#

import os
import pandas as pd
import matplotlib.pyplot as plt

import pvlib
from pvlib import iotools, location, irradiance
from pvlib.irradiance import (get_extra_radiation,
                              get_total_irradiance,
                              rtranspose_driesse_2023,
                              aoi,
                              )

from timeit import timeit

# %%
#
# Read a TMY3 file containing weather data and select needed columns
#

PVLIB_DIR = pvlib.__path__[0]
DATA_FILE = os.path.join(PVLIB_DIR, 'data', '723170TYA.CSV')

tmy, metadata = iotools.read_tmy3(DATA_FILE, coerce_year=1990,
                                  map_variables=True)

df = pd.DataFrame({'ghi': tmy['ghi'], 'dhi': tmy['dhi'], 'dni': tmy['dni'],
                   'temp_air': tmy['temp_air'],
                   'wind_speed': tmy['wind_speed'],
                   })

# %%
#
# Shift timestamps to middle of hour and then calculate sun positions
#

df.index = df.index - pd.Timedelta(minutes=30)

loc = location.Location.from_tmy(metadata)
solpos = loc.get_solarposition(df.index)

# %%
#
# Estimate global irradiance on a fixed-tilt array (forward transposition)
# The array is tilted 30 degrees an oriented 30 degrees east of south.
# The Perez-Driesse model is used for this to match the reverse transposition
# later, but the match is not perfect because the diffuse fraction of the data
# does not match the Erbs model.
#

TILT = 30
ORIENT = 150

df['dni_extra'] = get_extra_radiation(df.index)

total_irrad = get_total_irradiance(TILT, ORIENT,
                                   solpos.apparent_zenith, solpos.azimuth,
                                   df.dni, df.ghi, df.dhi,
                                   dni_extra=df.dni_extra,
                                   model='perez-driesse')

df['poa_global'] = total_irrad.poa_global
df['aoi'] = aoi(TILT, ORIENT, solpos.apparent_zenith, solpos.azimuth)

# %%
#
# Now estimate ghi from poa_global using reverse transposition
# This step uses a
#

df['ghi_rev'] = rtranspose_driesse_2023(TILT, ORIENT,
                                        solpos.apparent_zenith, solpos.azimuth,
                                        df.poa_global,
                                        dni_extra=df.dni_extra)

# %%
#
# Let's see how long this takes
#

def run_rtranspose():
    rtranspose_driesse_2023(TILT, ORIENT,
                            solpos.apparent_zenith, solpos.azimuth,
                            df.poa_global,
                            dni_extra=df.dni_extra)

elapsed = timeit('run_rtranspose()', number=1, globals=globals())

print('Elapsed time for reverse transposition: %.1f s' % elapsed)

# %%
#
# Errors occur mostly at high incidence angles.
#
df = df.sort_values('aoi')

plt.figure()
pc = plt.scatter(df['ghi'], df['ghi_rev'], c=df['aoi'], s=15,
                 cmap='jet', vmin=60, vmax=120)
plt.colorbar(label='AOI [°]', ax=plt.gca())
pc.set_alpha(0.5)
plt.grid(alpha=0.5)
plt.xlabel('GHI original [W/m²]')
plt.ylabel('GHI from POA [W/m²]')
plt.show()

