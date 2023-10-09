"""
A simple way to incorporate thermal inertia
===========================================

What can be simpler than a moving average?

"""

# %%
#
# Applying a moving average filter to the PV array operating conditions
# is a simple technique to compensate for the thermal inertia of the module,
# which delays and dampens temperature fluctuations.
# It is useful for simulating at small time steps, but even more useful for
# fitting models to field data as demonstrated in [1]_.
# The functions :py:func:`pvlib.temperature.faiman_dyn` and
# :py:func:`pvlib.temperature.fit_faiman_dyn` incorporate this moving average
# technique.
#
# This example reads a csv file containing one-minute average monitoring data.
# The function :py:func:`pvlib.temperature.fit_faiman_dyn` is used to determine
# the model parameters and the function :py:func:`pvlib.temperature.faiman_dyn`
# is used to demonstrate how well it worked.
#
# Contributed by Anton Driesse, PV Performance Labs, October 2023.
#
# References
# ----------
# .. [1] Driesse, A. (2022) "Module operating temperature model parameter
#    determination" DOI TBD
#

import os
import pandas as pd
import matplotlib.pyplot as plt

import pvlib
from pvlib.temperature import faiman, faiman_dyn, fit_faiman_dyn

# %%
#
# Read a CSV file containing one week of weather data and module temperature
#

PVLIB_DIR = pvlib.__path__[0]
DATA_FILE = os.path.join(PVLIB_DIR, 'data', 'tmod_sample_data_subset.csv')

df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

print(df.head())

# %%
#
# Estimate the dynamic Faiman model parameters using data where Gpoa > 10
# to eliminate night time data.
#
# The fitting procedure takes a simple approach to finding the optimal
# thermal_inertia: it tries a range of values and chooses the one that
# produces the lowest RMSE.
#

dff = df.copy()
dff = dff.where(df.poa_global > 10)

params, details = fit_faiman_dyn(dff.temp_pv, dff.poa_global,
                                 dff.temp_air, dff.wind_speed,
                                 thermal_inertia=(0, 15, 0.5),
                                 full_output=True)

for k, v in params.items():
    print('%-15s = %5.2f' % (k, v))

# %%
#
# With the full_output option you can obtain the results for all the values
# of thermal_inertia that were evaluated.  The optimal point is clearly visible
# below, but a minute shorter or longer actually doesn't make much difference
# in the RMSE.
#

plt.figure()
plt.plot(details.thermal_inertia, details.rmse, '.-')
plt.grid(alpha=0.5)
plt.xlabel('Thermal inertia')
plt.ylabel('RMSE')
plt.title('Optimal values: u0=%.2f, u1=%.2f, thermal_inertia=%.1f'
          % tuple(params.values()))
plt.show()

# %%
#
# Now calculate the modeled operating temperature of the PV modules.  The
# u0 and u1 values found for the dynamic model can be used with the
# regular Faiman model too, or translated to parameters for other models
# using :py:func:`pvlib.temperature.GenericLinearModel()`.
#

df['temp_pv_faiman'] = faiman(df.poa_global, df.temp_air, df.wind_speed,
                              u0=params['u0'], u1=params['u1'])
df['temp_pv_faiman_dyn'] = faiman_dyn(df.poa_global, df.temp_air,
                                      df.wind_speed, **params)

DAY = slice('2020-03-20 7:00', '2020-03-20 19:00')
# sphinx_gallery_thumbnail_number = 2
plt.figure()
plt.plot(df.temp_pv[DAY])
plt.plot(df.temp_pv_faiman[DAY], alpha=0.5, zorder=0)
plt.plot(df.temp_pv_faiman_dyn[DAY])
plt.legend(['measured', 'faiman', 'faiman_dyn'])
plt.grid(alpha=0.5)
plt.xlabel('2020-03-20')
plt.ylabel('PV temperature [C]')
plt.show()

# %%

dfs = df.sort_values('wind_speed')
plt.figure()
l1 = plt.plot(dfs['temp_pv'], dfs['temp_pv_faiman'], '.', color='C1')
l2 = plt.plot(dfs['temp_pv'], dfs['temp_pv_faiman_dyn'], '.', color='C2')
plt.legend(['faiman', 'faiman_dyn'])
l1[0].set_alpha(0.5)
l2[0].set_alpha(0.25)
plt.grid(alpha=0.5)
plt.xlabel('Measured temperature [°C]')
plt.ylabel('Modeled temperature [°C]')
plt.show()

# %%
#
# Both of the above graphs demonstrate that substantial improvement in modeled
# operating temperature is obtained by this simple technique.  Perhaps more
# important than this, however, is the fact that parameter values can be
# extracted from field data with minimal or no filtering.
#
