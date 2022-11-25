"""
Simulating PV systems using the ADR module efficiency model
===========================================================

Time series processing with the ADR model is fast and ... efficient!

(WORK IN PROGRESS)

"""

import os
import pandas as pd
import matplotlib.pyplot as plt

import pvlib
from pvlib.pvefficiency import adr

# %%
# Borrow the model parameters from the other example:
#

adr_parms = {'k_a': 0.99879,
             'k_d': -5.85188,
             'tc_d': 0.01939,
             'k_rs': 0.06962,
             'k_rsh': 0.21036
             }

# %%
#
# Read an existing PVWATTS simulation output file
# which contains all the input data we need to run an ADR simulation.
# this system is 4000 W nominal
# system losses are 14.08 %
# therefore P_STC = 3437 W
#

DATADIR = os.path.join(pvlib.__path__[0], 'data')
DATAFILE = os.path.join(DATADIR, 'pvwatts_8760_rackmount.csv')

df = pd.read_csv(DATAFILE, skiprows=17, nrows=8760)
df.columns = ['month', 'day', 'hour',
              'dni', 'dif', 't_amb', 'wind_speed',
              'poa_global', 't_cell', 'p_dc', 'p_ac']

df['year'] = 2019
DATECOLS = ['year', 'month', 'day', 'hour']
df.index = pd.to_datetime(df[DATECOLS])
df = df.drop(columns=DATECOLS)

# %%
#
# Simulating is done in two steps:
# first calculate the efficiency, then convert efficiency to power.
#

P_REF = 3437.   # (W)
G_REF = 1000.   # (W/m2)

df['eta_rel'] = adr(df['poa_global'], df['t_cell'], **adr_parms)

df['p_dc_adr'] = P_REF * df['eta_rel'] * (df['poa_global'] / G_REF)

# %%
#
# Compare the ADR simulated output to PVWATS for one day.
#
# NOTE: they are not supposed to be the same because the module simulated
# by PVWATTS is different from the our ADR example module.
#

DEMO_DAY = '2019-08-05'

plt.figure()
plt.plot(df['p_dc'][DEMO_DAY])
plt.plot(df['p_dc_adr'][DEMO_DAY])
plt.xticks(rotation=30)
plt.legend(['PVWATTS', 'ADR'])
plt.ylabel('Power [W]')
plt.show()

# %%
#
# The colors in the next graph show that the PVWATTS module probably has
# a larger temperature coefficient than the ADR example module.
#

plt.figure()
plt.scatter(df['p_dc'], df['p_dc_adr'],
            c=df['t_cell'], alpha=.3, cmap='jet')
plt.plot([0, 4000], [0, 4000], 'k', alpha=.5)
plt.xlabel('PVWATTS DC array output [W]')
plt.ylabel('ADR modelled DC array output [W]')
plt.colorbar(label='T_cell', ax=plt.gca())
plt.show()

# %%
#
# References
# ----------
# .. [1] A. Driesse and J. S. Stein, "From IEC 61853 power measurements
#    to PV system simulations", Sandia Report No. SAND2020-3877, 2020.
#
# .. [2] A. Driesse, M. Theristis and J. S. Stein, "A New Photovoltaic Module
#    Efficiency Model for Energy Prediction and Rating," in IEEE Journal
#    of Photovoltaics, vol. 11, no. 2, pp. 527-534, March 2021,
#    doi: 10.1109/JPHOTOV.2020.3045677.
#
