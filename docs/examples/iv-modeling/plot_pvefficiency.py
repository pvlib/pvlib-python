"""
Using the ADR PV efficiency model
=================================

Examples of getting the ADR PV efficiency model parameters
and using the model for system simulation.

(WORK IN PROGRESS)

"""
import os
from io import StringIO
import pandas as pd
import matplotlib.pyplot as plt

import pvlib
from pvlib.pvefficiency import fit_pvefficiency_adr, adr

iec61853data = '''
    irradiance  temperature     p_mp
0          100         15.0   30.159
1          200         15.0   63.057
2          400         15.0  129.849
3          600         15.0  197.744
4          800         15.0  264.825
5         1000         15.0  330.862
6          100         25.0   29.250
7          200         25.0   61.137
8          400         25.0  126.445
9          600         25.0  192.278
10         800         25.0  257.561
11        1000         25.0  322.305
12        1100         25.0  354.174
13         100         50.0   26.854
14         200         50.0   56.698
15         400         50.0  117.062
16         600         50.0  177.959
17         800         50.0  238.626
18        1000         50.0  298.954
19        1100         50.0  328.413
20         100         75.0   24.074
21         200         75.0   51.103
22         400         75.0  106.546
23         600         75.0  162.966
24         800         75.0  218.585
25        1000         75.0  273.651
26        1100         75.0  301.013
'''
df = pd.read_csv(StringIO(iec61853data), delim_whitespace=True)

P_STC = 322.305


def pmp2eta(g, p, p_stc):
    return p / p_stc / (g / 1000)


def eta2pmp(g, eta, p_stc):
    return eta * p_stc * (g / 1000)


eta_rel = pmp2eta(df.irradiance, df.p_mp, P_STC)

adr_parms = fit_pvefficiency_adr(df.irradiance, df.temperature, eta_rel)

eta_adr = adr(df.irradiance, df.temperature, **adr_parms)

plt.figure()
plt.plot(df.irradiance, eta_rel, 'oc')
plt.plot(df.irradiance, eta_adr, '.k')
plt.legend(['Lab measurements', 'ADR model fit'])
plt.xlabel('Irradiance [W/m²]')

# %% Cell 1

# this system is 4000 W nominal
# system losses are 14.08 %
# therefore P_STC = 3437 W

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

df['eta_adr'] = adr(df['poa_global'], df['t_cell'], **adr_parms)
df['p_dc_adr'] = eta2pmp(df['poa_global'], df['eta_adr'], p_stc=3437)

# %% Cell 2

DEMO_DAY = '2019-08-05'

plt.figure()
plt.plot(df['p_dc'][DEMO_DAY])
plt.plot(df['p_dc_adr'][DEMO_DAY])
plt.xticks(rotation=30)
plt.legend(['PVWATTS', 'ADR'])
plt.ylabel('Power [W]')

# %% Cell 3

plt.figure()
plt.scatter(df['p_dc'], df['p_dc_adr'],
            c=df['t_cell'], alpha=.3, cmap='jet')
plt.plot([0, 4000], [0, 4000], 'k', alpha=.5)
plt.xlabel('PVWATTS DC array output [W]')
plt.ylabel('ADR modelled DC array output [W]')
