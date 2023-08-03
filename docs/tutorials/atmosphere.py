"""
``atmosphere.py`` tutorial
==========================
"""

# %%
# This tutorial needs your help to make it better!
#
# This tutorial requires pvlib > 0.6.0.
#
# Authors:
#
# - Will Holmgren (@wholmgren), University of Arizona. 2015, March 2016,
#   August 2018.


import datetime

import matplotlib.pyplot as plt
import pandas as pd

from pvlib import solarposition, atmosphere
from pvlib.location import Location

# %%

tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
print(tus)

times = pd.date_range(start=datetime.datetime(2014, 6, 24),
                      end=datetime.datetime(2014, 6, 25),
                      freq='1Min', tz=tus.tz)

solpos = solarposition.get_solarposition(times, tus.latitude, tus.longitude)
print(solpos.head())
solpos.plot()

# %%

atmosphere.get_relative_airmass(solpos['zenith']) \
    .plot(label='kastenyoung1989, zenith')
atmosphere.get_relative_airmass(solpos['apparent_zenith']) \
    .plot(label='kastenyoung1989, app. zenith')
atmosphere.get_relative_airmass(solpos['zenith'], model='young1994') \
    .plot(label='young1994, zenith')
atmosphere.get_relative_airmass(solpos['zenith'], model='simple') \
    .plot(label='simple, zenith')
plt.legend()
plt.ylabel('Airmass')
plt.ylim(0, 100)

# %%

plt.plot(solpos['zenith'],
         atmosphere.get_relative_airmass(solpos['zenith'], model='simple'),
         label='simple')
plt.plot(solpos['zenith'],
         atmosphere.get_relative_airmass(solpos['apparent_zenith']),
         label='default')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xlabel('Zenith angle (deg)')
plt.ylabel('Airmass')
plt.legend()
