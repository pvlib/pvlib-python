# coding: utf-8

# # ``atmosphere.py`` tutorial
# 
# This tutorial needs your help to make it better!
# 
# This tutorial has been tested against the following package versions:
# * pvlib 0.2.0
# * Python 2.7.10
# * IPython 3.2
# * Pandas 0.16.2
# 
# It should work with other Python and Pandas versions. It requires pvlib > 0.2.0 and IPython > 3.0.
# 
# Authors:
# * Will Holmgren (@wholmgren), University of Arizona. 2015.
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
    sns.set(rc={"figure.figsize": (12, 6)})
except ImportError:
    print('We suggest you install seaborn using conda or pip and rerun this cell')

# built in python modules
import datetime
import logging
import os
import inspect

# python add-ons
import numpy as np
import pandas as pd

import pvlib
from pvlib.location import Location

tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')

print(tus)

times = pd.date_range(start=datetime.datetime(2014,6,24), end=datetime.datetime(2014,6,25), freq='1Min')

pyephem_ephem = pvlib.solarposition.get_solarposition(times, tus, method='pyephem')
print(pyephem_ephem.head())
pyephem_ephem.plot()

pyephem_ephem = pvlib.solarposition.get_solarposition(times.tz_localize(tus.tz), tus, method='pyephem')
print(pyephem_ephem.head())
pyephem_ephem.plot()

pvlib.atmosphere.relativeairmass(pyephem_ephem['zenith']).plot()
pvlib.atmosphere.relativeairmass(pyephem_ephem['apparent_zenith']).plot()
pvlib.atmosphere.relativeairmass(pyephem_ephem['zenith'], model='young1994').plot()
pvlib.atmosphere.relativeairmass(pyephem_ephem['zenith'], model='simple').plot()
plt.legend()
plt.ylim(0,100)

plt.plot(pyephem_ephem['zenith'], pvlib.atmosphere.relativeairmass(pyephem_ephem['zenith'], model='simple'), label='simple')
plt.plot(pyephem_ephem['zenith'], pvlib.atmosphere.relativeairmass(pyephem_ephem['apparent_zenith']), label='default')
plt.xlim(0,100)
plt.ylim(0,100)
plt.legend()



