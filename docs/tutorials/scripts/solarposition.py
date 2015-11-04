# coding: utf-8

# # ``solarposition.py`` tutorial
# 
# This tutorial needs your help to make it better!
# 
# Table of contents:
# 1. [Setup](#Setup)
# 2. [SPA output](#SPA-output)
# 2. [Speed tests](#Speed-tests)
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
# * Will Holmgren (@wholmgren), University of Arizona. July 2014, July 2015.

# ## Setup
import datetime

# scientific python add-ons
import numpy as np
import pandas as pd

# plotting stuff
# first line makes the plots appear in the notebook
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
# seaborn makes your plots look better
try:
    import seaborn as sns
    sns.set(rc={"figure.figsize": (12, 6)})
except ImportError:
    print('We suggest you install seaborn using conda or pip and rerun this cell')

# finally, we import the pvlib library
import pvlib

import pvlib
from pvlib.location import Location


# ## SPA output
tus = Location(32.2, -111, 'US/Arizona', 700, 'Tucson')
print(tus)
golden = Location(39.742476, -105.1786, 'America/Denver', 1830, 'Golden')
print(golden)
golden_mst = Location(39.742476, -105.1786, 'MST', 1830, 'Golden MST')
print(golden_mst)
berlin = Location(52.5167, 13.3833, 'Europe/Berlin', 34, 'Berlin')
print(berlin)

times = pd.date_range(start=datetime.datetime(2014,6,23), end=datetime.datetime(2014,6,24), freq='1Min')
times_loc = times.tz_localize(tus.pytz)

times

pyephemout = pvlib.solarposition.pyephem(times, tus)
spaout = pvlib.solarposition.spa_python(times, tus)

reload(pvlib.solarposition)
pyephemout = pvlib.solarposition.pyephem(times_loc, tus)
spaout = pvlib.solarposition.spa_python(times_loc, tus)

pyephemout['elevation'].plot(label='pyephem')
pyephemout['apparent_elevation'].plot(label='pyephem apparent')
spaout['elevation'].plot(label='spa')
plt.legend(ncol=2)
plt.title('elevation')

print('pyephem')
print(pyephemout.head())
print('spa')
print(spaout.head())

plt.figure()
pyephemout['elevation'].plot(label='pyephem')
spaout['elevation'].plot(label='spa')
(pyephemout['elevation'] - spaout['elevation']).plot(label='diff')
plt.legend(ncol=3)
plt.title('elevation')

plt.figure()
pyephemout['apparent_elevation'].plot(label='pyephem apparent')
spaout['elevation'].plot(label='spa')
(pyephemout['apparent_elevation'] - spaout['elevation']).plot(label='diff')
plt.legend(ncol=3)
plt.title('elevation')

plt.figure()
pyephemout['apparent_zenith'].plot(label='pyephem apparent')
spaout['zenith'].plot(label='spa')
(pyephemout['apparent_zenith'] - spaout['zenith']).plot(label='diff')
plt.legend(ncol=3)
plt.title('zenith')

plt.figure()
pyephemout['apparent_azimuth'].plot(label='pyephem apparent')
spaout['azimuth'].plot(label='spa')
(pyephemout['apparent_azimuth'] - spaout['azimuth']).plot(label='diff')
plt.legend(ncol=3)
plt.title('azimuth')

reload(pvlib.solarposition)
pyephemout = pvlib.solarposition.pyephem(times, tus)
spaout = pvlib.solarposition.spa_python(times, tus)

pyephemout['elevation'].plot(label='pyephem')
pyephemout['apparent_elevation'].plot(label='pyephem apparent')
spaout['elevation'].plot(label='spa')
plt.legend(ncol=3)
plt.title('elevation')

print('pyephem')
print(pyephemout.head())
print('spa')
print(spaout.head())

reload(pvlib.solarposition)
pyephemout = pvlib.solarposition.pyephem(times, golden)
spaout = pvlib.solarposition.spa_python(times, golden)

pyephemout['elevation'].plot(label='pyephem')
pyephemout['apparent_elevation'].plot(label='pyephem apparent')
spaout['elevation'].plot(label='spa')
plt.legend(ncol=2)
plt.title('elevation')

print('pyephem')
print(pyephemout.head())
print('spa')
print(spaout.head())

pyephemout = pvlib.solarposition.pyephem(times, golden)
ephemout = pvlib.solarposition.ephemeris(times, golden)

pyephemout['elevation'].plot(label='pyephem')
pyephemout['apparent_elevation'].plot(label='pyephem apparent')
ephemout['elevation'].plot(label='ephem')
plt.legend(ncol=2)
plt.title('elevation')

print('pyephem')
print(pyephemout.head())
print('ephem')
print(ephemout.head())

loc = berlin

pyephemout = pvlib.solarposition.pyephem(times, loc)
ephemout = pvlib.solarposition.ephemeris(times, loc)

pyephemout['elevation'].plot(label='pyephem')
pyephemout['apparent_elevation'].plot(label='pyephem apparent')
ephemout['elevation'].plot(label='ephem')
ephemout['apparent_elevation'].plot(label='ephem apparent')
plt.legend(ncol=2)
plt.title('elevation')

print('pyephem')
print(pyephemout.head())
print('ephem')
print(ephemout.head())

pyephemout['elevation'].plot(label='pyephem')
pyephemout['apparent_elevation'].plot(label='pyephem apparent')
ephemout['elevation'].plot(label='ephem')
ephemout['apparent_elevation'].plot(label='ephem apparent')
plt.legend(ncol=2)
plt.title('elevation')
plt.xlim(pd.Timestamp('2015-06-28 03:00:00+02:00'), pd.Timestamp('2015-06-28 06:00:00+02:00'))
plt.ylim(-10,10)

loc = berlin
times = pd.DatetimeIndex(start=datetime.date(2015,3,28), end=datetime.date(2015,3,29), freq='5min')

pyephemout = pvlib.solarposition.pyephem(times, loc)
ephemout = pvlib.solarposition.ephemeris(times, loc)

pyephemout['elevation'].plot(label='pyephem')
pyephemout['apparent_elevation'].plot(label='pyephem apparent')
ephemout['elevation'].plot(label='ephem')
plt.legend(ncol=2)
plt.title('elevation')

plt.figure()
pyephemout['azimuth'].plot(label='pyephem')
ephemout['azimuth'].plot(label='ephem')
plt.legend(ncol=2)
plt.title('azimuth')

print('pyephem')
print(pyephemout.head())
print('ephem')
print(ephemout.head())

loc = berlin
times = pd.DatetimeIndex(start=datetime.date(2015,3,30), end=datetime.date(2015,3,31), freq='5min')

pyephemout = pvlib.solarposition.pyephem(times, loc)
ephemout = pvlib.solarposition.ephemeris(times, loc)

pyephemout['elevation'].plot(label='pyephem')
pyephemout['apparent_elevation'].plot(label='pyephem apparent')
ephemout['elevation'].plot(label='ephem')
plt.legend(ncol=2)
plt.title('elevation')

plt.figure()
pyephemout['azimuth'].plot(label='pyephem')
ephemout['azimuth'].plot(label='ephem')
plt.legend(ncol=2)
plt.title('azimuth')

print('pyephem')
print(pyephemout.head())
print('ephem')
print(ephemout.head())

loc = berlin
times = pd.DatetimeIndex(start=datetime.date(2015,6,28), end=datetime.date(2015,6,29), freq='5min')

pyephemout = pvlib.solarposition.pyephem(times, loc)
ephemout = pvlib.solarposition.ephemeris(times, loc)

pyephemout['elevation'].plot(label='pyephem')
pyephemout['apparent_elevation'].plot(label='pyephem apparent')
ephemout['elevation'].plot(label='ephem')
plt.legend(ncol=2)
plt.title('elevation')

plt.figure()
pyephemout['azimuth'].plot(label='pyephem')
ephemout['azimuth'].plot(label='ephem')
plt.legend(ncol=2)
plt.title('azimuth')

print('pyephem')
print(pyephemout.head())
print('ephem')
print(ephemout.head())


# ## Speed tests
get_ipython().run_cell_magic('timeit', '', '\npyephemout = pvlib.solarposition.pyephem(times, loc)\n#ephemout = pvlib.solarposition.ephemeris(times, loc)')

get_ipython().run_cell_magic('timeit', '', '\n#pyephemout = pvlib.solarposition.pyephem(times, loc)\nephemout = pvlib.solarposition.ephemeris(times, loc)')

get_ipython().run_cell_magic('timeit', '', "\n#pyephemout = pvlib.solarposition.pyephem(times, loc)\nephemout = pvlib.solarposition.get_solarposition(times, loc, method='nrel_numpy')")


# This numba test will only work properly if you have installed numba. 
get_ipython().run_cell_magic('timeit', '', "\n#pyephemout = pvlib.solarposition.pyephem(times, loc)\nephemout = pvlib.solarposition.get_solarposition(times, loc, method='nrel_numba')")


# The numba calculation takes a long time the first time that it's run because it uses LLVM to compile the Python code to machine code. After that it's about 4-10 times faster depending on your machine. You can pass a ``numthreads`` argument to this function. The optimum ``numthreads`` depends on your machine and is equal to 4 by default.
get_ipython().run_cell_magic('timeit', '', "\n#pyephemout = pvlib.solarposition.pyephem(times, loc)\nephemout = pvlib.solarposition.get_solarposition(times, loc, method='nrel_numba', numthreads=16)")

get_ipython().run_cell_magic('timeit', '', "\nephemout = pvlib.solarposition.spa_python(times, loc, how='numba', numthreads=16)")



