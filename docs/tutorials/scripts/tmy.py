# coding: utf-8

# # TMY tutorial
# 
# This tutorial shows how to use the ``pvlib.tmy`` module to read data from TMY2 and TMY3 files.
# 
# This tutorial has been tested against the following package versions:
# * pvlib 0.2.1
# * Python 2.7.10
# * IPython 3.2
# * pandas 0.16.2
# 
# Authors:
# * Will Holmgren (@wholmgren), University of Arizona. July 2014, July 2015.

# ## Import modules
# built in python modules
import datetime
import os
import inspect

# python add-ons
import numpy as np
import pandas as pd

# plotting libraries
get_ipython().magic('matplotlib inline')
import matplotlib.pyplot as plt
try:
    import seaborn as sns
except ImportError:
    pass

import pvlib


# pvlib comes packaged with a TMY2 and a TMY3 data file.
# Find the absolute file path to your pvlib installation
pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))


# Import the TMY data using the functions in the ``pvlib.tmy`` module.
tmy3_data, tmy3_metadata = pvlib.tmy.readtmy3(os.path.join(pvlib_abspath, 'data', '703165TY.csv'))
tmy2_data, tmy2_metadata = pvlib.tmy.readtmy2(os.path.join(pvlib_abspath, 'data', '12839.tm2'))


# Print the TMY3 metadata and the first 5 lines of the data.
print(tmy3_metadata)
tmy3_data.head(5)

tmy3_data['GHI'].plot()


# The TMY readers have an optional argument to coerce the year to a single value.
tmy3_data, tmy3_metadata = pvlib.tmy.readtmy3(os.path.join(pvlib_abspath, 'data', '703165TY.csv'), coerce_year=1987)

tmy3_data['GHI'].plot()


# Here's the TMY2 data.
print(tmy2_metadata)
print(tmy2_data.head())


# Finally, the TMY readers can access TMY files directly from the NREL website.
tmy3_data, tmy3_metadata = pvlib.tmy.readtmy3('http://rredc.nrel.gov/solar/old_data/nsrdb/1991-2005/data/tmy3/722740TYA.CSV', coerce_year=2015)

tmy3_data['GHI'].plot(figsize=(12,6))
plt.title('Tucson TMY GHI')



