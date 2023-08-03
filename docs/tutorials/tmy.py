"""
TMY tutorial
============
This tutorial shows how to use the :py:mod:`pvlib.tmy` module to read data
from TMY2 and TMY3 files.
"""

# %%
# This tutorial has been tested against the following package versions:
#
# - pvlib 0.3.0
# - Python 3.5.1
# - IPython 4.1
# - pandas 0.18.0
#
# Authors:
#
# - Will Holmgren (@wholmgren), University of Arizona. July 2014, July 2015,
#   March 2016.
#
# Import modules
# --------------

import os
import inspect

import pvlib


# %%
# pvlib comes packaged with a TMY2 and a TMY3 data file.
# TODO: :pull:`1763`

# Find the absolute file path to your pvlib installation
pvlib_abspath = os.path.dirname(os.path.abspath(inspect.getfile(pvlib)))


# %%
# Import the TMY data using the functions in the ``pvlib.iotools`` module.

tmy3_data, tmy3_metadata = pvlib.iotools.read_tmy3(
    os.path.join(pvlib_abspath, "data", "703165TY.csv")
)
tmy2_data, tmy2_metadata = pvlib.iotools.read_tmy2(
    os.path.join(pvlib_abspath, "data", "12839.tm2")
)


# %%
# Print the TMY3 metadata and the first 5 lines of the data.

print(tmy3_metadata)
tmy3_data.head(5)


# %%
tmy3_data["GHI"].plot()


# %%
# The TMY readers have an optional argument to coerce the year to a single
# value.

tmy3_data, tmy3_metadata = pvlib.iotools.read_tmy3(
    os.path.join(pvlib_abspath, "data", "703165TY.csv"), coerce_year=1987
)


# %%
tmy3_data["GHI"].plot()


# %%
# Here's the TMY2 data.

print(tmy2_metadata)
print(tmy2_data.head())
