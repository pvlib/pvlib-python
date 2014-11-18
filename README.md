PVLIB_Python
============

[![TravisCI](https://travis-ci.org/UARENForecasting/PVLIB_Python.svg)](https://travis-ci.org/UARENForecasting/PVLIB_Python)
[![Coverage Status](https://img.shields.io/coveralls/UARENForecasting/PVLIB_Python.svg)](https://coveralls.io/r/UARENForecasting/PVLIB_Python)
[![Documentation Status](https://readthedocs.org/projects/uarenforecasting-pvlib-python/badge/?version=latest)](http://uarenforecasting-pvlib-python.readthedocs.org/en/latest/)

This repo is a fork of the [Sandia PVLIB_Python](https://github.com/Sandia-Labs/PVLIB_Python) project.

It provides a set of documented functions for simulating the performance of photovoltaic energy systems. The toolbox was originally developed in MATLAB at Sandia National Laboratories and it implements many of the models and methods developed at the Labs. 

We make some contributions to the Sandia develop branch, and we pull some commits back into our fork. We hope that the Sandia PVLIB_Python becomes the community standard, but we had some different ideas about how things should be done and we needed to make progress faster than the collaborative cycle would allow. See below for a partial list of differences. 

We use this library to generate solar power forecasts for TEP, APS, and other SVERI utilities, and to perform grid integration and variability studies for SVERI. For more information, see [https://forecasting.uaren.org](https://forecasting.uaren.org) and [https://sveri.uaren.org](https://sveri.uaren.org).

The primary drawback to using this library over the official library is that, well, it's not the official Sandia library. Another drawback is that the structure of this library will look a lot different to people coming from the MATLAB world, which is either a good thing or a bad thing depending on your perspective. Keep the following in mind as you consider using or contributing our fork:

* We hope to keep the projects as similar as possible to make it easier for people to experiment with our fork. 
* The Sandia repo should be the default repo for the user community.
* Developers should strongly consider contributing to the official Sandia repo rather than, or in addition to, our fork.
* Reread the above point (...waiting...) before the following (...waiting again...): community contributions in the form of PRs, issues, wikis, docs, tutorials, thoughts, etc are all welcomed and we will try hard to address them in a timely manner.
* All code contributions must be documented, tested, PEP8 compliant, and python 3 compatible. Pythonic code is easier to use, easier to maintain, and faster to develop. Adhering to PEP8 guidelines allows other python developers read code more quickly and accurately. 

That being said, we welcome your thoughts and contributions to our fork.


Code differences
================
Here are some of the major differences between our fork and the official python project. Note that some of these differences have been resolved in the Sandia [develop branch](https://github.com/Sandia-Labs/PVLIB_Python/tree/develop). We have not attempted to catalog the differences with the MATLAB code.

Library wide changes:
* Remove ``pvl_`` from module names.
* Consolidation of similar modules. For example, functions from ``pvl_clearsky_ineichen.py`` and ``pvl_clearsky_haurwitz.py`` have been consolidated into ``clearsky.py``. 
* Removed ``Vars=Locals(); Expect...; var=pvl\_tools.Parse(Vars,Expect);`` pattern. Very few tests of input validitity remain. Garbage in, garbage or ``nan`` out.
* Removing unnecssary and sometimes undesired behavior such as setting maximum zenith=90 or airmass=0. Instead, we make extensive use of ``nan`` values.
* Adding logging calls, removing print calls.
* Code in reviewed modules is mostly PEP8 compliant.
* All code is Python 3 compatible (see testing).
* Changing function and module names so that they do not conflict.
* Added ``/pvlib/data`` for lookup tables, test, and tutorial data.
* Return one DataFrame instead of a tuple of DataFrames.

More specific changes:
* Add PyEphem option to solar position calculations. 
* ``irradiance.py`` has more AOI, projection, and irradiance sum and calculation functions
* TMY data is not forced to 1987.
* Locations are now ``pvlib.location.Location`` objects, not structs.
* Specify time zones using a string from the standard IANA Time Zone Database naming conventions or using a pytz.timezone instead of an integer GMT offset. We may add dateutils support in the future.
* ``clearsky.ineichen`` supports interpolating monthly Linke Turbidities to daily resolution.

Documentation:
* Using readthedocs for documentation hosting.
* Many typos and formatting errors corrected.
* Documentation source code and tutorials live in ``/`` rather than ``/pvlib/docs``.
* Additional tutorials in ``/docs/tutorials``.

Testing:
* Tests are cleaner and more thorough. They are still no where near complete.
* Using Coveralls to measure test coverage. 
* Using TravisCI for automated testing.
* Using ``nosetests`` for more concise test code. 


Quick Start
===========

Installation
------------
Hopefully you're using [virtualenv](http://virtualenv.readthedocs.org/en/latest/) and [virtualenvwrapper](http://virtualenvwrapper.readthedocs.org). To install, run

```
pip install git+https://github.com/UARENForecasting/PVLIB_Python.git
```

Alternatively, ``git clone`` this repository and run

```
pip install .
```

Add ``-e`` to install in [develop mode](http://pip.readthedocs.org/en/latest/reference/pip_install.html#editable-installs).

Usage Example
-------------
```
# built-in imports
import sys
import datetime

# add-on imports
import pandas as pd

# pvlib imports
from pvlib.location import Location
import pvlib.solarposition
import pvlib.clearsky

# make a location
tus = Location(32.2, -111, 'MST', 700)

# make a pandas DatetimeIndex for some day
times = pd.date_range(start=datetime.datetime(2014,6,24), end=datetime.datetime(2014,6,25), freq='1Min')

# calculate the solar position
solpos = pvlib.solarposition.get_solarposition(times, tus, method='pyephem')
solpos.plot()

# calculate clear sky data
tus_cs = pvlib.clearsky.ineichen(times, tus, airmass_model='young1994')
tus_cs.plot()
```

Until the code is tested more thoroughly, you might find it useful to add:
```
import logging
logging.getLogger('pvlib').setLevel(logging.DEBUG) # or at least INFO
```


Testing
============
First, make sure the package is installed in develop mode or run ``python setup.py build_ext --inplace`` to properly compile the ``spa_py.c`` code. Testing can be accomplished by running nosetests on the pvlib directory (or pvlib/tests):
```
nosetests -v pvlib
```
Unit test code should be placed in the ``pvlib/test`` directory. Each module should have its own test module. 
