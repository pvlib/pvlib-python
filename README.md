pvlib-python
============

[![TravisCI](https://travis-ci.org/pvlib/pvlib-python.svg?branch=master)](https://travis-ci.org/pvlib/pvlib-python)
[![Coverage Status](https://img.shields.io/coveralls/pvlib/pvlib-python.svg)](https://coveralls.io/r/pvlib/pvlib-python)
[![Documentation Status](https://readthedocs.org/projects/pvlib-python/badge/?version=latest)](http://pvlib-python.readthedocs.org/en/latest/)


pvlib-python provides a set of documented functions for simulating the performance of photovoltaic energy systems. The toolbox was originally developed in MATLAB at Sandia National Laboratories and it implements many of the models and methods developed at the Labs. More information on Sandia Labs PV performance modelling programs can be found at https://pvpmc.sandia.gov/. 

Documentation
=============

Full documentation can be found at [readthedocs](http://pvlib-python.readthedocs.org/en/latest/).

Development
===========

We need your help to make pvlib-python a great tool! Please see the [Development information wiki](https://github.com/pvlib/pvlib-python/wiki/Development-information) for more on how you can contribute.

Quick Start
===========

Installation
------------
Hopefully you're using [virtualenv](http://virtualenv.readthedocs.org/en/latest/) and [virtualenvwrapper](http://virtualenvwrapper.readthedocs.org). To install, run

```
pip install git+https://github.com/pvlib/pvlib-python.git
```

Alternatively, ``git clone`` this repository, ``cd`` into it, and run

```
pip install .
```

Add ``-e`` to install in [develop mode](http://pip.readthedocs.org/en/latest/reference/pip_install.html#editable-installs).

To use the NREL SPA algorithm, a pip install from the web cannot be used. Instead: 

1. Download the pvlib repository from https://github.com/pvlib/pvlib-python.git
2. Download the SPA files from [NREL](http://www.nrel.gov/midc/spa/)
3. Copy the SPA files into ``pvlib-python/pvlib/spa_c_files`` 
4. From the ``pvlib-python`` directory, run ``pip uninstall pvlib`` followed by ``pip install . ``


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

License
=======
3 clause BSD.


Testing
============
First, make sure the package is installed in develop mode or run ``python setup.py build_ext --inplace`` to properly compile the ``spa_py.c`` code. Testing can be accomplished by running nosetests on the pvlib directory (or pvlib/tests):
```
nosetests -v pvlib
```
Unit test code should be placed in the ``pvlib/test`` directory. Each module should have its own test module. 


Compatibility
=============

pvlib-python is compatible with Python versions 2.7, 3.3, and 3.4, and pandas versions 0.13.1 through 0.16.


Code Transition
================
Here are some of the major differences between the latest build and the original  Sandia PVLIB\_Python project. 

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
