pvlib-python
============

[![TravisCI](https://travis-ci.org/pvlib/pvlib-python.svg?branch=master)](https://travis-ci.org/pvlib/pvlib-python)
[![Coverage Status](https://img.shields.io/coveralls/pvlib/pvlib-python.svg)](https://coveralls.io/r/pvlib/pvlib-python)
[![Documentation Status](https://readthedocs.org/projects/pvlib-python/badge/?version=latest)](http://pvlib-python.readthedocs.org/en/latest/)


pvlib-python is a community supported tool that provides a set of documented functions for simulating the performance of photovoltaic energy systems. The toolbox was originally developed in MATLAB at Sandia National Laboratories and it implements many of the models and methods developed at the Labs. More information on Sandia Labs PV performance modeling programs can be found at https://pvpmc.sandia.gov/. We collaborate with the PVLIB-MATLAB project, but operate independently of it.

Documentation
=============

Full documentation can be found at [readthedocs](http://pvlib-python.readthedocs.org/en/latest/).


Contributing
============

We need your help to make pvlib-python a great tool! Please see the [Development information wiki](https://github.com/pvlib/pvlib-python/wiki/Development-information) for more on how you can contribute. The long-term success of pvlib-python requires community support.


Installation
============

If you have Python
------------------
To obtain the most recent stable release, just use ``pip``:

```
pip install pvlib-python
```


If you don't have Python
------------------------
The Anaconda Python distribution provides an easy way for new users to get started. Here's the short version:

1. Install the full Anaconda Python distribution available [here](). Anaconda includes all of the libraries that you'll need, including ``pandas``, ``numpy``, and ``scipy``.
2. Create a new ``conda`` environment for pvlib: ``conda create -n pvlib``
2. Activate the new environment: ``source activate pvlib``
2. Install pvlib: ``pip install pvlib-python``

You're now ready to start some version of the Python interpreter and use pvlib. The easiest way to start is with one of our IPython notebook tutorials:

1. Use the nbviewer website to choose a tutorial to experiment with. Go to [](), click on e.g. Tutorial.ipynb, and then click on the download symbol.
1. Start the IPython notebook: ``ipython notebook``. This should open a web browser with a file/folder listing. If not, navigate to ``http://localhost:8000``
2. In IPython Notebook, navigate to the file that you downloaded in step one and open it.
2. Use ``shift-enter`` to execute the notebook cell-by-cell. There is also a Play button that will execute all of the cells in the notebook.

Many good online resources exist for getting started with scientific Python. The [pandas tutorial]() is particularly good.


Working at the bleeding-edge
----------------------------
We strongly recommend working in a ``conda`` or ``virtualenv`` *virtual environment* (see the wiki or Google for more information). 
To install the very latest development versions, activate your new virtual environment, then run

```
pip install git+https://github.com/pvlib/pvlib-python.git
```


NREL SPA algorithm
------------------
pvlib-python is distributed with several validated, high-precision, and high-performance solar position calculators.
It also includes wrappers for the official NREL SPA algorithm.
To use the NREL SPA algorithm, a pip install from the web cannot be used. Instead: 

1. Download the pvlib repository from https://github.com/pvlib/pvlib-python.git
2. Download the SPA files from [NREL](http://www.nrel.gov/midc/spa/)
3. Copy the SPA files into ``pvlib-python/pvlib/spa_c_files`` 
4. From the ``pvlib-python`` directory, run ``pip uninstall pvlib`` followed by ``pip install . ``


Usage Example
=============
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


License
=======
3 clause BSD.


Compatibility
=============

pvlib-python is compatible with Python versions 2.7, 3.3, and 3.4, and pandas versions 0.13.1 through 0.16.2.


Testing
=======
Testing can easily be accomplished by running ``nosetests`` on the pvlib directory:
```
nosetests -v pvlib
```
Unit test code should be placed in the corresponding test module in the pvlib/test directory. Use ``pip`` or ``conda`` to install ``nose``. Developers must include comprehensive tests for any additions or modifications to pvlib.


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
