pvlib-python
============

[![TravisCI](https://travis-ci.org/pvlib/pvlib-python.svg?branch=master)](https://travis-ci.org/pvlib/pvlib-python)
[![Coverage Status](https://img.shields.io/coveralls/pvlib/pvlib-python.svg)](https://coveralls.io/r/pvlib/pvlib-python)
[![Documentation Status](https://readthedocs.org/projects/pvlib-python/badge/?version=latest)](http://pvlib-python.readthedocs.org/en/latest/)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.20562.svg)](http://dx.doi.org/10.5281/zenodo.20562)


pvlib-python is a community supported tool that provides a set of documented functions for simulating the performance of photovoltaic energy systems. The toolbox was originally developed in MATLAB at Sandia National Laboratories and it implements many of the models and methods developed at the Labs. More information on Sandia Labs PV performance modeling programs can be found at https://pvpmc.sandia.gov/. We collaborate with the PVLIB-MATLAB project, but operate independently of it.


Documentation
=============

Full documentation can be found at [readthedocs](http://pvlib-python.readthedocs.org/en/latest/).


Contributing
============

We need your help to make pvlib-python a great tool! Please see the [Contributing to pvlib-python wiki](https://github.com/pvlib/pvlib-python/wiki/Contributing-to-pvlib-python) for more on how you can contribute. The long-term success of pvlib-python requires substantial community support.


Installation
============

If you have Python
------------------
To obtain the most recent stable release, just use ``pip`` or ``conda``:

```
pip install pvlib
```

```
conda install -c http://conda.anaconda.org/pvlib pvlib
```

If your system complains that you don't have access privileges or asks for a password then you're trying to install pvlib into your system's Python distribution. This is a very bad idea and you should instead follow the **If you don't have Python** instructions below.


If you don't have Python
------------------------
There are many ways to install Python on your system, but the Anaconda Scientific Python distribution provides by far the easiest way for new users to get started. Anaconda includes all of the popular libraries that you'll need for pvlib, including Pandas, NumPy, and SciPy. "Anaconda installs cleanly into a single directory, does not require Administrator or root privileges, does not affect other Python installs on your system, or interfere with OSX Frameworks." -Anaconda Documentation.

1. Install the full Anaconda Scientific Python distribution available [here](https://store.continuum.io/cshop/anaconda/). 

2. Install pvlib: ``conda install -c http://conda.anaconda.org/pvlib pvlib``

If you have trouble, see the [Anaconda FAQ](http://docs.continuum.io/anaconda/faq.html), Google your error messages, or make a new issue on our [Issues page](https://github.com/pvlib/pvlib-python/issues).


Working at the bleeding edge
----------------------------
We strongly recommend working in a **virtual environment** if you're going to use the development versions of the code. There are many ways to use virtual environments in Python, but Anaconda again provides the easiest solution:

1. Create a new conda environment for pvlib and pre-install a handful of packages into the environment: ``conda create --name pvlibdev python pandas scipy ephem``
2. Activate the new environment: ``source activate pvlibdev``
2. Install the latest development version:
  2. If you don't plan to modify the source-code: ``pip install git+https://github.com/pvlib/pvlib-python.git``
  2. If you do plan to modify the source code: Use the GitHub GUI application or git command-line tool to clone this repository to your computer, then navigate your command-line to the top-level pvlib-python directory, then ``pip install -e .``
2. You may also consider installing additional packages into your development environment: ``conda install ipython-notebook nose seaborn``

The [conda documentation](http://conda.pydata.org/docs/using/index.html) has more information on how to use virtual environments.


NREL SPA algorithm
------------------
pvlib-python is distributed with several validated, high-precision, and high-performance solar position calculators.
It also includes wrappers for the official NREL SPA algorithm.
To use the NREL SPA algorithm, a pip install from the web cannot be used. Instead: 

1. Download the pvlib repository from https://github.com/pvlib/pvlib-python.git
2. Download the SPA files from [NREL](http://www.nrel.gov/midc/spa/)
3. Copy the SPA files into ``pvlib-python/pvlib/spa_c_files`` 
4. From the ``pvlib-python`` directory, run ``pip uninstall pvlib`` followed by ``pip install . ``


Usage
=====
You're now ready to start some version of the Python interpreter and use pvlib. The easiest way to start is with one of our IPython notebook tutorials:

1. Use the nbviewer website to choose a tutorial to experiment with. Go to our [nbviewer tutorial page](http://nbviewer.ipython.org/github/pvlib/pvlib-python/tree/master/docs/tutorials/), click on e.g. pvsystem.ipynb, and then click on the download symbol.
1. Start the IPython Notebook server: ``ipython notebook``. This should open a web browser with the IPython Notebook's file/folder listing. If not, navigate to the url shown in the command line history, likely ``http://localhost:8888``
2. In IPython Notebook, navigate to the file that you downloaded in step one and open it.
2. Use ``shift-enter`` to execute the notebook cell-by-cell. There is also a Play button that will execute all of the cells in the notebook.

You can also experiment with the following simple code in a new IPython notebook or any other Python interpreter:

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
solpos = pvlib.solarposition.get_solarposition(times, tus)
solpos.plot()

# calculate clear sky data
tus_cs = pvlib.clearsky.ineichen(times, tus, airmass_model='young1994')
tus_cs.plot()
```

Many good online resources exist for getting started with scientific Python.


License
=======
3 clause BSD.


Compatibility
=============

pvlib-python is compatible with Python versions 2.7, 3.3, and 3.4, and pandas versions 0.13.1 through 0.16.2.

For Linux + Python 3 users: The combination of Linux, Python 3, NumPy 1.9, and SciPy 0.15 has some bugs. The most common place for these bugs to show up when using pvlib-python is in calculating IV curve parameters using the ``singlediode`` function. Downgrade your NumPy to 1.8 and SciPy to 0.14, then install whatever version of pandas you want but without dependencies. The conda commands for this are:

```
conda install numpy=1.8 scipy=0.14
conda install pandas --no-deps
```


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
