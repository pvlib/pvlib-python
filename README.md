pythonic-PVLIB
============

The PV_LIB Toolbox provides a set of well-documented functions for simulating the performance of photovoltaic energy systems. The toolbox was originally developed at Sandia National Laboratories and it implements many of the models and methods developed at the Labs. The libary was originally developed for MATLAB, and then ported to python.

This fork of the first python port was made to refactor the code to be more pythonic. Pythonic code is easier to use, easier to maintain, and faster to develop. Adhering to PEP8 guidelines allows other python developers read code more quickly and accurately. The downside of our approach is that users of the MATLAB PVLIB will need to do more work to port their applications to python. 


Key differences
============
This refactoring is still a work in progress, but some of the major differences so far include:

* Replace Location "struct" with namedtuple.
* Return one DataFrame instead of a tuple of DataFrames.
* Specify time zones using the standard IANA Time Zone Database naming conventions instead of an integer GMT offset. 
* Add PyEphem option to solar position calculations. 
* Consolidation of similar modules. For example, functions from pvl\_clearsky\_ineichen.py and pvl\_clearsky\_haurwitz.py have been consolidated into clearsky.py. Similar consolidations have occured for airmass and solar position modules. POA modules are probably next.
* Removing Vars=Locals(); Expect...; var=pvl_tools.Parse(Vars,Expect); pattern. 
* Removing unnecssary and sometimes undesired behavior such as setting zenith=90 instead of allowing it to be negative.
* \_\_init\_\_.py imports have been removed.
* Added logging calls
* Code in reviewed modules is mostly PEP8 compliant.
* Changing function names so that do not conflict with module names.


Quick Start
============
Clone this library using your favorite git tool.
Add the pythonic-PVLIB path to PYTHONPATH, or use sys.append().

```
# built-in imports
import sys
from collections import namedtuple
import datetime

# add-on imports
import pandas as pd

# pvlib imports
sys.path.append('/home/will/pythonic-PVLIB/')
import pvlib.solarposition
import pvlib.clearsky

# make a location
Location = namedtuple('Location', ['latitude', 'longitude', 'altitude', 'tz'])
tus = Location(32.2, -111, 700, 'US/Arizona')

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
logging.getLogger('pvlib').setLevel(logging.DEBUG)
```
