pythonic-PVLIB
============

The PV_LIB Toolbox provides a set of well-documented functions for simulating the performance of photovoltaic energy systems. The toolbox was originally developed at Sandia National Laboratories and it implements many of the models and methods developed at the Labs. The libary was originally developed for MATLAB, and then ported to python.

This fork of the first python port was made to refactor the code to be more pythonic. Pythonic code is easier to use, easier to maintain, and faster to develop. Adhering to PEP8 guidelines allows other python developers read code more quickly and accurately. The downside of our approach is that users of the MATLAB PVLIB will need to do more work to port their applications to python. 


Key differences
============
This refactoring is still a work in progress, but some of the major differences so far include:

* Remove pvl_ from module names.
* Locations are now referred to as objects, not structs.
* Return one DataFrame instead of a tuple of DataFrames.
* Specify time zones using a string from the standard IANA Time Zone Database naming conventions or using a pytz.timezone instead of an integer GMT offset. 
* Add PyEphem option to solar position calculations. 
* Consolidation of similar modules. For example, functions from ```pvl_clearsky_ineichen.py``` and ```pvl_clearsky_haurwitz.py``` have been consolidated into ```clearsky.py```. Similar consolidations have occured for airmass, solar position, and diffuse irradiance modules.
* Removing Vars=Locals(); Expect...; var=pvl_tools.Parse(Vars,Expect); pattern. Very few tests of input validitity remain. 
* Removing unnecssary and sometimes undesired behavior such as setting maximum zenith=90.
* ```__init__.py``` imports have been removed.
* Adding logging calls.
* Code in reviewed modules is mostly PEP8 compliant.
* Changing function names so that do not conflict with module names.
* Not bothering with boilerplate unit test code such as ```unittest.main()```. 
* Removing most wildcard imports.
* Improved documentation here and there.


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


Testing
============
Testing can be accomplished by running nosetests on the pvlib directory (or pvlib/tests):
```
nosetests -v pvlib
```
Unit test code should be placed in the ```pvlib/test``` directory. Each module should have its own test module. 

