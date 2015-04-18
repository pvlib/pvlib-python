.. _comparison_pvlib_matlab:

****************************
Comparison with PVLIB_MATLAB
****************************

This document is under construction.
Please see our 
`PVSC 2014 paper <http://energy.sandia.gov/wp/wp-content/gallery/uploads/PV_LIB_Python_final_SAND2014-18444C.pdf>`_
and
`PVSC 2015 abstract <https://github.com/UARENForecasting/pvlib-pvsc2015/blob/master/pvlib_pvsc_42.pdf?raw=true>`_ 
for more information.

The pvlib-python license is BSD 3-clause,
the PVLIB\_MATLAB license is ??.

We want to keep developing the core functionality and algorithms 
of the Python and MATLAB projects roughly in parallel, 
but we're not making any promises at this point.
The PVLIB\_MATLAB and pvlib-python projects are currently developed 
by different teams that do not regularly work together. 
We hope to grow this collaboration in the future.
Do not expect feature parity between the libaries, only similarity.

Here are some of the major differences between the latest pvlib-python build 
and the original Sandia PVLIB\_Python project, but many of these
comments apply to the difference between pvlib-python and PVLIB\_MATLAB.


Library wide changes
~~~~~~~~~~~~~~~~~~~~

* Remove ``pvl_`` from module names.
* Consolidation of similar modules. For example, functions from ``pvl_clearsky_ineichen.py`` and ``pvl_clearsky_haurwitz.py`` have been consolidated into ``clearsky.py``. 
* Removed ``Vars=Locals(); Expect...; var=pvl\_tools.Parse(Vars,Expect);`` pattern. Very few tests of input validitity remain. Garbage in, garbage or ``nan`` out.
* Removing unnecssary and sometimes undesired behavior such as setting maximum zenith=90 or airmass=0. Instead, we make extensive use of ``nan`` values.
* Changing function and module names so that they do not conflict.
* Added ``/pvlib/data`` for lookup tables, test, and tutorial data.


More specific changes
~~~~~~~~~~~~~~~~~~~~~

* Add PyEphem option to solar position calculations. 
* ``irradiance.py`` has more AOI, projection, and irradiance sum and calculation functions
* Locations are now ``pvlib.location.Location`` objects, not structs.
* Specify time zones using a string from the standard IANA Time Zone Database naming conventions or using a pytz.timezone instead of an integer GMT offset. We may add dateutils support in the future.
* ``clearsky.ineichen`` supports interpolating monthly Linke Turbidities to daily resolution.
* Instead of requiring effective irradiance as an input, ``pvsystem.sapm``
  calculates and returns it based on input POA irradiance, AM, and AOI.

Documentation
~~~~~~~~~~~~~

* Using readthedocs for documentation hosting.
* Many typos and formatting errors corrected.
* Documentation source code and tutorials live in ``/`` rather than ``/pvlib/docs``.
* Additional tutorials in ``/docs/tutorials``.

Testing
~~~~~~~

* Tests are cleaner and more thorough. They are still no where near complete.
* Using Coveralls to measure test coverage. 
* Using TravisCI for automated testing.
* Using ``nosetests`` for more concise test code. 
