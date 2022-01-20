.. _comparison_pvlib_matlab:

****************************
Comparison with PVLIB MATLAB
****************************

PVLIB was originally developed as a library for MATLAB at Sandia
National Lab, and Sandia remains the official maintainer of the MATLAB
library. Sandia supported the initial Python port and
then released further project maintenance and development to the
`pvlib-python maintainers <https://github.com/orgs/pvlib/people>`_.

The pvlib-python maintainers collaborate with the PVLIB MATLAB
maintainers but operate independently. We'd all like to keep the core
functionality of the Python and MATLAB projects synchronized, but this
will require the efforts of the larger pvlib-python community, not just
the maintainers. Therefore, do not expect feature parity between the
libaries, only similarity.

The `PV_LIB Matlab help webpage <https://pvpmc.sandia.gov/PVLIB_Matlab_Help/>`_
is a good reference for this comparison.

Missing functions
~~~~~~~~~~~~~~~~~

See pvlib-python GitHub `issue #2
<https://github.com/pvlib/pvlib-python/issues/2>`_ for a list of
functions missing from the Python version of the library.

Major differences
~~~~~~~~~~~~~~~~~

* pvlib-python uses git version control to track all changes
  to the code. A summary of changes is included in the whatsnew file
  for each release. PVLIB MATLAB documents changes in Changelog.docx
* pvlib-python has a comprehensive test suite, whereas PVLIB MATLAB does
  not have a test suite at all. Specifically, pvlib-python

    * Uses TravisCI for automated testing on Linux.
    * Uses Appveyor for automated testing on Windows.
    * Uses Coveralls to measure test coverage.

* Using readthedocs for automated documentation building and hosting.
* Removed ``pvl_`` from module/function names.
* Consolidated similar functions into topical modules.
  For example, functions from ``pvl_clearsky_ineichen.m`` and
  ``pvl_clearsky_haurwitz.m`` have been consolidated into ``clearsky.py``.
* PVLIB MATLAB uses ``location`` structs as the input to some functions.
  pvlib-python just uses the lat, lon, etc. as inputs to the functions.
  Furthermore, pvlib-python replaces the structs with classes, and these classes
  have methods, such as :py:func:`~pvlib.location.Location.get_solarposition`,
  that automatically reference the appropriate data.
  See :ref:`modeling-paradigms` for more information.
* pvlib-python implements a handful of class designed to simplify the
  PV modeling process. These include :py:class:`~pvlib.location.Location`,
  :py:class:`~pvlib.pvsystem.PVSystem`,
  :py:class:`~pvlib.tracking.SingleAxisTracker`,
  and
  :py:class:`~pvlib.modelchain.ModelChain`.

Other differences
~~~~~~~~~~~~~~~~~

* Very few tests of input validitity exist in the Python code.
  We believe that the vast majority of these tests were not necessary.
  We also make use of Python's robust support for raising and catching
  exceptions.
* Removed unnecessary and sometimes undesired behavior such as setting
  maximum zenith=90 or airmass=0. Instead, we make extensive use of
  ``nan`` values in returned arrays.
* Implemented the NREL solar position calculation algorithm.
  Also added a PyEphem option to solar position calculations.
* Specify time zones using a string from the standard IANA Time Zone
  Database naming conventions or using a pytz.timezone instead of an
  integer GMT offset.
* ``clearsky.ineichen`` supports interpolating monthly
  Linke Turbidities to daily resolution.
* Instead of requiring effective irradiance as an input, ``pvsystem.sapm``
  calculates and returns it based on input POA irradiance, AM, and AOI.
* pvlib-python does not come with as much example data.
* pvlib-python does not currently implement as many algorithms as
  PVLIB MATLAB.

Documentation
~~~~~~~~~~~~~

* Using Sphinx to build the documentation,
  including dynamically created inline examples.
* Additional Jupyter tutorials in ``/docs/tutorials``.
