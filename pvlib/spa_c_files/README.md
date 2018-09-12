README
------

NREL provides a C implementation of the solar position algorithm described in
[Reda, I.; Andreas, A. (2003). Solar Position Algorithm for Solar Radiation Applications. 55 pp.; NREL Report No. TP-560-34302](http://www.nrel.gov/docs/fy08osti/34302.pdf).

This folder contains the files required to make NREL's C code accessible
to the ``pvlib-python`` package. We use the Cython package to wrap NREL's SPA 
implementation. 

** Due to licensing issues, the [SPA C files](http://www.nrel.gov/midc/spa) can
_not_ be included in the pvlib-python distribution. The SPA C files will be
downloaded when you build the Python extension. By using this module you agree
to the NREL license in SPA_NOTICE. **

There are a total of 5 files needed to compile the C code, described below:

* ``spa.c``: original C code from NREL 
* ``spa.h``: header file for spa.c
* ``cspa_py.pxd``: a cython header file which essentially tells cython which
  parts of the main header file to pay attention to
* ``spa_py.pyx``: the cython code used to define both functions in the python
  namespace. NOTE: It is possible to provide user access to other paramters of
  the SPA algorithm through modifying this file 
* ``setup.py``: a distutils file which performs the compiling of the cython code

The cython compilation process produces two files:
* ``spa_py.c``: an intermediate cython c file
* ``spa_py.so`` or ``spa_py.<cpyver-platform>.pyd``: the python module which
  can be imported into a namespace

To create the SPA Python extension, use the following shell command inside this
folder:

    $ python setup.py build_ext --inplace

Executing the build command will also download the ``spa.c`` and ``spa.h``
files from NREL and copy them into the ``pvlib/spa_c_files`` directory.

This build process may not be not compatible with Python-2.7.

There are four optional keyword arguments `delta_ut1=0`, `slope=30.0`,
`azm_rotation=-10`, `atmos_refract` that effect four optional return values
`incidence`, `suntransit`, `sunrise`, and `sunset`. If not given, the defaults
shown are used.

There is an example in `spa_py_example.py` that contains a test function called
`spa_calc_example` that users can use to check that the result is consistent
with expected values:

    >>> from spa_py_example import spa_calc_example
    >>> r = spa_calc_example()
    {
        'year': 2004,
        'month': 10,
        'day': 17,
        'hour': 12,
        'minute': 30,
        'second': 30.0,
        'delta_ut1': 0.0,
        'delta_t': 67.0,
        'time_zone': -7.0,
        'longitude': -105.1786,
        'latitude': 39.742476,
        'elevation': 1830.14,
        'pressure': 820.0,
        'temperature': 11.0,
        'slope': 30.0,
        'azm_rotation': -10.0,
        'atmos_refract': 0.5667,
        'function': 3,
        'e0': 39.59209464796398,
        'e': 39.60858878898177,
        'zenith': 50.39141121101823,
        'azimuth_astro': 14.311961805946808,
        'azimuth': 194.3119618059468,
        'incidence': 25.42168493680471,
        'suntransit': 11.765833793714224,
        'sunrise': 6.22578372122376,
        'sunset': 17.320379610556166
    }
