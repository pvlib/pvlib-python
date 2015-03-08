README
------

NREL provides a C implementation of the solar position algorithm 
described in 
[Reda, I.; Andreas, A. (2003). Solar Position Algorithm for Solar Radiation Applications. 55 pp.; NREL Report No. TP-560-34302](
http://www.nrel.gov/docs/fy08osti/34302.pdf).

This folder contains the files required to make NREL's C code accessible
to the ``pvlib-python`` package. We use the Cython package to wrap NREL's SPA 
implementation. 

** Due to licensing issues, you must download the NREL C files from their 
[website](http://www.nrel.gov/midc/spa) **

Download the ``spa.c`` and ``spa.h`` files from NREL, 
and copy them into the ``pvlib/spa_c_files`` directory. 

There are a total of 5 files needed to compile the C code, described below:

* ``spa.c``: original C code from NREL 
* ``spa.h``: header file for spa.c
* ``cspa_py.pxd``: a cython header file which essentially tells cython which parts of the main header file to pay attention to
* ``spa_py.pyx``: the cython code used to define both functions in the python namespace. NOTE: It is possible to provide user access to other paramters of the SPA algorithm through modifying this file 
* ``setup.py``: a distutils file which performs the compiling of the cython code

The cython compilation process produces two files:
* ``spa_py.c``: an intermediate cython c file
* ``spa_py.so``: the python module which can be imported into a namespace

To process the original 5 files, 
use the following shell command inside this folder
 
 $ python setup.py build_ext --inplace
