pvlib-python
============

[![TravisCI](https://travis-ci.org/pvlib/pvlib-python.svg?branch=master)](https://travis-ci.org/pvlib/pvlib-python)
[![Build status](https://ci.appveyor.com/api/projects/status/gr2eyhc84tvtkopk?svg=true)](https://ci.appveyor.com/project/wholmgren/pvlib-python-fv2to)
[![Coverage Status](https://img.shields.io/coveralls/pvlib/pvlib-python.svg)](https://coveralls.io/r/pvlib/pvlib-python)
[![Documentation Status](https://readthedocs.org/projects/pvlib-python/badge/?version=latest)](http://pvlib-python.readthedocs.org/en/latest/)
[![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.50141.svg)](http://dx.doi.org/10.5281/zenodo.50141)


pvlib-python is a community supported tool that provides a set of documented functions for simulating the performance of photovoltaic energy systems. The toolbox was originally developed in MATLAB at Sandia National Laboratories and it implements many of the models and methods developed at the Labs. More information on Sandia Labs PV performance modeling programs can be found at https://pvpmc.sandia.gov/. We collaborate with the PVLIB-MATLAB project, but operate independently of it.


Documentation
=============

Full documentation can be found at [readthedocs](http://pvlib-python.readthedocs.org/en/latest/).


Contributing
============

We need your help to make pvlib-python a great tool! Please see the [Contributing page](http://pvlib-python.readthedocs.org/en/latest/contributing.html) for more on how you can contribute. The long-term success of pvlib-python requires substantial community support.


Installation
============

To obtain the most recent stable release, just use ``pip`` or ``conda``:

```
pip install pvlib
```

```
conda install -c pvlib pvlib
```

Please see the [Installation page](http://pvlib-python.readthedocs.org/en/latest/installation.html) of the documentation for complete instructions.


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
You're now ready to start some version of the Python interpreter and use pvlib. The easiest way to start is with one of our [Jupyter](http://jupyter.org) notebook tutorials:

1. Use the nbviewer website to choose a tutorial to experiment with. Go to our [nbviewer tutorial page](http://nbviewer.jupyter.org/github/pvlib/pvlib-python/tree/master/docs/tutorials/), click on e.g. [``tmy_to_power.ipynb``](http://nbviewer.jupyter.org/github/pvlib/pvlib-python/blob/master/docs/tutorials/tmy_to_power.ipynb), and then click on the download symbol.
1. Start the Jupyter Notebook server: ``jupyter notebook``. This should open a web browser with the Jupyter Notebook's file/folder listing. If not, navigate to the url shown in the command line history, likely ``http://localhost:8888``
2. In Jupyter Notebook, navigate to the file that you downloaded in step one and open it.
2. Use ``shift-enter`` to execute the notebook cell-by-cell. There is also a Play button that will execute all of the cells in the notebook.

You can also use a Jupyter notebook or any other Python interpreter to experiment with the simple code in the [Package overview](http://pvlib-python.readthedocs.org/en/latest/package_overview.html) section of the documentation.

Many good online resources exist for getting started with scientific Python.


License
=======
3 clause BSD.


Compatibility
=============

pvlib-python is compatible with Python versions 2.7, 3.3, 3.4, 3.5 and Pandas versions 0.13.1 through 0.18. Note that our Numba-accelerated solar position algorithms have more specific version requirements that will be resolved by the Numba installer.

For Linux + Python 3 users: Continuum's Python 3.x SciPy conda package is not compiled properly and has a few bugs related to complex arithmetic. The most common place for these bugs to show up when using pvlib-python is in calculating IV curve parameters using the ``singlediode`` function. We reported [the issue](https://github.com/ContinuumIO/anaconda-issues/issues/425) to Continuum and are waiting for it to be fixed. In the meantime, you can compile your own SciPy distribution, or you can use this trick on Python 3.3 and 3.4 (not 3.5): Downgrade your NumPy to 1.8 and SciPy to 0.14, then install whatever version of pandas you want but without dependencies. The conda commands for this are:

```
conda install numpy=1.8 scipy=0.14
conda install pandas --no-deps
```

For Windows + Python 2.7 users: Continuum's Python 2.7 SciPy 0.16.1 and 0.17.0 packages are not compiled properly and will crash your Python interpreter if you use our Linke turbidity lookup function. See [Anaconda issue 650](https://github.com/ContinuumIO/anaconda-issues/issues/650) for more.


Testing
=======
Testing can easily be accomplished by running ``nosetests`` on the pvlib directory:
```
nosetests -v pvlib
```
Unit test code should be placed in the corresponding test module in the pvlib/test directory. Use ``pip`` or ``conda`` to install ``nose``. Developers must include comprehensive tests for any additions or modifications to pvlib.
