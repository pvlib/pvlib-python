Welcome to pvlib-python's documentation!
========================================

pvlib-python provides a set of documented functions for simulating 
the performance of photovoltaic energy systems. 
The toolbox was originally developed in MATLAB at 
Sandia National Laboratories and it implements many of the 
models and methods developed at the Labs. 
More information on Sandia Labs PV performance modeling programs 
can be found at https://pvpmc.sandia.gov/.

The source code for pvlib-python is hosted on 
`github <https://github.com/pvlib/pvlib-python>`_.

The github page also contains a valuable
`wiki <https://github.com/pvlib/pvlib-python/wiki>`_
with information on how you can contribute to pvlib-python development!

Please see the links above for details on the status of the pvlib-python
project. We are at an early stage in the development of this project,
so expect to see significant API changes in the next few releases.

This documentation focuses on providing a reference for all of 
the modules and functions available in pvlib-python.
For examples of how to use pvlib-python, please see the
`tutorials <http://nbviewer.ipython.org/github/pvlib/pvlib-python/tree/master/docs/tutorials/>`_.
Some of the tutorials were written with older
versions of pvlib-python and we would greatly appreciate your
help updating them!

.. note::

   This documentation assumes general familiarity with
   Python, NumPy, and Pandas. Google searches will yield many
   excellent tutorials for these packages. 
   
Please see our 
`PVSC 2014 paper <http://energy.sandia.gov/wp/wp-content/gallery/uploads/PV_LIB_Python_final_SAND2014-18444C.pdf>`_
and
`PVSC 2015 paper <https://github.com/pvlib/pvsc2015/blob/master/pvlib_pvsc_42.pdf>`_ 
(and the `notebook to reproduce the figures <http://nbviewer.ipython.org/github/pvlib/pvsc2015/blob/master/paper.ipynb>`_) for more information.

The GitHub wiki also has a page on `Projects and publications that use pvlib python <https://github.com/pvlib/pvlib-python/wiki/Projects-and-publications-that-use-pvlib-python>`_ for inspiration and listing of your application.

There is a :ref:`variable naming convention <variables_style_rules>` to ensure consistency throughout the library.

Installation
============

1. Follow Pandas' 
   `instructions <http://pandas.pydata.org/pandas-docs/stable/install.html>`_
   for installing the scientific python stack, including ``pip``.
#. ``pip install pvlib-python``


Contents
========

.. toctree::
   :maxdepth: 2
   
   self
   package_overview
   whatsnew
   usage
   modules
   classes
   comparison_pvlib_matlab
   variables_style_rules

   
Contents (potential new ToC)
============================

* What’s New
* Overview: Why pvlib?
    * Introduction
    * Modeling paradigms
    * User extensions
    * What it is not
    * Comparison with PVLIB_MATLAB
* Installation
* Examples
    * Using the principal components
        * atmosphere.py tutorial
        * solarposition.py tutorial
        * irradiance.py tutorial 
    * Specific datasets
        * TMY tutorial 
        * TMY data and diffuse irradiance model 
        * TMY to Power Tutorial 
    * System Modelling
        * pvsystem tutorial
        * Tracking
        * Modeling multiple systems with system losses
    * Validations
        * SAPM – Some simulations using the Sandia Array Performance Model.
        * IV curves – Make some IV curves based on this data
* Principal components
    * Modules
    * Classes
* API reference
* Support and Development
    * Getting support
    * How do I contribute?
* Frequently Asked Questions
* Citing and References
    * Citing pvlib
    * References used for the code
    * Publications and projects using pvlib
    * Credits


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

