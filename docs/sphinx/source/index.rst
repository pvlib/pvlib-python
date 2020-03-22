.. image:: _images/pvlib_logo_horiz.png
  :width: 600

pvlib python
============

pvlib python is a community supported tool that provides a set of
functions and classes for simulating the performance of photovoltaic
energy systems. pvlib python was originally ported from the PVLIB MATLAB
toolbox developed at Sandia National Laboratories and it implements many
of the models and methods developed at the Labs. More information on
Sandia Labs PV performance modeling programs can be found at
https://pvpmc.sandia.gov/. We collaborate with the PVLIB MATLAB project,
but operate independently of it.

The source code for pvlib python is hosted on `github
<https://github.com/pvlib/pvlib-python>`_.

Please see the :ref:`installation` page for installation help.

For examples of how to use pvlib python, please see
:ref:`package_overview` and our `Jupyter Notebook tutorials
<http://nbviewer.ipython.org/github/pvlib/pvlib-python/tree/master/docs/
tutorials/>`_. The documentation assumes general familiarity with
Python, NumPy, and Pandas. Google searches will yield many
excellent tutorials for these packages.

The pvlib python GitHub wiki has a `Projects and publications that use
pvlib python
<https://github.com/pvlib/pvlib-python/wiki/Projects-and-publications-
that-use-pvlib-python>`_ page for inspiration and listing of your
application.

There is a :ref:`variable naming convention <variables_style_rules>` to
ensure consistency throughout the library.

Citing pvlib python
===================

Many of the contributors to pvlib-python work in institutions where
citation metrics are used in performance or career evaluations. If you
use pvlib python in a published work, please cite:

  William F. Holmgren, Clifford W. Hansen, and Mark A. Mikofski.
  "pvlib python: a python package for modeling solar energy systems."
  Journal of Open Source Software, 3(29), 884, (2018).
  https://doi.org/10.21105/joss.00884

Please also cite the DOI corresponding to the specific version of
pvlib python that you used. pvlib python DOIs are listed at
`Zenodo.org <https://zenodo.org/search?page=1&size=20&q=conceptrecid:593284&all_versions&sort=-version>`_

Additional pvlib python publications include:

* J. S. Stein, “The photovoltaic performance modeling
  collaborative (PVPMC),” in Photovoltaic Specialists Conference, 2012.
* R.W. Andrews, J.S. Stein, C. Hansen, and D. Riley, “Introduction
  to the open source pvlib for python photovoltaic system
  modelling package,” in 40th IEEE Photovoltaic Specialist
  Conference, 2014.
  (`paper
  <http://energy.sandia.gov/wp/wp-content/gallery/uploads/PV_LIB_Python_final_SAND2014-18444C.pdf>`__)
* W.F. Holmgren, R.W. Andrews, A.T. Lorenzo, and J.S. Stein,
  “PVLIB Python 2015,” in 42nd Photovoltaic Specialists Conference, 2015.
  (`paper
  <https://github.com/pvlib/pvsc2015/blob/master/pvlib_pvsc_42.pdf>`__ and
  the `notebook to reproduce the figures
  <http://nbviewer.ipython.org/github/pvlib/pvsc2015/blob/master/paper.ipynb>`_)
* J.S. Stein, W.F. Holmgren, J. Forbess, and C.W. Hansen,
  "PVLIB: Open Source Photovoltaic Performance Modeling Functions
  for Matlab and Python," in 43rd Photovoltaic Specialists Conference, 2016.
* W.F. Holmgren and D.G. Groenendyk,
  "An Open Source Solar Power Forecasting Tool Using PVLIB-Python,"
  in 43rd Photovoltaic Specialists Conference, 2016.


NumFOCUS
========

pvlib python is a `NumFOCUS Affiliated Project <https://numfocus.org/sponsored-projects/affiliated-projects>`_

.. image:: https://i0.wp.com/numfocus.org/wp-content/uploads/2019/06/AffiliatedProject.png
  :target: https://numfocus.org/sponsored-projects/affiliated-projects
  :alt: NumFocus Affliated Projects


Contents
========

.. toctree::
   :maxdepth: 1

   package_overview
   introtutorial
   auto_examples/index
   whatsnew
   installation
   contributing
   pvsystem
   modelchain
   timetimezones
   clearsky
   forecasts
   api
   comparison_pvlib_matlab
   variables_style_rules
   singlediode


Indices and tables
==================

* :ref:`genindex`
* :ref:`search`
