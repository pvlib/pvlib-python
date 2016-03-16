.. _installation:

Installation
============

Installing pvlib-python is similar to installing most other scientific
python packages. The instructions below describe how to install
pvlib-python under a few different conditions. Most of the information
on the `Pandas installation page
<http://pandas.pydata.org/pandas-docs/stable/install.html>`_ is also
applicable to pvlib-python.

If you have Python
------------------

To obtain the most recent stable release of pvlib-python, use
`conda <http://conda.pydata.org/docs/>`_ or `pip <https://pip.pypa.io>`_::

    conda install -c pvlib pvlib

    pip install pvlib

If your system complains that you don't have access privileges or asks
for a password then you're probably trying to install pvlib into your
system's Python distribution. This is usually a bad idea and you should
instead follow the :ref:`nopython` instructions below.

.. _nopython:

If you don't have Python
------------------------

There are many ways to install Python on your system, but the Anaconda
Scientific Python distribution provides by far the easiest way for new
users to get started. Anaconda includes all of the popular libraries
that you'll need for pvlib, including Pandas, NumPy, and SciPy.
"Anaconda installs cleanly into a single directory, does not require
Administrator or root privileges, does not affect other Python installs
on your system, or interfere with OSX Frameworks." -Anaconda
Documentation.

#. Install the full Anaconda Scientific Python distribution available
   `here <https://store.continuum.io/cshop/anaconda/>`_
#. Install pvlib: ``conda install -c pvlib pvlib``

If you have trouble, see the `Anaconda
FAQ <http://docs.continuum.io/anaconda/faq.html>`_, Google your error
messages, or make a new issue on our `Issues
page <https://github.com/pvlib/pvlib-python/issues>`_.


Working at the bleeding edge
----------------------------

We strongly recommend working in a `virtual environment
<http://astropy.readthedocs.org/en/latest/development/workflow/
virtual_pythons.html>`_ if you're going to use the development versions
of the code. There are many ways to use virtual environments in Python,
but Anaconda again provides the easiest solution:

#. Create a new conda environment for pvlib and pre-install a
   handful of packages into the environment:
   ``conda create --name pvlibdev python pandas scipy``
#. Activate the new environment: ``source activate pvlibdev``
#. Install the latest development version of pvlib:

  #. If you don't plan to modify the source-code:
     ``pip install git+https://github.com/pvlib/pvlib-python.git``
  #. If you do plan to modify the source code:
     Use the GitHub GUI application or git command-line tool to
     clone this repository to your computer, then navigate your
     command-line to the top-level pvlib-python directory,
     then ``pip install -e .``

#. You may also consider installing additional packages into your
   development environment:
   ``conda install jupyter ipython seaborn nose flake8``

The `conda documentation
<http://conda.pydata.org/docs/using/index.html>`_ has more information
on how to use virtual environments. You can also add ``-h`` to most
conda commands to get help (e.g. ``conda -h`` or ``conda env -h``)
