.. _contributing:

Contributing
============

Encouraging more people to help develop pvlib-python is essential to our
success. Therefore, we want to make it easy and rewarding for you to
contribute.


Easy ways to contribute
~~~~~~~~~~~~~~~~~~~~~~~

Here are a few ideas for you can contribute, even if you are new to
pvlib-python, git, or Python:

* Make `GitHub issues <https://github.com/pvlib/pvlib-python/issues>`_
  and contribute to the conversation about how to resolve them.
* Read issues and pull requests that other people created and
  contribute to the conversation about how to resolve them.
* Improve the documentation and the unit tests.
* Improve the IPython/Jupyter Notebook tutorials or write new ones that
  demonstrate how to use pvlib-python in your area of expertise.
* If you have MATLAB experience, you can help us keep pvlib-python
  up to date with PVLIB_MATLAB or help us develop common unit tests.
  For more, see `Issue #2 <https://github.com/pvlib/pvlib-python/issues/2>`_
  and `Issue #3 <https://github.com/pvlib/pvlib-python/issues/3>`_.
* Tell your friends and colleagues about pvlib-python.
* Add your project to our
  `Projects and publications that use pvlib-python wiki
  <https://github.com/pvlib/pvlib-python/wiki/Projects-and-publications-
  that-use-pvlib-python>`_.


How to contribute new code
~~~~~~~~~~~~~~~~~~~~~~~~~~

Contributors to pvlib-python use GitHub's pull requests to add/modify
its source code. The GitHub pull request process can be intimidating for
new users, but you'll find that it becomes straightforward once you use
it a few times. Please let us know if you get stuck at any point in the
process. Here's an outline of the process:

#. Create a GitHub issue and get initial feedback from users and
   maintainers. If the issue is a bug report, please include the
   code needed to reproduce the problem.
#. Obtain the latest version of pvlib-python: Fork the pvlib-python
   project to your GitHub account, ``git clone`` your fork to your computer.
#. Make some or all of your changes/additions and ``git commit`` them to
   your local repository.
#. Share your changes with us via a pull request: ``git push`` your
   local changes to your GitHub fork, then go to GitHub make a pull
   request.

The Pandas project maintains an excellent `contributing page
<https://github.com/pydata/pandas/wiki/Contributing>`_ that goes into
detail on each of these steps. Also see GitHub's `Set Up Git
<https://help.github.com/articles/set-up-git/>`_ and `Using Pull
Requests <https://help.github.com/articles/using-pull-requests/>`_.

Note that you do not need to make all of your changes before creating a
pull request. Your pull requests will automatically be updated when you
commit new changes and push them to GitHub. This gives everybody an easy
way to comment on the code and can make the process more efficient.

We strongly recommend using virtual environments for development.
Virtual environments make it trivial to switch between different
versions of software. This `astropy guide
<http://astropy.readthedocs.org/en/latest/development/workflow/
virtual_pythons.html>`_ is a good reference for virtual environments. If
this is your first pull request, don't worry about using a virtual
environment.

You must include documentation and unit tests for any new or improved
code. We can provide help and advice on this after you start the pull
request.

The maintainers will follow same procedures, rather than making direct
commits to the main repo. Exceptions may be made for extremely minor
changes, such as fixing documentation typos.


Testing
~~~~~~~

pvlib's unit tests can easily be run by executing ``py.test`` on the
pvlib directory:

``pytest pvlib``

or, for a single module:

``pytest pvlib/test/test_clearsky.py``

or, for a single test:

``pytest pvlib/test/test_clearsky.py::test_ineichen_nans``

Use the ``--pdb`` flag to debug failures and avoid using ``print``.

New unit test code should be placed in the corresponding test module in
the pvlib/test directory.

Developers **must** include comprehensive tests for any additions or
modifications to pvlib.

pvlib-python contains 3 "layers" of code: functions, PVSystem/Location,
and ModelChain. Contributors will need to add tests that correspond to
the layer that they modify.

Functions
---------
Tests of core pvlib functions should ensure that the function returns
the desired output for a variety of function inputs. The tests should be
independent of other pvlib functions (see :issue:`394`). The tests
should ensure that all reasonable combinations of input types (floats,
nans, arrays, series, scalars, etc) work as expected. Remember that your
use case is likely not the only way that this function will be used, and
your input data may not be generic enough to fully test the function.
Write tests that cover the full range of validity of the algorithm.
It is also important to write tests that assert the return value of the
function or that the function throws an exception when input data is
beyond the range of algorithm validity.

PVSystem/Location
-----------------
The PVSystem and Location classes provide convenience wrappers around
the core pvlib functions. The tests in test_pvsystem.py and
test_location.py should ensure that the method calls correctly wrap the
function calls. Many PVSystem/Location methods pass one or more of their
object's attributes (e.g. PVSystem.module_parameters, Location.latitude)
to a function. Tests should ensure that attributes are passed correctly.
These tests should also ensure that the method returns some reasonable
data, though the precise values of the data should be covered by
function-specific tests discussed above.

We prefer to use the ``pytest-mock`` framework to write these tests. The
test below shows an example of testing the ``PVSystem.ashraeiam``
method. ``mocker`` is a ``pytest-mock`` object. ``mocker.spy`` adds
features to the ``pvsystem.ashraeiam`` *function* that keep track of how
it was called. Then a ``PVSystem`` object is created and the
``PVSystem.ashraeiam`` *method* is called in the usual way. The
``PVSystem.ashraeiam`` method is supposed to call the
``pvsystem.ashraeiam`` function with the angles supplied to the method
call and the value of ``b`` that we defined in ``module_parameters``.
The ``pvsystem.ashraeiam.assert_called_once_with`` tests that this does,
in fact, happen. Finally, we check that the output of the method call is
reasonable.

.. code-block:: python
    def test_PVSystem_ashraeiam(mocker):
        # mocker is a pytest-mock object.
        # mocker.spy adds code to a function to keep track of how it is called
        mocker.spy(pvsystem, 'ashraeiam')

        # set up inputs
        module_parameters = pd.Series({'b': 0.05})
        system = pvsystem.PVSystem(module_parameters=module_parameters)
        thetas = 1

        # call the method
        iam = system.ashraeiam(thetas)

        # did the method call the function as we expected?
        # mocker.spy added assert_called_once_with to the function
        pvsystem.ashraeiam.assert_called_once_with(thetas, b=0.05)

        # check that the output is reasonable, but no need to duplicate
        # the rigorous tests of the function
        assert iam < 1.

Avoid writing PVSystem/Location tests that depend sensitively on the
return value of a statement as a substitute for using mock. These tests
are sensitive to changes in the functions, which is *not* what we want
to test here, and are difficult to maintain.

ModelChain
----------
The tests in test_modelchain.py should ensure that
``ModelChain.__init__`` correctly configures the ModelChain object to
eventually run the selected models. A test should ensure that the
appropriate method is actually called in the course of
``ModelChain.run_model``. A test should ensure that the model selection
does have a reasonable effect on the subsequent calculations, though the
precise values of the data should be covered by the function tests
discussed above. ``pytest-mock`` can also be used for testing ``ModelChain``.

The example below shows how mock can be used to assert that the correct
PVSystem method is called through ``ModelChain.run_model``.

.. code-block:: python
    def test_modelchain_dc_model(mocker):
        # set up location and system for model chain
        location = location.Location(32, -111)
        system = pvsystem.PVSystem(module_parameters=some_sandia_mod_params,
                                   inverter_parameters=some_cecinverter_params)

        # mocker.spy adds code to the system.sapm method to keep track of how
        # it is called. use returned mock object m to make assertion later,
        # but see example above for alternative
        m = mocker.spy(system, 'sapm')

        # make and run the model chain
        mc = ModelChain(system, location,
                        aoi_model='no_loss', spectral_model='no_loss')
        times = pd.date_range('20160101 1200-0700', periods=2, freq='6H')
        mc.run_model(times)

        # assertion fails if PVSystem.sapm is not called once
        m.assert_called_once()

        # ensure that dc attribute now exists and is correct type
        assert isinstance(mc.dc, (pd.Series, pd.DataFrame))


This documentation
~~~~~~~~~~~~~~~~~~

If this documentation is unclear, help us improve it! Consider looking
at the `pandas
documentation <http://pandas.pydata.org/pandas-docs/version/0.18.1/
contributing.html>`_ for inspiration.
