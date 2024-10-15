.. _testing:

Testing and benchmarking
========================

.. _overview:

Overview
~~~~~~~~

Developers **must** include comprehensive tests for any additions or
modifications to pvlib. New unit test code should be placed in the
corresponding test module in the
`pvlib/tests <https://github.com/pvlib/pvlib-python/tree/main/pvlib/tests>`_
directory.

A pull request will automatically run the tests for you on a variety of
platforms (Linux, Mac, Windows) and python versions. However, it is
typically more efficient to run and debug the tests in your own local
environment.

To run the tests locally, install the ``test`` dependencies specified in the
`setup.py <https://github.com/pvlib/pvlib-python/blob/main/setup.py>`_
file. See :ref:`installation` instructions for more information.

pvlib's unit tests can easily be run by executing ``pytest`` on the
pvlib directory::

    pytest pvlib

or, for a single module::

    pytest pvlib/tests/test_clearsky.py

or, for a single test::

    pytest pvlib/tests/test_clearsky.py::test_ineichen_nans

We suggest using pytest's ``--pdb`` flag to debug test failures rather
than using ``print`` or ``logging`` calls. For example::

    pytest pvlib --pdb

will drop you into the
`pdb debugger <https://docs.python.org/3/library/pdb.html>`_ at the
location of a test failure. As described in :ref:`code-style`, pvlib
code does not use ``print`` or ``logging`` calls, and this also applies
to the test suite (with rare exceptions).

To include all network-dependent tests, include the ``--remote-data`` flag to
your ``pytest`` call::

    pytest pvlib --remote-data

And consider adding ``@pytest.mark.remote_data`` to any network dependent test
you submit for a PR.

pvlib-python contains 3 "layers" of code: functions, PVSystem/Location,
and ModelChain. Contributors will need to add tests that correspond to
the layers that they modify.


Functions
~~~~~~~~~

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
~~~~~~~~~~~~~~~~~

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
        module_parameters = {'b': 0.05}
        system = pvsystem.PVSystem(module_parameters=module_parameters)
        thetas = 1

        # call the method
        iam = system.ashraeiam(thetas)

        # did the method call the function as we expected?
        # mocker.spy added assert_called_once_with to the function
        pvsystem.ashraeiam.assert_called_once_with(thetas, b=module_parameters['b'])

        # check that the output is reasonable, but no need to duplicate
        # the rigorous tests of the function
        assert iam < 1.

Avoid writing PVSystem/Location tests that depend sensitively on the
return value of a statement as a substitute for using mock. These tests
are sensitive to changes in the functions, which is *not* what we want
to test here, and are difficult to maintain.


ModelChain
~~~~~~~~~~

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

        # use `assert m.call_count == num` if function should be called
        # more than once

        # ensure that dc attribute now exists and is correct type
        assert isinstance(mc.dc, (pd.Series, pd.DataFrame))


Benchmarking
~~~~~~~~~~~~

pvlib includes a small number of performance benchmarking tests. These
tests are run using the `airspeed velocity
<https://asv.readthedocs.io/en/stable/>`_ tool. We do not require new
performance tests for most contributions at this time. Pull request
reviewers will provide further information if a performance test is
necessary. See our `README
<https://github.com/pvlib/pvlib-python/tree/main/benchmarks/README.md>`_
for instructions on running the benchmarks.
