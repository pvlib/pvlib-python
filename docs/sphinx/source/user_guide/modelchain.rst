.. _modelchaindoc:

ModelChain
==========

The :py:class:`~.modelchain.ModelChain` class provides a high-level
interface for standardized PV modeling. The class aims to automate much
of the modeling process while providing flexibility and remaining
extensible. This guide aims to build users' understanding of the
ModelChain class. It assumes some familiarity with object-oriented
code in Python, but most information should be understandable even
without a solid understanding of classes.

A :py:class:`~.modelchain.ModelChain` has three components:

* a :py:class:`~.pvsystem.PVSystem` object, representing a collection of modules and inverters
* a :py:class:`~.location.Location` object, representing a location on the planet
* values for attributes that specify the model to be used for each step in the PV modeling
  process.

Modeling with a :py:class:`~.ModelChain` typically involves 3 steps:

1. Creating an instance of :py:class:`~pvlib.modelchain.ModelChain`.
2. Executing a ModelChain.run_model method with weather data as input. See
   :ref:`modelchain_runmodel` for a list of run_model methods.
3. Examining the model results that are stored in the ModelChain's
   :py:class:`ModelChain.results <pvlib.modelchain.ModelChainResult>` attribute.

A simple ModelChain example
---------------------------

Before delving into the intricacies of ModelChain, we provide a brief
example of the modeling steps using ModelChain. First, we import pvlib’s
objects, module data, and inverter data.

.. ipython:: python

    import pandas as pd
    import numpy as np

    # pvlib imports
    import pvlib

    from pvlib.pvsystem import PVSystem, FixedMount
    from pvlib.location import Location
    from pvlib.modelchain import ModelChain
    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # load some module and inverter specifications
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

Now we create a Location object, a Mount object, a PVSystem object, and a
ModelChain object.

.. ipython:: python

    location = Location(latitude=32.2, longitude=-110.9)
    system = PVSystem(surface_tilt=20, surface_azimuth=200,
                      module_parameters=sandia_module,
                      inverter_parameters=cec_inverter,
                      temperature_model_parameters=temperature_model_parameters)
    mc = ModelChain(system, location)

Printing a ModelChain object will display its models.

.. ipython:: python

    print(mc)

Next, we run a model with some simple weather data.

.. ipython:: python

    weather = pd.DataFrame([[1050, 1000, 100, 30, 5]],
                           columns=['ghi', 'dni', 'dhi', 'temp_air', 'wind_speed'],
                           index=[pd.Timestamp('20170401 1200', tz='US/Arizona')])

    mc.run_model(weather)

ModelChain stores the modeling results in the ``results`` attribute. The
``results`` attribute is an instance of :py:class:`~pvlib.modelchain.ModelChainResult`.
A few examples of attributes of :py:class:`~pvlib.modelchain.ModelChainResult`
are shown below.

.. ipython:: python

    mc.results.aoi

.. ipython:: python

    mc.results.cell_temperature

.. ipython:: python

    mc.results.dc

.. ipython:: python

    mc.results.ac

The remainder of this guide examines the ModelChain functionality and
explores common pitfalls.

Defining a ModelChain
---------------------

A :py:class:`~pvlib.modelchain.ModelChain` object is defined by:

1. The properties of its :py:class:`~pvlib.pvsystem.PVSystem`
   and :py:class:`~pvlib.location.Location` objects
2. The keyword arguments passed to it at construction

ModelChain uses the keyword arguments passed to it to determine the
models for the simulation. The documentation describes the allowed
values for each keyword argument. If a keyword argument is not supplied,
ModelChain will attempt to infer the correct set of models by inspecting
the Location and PVSystem attributes.

Below, we show some examples of how to define a ModelChain.

Let’s make the most basic Location and PVSystem objects and build from
there.

.. ipython:: python

    location = Location(32.2, -110.9)
    poorly_specified_system = PVSystem()
    print(location)
    print(poorly_specified_system)

These basic objects do not have enough information for ModelChain to be
able to automatically determine its set of models, so the ModelChain
will throw an error when we try to create it.

.. ipython:: python
   :okexcept:

    ModelChain(poorly_specified_system, location)

Next, we define a PVSystem with a module from the SAPM database and an
inverter from the CEC database. ModelChain will examine the PVSystem
object’s properties and determine that it should choose the SAPM DC
model, AC model, AOI loss model, and spectral loss model.

.. ipython:: python

    sapm_system = PVSystem(
        module_parameters=sandia_module,
        inverter_parameters=cec_inverter,
        temperature_model_parameters=temperature_model_parameters)
    mc = ModelChain(sapm_system, location)
    print(mc)

.. ipython:: python

    mc.run_model(weather);
    mc.results.ac

Alternatively, we could have specified single diode or PVWatts related
information in the PVSystem construction. Here we pass parameters for
PVWatts models to the PVSystem. ModelChain will automatically determine that
it should choose PVWatts DC and AC models. ModelChain still needs us to specify
``aoi_model`` and ``spectral_model`` keyword arguments because the
``system.module_parameters`` dictionary does not contain enough
information to determine which of those models to choose.

.. ipython:: python

    pvwatts_system = PVSystem(
        module_parameters={'pdc0': 240, 'gamma_pdc': -0.004},
        inverter_parameters={'pdc0': 240},
        temperature_model_parameters=temperature_model_parameters)
    mc = ModelChain(pvwatts_system, location,
                    aoi_model='physical', spectral_model='no_loss')
    print(mc)

.. ipython:: python

    mc.run_model(weather);
    mc.results.ac

User-supplied keyword arguments override ModelChain’s inspection
methods. For example, we can tell ModelChain to use different loss
functions for a PVSystem that contains SAPM-specific parameters.

.. ipython:: python

    sapm_system = PVSystem(
        module_parameters=sandia_module,
        inverter_parameters=cec_inverter,
        temperature_model_parameters=temperature_model_parameters)
    mc = ModelChain(sapm_system, location, aoi_model='physical', spectral_model='no_loss')
    print(mc)

.. ipython:: python

    mc.run_model(weather);
    mc.results.ac

Of course, these choices can also lead to failure when executing
:py:meth:`~pvlib.modelchain.ModelChain.run_model` if your system objects
do not contain the required parameters for running the model chain.

As a convenience, ModelChain includes two class methods that return a ModelChain
with models selected to be consistent with named PV system models:

* :py:meth:`~pvlib.modelchain.ModelChain.with_pvwatts`
* :py:meth:`~pvlib.modelchain.ModelChain.with_sapm`

Each "with" method returns a ModelChain using a Location and PVSystem. Parameters
used to define the PVSystem need to be consistent with the models specified by
the "with" method. Using location and sapm_system defined above:

.. ipython:: python

    mc = mc.with_sapm(sapm_system, location)
    print(mc)

    mc.run_model(weather)
    mc.results.dc


Demystifying ModelChain internals
---------------------------------

The ModelChain class has a lot going in inside it in order to make
users' code as simple as possible.

The key parts of ModelChain are:

    1. The ModelChain.run_model methods.
    2. A set of methods that wrap and call the PVSystem methods.
    3. A set of methods that can inspect user-supplied objects to infer
       the appropriate model when a model isn't specified by the user.

run_model methods
~~~~~~~~~~~~~~~~~

ModelChain provides three methods for executing the chain of models. The
methods allow for simulating the output of the PVSystem with different
input data:

* :py:meth:`~pvlib.modelchain.ModelChain.run_model`, use when ``weather``
  contains global horizontal, direct and diffuse horizontal irradiance
  (``'ghi'``, ``'dni'`` and ``'dhi'``).
* :py:meth:`~pvlib.modelchain.ModelChain.run_model_from_poa`, use when
  ``weather`` broadband direct, diffuse and total irradiance in the plane of array
  (``'poa_global'``, ``'poa_direct'``, ``'poa_diffuse'``).
* :py:meth:`~pvlib.modelchain.ModelChain.run_model_from_effective_irradiance`,
  use when ``weather`` contains spectrally- and reflection-adjusted total
  irradiance in the plane of array ('effective_irradiance').

To illustrate the use of the `run_model` method, assume that a user has GHI and DHI.
:py:meth:`~pvlib.modelchain.ModelChain.prepare_inputs` requires all three
irradiance components (GHI, DNI, and DHI). In this case, the user needs
to calculate DNI before using `run_model`. The :py:meth:`~pvlib.modelchain.ModelChain.complete_irradiance`
method is available for calculating the full set of GHI, DNI, or DHI if
only two of these three series are provided. See also :ref:`dniestmodels`
for methods and functions that can help fully define the irradiance inputs.

The :py:meth:`~pvlib.modelchain.ModelChain.run_model` method, shown below,
calls a series of methods to complete the modeling steps. The first
method, :py:meth:`~pvlib.modelchain.ModelChain.prepare_inputs`, computes
parameters such as solar position, airmass, angle of incidence, and
plane of array irradiance. Next, :py:meth:`~pvlib.modelchain.ModelChain.run_model` calls the
wrapper methods for AOI loss, spectral loss, effective irradiance, cell
temperature, DC power, AC power, and other losses. These methods are
assigned to generic names, as described in the next section.

.. ipython:: python

    mc.run_model??

The methods called by :py:meth:`~pvlib.modelchain.ModelChain.run_model`
store their results in the ``results`` attribute, which is an instance of
:py:class:`~pvlib.modelchain.ModelChainResult`. For example, :py:class:`~.ModelChainResult`
includes the following attributes: ``solar_position``, ``effective_irradiance``,
``cell_temperature``,  ``dc``, ``ac``. See :py:class:`~pvlib.modelchain.ModelChainResult`
for a full list of results attributes.


Wrapping methods into a unified API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Readers may notice that the source code of the :py:meth:`~pvlib.modelchain.ModelChain.run_model`
method is model-agnostic. :py:meth:`~pvlib.modelchain.ModelChain.run_model` calls generic methods
such as ``self.dc_model`` rather than a specific model such as
``pvwatts_dc``. So how does :py:meth:`~pvlib.modelchain.ModelChain.run_model` know what models
it’s supposed to run? The answer comes in two parts, and allows us to
explore more of the ModelChain API along the way.

First, ModelChain has a set of methods that wrap the PVSystem methods
that perform the calculations (or further wrap the pvsystem.py module’s
functions). Each of these methods takes the same arguments (``self``)
and sets the same attributes, thus creating a uniform API. For example,
the :py:meth:`~pvlib.modelchain.ModelChain.pvwatts_dc` method is shown below. Its only argument is
``self``, and it sets the ``dc`` attribute.

.. ipython:: python

    mc.pvwatts_dc??

The :py:meth:`~pvlib.modelchain.ModelChain.pvwatts_dc` method calls the pvwatts_dc method of the
PVSystem object that we supplied when we created the ModelChain instance,
using data that is stored in the ModelChain ``effective_irradiance`` and
``cell_temperature`` attributes. The :py:meth:`~pvlib.modelchain.ModelChain.pvwatts_dc` method assigns its
result to the ``dc`` attribute of the ModelChain's ``results`` object. The code
below shows a simple example of this.

.. ipython:: python

    # make the objects
    pvwatts_system = PVSystem(
        module_parameters={'pdc0': 240, 'gamma_pdc': -0.004},
        inverter_parameters={'pdc0': 240},
        temperature_model_parameters=temperature_model_parameters)
    mc = ModelChain(pvwatts_system, location,
                    aoi_model='no_loss', spectral_model='no_loss')

    # manually assign data to the attributes that ModelChain.pvwatts_dc will need.
    # for standard workflows, run_model would assign these attributes.
    mc.results.effective_irradiance = pd.Series(1000, index=[pd.Timestamp('20170401 1200-0700')])
    mc.results.cell_temperature = pd.Series(50, index=[pd.Timestamp('20170401 1200-0700')])

    # run ModelChain.pvwatts_dc and look at the result
    mc.pvwatts_dc();
    mc.results.dc

The :py:meth:`~pvlib.modelchain.ModelChain.sapm` method works in a manner similar
to the :py:meth:`~pvlib.modelchain.ModelChain.pvwatts_dc`
method. It calls the :py:meth:`~pvlib.pvsystem.PVSystem.sapm` method using stored data, then
assigns the result to the ``dc`` attribute of ``ModelChain.results``.
The :py:meth:`~pvlib.modelchain.ModelChain.sapm` method differs from the
:py:meth:`~pvlib.modelchain.ModelChain.pvwatts_dc` method in
a notable way: the PVSystem.sapm method returns a DataFrame with current,
voltage, and power results, rather than a simple Series
of power. The ModelChain methods for single diode models (e.g.,
:py:meth:`~pvlib.modelchain.ModelChain.desoto`) also return a DataFrame with
current, voltage and power, and a second DataFrame with the single diode
equation parameter values.

All ModelChain methods for DC output use the
:py:meth:`~pvlib.pvsystem.PVSystem.scale_voltage_current_power` method to scale
DC quantities to the output of the full PVSystem.

.. ipython:: python

    mc.sapm??

.. ipython:: python

    # make the objects
    sapm_system = PVSystem(
        module_parameters=sandia_module,
        inverter_parameters=cec_inverter,
        temperature_model_parameters=temperature_model_parameters)
    mc = ModelChain(sapm_system, location)

    # manually assign data to the attributes that ModelChain.sapm will need.
    # for standard workflows, run_model would assign these attributes.
    mc.results.effective_irradiance = pd.Series(1000, index=[pd.Timestamp('20170401 1200-0700')])
    mc.results.cell_temperature = pd.Series(50, index=[pd.Timestamp('20170401 1200-0700')])

    # run ModelChain.sapm and look at the result
    mc.sapm();
    mc.results.dc

We’ve established that the ``ModelChain.pvwatts_dc`` and
``ModelChain.sapm`` have the same API: they take the same arugments
(``self``) and they both set the ``dc`` attribute.\* Because the methods
have the same API, we can call them in the same way. ModelChain includes
a large number of methods that perform the same API-unification roles
for each modeling step.

Again, so how does :py:meth:`~pvlib.modelchain.ModelChain.run_model` know which
models it’s supposed to run?

At object construction, ModelChain assigns the desired model’s method
(e.g. ``ModelChain.pvwatts_dc``) to the corresponding generic attribute
(e.g. ``ModelChain.dc_model``) either with the value assigned to the ``dc_model``
parameter at construction, or by inference as described in the next
section.

.. ipython:: python

    pvwatts_system = PVSystem(
        module_parameters={'pdc0': 240, 'gamma_pdc': -0.004},
        inverter_parameters={'pdc0': 240},
        temperature_model_parameters=temperature_model_parameters)
    mc = ModelChain(pvwatts_system, location,
                    aoi_model='no_loss', spectral_model='no_loss')
    mc.dc_model.__func__

The ModelChain.run_model method can ignorantly call ``self.dc_module``
because the API is the same for all methods that may be assigned to this
attribute.

\* some readers may object that the API is *not* actually the same
because the type of the ``dc`` attribute is different (Series
vs. DataFrame)!

Inferring models
~~~~~~~~~~~~~~~~

When ModelChain's attributes are not assigned when the instance is created,
ModelChain can infer the appropriate model from data stored on the ``PVSystem``
object. ModelChain uses a set of methods (e.g., :py:meth:`~pvlib.modelchain.ModelChain.infer_dc_model`,
:py:meth:`~pvlib.modelchain.ModelChain.infer_ac_model`, etc.) that examine the
parameters on the user-supplied PVSystem object. The inference methods use set
logic to assign one of the model-specific methods, such as
:py:meth:`~pvlib.modelchain.ModelChain.sapm` or :py:meth:`~pvlib.modelchain.ModelChain.sandia_inverter`,
to the universal method names ``ModelChain.dc_model`` and ``ModelChain.ac_model``,
respectively. A few examples are shown below. Inference methods generally work
by inspecting the parameters for all required parameters for a corresponding
method.

.. ipython:: python

    mc.infer_dc_model??

.. ipython:: python

    mc.infer_ac_model??
    pvlib.modelchain._snl_params??
    pvlib.modelchain._adr_params??
    pvlib.modelchain._pvwatts_params??

ModelChain for a PVSystem with multiple Arrays
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The PVSystem can represent a PV system with a single array of modules, or
with multiple arrays (see :ref:`multiarray`). The same models are applied to
all ``PVSystem.array`` objects, so each ``Array`` must contain the appropriate model
parameters. For example, if ``ModelChain.dc_model='pvwatts'``, then each 
``Array.module_parameters`` must contain ``'pdc0'``.

When the PVSystem contains multiple arrays, ``ModelChain.results`` attributes
are tuples with length equal to the number of Arrays. Each tuple's elements
are in the same order as in ``PVSystem.arrays``.

.. ipython:: python

    from pvlib.pvsystem import Array
    location = Location(latitude=32.2, longitude=-110.9)
    inverter_parameters = {'pdc0': 10000, 'eta_inv_nom': 0.96}
    module_parameters = {'pdc0': 250, 'gamma_pdc': -0.004}
    array_one = Array(mount=FixedMount(surface_tilt=20, surface_azimuth=200),
                      module_parameters=module_parameters,
                      temperature_model_parameters=temperature_model_parameters,
                      modules_per_string=10, strings=2)
    array_two = Array(mount=FixedMount(surface_tilt=20, surface_azimuth=160),
                      module_parameters=module_parameters,
                      temperature_model_parameters=temperature_model_parameters,
                      modules_per_string=10, strings=2)
    system_two_arrays = PVSystem(arrays=[array_one, array_two],
                                 inverter_parameters={'pdc0': 8000})
    mc = ModelChain(system_two_arrays, location, aoi_model='no_loss',
                    spectral_model='no_loss')

    mc.run_model(weather)

    mc.results.dc
    mc.results.dc[0]

When ``weather`` is a single DataFrame, these data are broadcast and used
for all arrays. Weather data can be specified for each array, in which case
``weather`` needs to be a tuple or list of DataFrames in the same order as
the arrays of the PVSystem. To specify data separately for each array, provide a tuple
for ``weather`` where each element is a DataFrame containing the required data.

Air, module and cell temperatures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

The different run_model methods allow the ModelChain to be run starting with
different irradiance data. Similarly, ModelChain run_model methods can be used
with different temperature data as long as cell temperature can be determined.
Temperature data are passed in the ``weather`` DataFrame and can include:

* cell temperature (``'cell_temperature'``). If passed in ``weather`` no
  cell temperature model is run.
* module temperature (``'module_temperature'``), typically measured on the rear surface.
  If found in ``weather`` and ``ModelChain.temperature_model='sapm'`` 
  (either set directly or inferred), the :py:meth:`~pvlib.modelchain.ModelChain.sapm_temp`
  method is used to calculate cell temperature. If ``ModelChain.temperature_model``
  is set to any other model, ``'module_temperature'`` is ignored.
* ambient air temperature (``'temp_air'``). In this case ``ModelChain.temperature_model``
  is used to calculate cell temeprature.

Cell temperature models also can use irradiance as input. All cell
temperature models expect POA irradiance (``'poa_global'``) as  input. When
``weather`` contains ``'effective_irradiance'`` but not
``'poa_global'``, ``'effective_irradiance'`` is substituted for calculating
cell temperature.


User-defined models
-------------------

Users may also write their own functions and pass them as arguments to
ModelChain. The first argument of the function must be a ModelChain
instance. For example, the functions below implement the PVUSA model and
a wrapper function appropriate for use with ModelChain. This follows the
pattern of implementing the core models using the simplest possible
functions, and then implementing wrappers to make them easier to use in
specific applications. Of course, you could implement it in a single
function if you wanted to.

.. ipython:: python

    def pvusa(poa_global, wind_speed, temp_air, a, b, c, d):
        """
        Calculates system power according to the PVUSA equation
        P = I * (a + b*I + c*W + d*T)
        where
        P is the output power,
        I is the plane of array irradiance,
        W is the wind speed, and
        T is the temperature
        a, b, c, d are empirically derived parameters.
        """
        return poa_global * (a + b*poa_global + c*wind_speed + d*temp_air)


    def pvusa_mc_wrapper(mc):
        """
        Calculate the dc power and assign it to mc.results.dc
        Set up to iterate over arrays and total_irrad. mc.system.arrays is
        always a tuple. However, when there is a single array
        mc.results.total_irrad will be a Series (if multiple arrays,
        total_irrad will be a tuple). In this case we put total_irrad
        in a list so that we can iterate. If we didn't put total_irrad
        in a list, iteration will access each value of the Series, one
        at a time.
        The iteration returns a tuple. If there is a single array, the
        tuple is of length 1. As a convenience, pvlib unwraps tuples of length 1
        that are assigned to ModelChain.results attributes.
        Returning mc is optional, but enables method chaining.
        """
        if mc.system.num_arrays == 1:
            total_irrads = [mc.results.total_irrad]
        else:
            total_irrads = mc.results.total_irrad
        mc.results.dc = tuple(
            pvusa(total_irrad['poa_global'], mc.results.weather['wind_speed'],
                  mc.results.weather['temp_air'], array.module_parameters['a'],
                  array.module_parameters['b'], array.module_parameters['c'],
                  array.module_parameters['d'])
            for total_irrad, array
            in zip(total_irrads, mc.system.arrays))
        return mc


    def pvusa_ac_mc(mc):
        # keep it simple
        mc.results.ac = mc.results.dc
        return mc


    def no_loss_temperature(mc):
        # keep it simple
        mc.results.cell_temperature = mc.results.weather['temp_air']
        return mc


.. ipython:: python

    module_parameters = {'a': 0.2, 'b': 0.00001, 'c': 0.001, 'd': -0.00005}
    pvusa_system = PVSystem(module_parameters=module_parameters)

    mc = ModelChain(pvusa_system, location,
                    dc_model=pvusa_mc_wrapper, ac_model=pvusa_ac_mc,
                    temperature_model=no_loss_temperature,
                    aoi_model='no_loss', spectral_model='no_loss')

A ModelChain object uses Python’s functools.partial function to assign
itself as the argument to the user-supplied functions.

.. ipython:: python

    mc.dc_model.func

The end result is that ModelChain.run_model works as expected!

.. ipython:: python

    mc = mc.run_model(weather)
    mc.results.dc
