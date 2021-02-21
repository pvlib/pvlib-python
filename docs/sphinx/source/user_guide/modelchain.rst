
ModelChain
==========

The :py:class:`~.modelchain.ModelChain` class provides a high-level
interface for standardized PV modeling. The class aims to automate much
of the modeling process while providing user-control and remaining
extensible. This guide aims to build users' understanding of the
ModelChain class. It assumes some familiarity with object-oriented
code in Python, but most information should be understandable even
without a solid understanding of classes.

A :py:class:`~.modelchain.ModelChain` is composed of a
:py:class:`~.pvsystem.PVSystem` object and a
:py:class:`~.location.Location` object. A PVSystem object represents an
assembled collection of modules, inverters, etc., a Location object
represents a particular place on the planet, and a ModelChain object
describes the modeling chain used to calculate a system's output at that
location. The PVSystem and Location objects will be described in detail
in another guide.

Modeling with a :py:class:`~.ModelChain` typically involves 3 steps:

1. Creating the :py:class:`~.ModelChain`.
2. Executing the :py:meth:`ModelChain.run_model() <.ModelChain.run_model>`
   method with prepared weather data.
3. Examining the model results that :py:meth:`~.ModelChain.run_model`
   stored in attributes of the :py:class:`~.ModelChain`.

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

    from pvlib.pvsystem import PVSystem
    from pvlib.location import Location
    from pvlib.modelchain import ModelChain
    from pvlib.temperature import TEMPERATURE_MODEL_PARAMETERS
    temperature_model_parameters = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']

    # load some module and inverter specifications
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    cec_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')

    sandia_module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    cec_inverter = cec_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']

Now we create a Location object, a PVSystem object, and a ModelChain
object.

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

    mc.run_model(weather);

ModelChain stores the modeling results on a series of attributes. A few
examples are shown below.

.. ipython:: python

    mc.aoi

.. ipython:: python

    mc.cell_temperature

.. ipython:: python

    mc.dc

.. ipython:: python

    mc.ac

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
    mc.ac

Alternatively, we could have specified single diode or PVWatts related
information in the PVSystem construction. Here we pass PVWatts data to
the PVSystem. ModelChain will automatically determine that it should
choose PVWatts DC and AC models. ModelChain still needs us to specify
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
    mc.ac

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
    mc.ac

Of course, these choices can also lead to failure when executing
:py:meth:`~pvlib.modelchain.ModelChain.run_model` if your system objects
do not contain the required parameters for running the model.

Demystifying ModelChain internals
---------------------------------

The ModelChain class has a lot going in inside it in order to make
users' code as simple as possible.

The key parts of ModelChain are:

    1. The :py:meth:`ModelChain.run_model() <.ModelChain.run_model>` method
    2. A set of methods that wrap and call the PVSystem methods.
    3. A set of methods that inspect user-supplied objects to determine
       the appropriate default models.

run_model
~~~~~~~~~

Most users will only interact with the
:py:meth:`~pvlib.modelchain.ModelChain.run_model` method. The
:py:meth:`~pvlib.modelchain.ModelChain.run_model` method, shown below,
calls a series of methods to complete the modeling steps. The first
method, :py:meth:`~pvlib.modelchain.ModelChain.prepare_inputs`, computes
parameters such as solar position, airmass, angle of incidence, and
plane of array irradiance. The
:py:meth:`~pvlib.modelchain.ModelChain.prepare_inputs` method also
assigns default values for temperature (20 C)
and wind speed (0 m/s) if these inputs are not provided.
:py:meth:`~pvlib.modelchain.ModelChain.prepare_inputs` requires all irradiance
components (GHI, DNI, and DHI). See
:py:meth:`~pvlib.modelchain.ModelChain.complete_irradiance` and
:ref:`dniestmodels` for methods and functions that can help fully define
the irradiance inputs.

Next, :py:meth:`~pvlib.modelchain.ModelChain.run_model` calls the
wrapper methods for AOI loss, spectral loss, effective irradiance, cell
temperature, DC power, AC power, and other losses. These methods are
assigned to standard names, as described in the next section.

The methods called by :py:meth:`~pvlib.modelchain.ModelChain.run_model`
store their results in a series of ModelChain attributes: ``times``,
``solar_position``, ``airmass``, ``irradiance``, ``total_irrad``,
``effective_irradiance``, ``weather``, ``temps``, ``aoi``,
``aoi_modifier``, ``spectral_modifier``, ``dc``, ``ac``, ``losses``.

.. ipython:: python

    mc.run_model??

Finally, the :py:meth:`~pvlib.modelchain.ModelChain.complete_irradiance`
method is available for calculating the full set of GHI, DNI, or DHI if
only two of these three series are provided. The completed dataset can
then be passed to :py:meth:`~pvlib.modelchain.ModelChain.run_model`.

Wrapping methods into a unified API
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Readers may notice that the source code of the ModelChain.run_model
method is model-agnostic. ModelChain.run_model calls generic methods
such as ``self.dc_model`` rather than a specific model such as
``singlediode``. So how does the ModelChain.run_model know what models
it’s supposed to run? The answer comes in two parts, and allows us to
explore more of the ModelChain API along the way.

First, ModelChain has a set of methods that wrap the PVSystem methods
that perform the calculations (or further wrap the pvsystem.py module’s
functions). Each of these methods takes the same arguments (``self``)
and sets the same attributes, thus creating a uniform API. For example,
the ModelChain.pvwatts_dc method is shown below. Its only argument is
``self``, and it sets the ``dc`` attribute.

.. ipython:: python

    mc.pvwatts_dc??

The ModelChain.pvwatts_dc method calls the pvwatts_dc method of the
PVSystem object that we supplied using data that is stored in its own
``effective_irradiance`` and ``cell_temperature`` attributes. Then it assigns the
result to the ``dc`` attribute of the ModelChain object. The code below
shows a simple example of this.

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
    mc.effective_irradiance = pd.Series(1000, index=[pd.Timestamp('20170401 1200-0700')])
    mc.cell_temperature = pd.Series(50, index=[pd.Timestamp('20170401 1200-0700')])

    # run ModelChain.pvwatts_dc and look at the result
    mc.pvwatts_dc();
    mc.dc

The ModelChain.sapm method works similarly to the ModelChain.pvwatts_dc
method. It calls the PVSystem.sapm method using stored data, then
assigns the result to the ``dc`` attribute. The ModelChain.sapm method
differs from the ModelChain.pvwatts_dc method in three notable ways.
First, the PVSystem.sapm method expects different units for effective
irradiance, so ModelChain handles the conversion for us. Second, the
PVSystem.sapm method (and the PVSystem.singlediode method) returns a
DataFrame with current, voltage, and power parameters rather than a
simple Series of power. Finally, this current and voltage information
allows the SAPM and single diode model paths to support the concept of
modules in series and parallel, which is handled by the
PVSystem.scale_voltage_current_power method.

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
    mc.effective_irradiance = pd.Series(1000, index=[pd.Timestamp('20170401 1200-0700')])
    mc.cell_temperature = pd.Series(50, index=[pd.Timestamp('20170401 1200-0700')])

    # run ModelChain.sapm and look at the result
    mc.sapm();
    mc.dc

We’ve established that the ``ModelChain.pvwatts_dc`` and
``ModelChain.sapm`` have the same API: they take the same arugments
(``self``) and they both set the ``dc`` attribute.\* Because the methods
have the same API, we can call them in the same way. ModelChain includes
a large number of methods that perform the same API-unification roles
for each modeling step.

Again, so how does the ModelChain.run_model know which models it’s
supposed to run?

At object construction, ModelChain assigns the desired model’s method
(e.g. ``ModelChain.pvwatts_dc``) to the corresponding generic attribute
(e.g. ``ModelChain.dc_model``) using a method described in the next
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

How does ModelChain infer the appropriate model types? ModelChain uses a
series of methods (ModelChain.infer_dc_model, ModelChain.infer_ac_model,
etc.) that examine the user-supplied PVSystem object. The inference
methods use set logic to assign one of the model-specific methods, such
as ModelChain.sapm or ModelChain.snlinverter, to the universal method
names ModelChain.dc_model and ModelChain.ac_model. A few examples are
shown below.

.. ipython:: python

    mc.infer_dc_model??

.. ipython:: python

    mc.infer_ac_model??

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
        # calculate the dc power and assign it to mc.dc
        mc.dc = pvusa(mc.total_irrad['poa_global'], mc.weather['wind_speed'], mc.weather['temp_air'],
                      mc.system.module_parameters['a'], mc.system.module_parameters['b'],
                      mc.system.module_parameters['c'], mc.system.module_parameters['d'])

        # returning mc is optional, but enables method chaining
        return mc


    def pvusa_ac_mc(mc):
        # keep it simple
        mc.ac = mc.dc
        return mc


    def no_loss_temperature(mc):
        # keep it simple
        mc.cell_temperature = mc.weather['temp_air']
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

    mc.run_model(weather);
    mc.dc
