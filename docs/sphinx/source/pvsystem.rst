.. _pvsystemdoc:

PVSystem
========

.. ipython:: python
   :suppress:

    import pandas as pd
    from pvlib import pvsystem


The :py:class:`~pvlib.pvsystem.PVSystem` represents one inverter and the
PV modules that supply DC power to the inverter. A PV system may be on fixed
mounting or single axis trackers. The :py:class`~pvlib.pvsystem.PVSystem`
is supported by the :py:class:`~pvlib.pvsystem.Array` which represents the
PV modules in the :py:class`~pvlib.pvsystem.PVSystem`. An instance of
:py:class:`~pvlib.pvsystem.PVSystem` has a single inverter, but can have
multiple instances of :py:class:`~pvlib.pvsystem.Array`. Arrays can have
different tilt, orientation, and number or type of modules.

The :py:class:`~pvlib.pvsystem.PVSystem` class methods wrap many of the
functions in the :py:mod:`~pvlib.pvsystem` module. Similarly, the
:py:class:`~pvlib.pvsystem.Array` wraps several functions with its class
methods.  Methods that wrap functions have similar names as the wrapped function.
This practice simplifies the API for :py:class:`~pvlib.pvsystem.PVSystem`
and :py:class:`~pvlib.pvsystem.Array` methods by eliminating the need to specify
arguments that are stored as attributes of these classes, such as
module and inverter properties when calling PVSystem methods. Using
:py:class:`~pvlib.pvsystem.PVSystem` is not better or worse than using the
functions it wraps -- it is simply an alternative way of organizing
your data and calculations.

This guide aims to build understanding of the PVSystem class. It assumes
basic familiarity with object-oriented code in Python, but most
information should be understandable without a solid understanding of
classes. Keep in mind that `functions` are independent of objects,
while `methods` are attached to objects.

See :py:class:`~pvlib.modelchain.ModelChain` for an application of
PVSystem to time series modeling.


.. _designphilosophy:

Design philosophy
-----------------

The PVSystem class allows modelers to easily separate the data that
represents a PV system (e.g. tilt angle or module parameters) from the
data that influences the PV system (e.g. the weather).

The data that represents the PV system is *intrinsic*. The
data that influences the PV system is *extrinsic*.

Intrinsic data is stored in object attributes. For example, the parameters
that describe a PV system's inverter is stored in
`PVSystem.inverter_parameters`.

.. ipython:: python

    inverter_parameters = {'pdc0': 5000, 'eta_inv_nom': 0.96}
    system = pvsystem.PVSystem(inverter_parameters=inverter_parameters)
    print(system.inverter_parameters)

The parameters that describe a PV system's modules can be provided to
`PVSystem.module_parameters` (in the case of a single array) or by providing
a list of instances of the :py:class:`~pvlib.pvsystem.Array`:

.. ipython:: python

    module_parameters = {'pdc0': 5000, 'gamma_pdc': -0.004}
    system = pvsystem.PVSystem(module_parameters=module_parameters,
                               inverter_parameters=inverter_parameters)
    print(system.module_parameters)

.. ipython:: python

    module_parameters = {'pdc0': 5000, 'gamma_pdc': -0.004}
    array = pvsystem.Array(module_parameters=module_parameters)
    system = pvsystem.PVSystem(arrays=[array],
                               inverter_parameters=inverter_parameters)
    print(system.module_parameters)
    print(system.inverter_parameters)

Extrinsic data is passed to a PVSystem instance as method arguments. For example,
the :py:meth:`~pvlib.pvsystem.PVSystem.pvwatts_dc` method accepts extrinsic
data irradiance and temperature.

.. ipython:: python

    pdc = system.pvwatts_dc(1000, 30)
    print(pdc)

Methods attached to a PVSystem object wrap corresponding functions in
:py:mod:`~pvlib.pvsystem`. The methods simplify the argument list by
using data stored in the PVSystem attributes. Compare the
:py:meth:`~pvlib.pvsystem.PVSystem.pvwatts_dc` method signature to the
:py:func:`~pvlib.pvsystem.pvwatts_dc` function signature:

    * :py:meth:`PVSystem.pvwatts_dc(g_poa_effective, temp_cell) <pvlib.pvsystem.PVSystem.pvwatts_dc>`
    * :py:func:`pvwatts_dc(g_poa_effective, temp_cell, pdc0, gamma_pdc, temp_ref=25.) <pvlib.pvsystem.pvwatts_dc>`

How does this work? The :py:meth:`~pvlib.pvsystem.PVSystem.pvwatts_dc`
method looks in `PVSystem.module_parameters` for the `pdc0`, and
`gamma_pdc` arguments. Then the :py:meth:`PVSystem.pvwatts_dc
<pvlib.pvsystem.PVSystem.pvwatts_dc>` method calls the
:py:func:`pvsystem.pvwatts_dc <pvlib.pvsystem.pvwatts_dc>` function with
all of the arguments and returns the result to the user. Note that the
function includes a default value for the parameter `temp_ref`. This
default value may be overridden by specifying the `temp_ref` key in the
`PVSystem.module_parameters` dictionary.

.. ipython:: python

    system.module_parameters['temp_ref'] = 0
    # lower temp_ref should lead to lower DC power than calculated above
    pdc = system.pvwatts_dc(1000, 30)
    print(pdc)

Multiple methods may pull data from the same attribute. For example, the
`PVSystem.module_parameters` attribute is used by the DC model methods
as well as the incidence angle modifier methods.


.. _pvsystemattributes:

PVSystem attributes
-------------------

Here we review the most commonly used PVSystem and Array attributes.
Please see the :py:class:`~pvlib.pvsystem.PVSystem` and 
:py:class:`~pvlib.pvsystem.Array` class documentation for a
comprehensive list of attributes.

The first parameters which describe the DC part of a PV system are the tilt
and azimuth of the modules. In the case of a PV system with a single array,
these parameters can be specified using the `PVSystem.surface_tilt` and
`PVSystem.surface_azimuth` attributes. In the case of a PV system with
several arrays, the parameters are specified for each array using
the attributes `Array.surface_tilt` and `Array.surface_azimuth`.

The `surface_tilt` and `surface_azimuth` attributes are used in PVSystem
(or Array) methods such as :py:meth:`~pvlib.pvsystem.PVSystem.get_aoi` or
:py:meth:`~pvlib.pvsystem.Array.get_aoi`. The angle of incidence (AOI)
(AOI) calculations require `surface_tilt`, `surface_azimuth` and also
the extrinsic sun position. The :py:meth:`~pvlib.pvsystem.PVSystem.get_aoi` method
uses the `surface_tilt` and `surface_azimuth` attributes from its PVSystem
object, and so requires only `solar_zenith` and `solar_azimuth` as
arguments. The :py:meth:`~pvlib.pvsystem.Array.get_aoi` operates in a similar
manner.

.. ipython:: python

    # single south-facing array at 20 deg tilt
    system_one_array = pvsystem.PVSystem(surface_tilt=20, surface_azimuth=180)
    print(system_one_array.surface_tilt, system_one_array.surface_azimuth)

    # call get_aoi with solar_zenith, solar_azimuth
    aoi = system_one_array.get_aoi(solar_zenith=30, solar_azimuth=180)
    print(aoi)

.. ipython:: python

    # two arrays each at 30 deg tilt with different facing
    array_one = pvsystem.Array(surface_tilt=30, surface_azimuth=90)
    array_two = pvsystem.Array(surface_tilt=30, surface_azimuth=220)
    system_multiarray = pvsystem.PVSystem(arrays=[array_one, array_two])
    print(system_multiarray.num_arrays)
    # call get_aoi with solar_zenith, solar_azimuth
    aoi = system_multiarray.get_aoi(solar_zenith=30, solar_azimuth=180)
    print(aoi)

Note that when the PV system includes more than one array, the output of
:py:meth:`~pvlib.pvsystem.PVSystem.get_aoi` is a *tuple* with the order of the
elements corresponding to the order of the arrays. If the AOI is desired for
a specific array, :py:meth:`~pvlib.pvsystem.Array.get_aoi` returns the AOI for
the array represented by the method's object.

.. ipython:: python

    aoi = array_one.get_aoi(solar_zenith=30, solar_azimuth=180)
    print(aoi)

`module_parameters` and `inverter_parameters` contain the data
necessary for computing DC and AC power using one of the available
PVSystem methods. These attributes are typically specified using data from
the :py:func:`~pvlib.pvsystem.retrieve_sam` function:

.. ipython:: python

    # retrieve_sam returns a dict. the dict keys are module names,
    # and the values are model parameters for that module
    modules = pvsystem.retrieve_sam('cecmod')
    module_parameters = modules['Canadian_Solar_Inc__CS5P_220M']
    inverters = pvsystem.retrieve_sam('cecinverter')
    inverter_parameters = inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    system_one_array = pvsystem.PVSystem(module_parameters=module_parameters,
                                         inverter_parameters=inverter_parameters)

The module and/or inverter parameters can also be specified manually.
This is useful for specifying modules and inverters that are not
included in the supplied databases. It is also useful for specifying
systems for use with the PVWatts models, as demonstrated in
:ref:`designphilosophy`.

The `losses_parameters` attribute contains data that may be used with
methods that calculate system losses. At present, these methods include
only :py:meth:`PVSystem.pvwatts_losses
<pvlib.pvsystem.PVSystem.pvwatts_losses>` and
:py:func:`pvsystem.pvwatts_losses <pvlib.pvsystem.pvwatts_losses>`, but
we hope to add more related functions and methods in the future.

The attributes `modules_per_string` and `strings_per_inverter` are used
in the :py:meth:`~pvlib.pvsystem.PVSystem.scale_voltage_current_power`
method. Some DC power models in :py:class:`~pvlib.modelchain.ModelChain`
automatically call this method and make use of these attributes. As an
example, consider a system with a single array comprising 35 modules
arranged into 5 strings of 7 modules each.

.. ipython:: python

    system = pvsystem.PVSystem(modules_per_string=7, strings_per_inverter=5)
    # crude numbers from a single module
    data = pd.DataFrame({'v_mp': 8, 'v_oc': 10, 'i_mp': 5, 'i_x': 6,
                         'i_xx': 4, 'i_sc': 7, 'p_mp': 40}, index=[0])
    data_scaled = system.scale_voltage_current_power(data)
    print(data_scaled)


.. _multiarray:

PVSystem with multiple Arrays
-----------------------------

It is possible to model a system with multiple arrays by passing a list of
:py:class:`~pvlib.pvsystem.Array` to the :py:class:`~pvlib.pvsystem.PVSystem`
constructor. The :py:class:`~pvlib.pvsystem.Array` class includes those 
:py:class:`~pvlib.pvsystem.PVSystem` attributes that may differ from array
to array. These attributes include `surface_tilt`, `surface_azimuth`,
`module_parameters`, `temperature_model_parameters`, `modules_per_string`,
`strings_per_inverter`, `albedo`, `surface_type`, `module_type`, and
`racking_model`.

.. ipython:: python

    array_one = pvsystem.Array(surface_tilt=30, surface_azimuth=90)
    array_two = pvsystem.Array(surface_tilt=30, surface_azimuth=220)
    system = pvsystem.PVSystem(arrays=[array_one, array_two])
    system.num_arrays

When instantiating a :py:class:`~pvlib.pvsystem.PVSystem` with a list
of :py:class:`~pvlib.pvsystem.Array`, each parameter must be specified individually
for each array when the :py:class:`~pvlib.pvsystem.PVSystem` instances are constructed.
For example, if all arrays are at the same tilt you must specify that tilt for
every array. When using a list of :py:class:`~pvlib.pvsystem.Array` you shouldn't
also pass any attributes for Arrays to the `PVSystem` attributes; these values
are ignored.

The output of `PVSystem` methods and attributes changes when the system has
multiple arrays. Accessing any of Array attributes on the PVSystem object returns
return a tuple with the value of the attribute for each array, in
the same order as the `PVSystem.arrays` parameter. For example, using the system
constructed above:

.. ipython:: python

    system.surface_tilt
    system.surface_azimuth

Similarly, other `PVSystem` methods expect tuples as input and return tuples
for values that differ among arrays.

.. ipython:: python

    aoi = system.get_aoi(solar_zenith=30, solar_azimuth=180)
    print(aoi)
    system.get_iam(aoi)

.. _sat:

SingleAxisTracker
-----------------

The :py:class:`~pvlib.tracking.SingleAxisTracker` is a subclass of
:py:class:`~pvlib.pvsystem.PVSystem`. The SingleAxisTracker class
includes a few more keyword arguments and attributes that are specific
to trackers, plus the
:py:meth:`~pvlib.tracking.SingleAxisTracker.singleaxis` method. It also
overrides the `get_aoi` and `get_irradiance` methods.
