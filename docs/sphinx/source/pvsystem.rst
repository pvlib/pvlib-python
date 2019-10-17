.. _pvsystemdoc:

PVSystem
========

.. ipython:: python
   :suppress:

    import pandas as pd
    from pvlib import pvsystem


The :py:class:`~pvlib.pvsystem.PVSystem` class wraps many of the
functions in the :py:mod:`~pvlib.pvsystem` module. This simplifies the
API by eliminating the need for a user to specify arguments such as
module and inverter properties when calling PVSystem methods.
:py:class:`~pvlib.pvsystem.PVSystem` is not better or worse than the
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

Intrinsic data is stored in object attributes. For example, the data
that describes a PV system's module parameters is stored in
`PVSystem.module_parameters`.

.. ipython:: python

    module_parameters = {'pdc0': 10, 'gamma_pdc': -0.004}
    system = pvsystem.PVSystem(module_parameters=module_parameters)
    print(system.module_parameters)

Extrinsic data is passed to a PVSystem as method arguments. For example,
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

Here we review the most commonly used PVSystem attributes. Please see
the :py:class:`~pvlib.pvsystem.PVSystem` class documentation for a
comprehensive list.

The first PVSystem parameters are `surface_tilt` and `surface_azimuth`.
These parameters are used in PVSystem methods such as
:py:meth:`~pvlib.pvsystem.PVSystem.get_aoi` and
:py:meth:`~pvlib.pvsystem.PVSystem.get_irradiance`. Angle of incidence
(AOI) calculations require `surface_tilt`, `surface_azimuth` and also
the sun position. The :py:meth:`~pvlib.pvsystem.PVSystem.get_aoi` method
uses the `surface_tilt` and `surface_azimuth` attributes in its PVSystem
object, and so requires only `solar_zenith` and `solar_azimuth` as
arguments.

.. ipython:: python

    # 20 deg tilt, south-facing
    system = pvsystem.PVSystem(surface_tilt=20, surface_azimuth=180)
    print(system.surface_tilt, system.surface_azimuth)

    # call get_aoi with solar_zenith, solar_azimuth
    aoi = system.get_aoi(30, 180)
    print(aoi)


`module_parameters` and `inverter_parameters` contain the data
necessary for computing DC and AC power using one of the available
PVSystem methods. These are typically specified using data from
the :py:func:`~pvlib.pvsystem.retrieve_sam` function:

.. ipython:: python

    # retrieve_sam returns a dict. the dict keys are module names,
    # and the values are model parameters for that module
    modules = pvsystem.retrieve_sam('cecmod')
    module_parameters = modules['Canadian_Solar_Inc__CS5P_220M']
    inverters = pvsystem.retrieve_sam('cecinverter')
    inverter_parameters = inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    system = pvsystem.PVSystem(module_parameters=module_parameters, inverter_parameters=inverter_parameters)


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
example, consider a system with 35 modules arranged into 5 strings of 7
modules each.

.. ipython:: python

    system = pvsystem.PVSystem(modules_per_string=7, strings_per_inverter=5)
    # crude numbers from a single module
    data = pd.DataFrame({'v_mp': 8, 'v_oc': 10, 'i_mp': 5, 'i_x': 6,
                         'i_xx': 4, 'i_sc': 7, 'p_mp': 40}, index=[0])
    data_scaled = system.scale_voltage_current_power(data)
    print(data_scaled)


.. _sat:

SingleAxisTracker
-----------------

The :py:class:`~pvlib.tracking.SingleAxisTracker` is a subclass of
:py:class:`~pvlib.pvsystem.PVSystem`. The SingleAxisTracker class
includes a few more keyword arguments and attributes that are specific
to trackers, plus the
:py:meth:`~pvlib.tracking.SingleAxisTracker.singleaxis` method. It also
overrides the `get_aoi` and `get_irradiance` methods.