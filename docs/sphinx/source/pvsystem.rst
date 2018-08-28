.. _pvsystem:

PVSystem
========

The :py:class:`~.pvsystem.PVSystem` class wraps many of the functions in
the :py:mod:`~.pvsystem` module. This simplifies the API by eliminating
the need for a user to specify arguments such as module and
inverter properties when calling PVSystem methods.

This guide aims to build understanding of the PVSystem class. It assumes
basic familiarity with object-oriented code in Python, but most
information should be understandable without a solid understanding of
classes. Keep in mind that `functions` are independent of objects,
while `methods` are attached to objects.

See the :ref:`modelchain` documentation for a similar guide
focused on using PVSystem objects in time series modeling applications.


.. _designphilosophy:

Design philosophy
-----------------

The PVSystem class is designed to separate the data that represents a PV
system (e.g. tilt angle or module parameters) from the data that
influences the PV system (e.g. the weather).

The data that represents the PV system is *intrinsic*. The
data that influences the PV system is *extrinsic*.

Intrinsic data is stored in object attributes. For example, the data
that describes a PV system's module parameters is stored in
`PVSystem.module_parameters`.

.. ipython::

    module_parameters = {'pdc0': 10, 'gamma_pdc': -0.004}
    system = pvsystem.PVSystem(module_parameters=module_parameters)
    print(system.module_parameters)

Extrinsic data is passed to a PVSystem as method arguments. For example,
the :py:meth:`~pvsystem.PVSystem.pvwatts_dc` method accepts extrinsic
data irradiance and temperature.

.. ipython::

    pdc = system.pvwatts_dc(1000, 30)
    print(pdc)

Compare the :py:meth:`~pvsystem.PVSystem.pvwatts_dc` method signature
to the :py:func:`~pvsystem.pvwatts_dc` function signature.

:py:meth:`~pvsystem.PVSystem.pvwatts_dc`: ``pvwatts_dc(g_poa_effective, temp_cell)``
:py:func:`~pvsystem.pvwatts_dc`: ``pvwatts_dc(g_poa_effective, temp_cell, pdc0, gamma_pdc, temp_ref=25.)``

How does this work? The :py:meth:`~pvsystem.PVSystem.pvwatts_dc` method
looks in `PVSystem.module_parameters` for the `pdc0`, `gamma_pdc`
arguments. Then the :py:meth:`~pvsystem.PVSystem.pvwatts_dc` calls the
:py:func:`~pvsystem.pvwatts_dc` function with all of the arguments and
returns the result to the user. Note that the function includes a
default value for the parameter `temp_ref`. This default value may be
overridden by specifying the `temp_ref` key in the
`PVSystem.module_parameters` dictionary.

.. ipython::

    system.module_parameters['temp_ref'] = 0
    pdc = system.pvwatts_dc(1000, 30)
    print(pdc)

Multiple methods may pull data from the same attribute. For example, the
`PVSystem.module_parameters` attribute is used by the DC model methods
as well as the incidence angle modifier methods.


.. _pvsystemattributes:

PVSystem attributes
-------------------

Here we review the most commonly used PVSystem attributes.
Please see the :py:class:`~.pvsystem.PVSystem` class documentation for a
comprehensive list.

`module_parameters` and `inverter_parameters` contain the data
necessary for computing DC and AC power using one of the available
PVSystem methods. These are typically specified using data from
the :py:func:`~pvsystem.retreive_sam` function:

.. ipython::

    modules = pvsystem.retrieve_sam('cecmod')
    module_parameters = modules['Example_Module']
    inverters = pvsystem.retrieve_sam('cecinverter')
    inverter_parameters = inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']
    system = pvsystem.PVSystem(module_parameters=module_parameters, inverter_parameters=inverter_parameters)

As shown above, the parameters can also be specified manually.
This is useful for specifying modules and inverters that are not
included in the supplied databases. It is also useful for specifying
systems for use with the PVWatts models, as demonstrated in
:ref:`designphilosophy`.

The `losses_parameters` attribute contains data that may be used with
methods that calculate system losses. At present, this is only incudes
:py:meth:`~PVSystem.pvwatts_losses` and
:py:func:`~pvsystem.pvwatts_losses`, but we hope to add more functions
and methods in the future.

modules_per_string and strings_per_inverter

miscellaneous attributes


.. _sat:

SingleAxisTracker
-----------------

SingleAxisTracker is a subclass of PVSystem