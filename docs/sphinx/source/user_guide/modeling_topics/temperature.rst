.. _temperature:

Temperature models
==================

pvlib provides a variety of models for predicting the operating temperature
of a PV module from irradiance and weather inputs.  These models range from
simple empirical equations requiring just a few multiplications to more complex
thermal balance models with numerical integration.

Types of models
---------------

Temperature models predict one of two quantities:

- *module temperature*: the temperature as measured at the back surface
  of a PV module.  Easy to measure, but usually marginally less
  than the cell temperature which determines efficiency.
- *cell temperature*: the temperature of the PV cell itself.  The relevant
  temperature for PV modeling, but almost never measured directly.

Temperature models estimate these quantities using inputs like incident
irradiance, ambient temperature, and wind speed.  Each model also takes
a set of parameter values that represent how a PV module responds to
those inputs.

Parameter values generally depend on both the PV
module technologies, the mounting configuration of the module,
and on any weather parameters that are not included in the model.
Note that, despite models conventionally being associated with either
cell or module temperature, it is actually the parameter values that determine
which of the two temperatures are predicted, as they will produce the same
type of temperature from which they were originally derived.

Another aspect of temperature models is whether they account for
the thermal inertia of a PV module.  Temperature models are either:

- *steady-state*: the module is assumed to have been at the specified operating
  conditions for a sufficiently long time for its temperature to reach
  equilibrium.
- *transient*: the module's thermal inertia is included in the model,
  causing a lag in modeled temperature change following changes in the inputs.

Other effects that temperature models may consider include the
photoconversion efficiency and radiative cooling.

The temperature models currently available in pvlib are summarized in the
following table:

+----------------------------------------------+--------+------------+---------------------------------------------------------------------------+
| Model                                        | Type   | Transient? | Weather inputs                                                            |
|                                              |        |            +----------------+---------------------+------------+-----------------------+
|                                              |        |            | POA irradiance | Ambient temperature | Wind speed | Downwelling IR [#f1]_ |
+==============================================+========+============+================+=====================+============+=======================+
| :py:func:`~pvlib.temperature.faiman`         | either |            | ✓              | ✓                   | ✓          |                       |
+----------------------------------------------+--------+------------+----------------+---------------------+------------+-----------------------+
| :py:func:`~pvlib.temperature.faiman_rad`     | either |            | ✓              | ✓                   | ✓          | ✓                     |
+----------------------------------------------+--------+------------+----------------+---------------------+------------+-----------------------+
| :py:func:`~pvlib.temperature.fuentes`        | either | ✓          | ✓              | ✓                   | ✓          |                       |
+----------------------------------------------+--------+------------+----------------+---------------------+------------+-----------------------+
| :py:func:`~pvlib.temperature.generic_linear` | either |            | ✓              | ✓                   | ✓          |                       |
+----------------------------------------------+--------+------------+----------------+---------------------+------------+-----------------------+
| :py:func:`~pvlib.temperature.noct_sam`       | cell   |            | ✓              | ✓                   | ✓          |                       |
+----------------------------------------------+--------+------------+----------------+---------------------+------------+-----------------------+
| :py:func:`~pvlib.temperature.pvsyst_cell`    | cell   |            | ✓              | ✓                   | ✓          |                       |
+----------------------------------------------+--------+------------+----------------+---------------------+------------+-----------------------+
| :py:func:`~pvlib.temperature.ross`           | cell   |            | ✓              | ✓                   |            |                       |
+----------------------------------------------+--------+------------+----------------+---------------------+------------+-----------------------+
| :py:func:`~pvlib.temperature.sapm_cell`      | cell   |            | ✓              | ✓                   | ✓          |                       |
+----------------------------------------------+--------+------------+----------------+---------------------+------------+-----------------------+
| :py:func:`~pvlib.temperature.sapm_module`    | module |            | ✓              | ✓                   | ✓          |                       |
+----------------------------------------------+--------+------------+----------------+---------------------+------------+-----------------------+

.. [#f1] Downwelling infrared radiation.

In addition to the core models above, pvlib provides several other functions
for temperature modeling:

- :py:func:`~pvlib.temperature.prilliman`: an "add-on" model that reprocesses
  the output of a steady-state model to apply transient effects.
- :py:func:`~pvlib.temperature.sapm_cell_from_module`: a model for
  estimating cell temperature from module temperature.


Model parameters
----------------

Some temperature model functions provide default values for their parameters,
and several additional sets of temperature model parameter values are
available in :py:data:`pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS`.
However, these generic values may not be suitable for all modules and mounting
configurations. It should be noted that using the default parameter values for each 
model generally leads to different modules temperature predictions. This alone
does not mean one model is better than another; it's just evidence that the measurements
used to derive the default parameter values were taken on different PV systems in different
locations under different conditions.

Parameter values for one model (e.g. ``u0``, ``u1`` for :py:func:`~pvlib.temperature.faiman`)
can be converted to another model (e.g. ``u_c``, ``u_v`` for :py:func:`~pvlib.temperature.pvsyst_cell`)
using :py:class:`~pvlib.temperature.GenericLinearModel`.

Module-specific values can be obtained via testing, for example following
the IEC 61853-2 standard for the Faiman model; however, such values still do not capture 
the dependency of temperature on system design and other variables.

Currently, pvlib provides no functionality for fitting parameter values
using measured temperature.
