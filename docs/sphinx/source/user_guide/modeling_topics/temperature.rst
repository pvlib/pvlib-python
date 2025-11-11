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
  of a PV module.  Easy to measure, but usually a few degrees less
  than the cell temperature which determines efficiency.
- *cell temperature*: the temperature of the PV cell itself.  The relevant
  temperature for PV modeling, but almost never measured directly.

Cell temperature is typically thought to be slightly higher than module
temperature.
Temperature models estimate these quantities using inputs like incident
irradiance, ambient temperature, and wind speed.  Each model also takes
a set of parameter values that represent how a PV module responds to
those inputs.  Parameter values generally depend on both the PV
module technologies and the mounting conditions of the module.

Another way to classify temperature models is whether they account for
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

+-------------------------------------------+--------+------------+--------------------------------------------------------------------+
| Model                                     | Type   | Transient? | Inputs                                                             |
|                                           |        |            +----------------+---------------------+------------+----------------+
|                                           |        |            | POA irradiance | Ambient temperature | Wind speed | Downwelling IR |
+===========================================+========+============+================+=====================+============+================+
| :py:func:`~pvlib.temperature.faiman`      | either |            | ✓              | ✓                   | ✓          |                |
+-------------------------------------------+--------+------------+----------------+---------------------+------------+----------------+
| :py:func:`~pvlib.temperature.faiman_rad`  | either |            | ✓              | ✓                   | ✓          | ✓              |
+-------------------------------------------+--------+------------+----------------+---------------------+------------+----------------+
| :py:func:`~pvlib.temperature.fuentes`     | cell   | ✓          | ✓              | ✓                   | ✓          |                |
+-------------------------------------------+--------+------------+----------------+---------------------+------------+----------------+
| :py:func:`~pvlib.temperature.noct_sam`    | cell   |            | ✓              | ✓                   | ✓          |                |
+-------------------------------------------+--------+------------+----------------+---------------------+------------+----------------+
| :py:func:`~pvlib.temperature.pvsyst_cell` | cell   |            | ✓              | ✓                   | ✓          |                |
+-------------------------------------------+--------+------------+----------------+---------------------+------------+----------------+
| :py:func:`~pvlib.temperature.ross`        | cell   |            | ✓              | ✓                   |            |                |
+-------------------------------------------+--------+------------+----------------+---------------------+------------+----------------+
| :py:func:`~pvlib.temperature.sapm_cell`   | cell   |            | ✓              | ✓                   | ✓          |                |
+-------------------------------------------+--------+------------+----------------+---------------------+------------+----------------+
| :py:func:`~pvlib.temperature.sapm_module` | module |            | ✓              | ✓                   | ✓          |                |
+-------------------------------------------+--------+------------+----------------+---------------------+------------+----------------+


Model parameters
----------------

Some temperature model functions provide default values for their parameters,
and several additional sets of temperature model parameter values are
available in :py:data:`pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS`.
However, these generic values may not be suitable for all modules and mounting
configurations.

Module-specific values can be obtained via testing, for example following
the IEC 61853-2 standard.

Currently, pvlib provides no functionality for fitting parameter values
using measured temperature.


Other functions
---------------

pvlib also provides a few other functions for temperature modeling:

- :py:func:`~pvlib.temperature.prilliman`: an "add-on" model that reprocesses
  the output of a steady-state model to apply transient effects.
- :py:func:`~pvlib.temperature.sapm_cell_from_module`: a model for
  estimating cell temperature from module temperature.
- :py:func:`~pvlib.temperature.generic_linear`: a generic linear model form,
  equivalent to several conventional temperature models.
