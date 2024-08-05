.. _faq:

Frequently Asked Questions
==========================

General Questions
*****************

What is pvlib?
--------------

pvlib is a free and open-source python software library for modeling
the electrical performance of solar photovoltaic (PV) systems.  It provides
implementations of scientific models for many topics relevant for PV modeling.

For additional details about the project, see :ref:`package_overview`.
For examples of using pvlib, see :ref:`example_gallery`.


How does pvlib compare to other PV modeling tools like PVsyst or SAM?
---------------------------------------------------------------------

pvlib is similar to tools like PVsyst and SAM in that it can be used
for "weather-to-power" modeling to model system energy production
based on system configuration and a weather dataset.  However, pvlib
is also very different in that you use pvlib via python code instead
of via a GUI, which makes pvlib ideal for automating tasks.  pvlib
is also more of a toolbox or a framework to use
to build your own modeling process (although some pre-built workflows
are available as well).


Usage Questions
***************

All I have is GHI, how do I get to POA?
---------------------------------------

Going from GHI to plane of array (POA) irradiance is a two-step process. 
The first step is to
use a decomposition model (also called a separation model) to estimate the
DNI and DHI corresponding to your GHI.  For a list of decomposition
models available in pvlib, see :ref:`dniestmodels`.

The second step is to transpose those estimated DNI and DHI components into
POA components.  This is most easily done with the
:py:func:`pvlib.irradiance.get_total_irradiance` function.


Where can I get irradiance data for my simulation?
--------------------------------------------------

pvlib has a module called iotools which has several functions for
retrieving irradiance data as well as reading standard file formats
such as EPW, TMY2, and TMY3. For free irradiance data, you may
consider NREL's NSRDB which can be accessed using the
:py:func:`pvlib.iotools.get_psm3` function and is available for
North America. For Europe and Africa, you may consider looking into
CAMS (:py:func:`pvlib.iotools.get_cams`).
PVGIS (:py:func:`pvlib.iotools.get_pvgis_hourly`) is another option, which
provides irradiance from several different databases with near global coverage.
pvlib also has functions for accessing a plethora of ground-measured
irradiance datasets, including the BSRN, SURFRAD, SRML, and NREL's MIDC.


Can I use PVsyst (PAN/OND) files with pvlib?
--------------------------------------------

Although pvlib includes a function to read PAN and OND files
(:py:func:`~pvlib.iotools.read_panond`), it is up to the user to determine
whether and how the imported parameter values can be used with pvlib's models.
Easier use of these parameter files with the rest of pvlib may be added
in a future version.  Until then, these Google Group threads
(`one <https://groups.google.com/g/pvlib-python/c/PDDic0SS6ao/m/Z-WKj7C6BwAJ>`_
and `two <https://groups.google.com/g/pvlib-python/c/b1mf4Y1qHBY/m/tK2FBCJyBgAJ>`_)
may be useful for some users.


Why don't my simulation results make sense? 
-------------------------------------------

pvlib does not prevent you from using models improperly and generating
invalid results.  It is on you as the user to understand the models you
are using and to supply appropriate, correctly-formatted data.  One modeling error that beginners sometimes
make is improper time zone localization. Calculating solar
positions is often the first step of a modeling process
and this step relies on timestamps being localized to the correct time zone.
A telltale sign of improper time zones is a time shift between solar
position and the irradiance data (for example, ``solar_elevation``
peaks at a different time from clear-sky ``ghi``).
For more information on handling timezone correctly, see :ref:`timetimezones`.

More generally, inspecting the simulation results visually is a good first
step when investigating strange results.
Matplotlib and pandas have very powerful plotting capabilities that are great
for tracking down where things went wrong in a modeling process.  Try plotting
a few days of intermediate time series results in a single plot, looking for
inconsistencies like nonzero irradiance when the sun is below the horizon.
This will give you a clue of where to look for errors in your code.


I got a warning like ``RuntimeWarning: invalid value encountered in arccos``, what does it mean?
------------------------------------------------------------------------------------------------

It is fairly common to use pvlib models in conditions where they are not
applicable, for example attempting to calculate an IV curve at night.
In such cases the model failure doesn't really matter (nighttime values are
irrelevant), but the numerical packages that pvlib is built on
(e.g. `numpy <https://numpy.org>`_) emit warnings complaining about
`invalid value`, `divide by zero`, etc.  In these cases the warnings can
often be ignored without issue.

However, that's not always the case: sometimes these warnings are caused
by an error in your code, for example by giving a function inappropriate inputs.
So, these warnings don't necessarily indicate a problem, but you shouldn't
get in the habit of immediately discounting them either.


I got an error like ``X has no attribute Y``, what does it mean?
----------------------------------------------------------------

If you see a function in the pvlib documentation that doesn't seem to exist
in your pvlib installation, the documentation is likely for a different version
of pvlib.  You can check your installed pvlib version by running
``print(pvlib.__version__)`` in python.  To switch documentation versions, use
the `v:` version switcher widget in the bottom left corner of this page.

You can also upgrade your installed pvlib to the latest compatible version
with ``pip install -U pvlib``, but be sure to check the :ref:`whatsnew`
page to see the differences between versions.


The CEC table doesn't include my module or inverter, what should I do?
----------------------------------------------------------------------

The CEC tables for module and inverter parameters included in pvlib are periodically
copied from `SAM <https://github.com/NREL/SAM/tree/develop/deploy/libraries>`_,
so you can check the tables there for more up-to-date tables.

For modules, if even the SAM files don't include the module you're looking for
either, you can calculate CEC module model parameters from
datasheet information using :py:func:`pvlib.ivtools.sdm.fit_cec_sam`.


Which should I use, the CEC or the Sandia PV Module database?
-------------------------------------------------------------

The CEC PV module database contains parameters for significantly more
modules, and is more up to date, than the Sandia PV module database.
Therefore, the CEC PV module database is probably the more useful option
in most cases.  However, finding parameters for the specific module
being used is usually more important than which database they came from.

Besides which modules each database includes, another consideration is the
different modeling capabilities each parameter set provides.  The CEC model
produces a continuous IV curve while the Sandia model calculates only a few
specific points of interest on the curve.  For typical simulations where
only the maximum power point is of interest, either model will suffice.


How do I model a system with multiple inverters?
------------------------------------------------

Currently, pvlib's :ref:`modelchaindoc` and :ref:`pvsystemdoc` only support
simulating one inverter at a time.  To calculate total power for multiple inverters,
there are two options:

If the modules, mounting, stringing, and inverters are all identical for each
inverter, then you may simply simulate one inverter and multiply the
``ModelChainResult.ac`` by the number of inverters to get the total system output.

If the inverters or their arrays are not all identical,
define one ``PVSystem`` and ``ModelChain`` per inverter and
run the simulation for each of them individually.  From there you
can add up the inverter-level outputs to get the total system output.

