.. _package_overview:

Package Overview
================

Introduction
------------

The core mission of pvlib-python is to provide open, reliable,
interoperable, and benchmark implementations of PV system models.

There are at least as many opinions about how to model PV systems as
there are modelers of PV systems, so
pvlib-python provides several modeling paradigms.


.. _modeling-paradigms:

Modeling paradigms
------------------

The backbone of pvlib-python
is well-tested procedural code that implements PV system models.
pvlib-python also provides a collection of classes for users
that prefer object-oriented programming.
These classes can help users keep track of data in a more organized way,
provide some "smart" functions with more flexible inputs,
and simplify the modeling process for common situations.
The classes do not add any algorithms beyond what's available
in the procedural code, and most of the object methods
are simple wrappers around the corresponding procedural code.

Let's use each of these pvlib modeling paradigms
to calculate the yearly energy yield for a given hardware
configuration at a handful of sites listed below.

.. ipython:: python

    import pandas as pd
    import matplotlib.pyplot as plt

    # seaborn makes the plots look nicer
    import seaborn as sns
    sns.set_color_codes()

    times = pd.DatetimeIndex(start='2015', end='2016', freq='1h')

    # very approximate
    # latitude, longitude, name, altitude
    coordinates = [(30, -110, 'Tucson', 700),
                   (35, -105, 'Albuquerque', 1500),
                   (40, -120, 'San Francisco', 10),
                   (50, 10, 'Berlin', 34)]

    import pvlib

    # get the module and inverter specifications from SAM
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('sandiainverter')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208_208V__CEC_2014_']

    # specify constant ambient air temp and wind for simplicity
    temp_air = 20
    wind_speed = 0


Procedural
^^^^^^^^^^

The straightforward procedural code can be used for all modeling
steps in pvlib-python.

The following code demonstrates how to use the procedural code
to accomplish our system modeling goal:

.. ipython:: python

    system = {'module': module, 'inverter': inverter,
              'surface_azimuth': 180}

    energies = {}
    for latitude, longitude, name, altitude in coordinates:
        system['surface_tilt'] = latitude
        cs = pvlib.clearsky.ineichen(times, latitude, longitude, altitude=altitude)
        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
        dni_extra = pvlib.irradiance.extraradiation(times)
        dni_extra = pd.Series(dni_extra, index=times)
        airmass = pvlib.atmosphere.relativeairmass(solpos['apparent_zenith'])
        pressure = pvlib.atmosphere.alt2pres(altitude)
        am_abs = pvlib.atmosphere.absoluteairmass(airmass, pressure)
        aoi = pvlib.irradiance.aoi(system['surface_tilt'], system['surface_azimuth'],
                                   solpos['apparent_zenith'], solpos['azimuth'])
        total_irrad = pvlib.irradiance.total_irrad(system['surface_tilt'],
                                                   system['surface_azimuth'],
                                                   solpos['apparent_zenith'],
                                                   solpos['azimuth'],
                                                   cs['dni'], cs['ghi'], cs['dhi'],
                                                   dni_extra=dni_extra,
                                                   model='haydavies')
        temps = pvlib.pvsystem.sapm_celltemp(total_irrad['poa_global'],
                                             wind_speed, temp_air)
        dc = pvlib.pvsystem.sapm(module, total_irrad['poa_direct'],
                                 total_irrad['poa_diffuse'], temps['temp_cell'],
                                 am_abs, aoi)
        ac = pvlib.pvsystem.snlinverter(inverter, dc['v_mp'], dc['p_mp'])
        annual_energy = ac.sum()
        energies[name] = annual_energy

    energies = pd.Series(energies)

    # based on the parameters specified above, these are in W*hrs
    print(energies.round(0))

    energies.plot(kind='bar', rot=0)
    @savefig proc-energies.png width=6in
    plt.ylabel('Yearly energy yield (W hr)')

pvlib-python provides a :py:func:`~pvlib.modelchain.basic_chain`
function that implements much of the code above. Use this function with
a full understanding of what it is doing internally!

.. ipython:: python

    from pvlib.modelchain import basic_chain

    energies = {}
    for latitude, longitude, name, altitude in coordinates:
        dc, ac = basic_chain(times, latitude, longitude,
                             module, inverter,
                             altitude=altitude,
                             orientation_strategy='south_at_latitude_tilt')
        annual_energy = ac.sum()
        energies[name] = annual_energy

    energies = pd.Series(energies)

    # based on the parameters specified above, these are in W*hrs
    print(energies.round(0))

    energies.plot(kind='bar', rot=0)
    @savefig basic-chain-energies.png width=6in
    plt.ylabel('Yearly energy yield (W hr)')


Object oriented (Location, PVSystem, ModelChain)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first object oriented paradigm uses a model where a
:py:class:`~pvlib.pvsystem.PVSystem` object represents an assembled
collection of modules, inverters, etc., a
:py:class:`~pvlib.location.Location` object represents a particular
place on the planet, and a :py:class:`~pvlib.modelchain.ModelChain`
object describes the modeling chain used to calculate PV output at that
Location. This can be a useful paradigm if you prefer to think about the
PV system and its location as separate concepts or if you develop your
own ModelChain subclasses. It can also be helpful if you make extensive
use of Location-specific methods for other calculations.

The following code demonstrates how to use
:py:class:`~pvlib.location.Location`,
:py:class:`~pvlib.pvsystem.PVSystem`, and
:py:class:`~pvlib.modelchain.ModelChain`
objects to accomplish our system modeling goal:

.. ipython:: python

    from pvlib.pvsystem import PVSystem
    from pvlib.location import Location
    from pvlib.modelchain import ModelChain

    system = PVSystem(module_parameters=module,
                      inverter_parameters=inverter)

    energies = {}
    for latitude, longitude, name, altitude in coordinates:
        location = Location(latitude, longitude, name=name, altitude=altitude)
        # very experimental
        mc = ModelChain(system, location,
                        orientation_strategy='south_at_latitude_tilt')
        # model results (ac, dc) and intermediates (aoi, temps, etc.)
        # assigned as mc object attributes
        mc.run_model(times)
        annual_energy = mc.ac.sum()
        energies[name] = annual_energy

    energies = pd.Series(energies)

    # based on the parameters specified above, these are in W*hrs
    print(energies.round(0))

    energies.plot(kind='bar', rot=0)
    @savefig modelchain-energies.png width=6in
    plt.ylabel('Yearly energy yield (W hr)')


Object oriented (LocalizedPVSystem)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second object oriented paradigm uses a model where a
:py:class:`~pvlib.pvsystem.LocalizedPVSystem` represents a
PV system at a particular place on the planet. This can be a useful
paradigm if you're thinking about a power plant that already exists.

The following code demonstrates how to use a
:py:class:`~pvlib.pvsystem.LocalizedPVSystem`
object to accomplish our modeling goal:

.. ipython:: python

    from pvlib.pvsystem import LocalizedPVSystem

    energies = {}
    for latitude, longitude, name, altitude in coordinates:
        localized_system = LocalizedPVSystem(module_parameters=module,
                                             inverter_parameters=inverter,
                                             surface_tilt=latitude,
                                             surface_azimuth=180,
                                             latitude=latitude,
                                             longitude=longitude,
                                             name=name,
                                             altitude=altitude)
        clearsky = localized_system.get_clearsky(times)
        solar_position = localized_system.get_solarposition(times)
        total_irrad = localized_system.get_irradiance(solar_position['apparent_zenith'],
                                                      solar_position['azimuth'],
                                                      clearsky['dni'],
                                                      clearsky['ghi'],
                                                      clearsky['dhi'])
        temps = localized_system.sapm_celltemp(total_irrad['poa_global'],
                                               wind_speed, temp_air)
        aoi = localized_system.get_aoi(solar_position['apparent_zenith'],
                                       solar_position['azimuth'])
        airmass = localized_system.get_airmass(solar_position=solar_position)
        dc = localized_system.sapm(total_irrad['poa_direct'],
                                   total_irrad['poa_diffuse'],
                                   temps['temp_cell'],
                                   airmass['airmass_absolute'],
                                   aoi)
        ac = localized_system.snlinverter(dc['v_mp'], dc['p_mp'])
        annual_energy = ac.sum()
        energies[name] = annual_energy

    energies = pd.Series(energies)

    # based on the parameters specified above, these are in W*hrs
    print(energies.round(0))

    energies.plot(kind='bar', rot=0)
    @savefig localized-pvsystem-energies.png width=6in
    plt.ylabel('Yearly energy yield (W hr)')


User extensions
---------------
There are many other ways to organize PV modeling code. We encourage you
to build on these paradigms and to share your experiences with the pvlib
community via issues and pull requests.


Getting support
---------------
The best way to get support is to make an issue on our
`GitHub issues page <https://github.com/pvlib/pvlib-python/issues>`_ .


How do I contribute?
--------------------
We're so glad you asked! Please see our
`wiki <https://github.com/pvlib/pvlib-python/wiki/Contributing-to-pvlib-python>`_
for information and instructions on how to contribute.
We really appreciate it!


Credits
-------
The pvlib-python community thanks Sandia National Lab
for developing PVLIB Matlab and for supporting
Rob Andrews of Calama Consulting to port the library to Python.
Will Holmgren thanks the DOE EERE Postdoctoral Fellowship program
for support.
The pvlib-python maintainers thank all of pvlib's contributors of issues
and especially pull requests.
The pvlib-python community thanks all of the
maintainers and contributors to the PyData stack.

