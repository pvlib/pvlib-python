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

    naive_times = pd.DatetimeIndex(start='2015', end='2016', freq='1h')

    # very approximate
    # latitude, longitude, name, altitude, timezone
    coordinates = [(30, -110, 'Tucson', 700, 'Etc/GMT+7'),
                   (35, -105, 'Albuquerque', 1500, 'Etc/GMT+7'),
                   (40, -120, 'San Francisco', 10, 'Etc/GMT+8'),
                   (50, 10, 'Berlin', 34, 'Etc/GMT-1')]

    import pvlib

    # get the module and inverter specifications from SAM
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
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

    for latitude, longitude, name, altitude, timezone in coordinates:
        times = naive_times.tz_localize(timezone)
        system['surface_tilt'] = latitude
        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
        dni_extra = pvlib.irradiance.extraradiation(times)
        dni_extra = pd.Series(dni_extra, index=times)
        airmass = pvlib.atmosphere.relativeairmass(solpos['apparent_zenith'])
        pressure = pvlib.atmosphere.alt2pres(altitude)
        am_abs = pvlib.atmosphere.absoluteairmass(airmass, pressure)
        tl = pvlib.clearsky.lookup_linke_turbidity(times, latitude, longitude)
        cs = pvlib.clearsky.ineichen(solpos['apparent_zenith'], am_abs, tl,
                                     dni_extra=dni_extra, altitude=altitude)
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
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
            total_irrad['poa_direct'], total_irrad['poa_diffuse'],
            am_abs, aoi, module)
        dc = pvlib.pvsystem.sapm(effective_irradiance, temps['temp_cell'], module)
        ac = pvlib.pvsystem.snlinverter(dc['v_mp'], dc['p_mp'], inverter)
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
    for latitude, longitude, name, altitude, timezone in coordinates:
        dc, ac = basic_chain(naive_times.tz_localize(timezone),
                             latitude, longitude,
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
    for latitude, longitude, name, altitude, timezone in coordinates:
        location = Location(latitude, longitude, name=name, altitude=altitude,
                            tz=timezone)
        # very experimental
        mc = ModelChain(system, location,
                        orientation_strategy='south_at_latitude_tilt')
        # model results (ac, dc) and intermediates (aoi, temps, etc.)
        # assigned as mc object attributes
        mc.run_model(naive_times.tz_localize(timezone))
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
    for latitude, longitude, name, altitude, timezone in coordinates:
        localized_system = LocalizedPVSystem(module_parameters=module,
                                             inverter_parameters=inverter,
                                             surface_tilt=latitude,
                                             surface_azimuth=180,
                                             latitude=latitude,
                                             longitude=longitude,
                                             name=name,
                                             altitude=altitude,
                                             tz=timezone)
        times = naive_times.tz_localize(timezone)
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
        effective_irradiance = localized_system.sapm_effective_irradiance(
            total_irrad['poa_direct'], total_irrad['poa_diffuse'],
            airmass['airmass_absolute'], aoi)
        dc = localized_system.sapm(effective_irradiance, temps['temp_cell'])
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
