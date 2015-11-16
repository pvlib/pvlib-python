Package Overview
================

Introduction
------------

The core mission of pvlib-python is to provide open, reliable,
interoperable, and benchmark implementations of PV system models.

There are at least as many opinions about how to model PV systems as
there are modelers of PV systems, so 
pvlib-python provides several modeling paradigms.


Modeling paradigms
------------------

The backbone of pvlib-python
is well-tested procedural code that implements PV system models.
pvlib-python also provides a collection of classes for users
that prefer object-oriented programming.
These classes can help users keep track of data in a more organized way,
and can help to simplify the modeling process.
The classes do not add any functionality beyond the procedural code.
Most of the object methods are simple wrappers around the
corresponding procedural code.

Let's use each of these pvlib modeling paradigms
to calculate the yearly energy yield for a given hardware
configuration at a handful of sites listed below ::

    import pandas as pd
    
    times = pd.DatetimeIndex(start='2015', end='2016', freq='1h')
    
    # very approximate
    coordinates = [(30, -110, 'Tucson'),
                   (35, -105, 'Albuquerque'),
                   (40, -120, 'San Francisco'),
                   (50, 10, 'Berlin')]

None of these examples are complete!
Should replace the clear sky assumption with TMY or similar
(or leave as an exercise to the reader?).


Procedural
^^^^^^^^^^

Procedural code can be used to for all modeling steps in pvlib-python.

The following code demonstrates how to use the procedural code
to accomplish our system modeling goal: ::

    import pvlib
    
    system = {'module': module, 'inverter': inverter,
              'surface_azimuth': 180, **other_params}

    energies = {}
    for latitude, longitude, name in coordinates:
        cs = pvlib.clearsky.ineichen(times, latitude, longitude)
        solpos = pvlib.solarposition.get_solarposition(times, latitude, longitude)
        system['surface_tilt'] = latitude
        total_irrad = pvlib.irradiance.total_irradiance(**solpos, **cs, **system)
        temps = pvlib.pvsystem.sapm_celltemp(**total_irrad, **system)
        dc = pvlib.pvsystem.sapm(**temps, **total_irrad, **system)
        ac = pvlib.pvsystem.snlinverter(**system, **dc)
        annual_energy = power.sum()
        energies[name] = annual_energy
    
    #energies = pd.DataFrame(energies)
    #energies.plot()


Object oriented (Location, PVSystem, ModelChain)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The first object oriented paradigm uses a model where
a :class:`PVSystem <pvlib.pvsystem.PVSystem>` object represents an
assembled collection of modules, inverters, etc.,
a :class:`Location <pvlib.location.Location>` object represents a
particular place on the planet,
and a :class:`ModelChain <pvlib.modelchain.ModelChain>` object describes
the modeling chain used to calculate PV output at that Location.
This can be a useful paradigm if you prefer to think about
the PV system and its location as separate concepts or if
you develop your own ModelChain subclasses.
It can also be helpful if you make extensive use of Location-specific
methods for other calculations.

The following code demonstrates how to use
:class:`Location <pvlib.location.Location>`,
:class:`PVSystem <pvlib.pvsystem.PVSystem>`, and
:class:`ModelChain <pvlib.modelchain.ModelChain>`
objects to accomplish our system modeling goal: ::
    
    from pvlib.pvsystem import PVSystem
    from pvlib.location import Location
    from pvlib.modelchain import ModelChain
    
    system = PVSystem(module, inverter, **other_params)
    
    energies = {}
    for latitude, longitude, name in coordinates:
        location = Location(latitude, longitude)
        # not yet clear what, exactly, goes into ModelChain(s)
        mc = ModelChain(system, location, times,
                        'south_at_latitude', **other_modelchain_params)
        output = mc.run_model()
        annual_energy = output['power'].sum()
        energies[name] = annual_energy
    
    #energies = pd.DataFrame(energies)
    #energies.plot()


Object oriented (LocalizedPVSystem)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The second object oriented paradigm uses a model where a 
:class:`LocalizedPVSystem <pvlib.pvsystem.LocalizedPVSystem>` represents a
PV system at a particular place on the planet.
This can be a useful paradigm if you're thinking about
a power plant that already exists.

The following code demonstrates how to use a
:class:`LocalizedPVSystem <pvlib.pvsystem.LocalizedPVSystem>`
object to accomplish our modeling goal: ::

    from pvlib.pvsystem import PVSystem, LocalizedPVSystem

    base_system = PVSystem(module, inverter, **other_system_params)

    energies = {}
    for latitude, longitude, name in coordinates:
        localized_system = base_system.localize(latitude, longitude, name=name)
        localized_system.surface_tilt = latitude
        cs = localized_system.get_clearsky(times)
        solpos = localized_system.get_solarposition(times)
        total_irrad = localized_system.get_irradiance(times, **solpos, **cs)
        power = localized_system.get_power(stuff)
        annual_energy = power.sum()
        energies[name] = annual_energy
    
    #energies = pd.DataFrame(energies)
    #energies.plot()


User extensions
---------------
There are many other ways to organize PV modeling code. 
The pvlib-python developers encourage users to build on one of
these paradigms and to share their experiences.
