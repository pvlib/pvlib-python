Package Overview
================

The core mission of pvlib-python is to provide open, reliable,
interoperable, and benchmark implementations of PV modeling algorithms.

There are at least as many opinions about how to model PV systems as
there are modelers of PV systems, so 
pvlib-python provides several modeling paradigms.


Modeling paradigms
------------------

The backbone of pvlib-python
is well-tested procedural code that implements these algorithms.
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

Procedural method ::
    import pvlib as pvl
    
    system = {'module': module, 'inverter': inverter,
              'surface_azimuth': 180, **other_params}

    energies = {}
    for latitude, longitude, name in coordinates:
        cs = pvl.clearsky.ineichen(times, latitude, longitude)
        solpos = pvl.solarposition.get_solarposition(times, latitude, longitude)
        system['surface_tilt'] = latitude
        total_irrad = pvl.irradiance.total_irradiance(**solpos, **cs, **system)
        temps = pvl.pvsystem.sapm_celltemp(**total_irrad, **system)
        dc = pvl.pvsystem.sapm(**temps, **total_irrad, **system)
        ac = pvl.pvsystem.snlinverter(**system, **dc)
        annual_energy = power.sum()
        energies[name] = annual_energy


Object oriented method using
:class:`pvlib.location.Location`,
:class:`pvlib.pvsystem.PVSystem`, and
:class:`pvlib.modelchain.ModelChain` ::
    
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


Object oriented method using
:class:`pvlib.pvsystem.LocalizedPVSystem` ::

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

