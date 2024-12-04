.. _introtutorial:

Intro Tutorial
==============

This page contains introductory examples of pvlib python usage.

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

    import pvlib
    import pandas as pd
    import matplotlib.pyplot as plt

    # latitude, longitude, name, altitude, timezone
    coordinates = [
        (32.2, -111.0, 'Tucson', 700, 'Etc/GMT+7'),
        (35.1, -106.6, 'Albuquerque', 1500, 'Etc/GMT+7'),
        (37.8, -122.4, 'San Francisco', 10, 'Etc/GMT+8'),
        (52.5, 13.4, 'Berlin', 34, 'Etc/GMT-1'),
    ]

    # get the module and inverter specifications from SAM
    sandia_modules = pvlib.pvsystem.retrieve_sam('SandiaMod')
    sapm_inverters = pvlib.pvsystem.retrieve_sam('cecinverter')
    module = sandia_modules['Canadian_Solar_CS5P_220M___2009_']
    inverter = sapm_inverters['ABB__MICRO_0_25_I_OUTD_US_208__208V_']
    temperature_model_parameters = pvlib.temperature.TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']


In order to retrieve meteorological data for the simulation, we can make use of
the :ref:`iotools` module. In this example we will be using PVGIS, one of the
data sources available, to retrieve a Typical Meteorological Year (TMY) which
includes irradiation, temperature and wind speed.

.. ipython:: python

    tmys = []
    for location in coordinates:
        latitude, longitude, name, altitude, timezone = location
        weather = pvlib.iotools.get_pvgis_tmy(latitude, longitude)[0]
        weather.index.name = "utc_time"
        tmys.append(weather)


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

    for location, weather in zip(coordinates, tmys):
        latitude, longitude, name, altitude, timezone = location
        system['surface_tilt'] = latitude
        solpos = pvlib.solarposition.get_solarposition(
            time=weather.index,
            latitude=latitude,
            longitude=longitude,
            altitude=altitude,
            temperature=weather["temp_air"],
            pressure=pvlib.atmosphere.alt2pres(altitude),
        )
        dni_extra = pvlib.irradiance.get_extra_radiation(weather.index)
        airmass = pvlib.atmosphere.get_relative_airmass(solpos['apparent_zenith'])
        pressure = pvlib.atmosphere.alt2pres(altitude)
        am_abs = pvlib.atmosphere.get_absolute_airmass(airmass, pressure)
        aoi = pvlib.irradiance.aoi(
            system['surface_tilt'],
            system['surface_azimuth'],
            solpos["apparent_zenith"],
            solpos["azimuth"],
        )
        total_irradiance = pvlib.irradiance.get_total_irradiance(
            system['surface_tilt'],
            system['surface_azimuth'],
            solpos['apparent_zenith'],
            solpos['azimuth'],
            weather['dni'],
            weather['ghi'],
            weather['dhi'],
            dni_extra=dni_extra,
            model='haydavies',
        )
        cell_temperature = pvlib.temperature.sapm_cell(
            total_irradiance['poa_global'],
            weather["temp_air"],
            weather["wind_speed"],
            **temperature_model_parameters,
        )
        effective_irradiance = pvlib.pvsystem.sapm_effective_irradiance(
            total_irradiance['poa_direct'],
            total_irradiance['poa_diffuse'],
            am_abs,
            aoi,
            module,
        )
        dc = pvlib.pvsystem.sapm(effective_irradiance, cell_temperature, module)
        ac = pvlib.inverter.sandia(dc['v_mp'], dc['p_mp'], inverter)
        annual_energy = ac.sum()
        energies[name] = annual_energy

    energies = pd.Series(energies)

    # based on the parameters specified above, these are in W*hrs
    print(energies)

    energies.plot(kind='bar', rot=0)
    @savefig proc-energies.png width=6in
    plt.ylabel('Yearly energy yield (W hr)')
    @suppress
    plt.close();


.. _object-oriented:

Object oriented (Location, Mount, Array, PVSystem, ModelChain)
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The object oriented paradigm uses a model with three main concepts:

1. System design (modules, inverters etc.) is represented by
   :py:class:`~pvlib.pvsystem.PVSystem`, :py:class:`~pvlib.pvsystem.Array`,
   and :py:class:`~pvlib.pvsystem.FixedMount`
   /:py:class:`~pvlib.pvsystem.SingleAxisTrackerMount` objects.
2. A particular place on the planet is represented by a
   :py:class:`~pvlib.location.Location` object.
3. The modeling chain used to calculate power output for a particular
   system and location is represented by a
   :py:class:`~pvlib.modelchain.ModelChain` object.

This can be a useful paradigm if you prefer to think about the
PV system and its location as separate concepts or if you develop your
own ModelChain subclasses. It can also be helpful if you make extensive
use of Location-specific methods for other calculations.

The following code demonstrates how to use
:py:class:`~pvlib.location.Location`,
:py:class:`~pvlib.pvsystem.PVSystem`, and
:py:class:`~pvlib.modelchain.ModelChain` objects to accomplish our
system modeling goal. ModelChain objects provide convenience methods
that can provide default selections for models and can also fill
necessary input with modeled data. For example, no air temperature
or wind speed data is provided in the input *weather* DataFrame,
so the ModelChain object defaults to 20 C and 0 m/s. Also, no irradiance
transposition model is specified (keyword argument `transposition_model` for
ModelChain) so the ModelChain defaults to the `haydavies` model. In this
example, ModelChain infers the DC power model from the module provided
by examining the parameters defined for the module.

.. ipython:: python

    from pvlib.pvsystem import PVSystem, Array, FixedMount
    from pvlib.location import Location
    from pvlib.modelchain import ModelChain

    energies = {}
    for location, weather in zip(coordinates, tmys):
        latitude, longitude, name, altitude, timezone = location
        location = Location(
            latitude,
            longitude,
            name=name,
            altitude=altitude,
            tz=timezone,
        )
        mount = FixedMount(surface_tilt=latitude, surface_azimuth=180)
        array = Array(
            mount=mount,
            module_parameters=module,
            temperature_model_parameters=temperature_model_parameters,
        )
        system = PVSystem(arrays=[array], inverter_parameters=inverter)
        mc = ModelChain(system, location)
        mc.run_model(weather)
        annual_energy = mc.results.ac.sum()
        energies[name] = annual_energy

    energies = pd.Series(energies)

    # based on the parameters specified above, these are in W*hrs
    print(energies)

    energies.plot(kind='bar', rot=0)
    @savefig modelchain-energies.png width=6in
    plt.ylabel('Yearly energy yield (W hr)')
    @suppress
    plt.close();
