"""
The ``modelchain`` module contains functions and classes that combine
many of the PV power modeling steps. These tools make it easy to
get started with pvlib and demonstrate standard ways to use the
library. With great power comes great responsibility: users should take
the time to read the source code for the module.
"""

import pandas as pd

from pvlib import solarposition, pvsystem, clearsky, atmosphere
import pvlib.irradiance  # avoid name conflict with full import


def basic_chain(times, latitude, longitude,
                module_parameters, inverter_parameters,
                irradiance=None, weather=None,
                surface_tilt=None, surface_azimuth=None,
                orientation_strategy=None,
                transposition_model='haydavies',
                solar_position_method='nrel_numpy',
                airmass_model='kastenyoung1989',
                **kwargs):
    """
    A function that computes all of the modeling steps necessary for
    calculating power or energy for a PV system at a given location.

    Parameters
    ----------
    system : dict
        The connected set of modules, inverters, etc.
        keys must include latitude, longitude, module_parameters

    location : dict
        The physical location at which to evaluate the model.

    times : DatetimeIndex
        Times at which to evaluate the model.

    orientation_strategy : None or str
        The strategy for aligning the modules.
        If not None, sets the ``surface_azimuth`` and ``surface_tilt``
        properties of the ``system``.

    clearsky_model : str
        Passed to location.get_clearsky.

    transposition_model : str
        Passed to system.get_irradiance.

    solar_position_method : str
        Passed to location.get_solarposition.

    **kwargs
        Arbitrary keyword arguments.
        See code for details.
    """

    # use surface_tilt and surface_azimuth if provided,
    # otherwise set them using the orientation_strategy
    if surface_tilt is not None and surface_azimuth is not None:
        pass
    elif orientation_strategy is not None:
        surface_tilt, surface_azimuth = \
            get_orientation(orientation_strategy, latitude=latitude)
    else:
        raise ValueError('orientation_strategy or surface_tilt and ' +
                         'surface_azimuth must be provided')

    times = times

    solar_position = solarposition.get_solarposition(times, latitude,
                                                     longitude, **kwargs)

    # possible error with using apparent zenith with some models
    airmass = atmosphere.relativeairmass(solar_position['apparent_zenith'],
                                         model=airmass_model)
    airmass = atmosphere.absoluteairmass(airmass,
                                         kwargs.get('pressure', 101325.))
    dni_extra = pvlib.irradiance.extraradiation(solar_position.index)
    dni_extra = pd.Series(dni_extra, index=solar_position.index)

    aoi = pvlib.irradiance.aoi(surface_tilt, surface_azimuth,
                               solar_position['apparent_zenith'],
                               solar_position['azimuth'])

    if irradiance is None:
        irradiance = clearsky.ineichen(
            solar_position.index,
            latitude,
            longitude,
            zenith_data=solar_position['apparent_zenith'],
            airmass_data=airmass)

    total_irrad = pvlib.irradiance.total_irrad(
        surface_tilt,
        surface_azimuth,
        solar_position['apparent_zenith'],
        solar_position['azimuth'],
        irradiance['dni'],
        irradiance['ghi'],
        irradiance['dhi'],
        model=transposition_model,
        dni_extra=dni_extra)

    if weather is None:
        weather = {'wind_speed': 0, 'temp_air': 20}

    temps = pvsystem.sapm_celltemp(total_irrad['poa_global'],
                                   weather['wind_speed'],
                                   weather['temp_air'])

    dc = pvsystem.sapm(module_parameters, total_irrad['poa_direct'],
                       total_irrad['poa_diffuse'],
                       temps['temp_cell'],
                       airmass,
                       aoi)

    ac = pvsystem.snlinverter(inverter_parameters, dc['v_mp'], dc['p_mp'])

    return dc, ac


def get_orientation(strategy, **kwargs):
    """
    Determine a PV system's surface tilt and surface azimuth
    using a named strategy.

    Parameters
    ----------
    strategy: str
        The orientation strategy.
        Allowed strategies include 'flat', 'south_at_latitude_tilt'.
    **kwargs:
        Strategy-dependent keyword arguments. See code for details.

    Returns
    -------
    surface_tilt, surface_azimuth
    """

    if strategy == 'south_at_latitude_tilt':
        surface_azimuth = 180
        surface_tilt = kwargs['latitude']
    elif strategy == 'flat':
        surface_azimuth = 180
        surface_tilt = 0
    else:
        raise ValueError('invalid orientation strategy. strategy must ' +
                         'be one of south_at_latitude, flat,')

    return surface_tilt, surface_azimuth


class ModelChain(object):
    """
    A class that represents all of the modeling steps necessary for
    calculating power or energy for a PV system at a given location.

    Consider an abstract base class.

    Parameters
    ----------
    system : PVSystem
        The connected set of modules, inverters, etc.

    location : location
        The physical location at which to evaluate the model.

    times : DatetimeIndex
        Times at which to evaluate the model.

    orientation_strategy : None or str
        The strategy for aligning the modules.
        If not None, sets the ``surface_azimuth`` and ``surface_tilt``
        properties of the ``system``.
        Allowed strategies include 'flat', 'south_at_latitude_tilt'.

    clearsky_model : str
        Passed to location.get_clearsky.

    transposition_model : str
        Passed to system.get_irradiance.

    solar_position_method : str
        Passed to location.get_solarposition.

    **kwargs
        Arbitrary keyword arguments.
        Included for compatibility, but not used.

    See also
    --------
    location.Location
    pvsystem.PVSystem
    """

    def __init__(self, system, location,
                 orientation_strategy='south_at_latitude_tilt',
                 clearsky_model='ineichen',
                 transposition_model='haydavies',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 **kwargs):

        self.system = system
        self.location = location
        self.clearsky_model = clearsky_model
        self.transposition_model = transposition_model
        self.solar_position_method = solar_position_method
        self.airmass_model = airmass_model

        # calls setter
        self.orientation_strategy = orientation_strategy

    @property
    def orientation_strategy(self):
        return self._orientation_strategy

    @orientation_strategy.setter
    def orientation_strategy(self, strategy):
        if strategy == 'None':
            strategy = None

        if strategy is not None:
            self.system.surface_tilt, self.system.surface_azimuth = \
                get_orientation(strategy, latitude=self.location.latitude)

        self._orientation_strategy = strategy

    def run_model(self, times, irradiance=None, weather=None):
        """
        Run the model.

        Parameters
        ----------
        times : DatetimeIndex
        irradiance : None or DataFrame
            If None, calculates clear sky data.
            Columns must be 'dni', 'ghi', 'dhi'
        weather : None or DataFrame
            If None, assumes air temperature is 20 C and
            wind speed is 0 m/s.
            Columns must be 'wind_speed', 'temp_air'.

        Returns
        -------
        output : DataFrame
            Some combination of AC power, DC power, POA irrad, etc.

        Assigns attributes: times, solar_position, airmass, irradiance,
        total_irrad, weather, temps, aoi, dc, ac
        """
        self.times = times

        self.solar_position = self.location.get_solarposition(self.times)

        self.airmass = self.location.get_airmass(
            solar_position=self.solar_position, model=self.airmass_model)

        if irradiance is None:
            irradiance = self.location.get_clearsky(
                self.solar_position.index, self.clearsky_model,
                zenith_data=self.solar_position['apparent_zenith'],
                airmass_data=self.airmass['airmass_absolute'])
        self.irradiance = irradiance

        self.total_irrad = self.system.get_irradiance(
            self.solar_position['apparent_zenith'],
            self.solar_position['azimuth'],
            self.irradiance['dni'],
            self.irradiance['ghi'],
            self.irradiance['dhi'],
            model=self.transposition_model)

        if weather is None:
            weather = {'wind_speed': 0, 'temp_air': 20}
        self.weather = weather

        self.temps = self.system.sapm_celltemp(self.total_irrad['poa_global'],
                                               self.weather['wind_speed'],
                                               self.weather['temp_air'])

        self.aoi = self.system.get_aoi(self.solar_position['apparent_zenith'],
                                       self.solar_position['azimuth'])

        self.dc = self.system.sapm(self.total_irrad['poa_direct'],
                                   self.total_irrad['poa_diffuse'],
                                   self.temps['temp_cell'],
                                   self.airmass['airmass_absolute'],
                                   self.aoi)

        self.ac = self.system.snlinverter(self.dc['v_mp'], self.dc['p_mp'])

        return self.dc, self.ac


    def model_system(self):
        """
        Model the system?

        I'm just copy/pasting example code...

        Returns
        -------
        ???
        """

        final_output = self.run_model()
        input = self.prettify_input()
        modeling_steps = self.get_modeling_steps()



class MoreSpecificModelChain(ModelChain):
    """
    Something more specific.
    """
    def __init__(self, *args, **kwargs):
        super(MoreSpecificModelChain, self).__init__(**kwargs)

    def run_model(self):
        # overrides the parent ModelChain method
        pass