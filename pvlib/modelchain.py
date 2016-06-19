"""
The ``modelchain`` module contains functions and classes that combine
many of the PV power modeling steps. These tools make it easy to
get started with pvlib and demonstrate standard ways to use the
library. With great power comes great responsibility: users should take
the time to read the source code for the module.
"""

from functools import partial

import pandas as pd

from pvlib import solarposition, pvsystem, clearsky, atmosphere
from pvlib.tracking import SingleAxisTracker
import pvlib.irradiance  # avoid name conflict with full import


def basic_chain(times, latitude, longitude,
                module_parameters, inverter_parameters,
                irradiance=None, weather=None,
                surface_tilt=None, surface_azimuth=None,
                orientation_strategy=None,
                transposition_model='haydavies',
                solar_position_method='nrel_numpy',
                airmass_model='kastenyoung1989',
                altitude=None, pressure=None,
                **kwargs):
    """
    An experimental function that computes all of the modeling steps
    necessary for calculating power or energy for a PV system at a given
    location.

    Parameters
    ----------
    times : DatetimeIndex
        Times at which to evaluate the model.

    latitude : float.
        Positive is north of the equator.
        Use decimal degrees notation.

    longitude : float.
        Positive is east of the prime meridian.
        Use decimal degrees notation.

    module_parameters : None, dict or Series
        Module parameters as defined by the SAPM.

    inverter_parameters : None, dict or Series
        Inverter parameters as defined by the CEC.

    irradiance : None or DataFrame
        If None, calculates clear sky data.
        Columns must be 'dni', 'ghi', 'dhi'.

    weather : None or DataFrame
        If None, assumes air temperature is 20 C and
        wind speed is 0 m/s.
        Columns must be 'wind_speed', 'temp_air'.

    surface_tilt : float or Series
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : float or Series
        Surface azimuth angles in decimal degrees.
        The azimuth convention is defined
        as degrees east of north
        (North=0, South=180, East=90, West=270).

    orientation_strategy : None or str
        The strategy for aligning the modules.
        If not None, sets the ``surface_azimuth`` and ``surface_tilt``
        properties of the ``system``.

    transposition_model : str
        Passed to system.get_irradiance.

    solar_position_method : str
        Passed to location.get_solarposition.

    airmass_model : str
        Passed to location.get_airmass.

    altitude : None or float
        If None, computed from pressure. Assumed to be 0 m
        if pressure is also None.

    pressure : None or float
        If None, computed from altitude. Assumed to be 101325 Pa
        if altitude is also None.

    **kwargs
        Arbitrary keyword arguments.
        See code for details.

    Returns
    -------
    output : (dc, ac)
        Tuple of DC power (with SAPM parameters) (DataFrame) and AC
        power (Series).
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

    if altitude is None and pressure is None:
        altitude = 0.
        pressure = 101325.
    elif altitude is None:
        altitude = atmosphere.pres2alt(pressure)
    elif pressure is None:
        pressure = atmosphere.alt2pres(altitude)

    solar_position = solarposition.get_solarposition(times, latitude,
                                                     longitude,
                                                     altitude=altitude,
                                                     pressure=pressure,
                                                     **kwargs)

    # possible error with using apparent zenith with some models
    airmass = atmosphere.relativeairmass(solar_position['apparent_zenith'],
                                         model=airmass_model)
    airmass = atmosphere.absoluteairmass(airmass, pressure)
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
            airmass_data=airmass,
            altitude=altitude)

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
    An experimental class that represents all of the modeling steps
    necessary for calculating power or energy for a PV system at a given
    location using the SAPM.

    CEC module specifications and the single diode model are not yet
    supported.

    Parameters
    ----------
    system : PVSystem
        A :py:class:`~pvlib.pvsystem.PVSystem` object that represents
        the connected set of modules, inverters, etc.

    location : Location
        A :py:class:`~pvlib.location.Location` object that represents
        the physical location at which to evaluate the model.

    orientation_strategy : None or str
        The strategy for aligning the modules. If not None, sets the
        ``surface_azimuth`` and ``surface_tilt`` properties of the
        ``system``. Allowed strategies include 'flat',
        'south_at_latitude_tilt'. Ignored for SingleAxisTracker systems.

    clearsky_model : str
        Passed to location.get_clearsky.

    transposition_model : str
        Passed to system.get_irradiance.

    solar_position_method : str
        Passed to location.get_solarposition.

    airmass_model : str
        Passed to location.get_airmass.

    dc_model: None, str, or function
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'sapm',
        'singlediode', 'pvwatts'. The ModelChain instance will be passed
        as the first argument to a user-defined function.

    ac_model: None, str, or function
        If None, the model will be inferred from the contents of
        system.inverter_parameters. Valid strings are 'snlinverter',
        'adrinverter', 'pvwatts'. The ModelChain instance will be passed
        as the first argument to a user-defined function.

    aoi_model: None, str, or function
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'physical',
        'ashrae'. The ModelChain instance will be passed
        as the first argument to a user-defined function.

    temp_model: None, str, or function
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'sapm'.
        The ModelChain instance will be passed
        as the first argument to a user-defined function.

    **kwargs
        Arbitrary keyword arguments.
        Included for compatibility, but not used.
    """

    def __init__(self, system, location,
                 orientation_strategy='south_at_latitude_tilt',
                 clearsky_model='ineichen',
                 transposition_model='haydavies',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 dc_model=None, ac_model=None, aoi_model=None,
                 temp_model=None,
                 **kwargs):

        self.system = system
        self.location = location
        self.clearsky_model = clearsky_model
        self.transposition_model = transposition_model
        self.solar_position_method = solar_position_method
        self.airmass_model = airmass_model

        # calls setters
        self.dc_model = dc_model
        self.ac_model = ac_model

#         if temp_model is None:
#             self.temp_model = self.infer_temp_model(self.system)
#         elif isinstance(temp_model, str):
#             temp_model = temp_model.lower()
#             if temp_model == 'sapm':
#                 raise NotImplementedError
#             elif temp_model == 'adrinverter':
#                 raise NotImplementedError
#             elif temp_model == 'pvwatts':
#                 raise NotImplementedError
#             else:
#                 raise ValueError(temp_model +
#                                  ' is not a valid temperature model')
#         else:
#             self.temp_model = temp_model
#
#         if aoi_model is None:
#             self.aoi_model = self.infer_aoi_model(self.system)
#         elif isinstance(aoi_model, str):
#             aoi_model = aoi_model.lower()
#             if aoi_model == 'ashrae':
#                 raise NotImplementedError
#             elif aoi_model == 'physical':
#                 raise NotImplementedError
#             else:
#                 raise ValueError(aoi_model +
#                                  ' is not a valid aoi loss model')
#         else:
#             self.aoi_model = aoi_model

        self.orientation_strategy = orientation_strategy

    def __repr__(self):
        return ('ModelChain for: ' + str(self.system) +
                ' orientation_startegy: ' + str(self.orientation_strategy) +
                ' clearsky_model: ' + str(self.clearsky_model) +
                ' transposition_model: ' + str(self.transposition_model) +
                ' solar_position_method: ' + str(self.solar_position_method) +
                ' airmass_model: ' + str(self.airmass_model))

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

    @property
    def dc_model(self):
        return self._dc_model

    @dc_model.setter
    def dc_model(self, model):
        if model is None:
            self._dc_model = self.infer_dc_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'sapm':
                self._dc_model = self.sapm
            elif model == 'singlediode':
                self._dc_model = self.singlediode
            elif model == 'pvwatts':
                self._dc_model = self.pvwatts
            else:
                raise ValueError(model + ' is not a valid DC power model')
        else:
            self._dc_model = partial(model, self)

    def infer_dc_model(self):
        params = set(self.system.module_parameters.keys())
        if set(['A0', 'A1', 'C7']) <= params:
            return self.sapm
        elif set(['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s']) <= params:
            return self.singlediode
        elif set(['temp_co']) <= params:
            return self.pvwatts_dc
        else:
            raise ValueError('could not infer DC model from system.module_parameters')

    def sapm(self):
        self.temps = self.system.sapm_celltemp(self.total_irrad['poa_global'],
                                               self.weather['wind_speed'],
                                               self.weather['temp_air'])

        self.dc = self.system.sapm(self.total_irrad['poa_direct'],
                                   self.total_irrad['poa_diffuse'],
                                   self.temps['temp_cell'],
                                   self.airmass['airmass_absolute'],
                                   self.aoi)

        self.dc = self.system.scale_voltage_current_power(self.dc)

        return self

    def singlediode(self):
        # not sure about the aoi modifiers
        self.aoi_mod = self.system.ashraeiam(self.aoi).fillna(0)
        self.total_irrad['poa_global_aoi'] = (
            self.total_irrad['poa_direct'] * self.aoi_mod +
            self.total_irrad['poa_diffuse'])

        self.temps = self.system.sapm_celltemp(
            self.total_irrad['poa_global_aoi'],
            self.weather['wind_speed'],
            self.weather['temp_air'])

        (photocurrent, saturation_current, resistance_series,
         resistance_shunt, nNsVth) = (
            self.system.calcparams_desoto(self.total_irrad['poa_global_aoi'],
                                          self.temps['temp_cell']))

        self.desoto = (photocurrent, saturation_current, resistance_series,
                       resistance_shunt, nNsVth)

        self.dc = self.system.singlediode(
            photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)

        self.dc = self.system.scale_voltage_current_power(self.dc)

        return self

    def pvwatts_dc(self):
        raise NotImplementedError
        return self

    @property
    def ac_model(self):
        return self._ac_model

    @ac_model.setter
    def ac_model(self, model):
        if model is None:
            self._ac_model = self.infer_ac_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'snlinverter':
                self._ac_model = self.snlinverter
            elif model == 'adrinverter':
                raise NotImplementedError
            elif model == 'pvwatts':
                raise NotImplementedError
            else:
                raise ValueError(model + ' is not a valid AC power model')
        else:
            self._ac_model = partial(model, self)

    def infer_ac_model(self):
        params = set(self.system.inverter_parameters.keys())
        if set(['C0', 'C1', 'C2']) <= params:
            return self.snlinverter
        else:
            raise ValueError('could not infer AC model from system.inverter_parameters')

    def snlinverter(self):
        self.ac = self.system.snlinverter(self.dc['v_mp'], self.dc['p_mp'])
        return self

    def adrinverter(self):
        raise NotImplementedError
        return self

    def pvwatts_inverter(self):
        raise NotImplementedError
        return self


    def prepare_inputs(self, times, irradiance=None, weather=None):
        """
        Prepare the solar position, irradiance, and weather inputs to
        the model.
        Parameters
        ----------
        times : DatetimeIndex
            Times at which to evaluate the model.
        irradiance : None or DataFrame
            If None, calculates clear sky data.
            Columns must be 'dni', 'ghi', 'dhi'.
        weather : None or DataFrame
            If None, assumes air temperature is 20 C and
            wind speed is 0 m/s.
            Columns must be 'wind_speed', 'temp_air'.
        Returns
        -------
        self
        Assigns attributes: times, solar_position, airmass, irradiance,
        total_irrad, weather, aoi
        """

        self.times = times

        self.solar_position = self.location.get_solarposition(self.times)

        self.airmass = self.location.get_airmass(
            solar_position=self.solar_position, model=self.airmass_model)

        self.aoi = self.system.get_aoi(self.solar_position['apparent_zenith'],
                                       self.solar_position['azimuth'])

        if irradiance is None:
            irradiance = self.location.get_clearsky(
                self.solar_position.index, self.clearsky_model,
                zenith_data=self.solar_position['apparent_zenith'],
                airmass_data=self.airmass['airmass_absolute'])
        self.irradiance = irradiance

        # PVSystem.get_irradiance and SingleAxisTracker.get_irradiance
        # have different method signatures, so use partial to handle
        # the differences.
        if isinstance(self.system, SingleAxisTracker):
            self.tracking = self.system.singleaxis(
                self.solar_position['apparent_zenith'],
                self.solar_position['azimuth'])
            self.tracking['surface_tilt'] = (
                self.tracking['surface_tilt']
                    .fillna(self.system.axis_tilt))
            self.tracking['surface_azimuth'] = (
                self.tracking['surface_azimuth']
                    .fillna(self.system.axis_azimuth))
            get_irradiance = partial(
                self.system.get_irradiance,
                surface_tilt=self.tracking['surface_tilt'],
                surface_azimuth=self.tracking['surface_azimuth'],
                solar_zenith=self.solar_position['apparent_zenith'],
                solar_azimuth=self.solar_position['azimuth'])
        else:
            get_irradiance = partial(
                self.system.get_irradiance,
                self.solar_position['apparent_zenith'],
                self.solar_position['azimuth'])

        self.total_irrad = get_irradiance(
            self.irradiance['dni'],
            self.irradiance['ghi'],
            self.irradiance['dhi'],
            model=self.transposition_model)

        if weather is None:
            weather = {'wind_speed': 0, 'temp_air': 20}
        self.weather = weather

        return self


    def run_model(self, times, irradiance=None, weather=None):
        """
        Run the model.

        Parameters
        ----------
        times : DatetimeIndex
            Times at which to evaluate the model.

        irradiance : None or DataFrame
            If None, calculates clear sky data.
            Columns must be 'dni', 'ghi', 'dhi'.

        weather : None or DataFrame
            If None, assumes air temperature is 20 C and
            wind speed is 0 m/s.
            Columns must be 'wind_speed', 'temp_air'.

        Returns
        -------
        self

        Assigns attributes: times, solar_position, airmass, irradiance,
        total_irrad, weather, temps, aoi, dc, ac
        """
        self.prepare_inputs(times, irradiance, weather)

        self.dc_model()

        self.ac_model()

        return self
