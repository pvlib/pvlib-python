"""
The ``modelchain`` module contains functions and classes that combine
many of the PV power modeling steps. These tools make it easy to
get started with pvlib and demonstrate standard ways to use the
library. With great power comes great responsibility: users should take
the time to read the source code for the module.
"""

from functools import partial
import logging
import warnings
import pandas as pd

from pvlib import (solarposition, pvsystem, clearsky, atmosphere, tools)
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
        properties of the ``system``. Allowed strategies include 'flat',
        'south_at_latitude_tilt'. Ignored for SingleAxisTracker systems.

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
        linke_turbidity = clearsky.lookup_linke_turbidity(
            solar_position.index, latitude, longitude)
        irradiance = clearsky.ineichen(
            solar_position['apparent_zenith'],
            airmass,
            linke_turbidity,
            altitude=altitude,
            dni_extra=dni_extra
            )

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

    effective_irradiance = pvsystem.sapm_effective_irradiance(
        total_irrad['poa_direct'], total_irrad['poa_diffuse'], airmass, aoi,
        module_parameters)

    dc = pvsystem.sapm(effective_irradiance, temps['temp_cell'],
                       module_parameters)

    ac = pvsystem.snlinverter(dc['v_mp'], dc['p_mp'], inverter_parameters)

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
        system.inverter_parameters and system.module_parameters. Valid
        strings are 'snlinverter', 'adrinverter' (not implemented),
        'pvwatts'. The ModelChain instance will be passed as the first
        argument to a user-defined function.

    aoi_model: None, str, or function
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'physical',
        'ashrae', 'sapm', 'no_loss'. The ModelChain instance will be
        passed as the first argument to a user-defined function.

    spectral_model: None, str, or function
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'sapm',
        'first_solar' (not implemented), 'no_loss'. The ModelChain
        instance will be passed as the first argument to a user-defined
        function.

    temp_model: str or function
        Valid strings are 'sapm'. The ModelChain instance will be passed
        as the first argument to a user-defined function.

    losses_model: str or function
        Valid strings are 'pvwatts', 'no_loss'. The ModelChain instance
        will be passed as the first argument to a user-defined function.

    **kwargs
        Arbitrary keyword arguments. Included for compatibility, but not
        used.
    """

    def __init__(self, system, location,
                 orientation_strategy='south_at_latitude_tilt',
                 clearsky_model='ineichen',
                 transposition_model='haydavies',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 dc_model=None, ac_model=None, aoi_model=None,
                 spectral_model=None, temp_model='sapm',
                 losses_model='no_loss',
                 name=None,
                 **kwargs):

        self.name = name
        self.system = system
        self.location = location
        self.clearsky_model = clearsky_model
        self.transposition_model = transposition_model
        self.solar_position_method = solar_position_method
        self.airmass_model = airmass_model

        # calls setters
        self.dc_model = dc_model
        self.ac_model = ac_model
        self.aoi_model = aoi_model
        self.spectral_model = spectral_model
        self.temp_model = temp_model
        self.losses_model = losses_model
        self.orientation_strategy = orientation_strategy

        self.weather = None
        self.times = None
        self.solar_position = None

    def __repr__(self):
        attrs = [
            'name', 'orientation_strategy', 'clearsky_model',
            'transposition_model', 'solar_position_method',
            'airmass_model', 'dc_model', 'ac_model', 'aoi_model',
            'spectral_model', 'temp_model', 'losses_model'
            ]

        def getmcattr(self, attr):
            """needed to avoid recursion in property lookups"""
            out = getattr(self, attr)
            try:
                out = out.__name__
            except AttributeError:
                pass
            return out

        return ('ModelChain: \n  ' + '\n  '.join(
            (attr + ': ' + getmcattr(self, attr) for attr in attrs)))

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
                self._dc_model = self.pvwatts_dc
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
        elif set(['pdc0', 'gamma_pdc']) <= params:
            return self.pvwatts_dc
        else:
            raise ValueError('could not infer DC model from ' +
                             'system.module_parameters')

    def sapm(self):
        self.dc = self.system.sapm(self.effective_irradiance/1000.,
                                   self.temps['temp_cell'])

        self.dc = self.system.scale_voltage_current_power(self.dc)

        return self

    def singlediode(self):
        (photocurrent, saturation_current, resistance_series,
         resistance_shunt, nNsVth) = (
            self.system.calcparams_desoto(self.effective_irradiance,
                                          self.temps['temp_cell']))

        self.desoto = (photocurrent, saturation_current, resistance_series,
                       resistance_shunt, nNsVth)

        self.dc = self.system.singlediode(
            photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)

        self.dc = self.system.scale_voltage_current_power(self.dc).fillna(0)

        return self

    def pvwatts_dc(self):
        self.dc = self.system.pvwatts_dc(self.effective_irradiance,
                                         self.temps['temp_cell'])
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
                self._ac_model = self.pvwatts_inverter
            else:
                raise ValueError(model + ' is not a valid AC power model')
        else:
            self._ac_model = partial(model, self)

    def infer_ac_model(self):
        inverter_params = set(self.system.inverter_parameters.keys())
        module_params = set(self.system.module_parameters.keys())
        if set(['C0', 'C1', 'C2']) <= inverter_params:
            return self.snlinverter
        elif set(['pdc0']) <= module_params:
            return self.pvwatts_inverter
        else:
            raise ValueError('could not infer AC model from ' +
                             'system.inverter_parameters')

    def snlinverter(self):
        self.ac = self.system.snlinverter(self.dc['v_mp'], self.dc['p_mp'])
        return self

    def adrinverter(self):
        raise NotImplementedError
        return self

    def pvwatts_inverter(self):
        self.ac = self.system.pvwatts_ac(self.dc).fillna(0)
        return self

    @property
    def aoi_model(self):
        return self._aoi_model

    @aoi_model.setter
    def aoi_model(self, model):
        if model is None:
            self._aoi_model = self.infer_aoi_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'ashrae':
                self._aoi_model = self.ashrae_aoi_loss
            elif model == 'physical':
                self._aoi_model = self.physical_aoi_loss
            elif model == 'sapm':
                self._aoi_model = self.sapm_aoi_loss
            elif model == 'no_loss':
                self._aoi_model = self.no_aoi_loss
            else:
                raise ValueError(model + ' is not a valid aoi loss model')
        else:
            self._aoi_model = partial(model, self)

    def infer_aoi_model(self):
        params = set(self.system.module_parameters.keys())
        if set(['K', 'L', 'n']) <= params:
            return self.physical_aoi_loss
        elif set(['B5', 'B4', 'B3', 'B2', 'B1', 'B0']) <= params:
            return self.sapm_aoi_loss
        elif set(['b']) <= params:
            return self.ashrae_aoi_loss
        else:
            raise ValueError('could not infer AOI model from ' +
                             'system.module_parameters')

    def ashrae_aoi_loss(self):
        self.aoi_modifier = self.system.ashraeiam(self.aoi)
        return self

    def physical_aoi_loss(self):
        self.aoi_modifier = self.system.physicaliam(self.aoi)
        return self

    def sapm_aoi_loss(self):
        self.aoi_modifier = self.system.sapm_aoi_loss(self.aoi)
        return self

    def no_aoi_loss(self):
        self.aoi_modifier = 1
        return self

    @property
    def spectral_model(self):
        return self._spectral_model

    @spectral_model.setter
    def spectral_model(self, model):
        if model is None:
            self._spectral_model = self.infer_spectral_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'first_solar':
                raise NotImplementedError
            elif model == 'sapm':
                self._spectral_model = self.sapm_spectral_loss
            elif model == 'no_loss':
                self._spectral_model = self.no_spectral_loss
            else:
                raise ValueError(model + ' is not a valid spectral loss model')
        else:
            self._spectral_model = partial(model, self)

    def infer_spectral_model(self):
        params = set(self.system.module_parameters.keys())
        if set(['A4', 'A3', 'A2', 'A1', 'A0']) <= params:
            return self.sapm_spectral_loss
        else:
            raise ValueError('could not infer spectral model from ' +
                             'system.module_parameters')

    def first_solar_spectral_loss(self):
        raise NotImplementedError

    def sapm_spectral_loss(self):
        self.spectral_modifier = self.system.sapm_spectral_loss(
            self.airmass['airmass_absolute'])
        return self

    def no_spectral_loss(self):
        self.spectral_modifier = 1
        return self

    @property
    def temp_model(self):
        return self._temp_model

    @temp_model.setter
    def temp_model(self, model):
        if model is None:
            self._temp_model = self.infer_temp_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'sapm':
                self._temp_model = self.sapm_temp
            else:
                raise ValueError(model + ' is not a valid temp model')
        else:
            self._temp_model = partial(model, self)

    def infer_temp_model(self):
        raise NotImplementedError

    def sapm_temp(self):
        self.temps = self.system.sapm_celltemp(self.total_irrad['poa_global'],
                                               self.weather['wind_speed'],
                                               self.weather['temp_air'])
        return self

    @property
    def losses_model(self):
        return self._losses_model

    @losses_model.setter
    def losses_model(self, model):
        if model is None:
            self._losses_model = self.infer_losses_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'pvwatts':
                self._losses_model = self.pvwatts_losses
            elif model == 'no_loss':
                self._losses_model = self.no_extra_losses
            else:
                raise ValueError(model + ' is not a valid losses model')
        else:
            self._losses_model = partial(model, self)

    def infer_losses_model(self):
        raise NotImplementedError

    def pvwatts_losses(self):
        self.losses = (100 - self.system.pvwatts_losses()) / 100.
        self.ac *= self.losses
        return self

    def no_extra_losses(self):
        self.losses = 1
        return self

    def effective_irradiance_model(self):
        fd = self.system.module_parameters.get('FD', 1.)
        self.effective_irradiance = self.spectral_modifier * (
            self.total_irrad['poa_direct']*self.aoi_modifier +
            fd*self.total_irrad['poa_diffuse'])
        return self

    def complete_irradiance(self, times=None, weather=None):
        """
        Determine the missing irradiation columns. Only two of the
        following data columns (dni, ghi, dhi) are needed to calculate
        the missing data.

        This function is not safe at the moment. Results can be too high
        or negative. Please contribute and help to improve this function
        on https://github.com/pvlib/pvlib-python

        Parameters
        ----------
        times : DatetimeIndex
            Times at which to evaluate the model. Can be None if
            attribute `times` is already set.
        weather : pandas.DataFrame
            Table with at least two columns containing one of the
            following data sets: dni, dhi, ghi. Can be None if attribute
            `weather` is already set.

        Returns
        -------
        self

        Assigns attributes: times, weather

        Examples
        --------
        This example does not work until the parameters `my_system`,
        `my_location`, `my_datetime` and `my_weather` are not defined
        properly but shows the basic idea how this method can be used.

        >>> from pvlib.modelchain import ModelChain

        >>> # my_weather containing 'dhi' and 'ghi'.
        >>> mc = ModelChain(my_system, my_location)  # doctest: +SKIP
        >>> mc.complete_irradiance(my_datetime, my_weather)  # doctest: +SKIP
        >>> mc.run_model()  # doctest: +SKIP

        >>> # my_weather containing 'dhi', 'ghi' and 'dni'.
        >>> mc = ModelChain(my_system, my_location)  # doctest: +SKIP
        >>> mc.run_model(my_datetime, my_weather)  # doctest: +SKIP
        """
        if weather is not None:
            self.weather = weather
        if times is not None:
            self.times = times
        self.solar_position = self.location.get_solarposition(self.times)
        icolumns = set(self.weather.columns)
        wrn_txt = ("This function is not safe at the moment.\n" +
                   "Results can be too high or negative.\n" +
                   "Help to improve this function on github:\n" +
                   "https://github.com/pvlib/pvlib-python \n")
        warnings.warn(wrn_txt, UserWarning)
        if {'ghi', 'dhi'} <= icolumns and 'dni' not in icolumns:
            logging.debug('Estimate dni from ghi and dhi')
            self.weather.loc[:, 'dni'] = (
                (self.weather.loc[:, 'ghi'] - self.weather.loc[:, 'dhi']) /
                tools.cosd(self.solar_position.loc[:, 'zenith']))
        elif {'dni', 'dhi'} <= icolumns and 'ghi' not in icolumns:
            logging.debug('Estimate ghi from dni and dhi')
            self.weather.loc[:, 'ghi'] = (
                self.weather.dni * tools.cosd(self.solar_position.zenith) +
                self.weather.dhi)
        elif {'dni', 'ghi'} <= icolumns and 'dhi' not in icolumns:
            logging.debug('Estimate dhi from dni and ghi')
            self.weather.loc[:, 'dhi'] = (
                self.weather.ghi - self.weather.dni *
                tools.cosd(self.solar_position.zenith))

        return self

    def prepare_inputs(self, times=None, irradiance=None, weather=None):
        """
        Prepare the solar position, irradiance, and weather inputs to
        the model.

        Parameters
        ----------
        times : DatetimeIndex
            Times at which to evaluate the model. Can be None if
            attribute `times` is already set.
        irradiance : None or DataFrame
            This parameter is deprecated. Please use `weather` instead.
        weather : None or DataFrame
            If None, the weather attribute is used. If the weather
            attribute is also None assumes air temperature is 20 C, wind
            speed is 0 m/s and irradiation calculated from clear sky
            data. Column names must be 'wind_speed', 'temp_air', 'dni',
            'ghi', 'dhi'. Do not pass incomplete irradiation data. Use
            method
            :py:meth:`~pvlib.modelchain.ModelChain.complete_irradiance`
            instead.

        Returns
        -------
        self

        Assigns attributes: times, solar_position, airmass, total_irrad,
        aoi
        """
        if weather is not None:
            self.weather = weather
        if self.weather is None:
            self.weather = pd.DataFrame()

        # The following part could be removed together with the irradiance
        # parameter at version v0.5 or v0.6.
        # **** Begin ****
        wrn_txt = ("The irradiance parameter will be removed soon.\n" +
                   "Please use the weather parameter to pass a DataFrame " +
                   "with irradiance (ghi, dni, dhi), wind speed and " +
                   "temp_air.\n")
        if irradiance is not None:
            warnings.warn(wrn_txt, FutureWarning)
            for column in irradiance.columns:
                self.weather[column] = irradiance[column]
        # **** End ****

        if times is not None:
            self.times = times

        self.solar_position = self.location.get_solarposition(self.times)

        self.airmass = self.location.get_airmass(
            solar_position=self.solar_position, model=self.airmass_model)

        self.aoi = self.system.get_aoi(self.solar_position['apparent_zenith'],
                                       self.solar_position['azimuth'])

        if not any([x in ['ghi', 'dni', 'dhi'] for x in self.weather.columns]):
            self.weather[['ghi', 'dni', 'dhi']] = self.location.get_clearsky(
                self.solar_position.index, self.clearsky_model,
                zenith_data=self.solar_position['apparent_zenith'],
                airmass_data=self.airmass['airmass_absolute'])

        if not {'ghi', 'dni', 'dhi'} <= set(self.weather.columns):
            raise ValueError(
                "Uncompleted irradiance data set. Please check you input " +
                "data.\nData set needs to have 'dni', 'dhi' and 'ghi'.\n" +
                "Detected data: {0}".format(list(self.weather.columns)))

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
            self.weather['dni'],
            self.weather['ghi'],
            self.weather['dhi'],
            airmass=self.airmass['airmass_relative'],
            model=self.transposition_model)

        if self.weather.get('wind_speed') is None:
            self.weather['wind_speed'] = 0
        if self.weather.get('temp_air') is None:
            self.weather['temp_air'] = 20
        return self

    def run_model(self, times=None, irradiance=None, weather=None):
        """
        Run the model.

        Parameters
        ----------
        times : DatetimeIndex
            Times at which to evaluate the model. Can be None if
            attribute `times` is already set.
        irradiance : None or DataFrame
            This parameter is deprecated. Please use `weather` instead.
        weather : None or DataFrame
            If None, assumes air temperature is 20 C, wind speed is 0
            m/s and irradiation calculated from clear sky data. Column
            names must be 'wind_speed', 'temp_air', 'dni', 'ghi', 'dhi'.
            Do not pass incomplete irradiation data. Use method
            :py:meth:`~pvlib.modelchain.ModelChain.complete_irradiance`
            instead.

        Returns
        -------
        self

        Assigns attributes: times, solar_position, airmass, irradiance,
        total_irrad, effective_irradiance, weather, temps, aoi,
        aoi_modifier, spectral_modifier, dc, ac, losses.
        """

        self.prepare_inputs(times, irradiance, weather)
        self.aoi_model()
        self.spectral_model()
        self.effective_irradiance_model()
        self.temp_model()
        self.dc_model()
        self.ac_model()
        self.losses_model()

        return self
