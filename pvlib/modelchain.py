"""
The ``modelchain`` module contains functions and classes that combine
many of the PV power modeling steps. These tools make it easy to
get started with pvlib and demonstrate standard ways to use the
library. With great power comes great responsibility: users should take
the time to read the source code for the module.
"""

from functools import partial
import itertools
import warnings
import pandas as pd
from dataclasses import dataclass, field
from typing import Union, Tuple, Optional, TypeVar

from pvlib import (atmosphere, clearsky, inverter, pvsystem, solarposition,
                   temperature, tools)
from pvlib.tracking import SingleAxisTracker
import pvlib.irradiance  # avoid name conflict with full import
from pvlib.pvsystem import _DC_MODEL_PARAMS
from pvlib._deprecation import pvlibDeprecationWarning
from pvlib.tools import _build_kwargs

# keys that are used to detect input data and assign data to appropriate
# ModelChain attribute
# for ModelChain.weather
WEATHER_KEYS = ('ghi', 'dhi', 'dni', 'wind_speed', 'temp_air',
                'precipitable_water')

# for ModelChain.total_irrad
POA_KEYS = ('poa_global', 'poa_direct', 'poa_diffuse')

# Optional keys to communicate temperature data. If provided,
# 'cell_temperature' overrides ModelChain.temperature_model and sets
# ModelChain.cell_temperature to the data. If 'module_temperature' is provdied,
# overrides ModelChain.temperature_model with
# pvlib.temperature.sapm_celL_from_module
TEMPERATURE_KEYS = ('module_temperature', 'cell_temperature')

DATA_KEYS = WEATHER_KEYS + POA_KEYS + TEMPERATURE_KEYS

# these dictionaries contain the default configuration for following
# established modeling sequences. They can be used in combination with
# basic_chain and ModelChain. They are used by the ModelChain methods
# ModelChain.with_pvwatts, ModelChain.with_sapm, etc.

# pvwatts documentation states that it uses the following reference for
# a temperature model: Fuentes, M. K. (1987). A Simplified Thermal Model
# for Flat-Plate Photovoltaic Arrays. SAND85-0330. Albuquerque, NM:
# Sandia National Laboratories. Accessed September 3, 2013:
# http://prod.sandia.gov/techlib/access-control.cgi/1985/850330.pdf
# pvlib python does not implement that model, so use the SAPM instead.
PVWATTS_CONFIG = dict(
    dc_model='pvwatts', ac_model='pvwatts', losses_model='pvwatts',
    transposition_model='perez', aoi_model='physical',
    spectral_model='no_loss', temperature_model='sapm'
)

SAPM_CONFIG = dict(
    dc_model='sapm', ac_model='sandia', losses_model='no_loss',
    aoi_model='sapm', spectral_model='sapm', temperature_model='sapm'
)


def basic_chain(times, latitude, longitude,
                module_parameters, temperature_model_parameters,
                inverter_parameters,
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
        Module parameters as defined by the SAPM. See pvsystem.sapm for
        details.

    temperature_model_parameters : None, dict or Series.
        Temperature model parameters as defined by the SAPM.
        See temperature.sapm_cell for details.

    inverter_parameters : None, dict or Series
        Inverter parameters as defined by the CEC. See
        :py:func:`inverter.sandia` for details.

    irradiance : None or DataFrame, default None
        If None, calculates clear sky data.
        Columns must be 'dni', 'ghi', 'dhi'.

    weather : None or DataFrame, default None
        If None, assumes air temperature is 20 C and
        wind speed is 0 m/s.
        Columns must be 'wind_speed', 'temp_air'.

    surface_tilt : None, float or Series, default None
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : None, float or Series, default None
        Surface azimuth angles in decimal degrees.
        The azimuth convention is defined
        as degrees east of north
        (North=0, South=180, East=90, West=270).

    orientation_strategy : None or str, default None
        The strategy for aligning the modules.
        If not None, sets the ``surface_azimuth`` and ``surface_tilt``
        properties of the ``system``. Allowed strategies include 'flat',
        'south_at_latitude_tilt'. Ignored for SingleAxisTracker systems.

    transposition_model : str, default 'haydavies'
        Passed to system.get_irradiance.

    solar_position_method : str, default 'nrel_numpy'
        Passed to solarposition.get_solarposition.

    airmass_model : str, default 'kastenyoung1989'
        Passed to atmosphere.relativeairmass.

    altitude : None or float, default None
        If None, computed from pressure. Assumed to be 0 m
        if pressure is also None.

    pressure : None or float, default None
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
        raise ValueError('orientation_strategy or surface_tilt and '
                         'surface_azimuth must be provided')

    if altitude is None and pressure is None:
        altitude = 0.
        pressure = 101325.
    elif altitude is None:
        altitude = atmosphere.pres2alt(pressure)
    elif pressure is None:
        pressure = atmosphere.alt2pres(altitude)

    solar_position = solarposition.get_solarposition(
        times, latitude, longitude, altitude=altitude, pressure=pressure,
        method=solar_position_method, **kwargs)

    # possible error with using apparent zenith with some models
    airmass = atmosphere.get_relative_airmass(
        solar_position['apparent_zenith'], model=airmass_model)
    airmass = atmosphere.get_absolute_airmass(airmass, pressure)
    dni_extra = pvlib.irradiance.get_extra_radiation(solar_position.index)

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

    total_irrad = pvlib.irradiance.get_total_irradiance(
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

    cell_temperature = temperature.sapm_cell(
        total_irrad['poa_global'], weather['temp_air'], weather['wind_speed'],
        temperature_model_parameters['a'], temperature_model_parameters['b'],
        temperature_model_parameters['deltaT'])

    effective_irradiance = pvsystem.sapm_effective_irradiance(
        total_irrad['poa_direct'], total_irrad['poa_diffuse'], airmass, aoi,
        module_parameters)

    dc = pvsystem.sapm(effective_irradiance, cell_temperature,
                       module_parameters)

    ac = inverter.sandia(dc['v_mp'], dc['p_mp'], inverter_parameters)

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
        raise ValueError('invalid orientation strategy. strategy must '
                         'be one of south_at_latitude, flat,')

    return surface_tilt, surface_azimuth


@dataclass
class ModelChainResult:
    _T = TypeVar('T')
    PerArray = Union[_T, Tuple[_T, ...]]
    """Type for fields that vary between arrays"""
    # system-level information
    solar_position: Optional[pd.DataFrame] = field(default=None)
    airmass: Optional[pd.DataFrame] = field(default=None)
    ac: Optional[pd.Series] = field(default=None)
    # per DC array information
    tracking: Optional[pd.DataFrame] = field(default=None)
    total_irrad: Optional[PerArray[pd.DataFrame]] = field(default=None)
    aoi: Optional[PerArray[pd.Series]] = field(default=None)
    aoi_modifier: Optional[PerArray[pd.Series]] = field(default=None)
    spectral_modifier: Optional[PerArray[pd.Series]] = field(default=None)
    cell_temperature: Optional[PerArray[pd.Series]] = field(default=None)
    effective_irradiance: Optional[PerArray[pd.Series]] = field(default=None)
    dc: Optional[PerArray[Union[pd.Series, pd.DataFrame]]] = \
        field(default=None)
    diode_params: Optional[PerArray[pd.DataFrame]] = field(default=None)


class ModelChain:
    """
    The ModelChain class to provides a standardized, high-level
    interface for all of the modeling steps necessary for calculating PV
    power from a time series of weather inputs. The same models are applied
    to all ``pvsystem.Array`` objects, so each Array must contain the
    appropriate model parameters. For example, if ``dc_model='pvwatts'``,
    then each ``Array.module_parameters`` must contain ``'pdc0'``.

    See https://pvlib-python.readthedocs.io/en/stable/modelchain.html
    for examples.

    Parameters
    ----------
    system : PVSystem
        A :py:class:`~pvlib.pvsystem.PVSystem` object that represents
        the connected set of modules, inverters, etc.

    location : Location
        A :py:class:`~pvlib.location.Location` object that represents
        the physical location at which to evaluate the model.

    orientation_strategy : None or str, default None
        The strategy for aligning the modules. If not None, sets the
        ``surface_azimuth`` and ``surface_tilt`` properties of the
        ``system``. Allowed strategies include 'flat',
        'south_at_latitude_tilt'. Ignored for SingleAxisTracker systems.

    clearsky_model : str, default 'ineichen'
        Passed to location.get_clearsky.

    transposition_model : str, default 'haydavies'
        Passed to system.get_irradiance.

    solar_position_method : str, default 'nrel_numpy'
        Passed to location.get_solarposition.

    airmass_model : str, default 'kastenyoung1989'
        Passed to location.get_airmass.

    dc_model: None, str, or function, default None
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'sapm',
        'desoto', 'cec', 'pvsyst', 'pvwatts'. The ModelChain instance will
        be passed as the first argument to a user-defined function.

    ac_model: None, str, or function, default None
        If None, the model will be inferred from the contents of
        system.inverter_parameters and system.module_parameters. Valid
        strings are 'sandia', 'adr', 'pvwatts'. The
        ModelChain instance will be passed as the first argument to a
        user-defined function.

    aoi_model: None, str, or function, default None
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'physical',
        'ashrae', 'sapm', 'martin_ruiz', 'no_loss'. The ModelChain instance
        will be passed as the first argument to a user-defined function.

    spectral_model: None, str, or function, default None
        If None, the model will be inferred from the contents of
        system.module_parameters. Valid strings are 'sapm',
        'first_solar', 'no_loss'. The ModelChain instance will be passed
        as the first argument to a user-defined function.

    temperature_model: None, str or function, default None
        Valid strings are 'sapm', 'pvsyst', 'faiman', and 'fuentes'.
        The ModelChain instance will be passed as the first argument to a
        user-defined function.

    losses_model: str or function, default 'no_loss'
        Valid strings are 'pvwatts', 'no_loss'. The ModelChain instance
        will be passed as the first argument to a user-defined function.

    name: None or str, default None
        Name of ModelChain instance.
    """

    # list of deprecated attributes
    _deprecated_attrs = ['solar_position', 'airmass', 'total_irrad',
                         'aoi', 'aoi_modifier', 'spectral_modifier',
                         'cell_temperature', 'effective_irradiance',
                         'dc', 'ac', 'diode_params', 'tracking']

    def __init__(self, system, location,
                 orientation_strategy=None,
                 clearsky_model='ineichen',
                 transposition_model='haydavies',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 dc_model=None, ac_model=None, aoi_model=None,
                 spectral_model=None, temperature_model=None,
                 losses_model='no_loss', name=None):

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
        self.temperature_model = temperature_model

        self.losses_model = losses_model
        self.orientation_strategy = orientation_strategy

        self.weather = None
        self.times = None

        self.results = ModelChainResult()

    def __getattr__(self, key):
        if key in ModelChain._deprecated_attrs:
            msg = f'ModelChain.{key} is deprecated and will' \
                  f' be removed in v0.10. Use' \
                  f' ModelChain.results.{key} instead'
            warnings.warn(msg, pvlibDeprecationWarning)
            return getattr(self.results, key)
        # __getattr__ is only called if __getattribute__ fails.
        # In that case we should check if key is a deprecated attribute,
        # and fail with an AttributeError if it is not.
        raise AttributeError

    def __setattr__(self, key, value):
        if key in ModelChain._deprecated_attrs:
            msg = f'ModelChain.{key} is deprecated from v0.9. Use' \
                  f' ModelChain.results.{key} instead'
            warnings.warn(msg, pvlibDeprecationWarning)
            setattr(self.results, key, value)
        else:
            super().__setattr__(key, value)

    @classmethod
    def with_pvwatts(cls, system, location,
                     orientation_strategy=None,
                     clearsky_model='ineichen',
                     airmass_model='kastenyoung1989',
                     name=None,
                     **kwargs):
        """
        ModelChain that follows the PVWatts methods.

        Parameters
        ----------
        system : PVSystem
            A :py:class:`~pvlib.pvsystem.PVSystem` object that represents
            the connected set of modules, inverters, etc.

        location : Location
            A :py:class:`~pvlib.location.Location` object that represents
            the physical location at which to evaluate the model.

        orientation_strategy : None or str, default None
            The strategy for aligning the modules. If not None, sets the
            ``surface_azimuth`` and ``surface_tilt`` properties of the
            ``system``. Allowed strategies include 'flat',
            'south_at_latitude_tilt'. Ignored for SingleAxisTracker systems.

        clearsky_model : str, default 'ineichen'
            Passed to location.get_clearsky.

        airmass_model : str, default 'kastenyoung1989'
            Passed to location.get_airmass.

        name: None or str, default None
            Name of ModelChain instance.

        **kwargs
            Parameters supplied here are passed to the ModelChain
            constructor and take precedence over the default
            configuration.

        Examples
        --------
        >>> module_parameters = dict(gamma_pdc=-0.003, pdc0=4500)
        >>> inverter_parameters = dict(pac0=4000)
        >>> tparams = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        >>> system = PVSystem(surface_tilt=30, surface_azimuth=180,
        ...     module_parameters=module_parameters,
        ...     inverter_parameters=inverter_parameters,
        ...     temperature_model_parameters=tparams)
        >>> location = Location(32.2, -110.9)
        >>> ModelChain.with_pvwatts(system, location)
        ModelChain:
          name: None
          orientation_strategy: None
          clearsky_model: ineichen
          transposition_model: perez
          solar_position_method: nrel_numpy
          airmass_model: kastenyoung1989
          dc_model: pvwatts_dc
          ac_model: pvwatts_inverter
          aoi_model: physical_aoi_loss
          spectral_model: no_spectral_loss
          temperature_model: sapm_temp
          losses_model: pvwatts_losses
        """  # noqa: E501
        config = PVWATTS_CONFIG.copy()
        config.update(kwargs)
        return ModelChain(
            system, location,
            orientation_strategy=orientation_strategy,
            clearsky_model=clearsky_model,
            airmass_model=airmass_model,
            name=name,
            **config
        )

    @classmethod
    def with_sapm(cls, system, location,
                  orientation_strategy=None,
                  clearsky_model='ineichen',
                  transposition_model='haydavies',
                  solar_position_method='nrel_numpy',
                  airmass_model='kastenyoung1989',
                  name=None,
                  **kwargs):
        """
        ModelChain that follows the Sandia Array Performance Model
        (SAPM) methods.

        Parameters
        ----------
        system : PVSystem
            A :py:class:`~pvlib.pvsystem.PVSystem` object that represents
            the connected set of modules, inverters, etc.

        location : Location
            A :py:class:`~pvlib.location.Location` object that represents
            the physical location at which to evaluate the model.

        orientation_strategy : None or str, default None
            The strategy for aligning the modules. If not None, sets the
            ``surface_azimuth`` and ``surface_tilt`` properties of the
            ``system``. Allowed strategies include 'flat',
            'south_at_latitude_tilt'. Ignored for SingleAxisTracker systems.

        clearsky_model : str, default 'ineichen'
            Passed to location.get_clearsky.

        transposition_model : str, default 'haydavies'
            Passed to system.get_irradiance.

        solar_position_method : str, default 'nrel_numpy'
            Passed to location.get_solarposition.

        airmass_model : str, default 'kastenyoung1989'
            Passed to location.get_airmass.

        name: None or str, default None
            Name of ModelChain instance.

        **kwargs
            Parameters supplied here are passed to the ModelChain
            constructor and take precedence over the default
            configuration.

        Examples
        --------
        >>> mods = pvlib.pvsystem.retrieve_sam('sandiamod')
        >>> invs = pvlib.pvsystem.retrieve_sam('cecinverter')
        >>> module_parameters = mods['Canadian_Solar_CS5P_220M___2009_']
        >>> inverter_parameters = invs['ABB__MICRO_0_25_I_OUTD_US_240__240V_']
        >>> tparams = TEMPERATURE_MODEL_PARAMETERS['sapm']['open_rack_glass_glass']
        >>> system = PVSystem(surface_tilt=30, surface_azimuth=180,
        ...     module_parameters=module_parameters,
        ...     inverter_parameters=inverter_parameters,
        ...     temperature_model_parameters=tparams)
        >>> location = Location(32.2, -110.9)
        >>> ModelChain.with_sapm(system, location)
        ModelChain:
          name: None
          orientation_strategy: None
          clearsky_model: ineichen
          transposition_model: haydavies
          solar_position_method: nrel_numpy
          airmass_model: kastenyoung1989
          dc_model: sapm
          ac_model: snlinverter
          aoi_model: sapm_aoi_loss
          spectral_model: sapm_spectral_loss
          temperature_model: sapm_temp
          losses_model: no_extra_losses
        """  # noqa: E501
        config = SAPM_CONFIG.copy()
        config.update(kwargs)
        return ModelChain(
            system, location,
            orientation_strategy=orientation_strategy,
            clearsky_model=clearsky_model,
            transposition_model=transposition_model,
            solar_position_method=solar_position_method,
            airmass_model=airmass_model,
            name=name,
            **config
        )

    def __repr__(self):
        attrs = [
            'name', 'orientation_strategy', 'clearsky_model',
            'transposition_model', 'solar_position_method',
            'airmass_model', 'dc_model', 'ac_model', 'aoi_model',
            'spectral_model', 'temperature_model', 'losses_model'
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
            f'{attr}: {getmcattr(self, attr)}' for attr in attrs))

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
        # guess at model if None
        if model is None:
            self._dc_model, model = self.infer_dc_model()

        # Set model and validate parameters
        if isinstance(model, str):
            model = model.lower()
            if model in _DC_MODEL_PARAMS.keys():
                # validate module parameters
                missing_params = (
                    _DC_MODEL_PARAMS[model] -
                    _common_keys(self.system.module_parameters))
                if missing_params:  # some parameters are not in module.keys()
                    raise ValueError(model + ' selected for the DC model but '
                                     'one or more Arrays are missing '
                                     'one or more required parameters '
                                     ' : ' + str(missing_params))
                if model == 'sapm':
                    self._dc_model = self.sapm
                elif model == 'desoto':
                    self._dc_model = self.desoto
                elif model == 'cec':
                    self._dc_model = self.cec
                elif model == 'pvsyst':
                    self._dc_model = self.pvsyst
                elif model == 'pvwatts':
                    self._dc_model = self.pvwatts_dc
            else:
                raise ValueError(model + ' is not a valid DC power model')
        else:
            self._dc_model = partial(model, self)

    def infer_dc_model(self):
        """Infer DC power model from Array module parameters."""
        params = _common_keys(self.system.module_parameters)
        if {'A0', 'A1', 'C7'} <= params:
            return self.sapm, 'sapm'
        elif {'a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s',
              'Adjust'} <= params:
            return self.cec, 'cec'
        elif {'a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref', 'R_s'} <= params:
            return self.desoto, 'desoto'
        elif {'gamma_ref', 'mu_gamma', 'I_L_ref', 'I_o_ref', 'R_sh_ref',
              'R_sh_0', 'R_sh_exp', 'R_s'} <= params:
            return self.pvsyst, 'pvsyst'
        elif {'pdc0', 'gamma_pdc'} <= params:
            return self.pvwatts_dc, 'pvwatts'
        else:
            raise ValueError('could not infer DC model from '
                             'system.module_parameters. Check '
                             'system.module_parameters or explicitly '
                             'set the model with the dc_model kwarg.')

    def sapm(self):
        self.results.dc = self.system.sapm(self.results.effective_irradiance,
                                           self.results.cell_temperature)

        self.results.dc = self.system.scale_voltage_current_power(
            self.results.dc)

        return self

    def _singlediode(self, calcparams_model_function):
        def _make_diode_params(photocurrent, saturation_current,
                               resistance_series, resistance_shunt,
                               nNsVth):
            return pd.DataFrame(
                {'I_L': photocurrent, 'I_o': saturation_current,
                 'R_s': resistance_series, 'R_sh': resistance_shunt,
                 'nNsVth': nNsVth}
            )
        params = calcparams_model_function(self.results.effective_irradiance,
                                           self.results.cell_temperature,
                                           unwrap=False)
        self.results.diode_params = tuple(itertools.starmap(
            _make_diode_params, params))
        self.results.dc = tuple(itertools.starmap(
            self.system.singlediode, params))
        self.results.dc = self.system.scale_voltage_current_power(
            self.results.dc,
            unwrap=False
        )
        self.results.dc = tuple(dc.fillna(0) for dc in self.results.dc)
        # If the system has one Array, unwrap the single return value
        # to preserve the original behavior of ModelChain
        if self.system.num_arrays == 1:
            self.results.diode_params = self.results.diode_params[0]
            self.results.dc = self.results.dc[0]
        return self

    def desoto(self):
        return self._singlediode(self.system.calcparams_desoto)

    def cec(self):
        return self._singlediode(self.system.calcparams_cec)

    def pvsyst(self):
        return self._singlediode(self.system.calcparams_pvsyst)

    def pvwatts_dc(self):
        """Calculate DC power using the PVWatts model.

        Results are stored in ModelChain.results.dc. DC power is computed
        from PVSystem.module_parameters['pdc0'] and then scaled by
        PVSystem.modules_per_string and PVSystem.strings_per_inverter.

        Returns
        -------
        self

        See also
        --------
        pvlib.pvsystem.PVSystem.pvwatts_dc
        pvlib.pvsystem.PVSystem.scale_voltage_current_power
        """
        self.results.dc = self.system.pvwatts_dc(
            self.results.effective_irradiance, self.results.cell_temperature)
        if isinstance(self.results.dc, tuple):
            temp = tuple(
                pd.DataFrame(s, columns=['p_mp']) for s in self.results.dc)
        else:
            temp = pd.DataFrame(self.results.dc, columns=['p_mp'])
        scaled = self.system.scale_voltage_current_power(temp)
        if isinstance(scaled, tuple):
            self.results.dc = tuple(s['p_mp'] for s in scaled)
        else:
            self.results.dc = scaled['p_mp']
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
            if model == 'sandia':
                self._ac_model = self.snlinverter
            elif model == 'sandia_multi':
                self._ac_model = self.sandia_multi_inverter
            elif model in 'adr':
                self._ac_model = self.adrinverter
            elif model == 'pvwatts':
                self._ac_model = self.pvwatts_inverter
            elif model == 'pvwatts_multi':
                self._ac_model = self.pvwatts_multi_inverter
            else:
                raise ValueError(model + ' is not a valid AC power model')
        else:
            self._ac_model = partial(model, self)

    def infer_ac_model(self):
        """Infer AC power model from system attributes."""
        inverter_params = set(self.system.inverter_parameters.keys())
        if self.system.num_arrays > 1:
            return self._infer_ac_model_multi(inverter_params)
        if _snl_params(inverter_params):
            return self.snlinverter
        if _adr_params(inverter_params):
            return self.adrinverter
        if _pvwatts_params(inverter_params):
            return self.pvwatts_inverter
        raise ValueError('could not infer AC model from '
                         'system.inverter_parameters. Check '
                         'system.inverter_parameters or explicitly '
                         'set the model with the ac_model kwarg.')

    def _infer_ac_model_multi(self, inverter_params):
        if _snl_params(inverter_params):
            return self.sandia_multi_inverter
        elif _pvwatts_params(inverter_params):
            return self.pvwatts_multi_inverter
        raise ValueError('could not infer multi-array AC model from '
                         'system.inverter_parameters. Only sandia and pvwatts '
                         'inverter models support multiple '
                         'Arrays. Check system.inverter_parameters or '
                         'explicitly set the model with the ac_model kwarg.')

    def sandia_multi_inverter(self):
        self.results.ac = self.system.sandia_multi(
            _tuple_from_dfs(self.results.dc, 'v_mp'),
            _tuple_from_dfs(self.results.dc, 'p_mp')
        )
        return self

    def pvwatts_multi_inverter(self):
        self.results.ac = self.system.pvwatts_multi(self.results.dc)
        return self

    def snlinverter(self):
        self.results.ac = self.system.snlinverter(self.results.dc['v_mp'],
                                                  self.results.dc['p_mp'])
        return self

    def adrinverter(self):
        self.results.ac = self.system.adrinverter(self.results.dc['v_mp'],
                                                  self.results.dc['p_mp'])
        return self

    def pvwatts_inverter(self):
        self.results.ac = self.system.pvwatts_ac(self.results.dc).fillna(0)
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
            elif model == 'martin_ruiz':
                self._aoi_model = self.martin_ruiz_aoi_loss
            elif model == 'no_loss':
                self._aoi_model = self.no_aoi_loss
            else:
                raise ValueError(model + ' is not a valid aoi loss model')
        else:
            self._aoi_model = partial(model, self)

    def infer_aoi_model(self):
        params = _common_keys(self.system.module_parameters)
        if {'K', 'L', 'n'} <= params:
            return self.physical_aoi_loss
        elif {'B5', 'B4', 'B3', 'B2', 'B1', 'B0'} <= params:
            return self.sapm_aoi_loss
        elif {'b'} <= params:
            return self.ashrae_aoi_loss
        elif {'a_r'} <= params:
            return self.martin_ruiz_aoi_loss
        else:
            raise ValueError('could not infer AOI model from '
                             'system.module_parameters. Check that the '
                             'module_parameters for all Arrays in '
                             'system.arrays contain parameters for '
                             'the physical, aoi, ashrae or martin_ruiz model; '
                             'explicitly set the model with the aoi_model '
                             'kwarg; or set aoi_model="no_loss".')

    def ashrae_aoi_loss(self):
        self.results.aoi_modifier = self.system.get_iam(
            self.results.aoi, iam_model='ashrae')
        return self

    def physical_aoi_loss(self):
        self.results.aoi_modifier = self.system.get_iam(self.results.aoi,
                                                        iam_model='physical')
        return self

    def sapm_aoi_loss(self):
        self.results.aoi_modifier = self.system.get_iam(self.results.aoi,
                                                        iam_model='sapm')
        return self

    def martin_ruiz_aoi_loss(self):
        self.results.aoi_modifier = self.system.get_iam(
            self.results.aoi,
            iam_model='martin_ruiz')
        return self

    def no_aoi_loss(self):
        if self.system.num_arrays == 1:
            self.results.aoi_modifier = 1.0
        else:
            self.results.aoi_modifier = (1.0,) * self.system.num_arrays
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
                self._spectral_model = self.first_solar_spectral_loss
            elif model == 'sapm':
                self._spectral_model = self.sapm_spectral_loss
            elif model == 'no_loss':
                self._spectral_model = self.no_spectral_loss
            else:
                raise ValueError(model + ' is not a valid spectral loss model')
        else:
            self._spectral_model = partial(model, self)

    def infer_spectral_model(self):
        """Infer spectral model from system attributes."""
        params = _common_keys(self.system.module_parameters)
        if {'A4', 'A3', 'A2', 'A1', 'A0'} <= params:
            return self.sapm_spectral_loss
        elif ((('Technology' in params or
                'Material' in params) and
               (self.system._infer_cell_type() is not None)) or
              'first_solar_spectral_coefficients' in params):
            return self.first_solar_spectral_loss
        else:
            raise ValueError('could not infer spectral model from '
                             'system.module_parameters. Check that the '
                             'module_parameters for all Arrays in '
                             'system.arrays contain valid '
                             'first_solar_spectral_coefficients, a valid '
                             'Material or Technology value, or set '
                             'spectral_model="no_loss".')

    def first_solar_spectral_loss(self):
        self.results.spectral_modifier = self.system.first_solar_spectral_loss(
            self.weather['precipitable_water'],
            self.results.airmass['airmass_absolute'])
        return self

    def sapm_spectral_loss(self):
        self.results.spectral_modifier = self.system.sapm_spectral_loss(
            self.results.airmass['airmass_absolute'])
        return self

    def no_spectral_loss(self):
        if self.system.num_arrays == 1:
            self.results.spectral_modifier = 1
        else:
            self.results.spectral_modifier = (1,) * self.system.num_arrays
        return self

    @property
    def temperature_model(self):
        return self._temperature_model

    @temperature_model.setter
    def temperature_model(self, model):
        if model is None:
            self._temperature_model = self.infer_temperature_model()
        elif isinstance(model, str):
            model = model.lower()
            if model == 'sapm':
                self._temperature_model = self.sapm_temp
            elif model == 'pvsyst':
                self._temperature_model = self.pvsyst_temp
            elif model == 'faiman':
                self._temperature_model = self.faiman_temp
            elif model == 'fuentes':
                self._temperature_model = self.fuentes_temp
            else:
                raise ValueError(model + ' is not a valid temperature model')
            # check system.temperature_model_parameters for consistency
            name_from_params = self.infer_temperature_model().__name__
            if self._temperature_model.__name__ != name_from_params:
                raise ValueError(
                    f'Temperature model {self._temperature_model.__name__} is '
                    f'inconsistent with PVSystem temperature model '
                    f'parameters. All Arrays in system.arrays must have '
                    f'consistent parameters. Common temperature model '
                    f'parameters: '
                    f'{_common_keys(self.system.temperature_model_parameters)}'
                )
        else:
            self._temperature_model = partial(model, self)

    def infer_temperature_model(self):
        """Infer temperature model from system attributes."""
        params = _common_keys(self.system.temperature_model_parameters)
        # remove or statement in v0.9
        if {'a', 'b', 'deltaT'} <= params or (
                not params and self.system.racking_model is None
                and self.system.module_type is None):
            return self.sapm_temp
        elif {'u_c', 'u_v'} <= params:
            return self.pvsyst_temp
        elif {'u0', 'u1'} <= params:
            return self.faiman_temp
        elif {'noct_installed'} <= params:
            return self.fuentes_temp
        else:
            raise ValueError(f'could not infer temperature model from '
                             f'system.temperature_model_parameters. Check '
                             f'that all Arrays in system.arrays have '
                             f'parameters for the same temperature model. '
                             f'Common temperature model parameters: {params}.')

    def _set_celltemp(self, model):
        """Set self.results.cell_temperature using the given cell
        temperature model.

        Parameters
        ----------
        model : function
            A function that takes POA irradiance, air temperature, and
            wind speed and returns cell temperature. `model` must accept
            tuples or single values for each parameter where each element of
            the tuple is the value for a different array in the system
            (see :py:class:`pvlib.pvsystem.PVSystem` for more information).

        Returns
        -------
        self
        """

        poa = _irrad_for_celltemp(self.results.total_irrad,
                                  self.results.effective_irradiance)
        temp_air = _tuple_from_dfs(self.weather, 'temp_air')
        wind_speed = _tuple_from_dfs(self.weather, 'wind_speed')
        self.results.cell_temperature = model(poa, temp_air, wind_speed)
        return self

    def sapm_temp(self):
        return self._set_celltemp(self.system.sapm_celltemp)

    def pvsyst_temp(self):
        return self._set_celltemp(self.system.pvsyst_celltemp)

    def faiman_temp(self):
        return self._set_celltemp(self.system.faiman_celltemp)

    def fuentes_temp(self):
        return self._set_celltemp(self.system.fuentes_celltemp)

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
        if self.system.num_arrays > 1:
            for dc in self.results.dc:
                dc *= self.losses
        else:
            self.results.dc *= self.losses
        return self

    def no_extra_losses(self):
        self.losses = 1
        return self

    def effective_irradiance_model(self):
        def _eff_irrad(module_parameters, total_irrad, spect_mod, aoi_mod):
            fd = module_parameters.get('FD', 1.)
            return spect_mod * (total_irrad['poa_direct'] * aoi_mod +
                                fd * total_irrad['poa_diffuse'])
        if isinstance(self.results.total_irrad, tuple):
            self.results.effective_irradiance = tuple(
                _eff_irrad(array.module_parameters, ti, sm, am) for
                array, ti, sm, am in zip(
                    self.system.arrays, self.results.total_irrad,
                    self.results.spectral_modifier, self.results.aoi_modifier))
        else:
            self.results.effective_irradiance = _eff_irrad(
                self.system.module_parameters,
                self.results.total_irrad,
                self.results.spectral_modifier,
                self.results.aoi_modifier
            )
        return self

    def complete_irradiance(self, weather):
        """
        Determine the missing irradiation columns. Only two of the
        following data columns (dni, ghi, dhi) are needed to calculate
        the missing data.

        This function is not safe at the moment. Results can be too high
        or negative. Please contribute and help to improve this function
        on https://github.com/pvlib/pvlib-python

        Parameters
        ----------
        weather : DataFrame, or tuple or list of DataFrame
            Column names must be ``'dni'``, ``'ghi'``, ``'dhi'``,
            ``'wind_speed'``, ``'temp_air'``. All irradiance components
            are required. Air temperature of 20 C and wind speed
            of 0 m/s will be added to the DataFrame if not provided.
            If `weather` is a tuple it must be the same length as the number
            of Arrays in the system and the indices for each DataFrame must
            be the same.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            if the number of dataframes in `weather` is not the same as the
            number of Arrays in the system or if the indices of all elements
            of `weather` are not the same.

        Notes
        -----
        Assigns attributes: ``weather``

        Examples
        --------
        This example does not work until the parameters `my_system`,
        `my_location`, and `my_weather` are defined but shows the basic idea
        how this method can be used.

        >>> from pvlib.modelchain import ModelChain

        >>> # my_weather containing 'dhi' and 'ghi'.
        >>> mc = ModelChain(my_system, my_location)  # doctest: +SKIP
        >>> mc.complete_irradiance(my_weather)  # doctest: +SKIP
        >>> mc.run_model(mc.weather)  # doctest: +SKIP

        >>> # my_weather containing 'dhi', 'ghi' and 'dni'.
        >>> mc = ModelChain(my_system, my_location)  # doctest: +SKIP
        >>> mc.run_model(my_weather)  # doctest: +SKIP
        """
        weather = _to_tuple(weather)
        self._check_multiple_input(weather)
        # Don't use ModelChain._assign_weather() here because it adds
        # temperature and wind-speed columns which we do not need here.
        self.weather = _copy(weather)
        self._assign_times()
        self.results.solar_position = self.location.get_solarposition(
            self.times, method=self.solar_position_method)

        if isinstance(weather, tuple):
            for w in self.weather:
                self._complete_irradiance(w)
        else:
            self._complete_irradiance(self.weather)

        return self

    def _complete_irradiance(self, weather):
        icolumns = set(weather.columns)
        wrn_txt = ("This function is not safe at the moment.\n" +
                   "Results can be too high or negative.\n" +
                   "Help to improve this function on github:\n" +
                   "https://github.com/pvlib/pvlib-python \n")

        if {'ghi', 'dhi'} <= icolumns and 'dni' not in icolumns:
            clearsky = self.location.get_clearsky(
                weather.index, solar_position=self.results.solar_position)
            weather.loc[:, 'dni'] = pvlib.irradiance.dni(
                weather.loc[:, 'ghi'], weather.loc[:, 'dhi'],
                self.results.solar_position.zenith,
                clearsky_dni=clearsky['dni'],
                clearsky_tolerance=1.1)
        elif {'dni', 'dhi'} <= icolumns and 'ghi' not in icolumns:
            warnings.warn(wrn_txt, UserWarning)
            weather.loc[:, 'ghi'] = (
                weather.dhi + weather.dni *
                tools.cosd(self.results.solar_position.zenith)
            )
        elif {'dni', 'ghi'} <= icolumns and 'dhi' not in icolumns:
            warnings.warn(wrn_txt, UserWarning)
            weather.loc[:, 'dhi'] = (
                weather.ghi - weather.dni *
                tools.cosd(self.results.solar_position.zenith))

    def _prep_inputs_solar_pos(self, kwargs={}):
        """
        Assign solar position
        """
        self.results.solar_position = self.location.get_solarposition(
            self.times, method=self.solar_position_method,
            **kwargs)
        return self

    def _prep_inputs_airmass(self):
        """
        Assign airmass
        """
        self.results.airmass = self.location.get_airmass(
            solar_position=self.results.solar_position,
            model=self.airmass_model)
        return self

    def _prep_inputs_tracking(self):
        """
        Calculate tracker position and AOI
        """
        self.results.tracking = self.system.singleaxis(
            self.results.solar_position['apparent_zenith'],
            self.results.solar_position['azimuth'])
        self.results.tracking['surface_tilt'] = (
            self.results.tracking['surface_tilt']
                .fillna(self.system.axis_tilt))
        self.results.tracking['surface_azimuth'] = (
            self.results.tracking['surface_azimuth']
                .fillna(self.system.axis_azimuth))
        self.results.aoi = self.results.tracking['aoi']
        return self

    def _prep_inputs_fixed(self):
        """
        Calculate AOI for fixed tilt system
        """
        self.results.aoi = self.system.get_aoi(
            self.results.solar_position['apparent_zenith'],
            self.results.solar_position['azimuth'])
        return self

    def _verify_df(self, data, required):
        """ Checks data for column names in required

        Parameters
        ----------
        data : Dataframe
        required : List of str

        Raises
        ------
        ValueError if any of required are not in data.columns.
        """
        def _verify(data, index=None):
            if not set(required) <= set(data.columns):
                tuple_txt = "" if index is None else f"in element {index} "
                raise ValueError(
                    "Incomplete input data. Data needs to contain "
                    f"{required}. Detected data {tuple_txt}contains: "
                    f"{list(data.columns)}")
        if not isinstance(data, tuple):
            _verify(data)
        else:
            for (i, array_data) in enumerate(data):
                _verify(array_data, i)

    def _assign_weather(self, data):
        def _build_weather(data):
            key_list = [k for k in WEATHER_KEYS if k in data]
            weather = data[key_list].copy()
            if weather.get('wind_speed') is None:
                weather['wind_speed'] = 0
            if weather.get('temp_air') is None:
                weather['temp_air'] = 20
            return weather
        if not isinstance(data, tuple):
            self.weather = _build_weather(data)
        else:
            self.weather = tuple(
                _build_weather(weather) for weather in data
            )
        return self

    def _assign_total_irrad(self, data):
        def _build_irrad(data):
            key_list = [k for k in POA_KEYS if k in data]
            return data[key_list].copy()
        if isinstance(data, tuple):
            self.results.total_irrad = tuple(
                _build_irrad(irrad_data) for irrad_data in data
            )
            return self
        self.results.total_irrad = _build_irrad(data)
        return self

    def _assign_times(self):
        """Assign self.times according the the index of self.weather.

        If there are multiple DataFrames in self.weather then the index
        of the first one is assigned. It is assumed that the indices of
        each data frame in `weather` are the same. This can be verified
        by calling :py:func:`_all_same_index` or
        :py:meth:`self._check_multiple_weather` before calling this method.
        """
        if isinstance(self.weather, tuple):
            self.times = self.weather[0].index
        else:
            self.times = self.weather.index

    def prepare_inputs(self, weather):
        """
        Prepare the solar position, irradiance, and weather inputs to
        the model, starting with GHI, DNI and DHI.

        Parameters
        ----------
        weather : DataFrame, or tuple or list of DataFrame
            Required column names include ``'dni'``, ``'ghi'``, ``'dhi'``.
            Optional column names are ``'wind_speed'``, ``'temp_air'``; if not
            provided, air temperature of 20 C and wind speed
            of 0 m/s will be added to the DataFrame.

            If `weather` is a tuple or list, it must be of the same length and
            order as the Arrays of the ModelChain's PVSystem.

        Raises
        ------
        ValueError
            If any `weather` DataFrame(s) is missing an irradiance component.
        ValueError
            If `weather` is a tuple or list and the DataFrames it contains have
            different indices.
        ValueError
            If `weather` is a tuple or list with a different length than the
            number of Arrays in the system.

        Notes
        -----
        Assigns attributes: ``weather``, ``solar_position``, ``airmass``,
        ``total_irrad``, ``aoi``

        See also
        --------
        ModelChain.complete_irradiance
        """
        weather = _to_tuple(weather)
        self._check_multiple_input(weather, strict=False)
        self._verify_df(weather, required=['ghi', 'dni', 'dhi'])
        self._assign_weather(weather)
        self._assign_times()

        # build kwargs for solar position calculation
        try:
            press_temp = _build_kwargs(['pressure', 'temp_air'],
                                       weather[0] if isinstance(weather, tuple)
                                       else weather)
            press_temp['temperature'] = press_temp.pop('temp_air')
        except KeyError:
            pass

        self._prep_inputs_solar_pos(press_temp)
        self._prep_inputs_airmass()

        # PVSystem.get_irradiance and SingleAxisTracker.get_irradiance
        # and PVSystem.get_aoi and SingleAxisTracker.get_aoi
        # have different method signatures. Use partial to handle
        # the differences.
        if isinstance(self.system, SingleAxisTracker):
            self._prep_inputs_tracking()
            get_irradiance = partial(
                self.system.get_irradiance,
                self.results.tracking['surface_tilt'],
                self.results.tracking['surface_azimuth'],
                self.results.solar_position['apparent_zenith'],
                self.results.solar_position['azimuth'])
        else:
            self._prep_inputs_fixed()
            get_irradiance = partial(
                self.system.get_irradiance,
                self.results.solar_position['apparent_zenith'],
                self.results.solar_position['azimuth'])

        self.results.total_irrad = get_irradiance(
            _tuple_from_dfs(self.weather, 'dni'),
            _tuple_from_dfs(self.weather, 'ghi'),
            _tuple_from_dfs(self.weather, 'dhi'),
            airmass=self.results.airmass['airmass_relative'],
            model=self.transposition_model)

        return self

    def _check_multiple_input(self, data, strict=True):
        """Check that the number of elements in `data` is the same as
        the number of Arrays in `self.system`.

        In most cases if ``self.system.num_arrays`` is greater than 1 we
        want to raise an error when `data` is not a tuple; however, that
        behavior can be suppressed by setting ``strict=False``. This is
        useful for validating inputs such as GHI, DHI, DNI, wind speed, or
        air temperature that can be applied a ``PVSystem`` as a system-wide
        input. In this case we want to ensure that when a tuple is provided
        it has the same length as the number of Arrays, but we do not want
        to fail if the input is not a tuple.
        """
        if (not strict or self.system.num_arrays == 1) \
                and not isinstance(data, tuple):
            return
        if strict and not isinstance(data, tuple):
            raise TypeError("Input must be a tuple of length "
                            f"{self.system.num_arrays}, "
                            f"got {type(data).__name__}.")
        if len(data) != self.system.num_arrays:
            raise ValueError("Input must be same length as number of Arrays "
                             f"in system. Expected {self.system.num_arrays}, "
                             f"got {len(data)}.")
        _all_same_index(data)

    def prepare_inputs_from_poa(self, data):
        """
        Prepare the solar position, irradiance and weather inputs to
        the model, starting with plane-of-array irradiance.

        Parameters
        ----------
        data : DataFrame, or tuple or list of DataFrame
            Contains plane-of-array irradiance data. Required column names
            include ``'poa_global'``, ``'poa_direct'`` and ``'poa_diffuse'``.
            Columns with weather-related data are ssigned to the
            ``weather`` attribute.  If columns for ``'temp_air'`` and
            ``'wind_speed'`` are not provided, air temperature of 20 C and wind
            speed of 0 m/s are assumed.

            If list or tuple, must be of the same length and order as the
            Arrays of the ModelChain's PVSystem.

        Raises
        ------
        ValueError
             If the number of DataFrames passed in `data` is not the same
             as the number of Arrays in the system.

        Notes
        -----
        Assigns attributes: ``weather``, ``total_irrad``, ``solar_position``,
        ``airmass``, ``aoi``.

        See also
        --------
        pvlib.modelchain.ModelChain.prepare_inputs
        """
        data = _to_tuple(data)
        self._check_multiple_input(data)
        self._assign_weather(data)

        self._verify_df(data, required=['poa_global', 'poa_direct',
                                        'poa_diffuse'])
        self._assign_total_irrad(data)

        self._prep_inputs_solar_pos()
        self._prep_inputs_airmass()

        if isinstance(self.system, SingleAxisTracker):
            self._prep_inputs_tracking()
        else:
            self._prep_inputs_fixed()

        return self

    def _get_cell_temperature(self, data,
                              poa, temperature_model_parameters):
        """Extract the cell temperature data from a DataFrame.

        If 'cell_temperature' column exists in data then it is returned. If
        'module_temperature' column exists in data, then it is used with poa to
        calculate the cell temperature. If neither column exists then None is
        returned.

        Parameters
        ----------
        data : DataFrame (not a tuple of DataFrame)
        poa : Series (not a tuple of Series)

        Returns
        -------
        Series
        """
        if 'cell_temperature' in data:
            return data['cell_temperature']
        # cell_temperature is not in input. Calculate cell_temperature using
        # a temperature_model.
        # If module_temperature is in input data we can use the SAPM cell
        # temperature model.
        if (('module_temperature' in data) and
                (self.temperature_model == self.sapm_temp)):
            # use SAPM cell temperature model only
            return pvlib.temperature.sapm_cell_from_module(
                module_temperature=data['module_temperature'],
                poa_global=poa,
                deltaT=temperature_model_parameters['deltaT'])

    def _prepare_temperature_single_array(self, data, poa):
        """Set cell_temperature using a single data frame."""
        self.results.cell_temperature = self._get_cell_temperature(
            data,
            poa,
            self.system.temperature_model_parameters
        )
        if self.results.cell_temperature is None:
            self.temperature_model()
        return self

    def _prepare_temperature(self, data=None):
        """
        Sets cell_temperature using inputs in data and the specified
        temperature model.

        If 'data' contains 'cell_temperature', these values are assigned to
        attribute ``cell_temperature``. If 'data' contains 'module_temperature`
        and `temperature_model' is 'sapm', cell temperature is calculated using
        :py:func:`pvlib.temperature.sapm_cell_from_module`. Otherwise, cell
        temperature is calculated by 'temperature_model'.

        Parameters
        ----------
        data : DataFrame, default None
            May contain columns ``'cell_temperature'`` or
            ``'module_temperaure'``.

        Returns
        -------
        self

        Assigns attribute ``results.cell_temperature``.

        """
        poa = _irrad_for_celltemp(self.results.total_irrad,
                                  self.results.effective_irradiance)
        if not isinstance(data, tuple) and self.system.num_arrays > 1:
            # broadcast data to all arrays
            data = (data,) * self.system.num_arrays
        elif not isinstance(data, tuple):
            return self._prepare_temperature_single_array(data, poa)
        given_cell_temperature = tuple(itertools.starmap(
            self._get_cell_temperature,
            zip(data, poa, self.system.temperature_model_parameters)
        ))
        # If cell temperature has been specified for all arrays return
        # immediately and do not try to compute it.
        if all(cell_temp is not None for cell_temp in given_cell_temperature):
            self.results.cell_temperature = given_cell_temperature
            return self
        # Calculate cell temperature from weather data. If cell_temperature
        # has not been provided for some arrays then it is computed with
        # ModelChain.temperature_model(). Because this operates on all Arrays
        # simultaneously, 'poa_global' must be known for all arrays, including
        # those that have a known cell temperature.
        try:
            self._verify_df(self.results.total_irrad, ['poa_global'])
        except ValueError:
            # Provide a more informative error message. Because only
            # run_model_from_effective_irradiance() can get to this point
            # without known POA we can suggest a very specific remedy in the
            # error message.
            raise ValueError("Incomplete input data. Data must contain "
                             "'poa_global'. For systems with multiple Arrays "
                             "if you have provided 'cell_temperature' for "
                             "only a subset of Arrays you must provide "
                             "'poa_global' for all Arrays, including those "
                             "that have a known 'cell_temperature'.")
        self.temperature_model()
        # replace calculated cell temperature with temperature given in `data`
        # where available.
        self.results.cell_temperature = tuple(
            itertools.starmap(
                lambda given, modeled: modeled if given is None else given,
                zip(given_cell_temperature, self.results.cell_temperature)
            )
        )
        return self

    def run_model(self, weather):
        """
        Run the model chain starting with broadband global, diffuse and/or
        direct irradiance.

        Parameters
        ----------
        weather : DataFrame, or tuple or list of DataFrame
            Irradiance column names must include ``'dni'``, ``'ghi'``, and
            ``'dhi'``. If optional columns ``'temp_air'`` and ``'wind_speed'``
            are not provided, air temperature of 20 C and wind speed of 0 m/s
            are added to the DataFrame. If optional column
            ``'cell_temperature'`` is provided, these values are used instead
            of `temperature_model`. If optional column `module_temperature`
            is provided, `temperature_model` must be ``'sapm'``.

            If list or tuple, must be of the same length and order as the
            Arrays of the ModelChain's PVSystem.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of DataFrames in `data` is different than the number
            of Arrays in the PVSystem.
        ValueError
            If the DataFrames in `data` have different indexes.

        Notes
        -----
        Assigns attributes: ``solar_position``, ``airmass``, ``weather``,
        ``total_irrad``, ``aoi``, ``aoi_modifier``, ``spectral_modifier``,
        and ``effective_irradiance``, ``cell_temperature``, ``dc``, ``ac``,
        ``losses``, ``diode_params`` (if dc_model is a single diode model).

        See also
        --------
        pvlib.modelchain.ModelChain.run_model_from_poa
        pvlib.modelchain.ModelChain.run_model_from_effective_irradiance
        """
        weather = _to_tuple(weather)
        self.prepare_inputs(weather)
        self.aoi_model()
        self.spectral_model()
        self.effective_irradiance_model()

        self._run_from_effective_irrad(weather)

        return self

    def run_model_from_poa(self, data):
        """
        Run the model starting with broadband irradiance in the plane of array.

        Data must include direct, diffuse and total irradiance (W/m2) in the
        plane of array. Reflections and spectral adjustments are made to
        calculate effective irradiance (W/m2).

        Parameters
        ----------
        data : DataFrame, or tuple or list of DataFrame
            Required column names include ``'poa_global'``,
            ``'poa_direct'`` and ``'poa_diffuse'``. If optional columns
            ``'temp_air'`` and ``'wind_speed'`` are not provided, air
            temperature of 20 C and wind speed of 0 m/s are assumed.
            If optional column ``'cell_temperature'`` is provided, these values
            are used instead of `temperature_model`. If optional column
            ``'module_temperature'`` is provided, `temperature_model` must be
            ``'sapm'``.

            If the ModelChain's PVSystem has multiple arrays, `data` must be a
            list or tuple with the same length and order as the PVsystem's
            Arrays. Each element of `data` provides the irradiance and weather
            for the corresponding array.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of DataFrames in `data` is different than the number
            of Arrays in the PVSystem.
        ValueError
            If the DataFrames in `data` have different indexes.

        Notes
        -----
        Assigns attributes: ``solar_position``, ``airmass``, ``weather``,
        ``total_irrad``, ``aoi``, ``aoi_modifier``, ``spectral_modifier``,
        and ``effective_irradiance``, ``cell_temperature``, ``dc``, ``ac``,
        ``losses``, ``diode_params`` (if dc_model is a single diode model).

        See also
        --------
        pvlib.modelchain.ModelChain.run_model
        pvlib.modelchain.ModelChain.run_model_from_effective_irradiance
        """
        data = _to_tuple(data)
        self.prepare_inputs_from_poa(data)

        self.aoi_model()
        self.spectral_model()
        self.effective_irradiance_model()

        self._run_from_effective_irrad(data)

        return self

    def _run_from_effective_irrad(self, data=None):
        """
        Executes the temperature, DC, losses and AC models.

        Parameters
        ----------
        data : DataFrame, or tuple of DataFrame, default None
            If optional column ``'cell_temperature'`` is provided, these values
            are used instead of `temperature_model`. If optional column
            `module_temperature` is provided, `temperature_model` must be
            ``'sapm'``.

        Returns
        -------
        self

        Notes
        -----
        Assigns attributes:``cell_temperature``, ``dc``, ``ac``, ``losses``,
        ``diode_params`` (if dc_model is a single diode model).
        """
        self._prepare_temperature(data)
        self.dc_model()
        self.losses_model()
        self.ac_model()

        return self

    def run_model_from_effective_irradiance(self, data=None):
        """
        Run the model starting with effective irradiance in the plane of array.

        Effective irradiance is irradiance in the plane-of-array after any
        adjustments for soiling, reflections and spectrum.

        Parameters
        ----------
        data : DataFrame, or list or tuple of DataFrame
            Required column is ``'effective_irradiance'``.
            Optional columns include ``'cell_temperature'``,
            ``'module_temperature'`` and ``'poa_global'``.

            If the ModelChain's PVSystem has multiple arrays, `data` must be a
            list or tuple with the same length and order as the PVsystem's
            Arrays. Each element of `data` provides the irradiance and weather
            for the corresponding array.

        Returns
        -------
        self

        Raises
        ------
        ValueError
            If the number of DataFrames in `data` is different than the number
            of Arrays in the PVSystem.
        ValueError
            If the DataFrames in `data` have different indexes.

        Notes
        -----
        Optional ``data`` columns ``'cell_temperature'``,
        ``'module_temperature'`` and ``'poa_global'`` are used for determining
        cell temperature.

        * If optional column ``'cell_temperature'`` is present, these values
          are used and `temperature_model` is ignored.
        * If optional column ``'module_temperature'`` is preset,
          `temperature_model` must be ``'sapm'``.
        * Otherwise, cell temperature is calculated using `temperature_model`.

        The cell temperature models require plane-of-array irradiance as input.
        If optional column ``'poa_global'`` is present, these data are used.
        If ``'poa_global'`` is not present, ``'effective_irradiance'`` is used.

        Assigns attributes: ``weather``, ``total_irrad``,
        ``effective_irradiance``, ``cell_temperature``, ``dc``, ``ac``,
        ``losses``, ``diode_params`` (if dc_model is a single diode model).

        See also
        --------
        pvlib.modelchain.ModelChain.run_model
        pvlib.modelchain.ModelChain.run_model_from_poa
        """
        data = _to_tuple(data)
        self._check_multiple_input(data)
        self._assign_weather(data)
        self._assign_total_irrad(data)
        self.results.effective_irradiance = _tuple_from_dfs(
            data, 'effective_irradiance')
        self._run_from_effective_irrad(data)

        return self


def _irrad_for_celltemp(total_irrad, effective_irradiance):
    """
    Determine irradiance to use for cell temperature models, in order
    of preference 'poa_global' then 'effective_irradiance'

    Returns
    -------
    Series or tuple of Series
        tuple if total_irrad is a tuple of DataFrame

    """
    if isinstance(total_irrad, tuple):
        if all(['poa_global' in df for df in total_irrad]):
            return _tuple_from_dfs(total_irrad, 'poa_global')
        else:
            return effective_irradiance
    else:
        if 'poa_global' in total_irrad:
            return total_irrad['poa_global']
        else:
            return effective_irradiance


def _snl_params(inverter_params):
    """Return True if `inverter_params` includes parameters for the
    Sandia inverter model."""
    return {'C0', 'C1', 'C2'} <= inverter_params


def _adr_params(inverter_params):
    """Return True if `inverter_params` includes parameters for the ADR
    inverter model."""
    return {'ADRCoefficients'} <= inverter_params


def _pvwatts_params(inverter_params):
    """Return True if `inverter_params` includes parameters for the
    PVWatts inverter model."""
    return {'pdc0'} <= inverter_params


def _copy(data):
    """Return a copy of each DataFrame in `data` if it is a tuple,
    otherwise return a copy of `data`."""
    if not isinstance(data, tuple):
        return data.copy()
    return tuple(df.copy() for df in data)


def _all_same_index(data):
    """Raise a ValueError if all DataFrames in `data` do not have the
    same index."""
    indexes = map(lambda df: df.index, data)
    next(indexes, None)
    for index in indexes:
        if not index.equals(data[0].index):
            raise ValueError("Input DataFrames must have same index.")


def _common_keys(dicts):
    """Return the intersection of the set of keys for each dictionary
    in `dicts`"""
    def _keys(x):
        return set(x.keys())
    if isinstance(dicts, tuple):
        return set.intersection(*map(_keys, dicts))
    return _keys(dicts)


def _tuple_from_dfs(dfs, name):
    """Extract a column from each DataFrame in `dfs` if `dfs` is a tuple.

    Returns a tuple of Series if `dfs` is a tuple or a Series if `dfs` is
    a DataFrame.
    """
    if isinstance(dfs, tuple):
        return tuple(df[name] for df in dfs)
    else:
        return dfs[name]


def _to_tuple(x):
    if not isinstance(x, (tuple, list)):
        return x
    return tuple(x)
