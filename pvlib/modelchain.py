"""
The ``modelchain`` module contains functions and classes that combine
many of the PV power modeling steps. These tools make it easy to
get started with pvlib and demonstrate standard ways to use the
library. With great power comes great responsibility: users should take
the time to read the source code for the module.
"""

from functools import partial
import warnings
import pandas as pd

from pvlib import (atmosphere, clearsky, inverter, pvsystem, solarposition,
                   temperature, tools)
from pvlib.tracking import SingleAxisTracker
import pvlib.irradiance  # avoid name conflict with full import
from pvlib.pvsystem import _DC_MODEL_PARAMS
from pvlib._deprecation import pvlibDeprecationWarning
from pvlib.tools import _build_kwargs


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


class ModelChain(object):
    """
    The ModelChain class to provides a standardized, high-level
    interface for all of the modeling steps necessary for calculating PV
    power from a time series of weather inputs.

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
        Valid strings are 'sapm', 'pvsyst', and 'faiman'. The ModelChain
        instance will be passed as the first argument to a user-defined
        function.

    losses_model: str or function, default 'no_loss'
        Valid strings are 'pvwatts', 'no_loss'. The ModelChain instance
        will be passed as the first argument to a user-defined function.

    name: None or str, default None
        Name of ModelChain instance.

    **kwargs
        Arbitrary keyword arguments. Included for compatibility, but not
        used.
    """

    def __init__(self, system, location,
                 orientation_strategy=None,
                 clearsky_model='ineichen',
                 transposition_model='haydavies',
                 solar_position_method='nrel_numpy',
                 airmass_model='kastenyoung1989',
                 dc_model=None, ac_model=None, aoi_model=None,
                 spectral_model=None, temperature_model=None,
                 losses_model='no_loss', name=None, **kwargs):

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
        self.solar_position = None

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
            ('{}: {}'.format(attr, getmcattr(self, attr)) for attr in attrs)))

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
                missing_params = (_DC_MODEL_PARAMS[model]
                                  - set(self.system.module_parameters.keys()))
                if missing_params:  # some parameters are not in module.keys()
                    raise ValueError(model + ' selected for the DC model but '
                                     'one or more required parameters are '
                                     'missing : ' + str(missing_params))
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
        params = set(self.system.module_parameters.keys())
        if set(['A0', 'A1', 'C7']) <= params:
            return self.sapm, 'sapm'
        elif set(['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref',
                  'R_s', 'Adjust']) <= params:
            return self.cec, 'cec'
        elif set(['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref',
                  'R_s']) <= params:
            return self.desoto, 'desoto'
        elif set(['gamma_ref', 'mu_gamma', 'I_L_ref', 'I_o_ref',
                  'R_sh_ref', 'R_sh_0', 'R_sh_exp', 'R_s']) <= params:
            return self.pvsyst, 'pvsyst'
        elif set(['pdc0', 'gamma_pdc']) <= params:
            return self.pvwatts_dc, 'pvwatts'
        else:
            raise ValueError('could not infer DC model from '
                             'system.module_parameters. Check '
                             'system.module_parameters or explicitly '
                             'set the model with the dc_model kwarg.')

    def sapm(self):
        self.dc = self.system.sapm(self.effective_irradiance,
                                   self.cell_temperature)

        self.dc = self.system.scale_voltage_current_power(self.dc)

        return self

    def _singlediode(self, calcparams_model_function):
        (photocurrent, saturation_current, resistance_series,
         resistance_shunt, nNsVth) = (
            calcparams_model_function(self.effective_irradiance,
                                      self.cell_temperature))

        self.diode_params = pd.DataFrame({'I_L': photocurrent,
                                          'I_o': saturation_current,
                                          'R_s': resistance_series,
                                          'R_sh': resistance_shunt,
                                          'nNsVth': nNsVth})

        self.dc = self.system.singlediode(
            photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth)

        self.dc = self.system.scale_voltage_current_power(self.dc).fillna(0)

        return self

    def desoto(self):
        return self._singlediode(self.system.calcparams_desoto)

    def cec(self):
        return self._singlediode(self.system.calcparams_cec)

    def pvsyst(self):
        return self._singlediode(self.system.calcparams_pvsyst)

    def pvwatts_dc(self):
        self.dc = self.system.pvwatts_dc(self.effective_irradiance,
                                         self.cell_temperature)
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
            # TODO in v0.9: remove 'snlinverter', 'adrinverter'
            if model in ['sandia', 'snlinverter']:
                if model == 'snlinverter':
                    warnings.warn("ac_model = 'snlinverter' is deprecated and"
                                  " will be removed in v0.9; use"
                                  " ac_model = 'sandia' instead.",
                                  pvlibDeprecationWarning)
                self._ac_model = self.snlinverter
            elif model in ['adr', 'adrinverter']:
                if model == 'adrinverter':
                    warnings.warn("ac_model = 'adrinverter' is deprecated and"
                                  " will be removed in v0.9; use"
                                  " ac_model = 'adr' instead.",
                                  pvlibDeprecationWarning)
                self._ac_model = self.adrinverter
            elif model == 'pvwatts':
                self._ac_model = self.pvwatts_inverter
            else:
                raise ValueError(model + ' is not a valid AC power model')
        else:
            self._ac_model = partial(model, self)

    def infer_ac_model(self):
        inverter_params = set(self.system.inverter_parameters.keys())
        if set(['C0', 'C1', 'C2']) <= inverter_params:
            return self.snlinverter
        elif set(['ADRCoefficients']) <= inverter_params:
            return self.adrinverter
        elif set(['pdc0']) <= inverter_params:
            return self.pvwatts_inverter
        else:
            raise ValueError('could not infer AC model from '
                             'system.inverter_parameters. Check '
                             'system.inverter_parameters or explicitly '
                             'set the model with the ac_model kwarg.')

    def snlinverter(self):
        self.ac = self.system.snlinverter(self.dc['v_mp'], self.dc['p_mp'])
        return self

    def adrinverter(self):
        self.ac = self.system.adrinverter(self.dc['v_mp'], self.dc['p_mp'])
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
            elif model == 'martin_ruiz':
                self._aoi_model = self.martin_ruiz_aoi_loss
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
        elif set(['a_r']) <= params:
            return self.martin_ruiz_aoi_loss
        else:
            raise ValueError('could not infer AOI model from '
                             'system.module_parameters. Check that the '
                             'system.module_parameters contain parameters for '
                             'the physical, aoi, ashrae or martin_ruiz model; '
                             'explicitly set the model with the aoi_model '
                             'kwarg; or set aoi_model="no_loss".')

    def ashrae_aoi_loss(self):
        self.aoi_modifier = self.system.get_iam(self.aoi, iam_model='ashrae')
        return self

    def physical_aoi_loss(self):
        self.aoi_modifier = self.system.get_iam(self.aoi, iam_model='physical')
        return self

    def sapm_aoi_loss(self):
        self.aoi_modifier = self.system.get_iam(self.aoi, iam_model='sapm')
        return self

    def martin_ruiz_aoi_loss(self):
        self.aoi_modifier = self.system.get_iam(self.aoi,
                                                iam_model='martin_ruiz')
        return self

    def no_aoi_loss(self):
        self.aoi_modifier = 1.0
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
        params = set(self.system.module_parameters.keys())
        if set(['A4', 'A3', 'A2', 'A1', 'A0']) <= params:
            return self.sapm_spectral_loss
        elif ((('Technology' in params or
                'Material' in params) and
               (self.system._infer_cell_type() is not None)) or
              'first_solar_spectral_coefficients' in params):
            return self.first_solar_spectral_loss
        else:
            raise ValueError('could not infer spectral model from '
                             'system.module_parameters. Check that the '
                             'system.module_parameters contain valid '
                             'first_solar_spectral_coefficients, a valid '
                             'Material or Technology value, or set '
                             'spectral_model="no_loss".')

    def first_solar_spectral_loss(self):
        self.spectral_modifier = self.system.first_solar_spectral_loss(
            self.weather['precipitable_water'],
            self.airmass['airmass_absolute'])
        return self

    def sapm_spectral_loss(self):
        self.spectral_modifier = self.system.sapm_spectral_loss(
            self.airmass['airmass_absolute'])
        return self

    def no_spectral_loss(self):
        self.spectral_modifier = 1
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
            else:
                raise ValueError(model + ' is not a valid temperature model')
            # check system.temperature_model_parameters for consistency
            name_from_params = self.infer_temperature_model().__name__
            if self._temperature_model.__name__ != name_from_params:
                raise ValueError(
                    'Temperature model {} is inconsistent with '
                    'PVsystem.temperature_model_parameters {}'.format(
                        self._temperature_model.__name__,
                        self.system.temperature_model_parameters))
        else:
            self._temperature_model = partial(model, self)

    def infer_temperature_model(self):
        params = set(self.system.temperature_model_parameters.keys())
        # remove or statement in v0.9
        if set(['a', 'b', 'deltaT']) <= params or (
                not params and self.system.racking_model is None
                and self.system.module_type is None):
            return self.sapm_temp
        elif set(['u_c', 'u_v']) <= params:
            return self.pvsyst_temp
        elif set(['u0', 'u1']) <= params:
            return self.faiman_temp
        else:
            raise ValueError('could not infer temperature model from '
                             'system.temperature_module_parameters {}.'
                             .format(self.system.temperature_model_parameters))

    def sapm_temp(self):
        self.cell_temperature = self.system.sapm_celltemp(
            self.total_irrad['poa_global'], self.weather['temp_air'],
            self.weather['wind_speed'])
        return self

    def pvsyst_temp(self):
        self.cell_temperature = self.system.pvsyst_celltemp(
            self.total_irrad['poa_global'], self.weather['temp_air'],
            self.weather['wind_speed'])
        return self

    def faiman_temp(self):
        self.cell_temperature = self.system.faiman_celltemp(
            self.total_irrad['poa_global'], self.weather['temp_air'],
            self.weather['wind_speed'])
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
        self.dc *= self.losses
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
        weather : DataFrame
            Column names must be ``'dni'``, ``'ghi'``, ``'dhi'``,
            ``'wind_speed'``, ``'temp_air'``. All irradiance components
            are required. Air temperature of 20 C and wind speed
            of 0 m/s will be added to the DataFrame if not provided.

        Returns
        -------
        self

        Notes
        -----
        Assigns attributes: ``weather``

        Examples
        --------
        This example does not work until the parameters `my_system`,
        `my_location`, `my_datetime` and `my_weather` are not defined
        properly but shows the basic idea how this method can be used.

        >>> from pvlib.modelchain import ModelChain

        >>> # my_weather containing 'dhi' and 'ghi'.
        >>> mc = ModelChain(my_system, my_location)  # doctest: +SKIP
        >>> mc.complete_irradiance(my_weather)  # doctest: +SKIP
        >>> mc.run_model(mc.weather)  # doctest: +SKIP

        >>> # my_weather containing 'dhi', 'ghi' and 'dni'.
        >>> mc = ModelChain(my_system, my_location)  # doctest: +SKIP
        >>> mc.run_model(my_weather)  # doctest: +SKIP
        """
        self.weather = weather

        self.solar_position = self.location.get_solarposition(
            self.weather.index, method=self.solar_position_method)

        icolumns = set(self.weather.columns)
        wrn_txt = ("This function is not safe at the moment.\n" +
                   "Results can be too high or negative.\n" +
                   "Help to improve this function on github:\n" +
                   "https://github.com/pvlib/pvlib-python \n")

        if {'ghi', 'dhi'} <= icolumns and 'dni' not in icolumns:
            clearsky = self.location.get_clearsky(
                self.weather.index, solar_position=self.solar_position)
            self.weather.loc[:, 'dni'] = pvlib.irradiance.dni(
                self.weather.loc[:, 'ghi'], self.weather.loc[:, 'dhi'],
                self.solar_position.zenith,
                clearsky_dni=clearsky['dni'],
                clearsky_tolerance=1.1)
        elif {'dni', 'dhi'} <= icolumns and 'ghi' not in icolumns:
            warnings.warn(wrn_txt, UserWarning)
            self.weather.loc[:, 'ghi'] = (
                self.weather.dni * tools.cosd(self.solar_position.zenith) +
                self.weather.dhi)
        elif {'dni', 'ghi'} <= icolumns and 'dhi' not in icolumns:
            warnings.warn(wrn_txt, UserWarning)
            self.weather.loc[:, 'dhi'] = (
                self.weather.ghi - self.weather.dni *
                tools.cosd(self.solar_position.zenith))

        return self

    def prepare_inputs(self, weather):
        """
        Prepare the solar position, irradiance, and weather inputs to
        the model.

        Parameters
        ----------
        weather : DataFrame
            Column names must be ``'dni'``, ``'ghi'``, ``'dhi'``,
            ``'wind_speed'``, ``'temp_air'``. All irradiance components
            are required. Air temperature of 20 C and wind speed
            of 0 m/s will be added to the DataFrame if not provided.

        Notes
        -----
        Assigns attributes: ``solar_position``, ``airmass``,
        ``total_irrad``, ``aoi``

        See also
        --------
        ModelChain.complete_irradiance
        """

        if not {'ghi', 'dni', 'dhi'} <= set(weather.columns):
            raise ValueError(
                "Uncompleted irradiance data set. Please check your input "
                "data.\nData set needs to have 'dni', 'dhi' and 'ghi'.\n"
                "Detected data: {0}".format(list(weather.columns)))

        self.weather = weather

        self.times = self.weather.index
        try:
            kwargs = _build_kwargs(['pressure', 'temp_air'], weather)
            kwargs['temperature'] = kwargs.pop('temp_air')
        except KeyError:
            pass

        self.solar_position = self.location.get_solarposition(
            self.weather.index, method=self.solar_position_method,
            **kwargs)

        self.airmass = self.location.get_airmass(
            solar_position=self.solar_position, model=self.airmass_model)

        # PVSystem.get_irradiance and SingleAxisTracker.get_irradiance
        # and PVSystem.get_aoi and SingleAxisTracker.get_aoi
        # have different method signatures. Use partial to handle
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
            self.aoi = self.tracking['aoi']
            get_irradiance = partial(
                self.system.get_irradiance,
                self.tracking['surface_tilt'],
                self.tracking['surface_azimuth'],
                self.solar_position['apparent_zenith'],
                self.solar_position['azimuth'])
        else:
            self.aoi = self.system.get_aoi(
                self.solar_position['apparent_zenith'],
                self.solar_position['azimuth'])
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

    def run_model(self, weather):
        """
        Run the model.

        Parameters
        ----------
        weather : DataFrame
            Column names must be ``'dni'``, ``'ghi'``, ``'dhi'``,
            ``'wind_speed'``, ``'temp_air'``. All irradiance components
            are required. Air temperature of 20 C and wind speed
            of 0 m/s will be added to the DataFrame if not provided.

        Returns
        -------
        self

        Assigns attributes: ``solar_position``, ``airmass``, ``irradiance``,
        ``total_irrad``, ``effective_irradiance``, ``weather``,
        ``cell_temperature``, ``aoi``, ``aoi_modifier``, ``spectral_modifier``,
        ``dc``, ``ac``, ``losses``,
        ``diode_params`` (if dc_model is a single diode model)
        """
        self.prepare_inputs(weather)
        self.aoi_model()
        self.spectral_model()
        self.effective_irradiance_model()
        self.temperature_model()
        self.dc_model()
        self.losses_model()
        self.ac_model()

        return self
