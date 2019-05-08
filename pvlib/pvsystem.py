"""
The ``pvsystem`` module contains functions for modeling the output and
performance of PV modules and inverters.
"""

from __future__ import division

from collections import OrderedDict
import io
import os
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

import numpy as np
import pandas as pd

from pvlib import atmosphere, irradiance, tools, singlediode as _singlediode
from pvlib.tools import _build_kwargs
from pvlib.location import Location


# a dict of required parameter names for each DC power model
DC_MODEL_PARAMS = {
    'sapm': set([
        'A0', 'A1', 'A2', 'A3', 'A4', 'B0', 'B1', 'B2', 'B3',
        'B4', 'B5', 'C0', 'C1', 'C2', 'C3', 'C4', 'C5', 'C6',
        'C7', 'Isco', 'Impo', 'Aisc', 'Aimp', 'Bvoco',
        'Mbvoc', 'Bvmpo', 'Mbvmp', 'N', 'Cells_in_Series',
        'IXO', 'IXXO', 'FD']),
    'desoto': set([
        'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_s']),
    'cec': set([
        'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_s', 'Adjust']),
    'pvsyst': set([
        'gamma_ref', 'mu_gamma', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_sh_0', 'R_s', 'alpha_sc', 'EgRef',
        'cells_in_series']),
    'singlediode': set([
        'alpha_sc', 'a_ref', 'I_L_ref', 'I_o_ref',
        'R_sh_ref', 'R_s']),
    'pvwatts': set(['pdc0', 'gamma_pdc'])
}


TEMP_MODEL_PARAMS = {
    'sapm': {'open_rack_cell_glassback': (-3.47, -.0594, 3),
             'roof_mount_cell_glassback': (-2.98, -.0471, 1),
             'open_rack_cell_polymerback': (-3.56, -.0750, 3),
             'insulated_back_polymerback': (-2.81, -.0455, 0),
             'open_rack_polymer_thinfilm_steel': (-3.58, -.113, 3),
             '22x_concentrator_tracker': (-3.23, -.130, 13)},
    'pvsyst': {'freestanding': (29.0, 0), 'insulated': (15.0, 0)}
}

# not sure if this belongs in the pvsystem module.
# maybe something more like core.py? It may eventually grow to
# import a lot more functionality from other modules.
class PVSystem(object):
    """
    The PVSystem class defines a standard set of PV system attributes
    and modeling functions. This class describes the collection and
    interactions of PV system components rather than an installed system
    on the ground. It is typically used in combination with
    :py:class:`~pvlib.location.Location` and
    :py:class:`~pvlib.modelchain.ModelChain`
    objects.

    See the :py:class:`LocalizedPVSystem` class for an object model that
    describes an installed PV system.

    The class supports basic system topologies consisting of:

        * `N` total modules arranged in series
          (`modules_per_string=N`, `strings_per_inverter=1`).
        * `M` total modules arranged in parallel
          (`modules_per_string=1`, `strings_per_inverter=M`).
        * `NxM` total modules arranged in `M` strings of `N` modules each
          (`modules_per_string=N`, `strings_per_inverter=M`).

    The class is complementary to the module-level functions.

    The attributes should generally be things that don't change about
    the system, such the type of module and the inverter. The instance
    methods accept arguments for things that do change, such as
    irradiance and temperature.

    Parameters
    ----------
    surface_tilt: float or array-like, default 0
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth: float or array-like, default 180
        Azimuth angle of the module surface.
        North=0, East=90, South=180, West=270.

    albedo : None or float, default None
        The ground albedo. If ``None``, will attempt to use
        ``surface_type`` and ``irradiance.SURFACE_ALBEDOS``
        to lookup albedo.

    surface_type : None or string, default None
        The ground surface type. See ``irradiance.SURFACE_ALBEDOS``
        for valid values.

    module : None or string, default None
        The model name of the modules.
        May be used to look up the module_parameters dictionary
        via some other method.

    module_parameters : None, dict or Series, default None
        Module parameters as defined by the SAPM, CEC, or other.

    modules_per_string: int or float, default 1
        See system topology discussion above.

    strings_per_inverter: int or float, default 1
        See system topology discussion above.

    inverter : None or string, default None
        The model name of the inverters.
        May be used to look up the inverter_parameters dictionary
        via some other method.

    inverter_parameters : None, dict or Series, default None
        Inverter parameters as defined by the SAPM, CEC, or other.

    racking_model : None or string, default 'open_rack_cell_glassback'
        Used for cell and module temperature calculations.

    losses_parameters : None, dict or Series, default None
        Losses parameters as defined by PVWatts or other.

    name : None or string, default None

    **kwargs
        Arbitrary keyword arguments.
        Included for compatibility, but not used.

    See also
    --------
    pvlib.location.Location
    pvlib.tracking.SingleAxisTracker
    pvlib.pvsystem.LocalizedPVSystem
    """

    def __init__(self,
                 surface_tilt=0, surface_azimuth=180,
                 albedo=None, surface_type=None,
                 module=None, module_parameters=None,
                 modules_per_string=1, strings_per_inverter=1,
                 inverter=None, inverter_parameters=None,
                 racking_model='open_rack_cell_glassback',
                 losses_parameters=None, name=None, **kwargs):

        self.name = name

        self.surface_tilt = surface_tilt
        self.surface_azimuth = surface_azimuth

        # could tie these together with @property
        self.surface_type = surface_type
        if albedo is None:
            self.albedo = irradiance.SURFACE_ALBEDOS.get(surface_type, 0.25)
        else:
            self.albedo = albedo

        # could tie these together with @property
        self.module = module
        if module_parameters is None:
            self.module_parameters = {}
        else:
            self.module_parameters = module_parameters

        self.modules_per_string = modules_per_string
        self.strings_per_inverter = strings_per_inverter

        self.inverter = inverter
        if inverter_parameters is None:
            self.inverter_parameters = {}
        else:
            self.inverter_parameters = inverter_parameters

        if losses_parameters is None:
            self.losses_parameters = {}
        else:
            self.losses_parameters = losses_parameters

        self.racking_model = racking_model

    def __repr__(self):
        attrs = ['name', 'surface_tilt', 'surface_azimuth', 'module',
                 'inverter', 'albedo', 'racking_model']
        return ('PVSystem: \n  ' + '\n  '.join(
            ('{}: {}'.format(attr, getattr(self, attr)) for attr in attrs)))

    def get_aoi(self, solar_zenith, solar_azimuth):
        """Get the angle of incidence on the system.

        Parameters
        ----------
        solar_zenith : float or Series.
            Solar zenith angle.
        solar_azimuth : float or Series.
            Solar azimuth angle.

        Returns
        -------
        aoi : Series
            The angle of incidence
        """

        aoi = irradiance.aoi(self.surface_tilt, self.surface_azimuth,
                             solar_zenith, solar_azimuth)
        return aoi

    def get_irradiance(self, solar_zenith, solar_azimuth, dni, ghi, dhi,
                       dni_extra=None, airmass=None, model='haydavies',
                       **kwargs):
        """
        Uses the :py:func:`irradiance.get_total_irradiance` function to
        calculate the plane of array irradiance components on a tilted
        surface defined by ``self.surface_tilt``,
        ``self.surface_azimuth``, and ``self.albedo``.

        Parameters
        ----------
        solar_zenith : float or Series.
            Solar zenith angle.
        solar_azimuth : float or Series.
            Solar azimuth angle.
        dni : float or Series
            Direct Normal Irradiance
        ghi : float or Series
            Global horizontal irradiance
        dhi : float or Series
            Diffuse horizontal irradiance
        dni_extra : None, float or Series, default None
            Extraterrestrial direct normal irradiance
        airmass : None, float or Series, default None
            Airmass
        model : String, default 'haydavies'
            Irradiance model.

        kwargs
            Extra parameters passed to :func:`irradiance.total_irrad`.

        Returns
        -------
        poa_irradiance : DataFrame
            Column names are: ``total, beam, sky, ground``.
        """

        # not needed for all models, but this is easier
        if dni_extra is None:
            dni_extra = irradiance.get_extra_radiation(solar_zenith.index)

        if airmass is None:
            airmass = atmosphere.get_relative_airmass(solar_zenith)

        return irradiance.get_total_irradiance(self.surface_tilt,
                                               self.surface_azimuth,
                                               solar_zenith, solar_azimuth,
                                               dni, ghi, dhi,
                                               dni_extra=dni_extra,
                                               airmass=airmass,
                                               model=model,
                                               albedo=self.albedo,
                                               **kwargs)

    def ashraeiam(self, aoi):
        """
        Determine the incidence angle modifier using
        ``self.module_parameters['b']``, ``aoi``,
        and the :py:func:`ashraeiam` function.

        Uses default arguments if keys not in module_parameters.

        Parameters
        ----------
        aoi : numeric
            The angle of incidence in degrees.

        Returns
        -------
        modifier : numeric
            The AOI modifier.
        """
        kwargs = _build_kwargs(['b'], self.module_parameters)

        return ashraeiam(aoi, **kwargs)

    def physicaliam(self, aoi):
        """
        Determine the incidence angle modifier using ``aoi``,
        ``self.module_parameters['K']``,
        ``self.module_parameters['L']``,
        ``self.module_parameters['n']``,
        and the
        :py:func:`physicaliam` function.

        Uses default arguments if keys not in module_parameters.

        Parameters
        ----------
        aoi : numeric
            The angle of incidence in degrees.

        Returns
        -------
        modifier : numeric
            The AOI modifier.
        """
        kwargs = _build_kwargs(['K', 'L', 'n'], self.module_parameters)

        return physicaliam(aoi, **kwargs)

    def calcparams_desoto(self, effective_irradiance, temp_cell, **kwargs):
        """
        Use the :py:func:`calcparams_desoto` function, the input
        parameters and ``self.module_parameters`` to calculate the
        module currents and resistances.

        Parameters
        ----------
        effective_irradiance : numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series
            The average cell temperature of cells within a module in C.

        **kwargs
            See pvsystem.calcparams_desoto for details

        Returns
        -------
        See pvsystem.calcparams_desoto for details
        """

        kwargs = _build_kwargs(['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref',
                                'R_s', 'alpha_sc', 'EgRef', 'dEgdT',
                                'irrad_ref', 'temp_ref'],
                               self.module_parameters)

        return calcparams_desoto(effective_irradiance, temp_cell, **kwargs)

    def calcparams_cec(self, effective_irradiance, temp_cell, **kwargs):
        """
        Use the :py:func:`calcparams_cec` function, the input
        parameters and ``self.module_parameters`` to calculate the
        module currents and resistances.

        Parameters
        ----------
        effective_irradiance : numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series
            The average cell temperature of cells within a module in C.

        **kwargs
            See pvsystem.calcparams_cec for details

        Returns
        -------
        See pvsystem.calcparams_cec for details
        """

        kwargs = _build_kwargs(['a_ref', 'I_L_ref', 'I_o_ref', 'R_sh_ref',
                                'R_s', 'alpha_sc', 'Adjust', 'EgRef', 'dEgdT',
                                'irrad_ref', 'temp_ref'],
                               self.module_parameters)

        return calcparams_cec(effective_irradiance, temp_cell, **kwargs)

    def calcparams_pvsyst(self, effective_irradiance, temp_cell):
        """
        Use the :py:func:`calcparams_pvsyst` function, the input
        parameters and ``self.module_parameters`` to calculate the
        module currents and resistances.

        Parameters
        ----------
        effective_irradiance : numeric
            The irradiance (W/m2) that is converted to photocurrent.

        temp_cell : float or Series
            The average cell temperature of cells within a module in C.

        Returns
        -------
        See pvsystem.calcparams_pvsyst for details
        """

        kwargs = _build_kwargs(['gamma_ref', 'mu_gamma', 'I_L_ref', 'I_o_ref',
                                'R_sh_ref', 'R_sh_0', 'R_sh_exp',
                                'R_s', 'alpha_sc', 'EgRef',
                                'irrad_ref', 'temp_ref',
                                'cells_in_series'],
                               self.module_parameters)

        return calcparams_pvsyst(effective_irradiance, temp_cell, **kwargs)

    def sapm(self, effective_irradiance, temp_cell, **kwargs):
        """
        Use the :py:func:`sapm` function, the input parameters,
        and ``self.module_parameters`` to calculate
        Voc, Isc, Ix, Ixx, Vmp/Imp.

        Parameters
        ----------
        poa_direct : Series
            The direct irradiance incident upon the module (W/m^2).

        poa_diffuse : Series
            The diffuse irradiance incident on module.

        temp_cell : Series
            The cell temperature (degrees C).

        airmass_absolute : Series
            Absolute airmass.

        aoi : Series
            Angle of incidence (degrees).

        kwargs
            See pvsystem.sapm for details

        Returns
        -------
        See pvsystem.sapm for details
        """
        return sapm(effective_irradiance, temp_cell, self.module_parameters)

    def sapm_celltemp(self, irrad, wind, temp):
        """Uses :py:func:`sapm_celltemp` to calculate module and cell
        temperatures based on ``self.racking_model`` and
        the input parameters.

        Parameters
        ----------
        See pvsystem.sapm_celltemp for details

        Returns
        -------
        See pvsystem.sapm_celltemp for details
        """
        return sapm_celltemp(irrad, wind, temp, self.racking_model)

    def sapm_spectral_loss(self, airmass_absolute):
        """
        Use the :py:func:`sapm_spectral_loss` function, the input
        parameters, and ``self.module_parameters`` to calculate F1.

        Parameters
        ----------
        airmass_absolute : numeric
            Absolute airmass.

        Returns
        -------
        F1 : numeric
            The SAPM spectral loss coefficient.
        """
        return sapm_spectral_loss(airmass_absolute, self.module_parameters)

    def sapm_aoi_loss(self, aoi):
        """
        Use the :py:func:`sapm_aoi_loss` function, the input parameters,
        and ``self.module_parameters`` to calculate F2.

        Parameters
        ----------
        aoi : numeric
            Angle of incidence in degrees.

        Returns
        -------
        F2 : numeric
            The SAPM angle of incidence loss coefficient.
        """
        return sapm_aoi_loss(aoi, self.module_parameters)

    def sapm_effective_irradiance(self, poa_direct, poa_diffuse,
                                  airmass_absolute, aoi,
                                  reference_irradiance=1000):
        """
        Use the :py:func:`sapm_effective_irradiance` function, the input
        parameters, and ``self.module_parameters`` to calculate
        effective irradiance.

        Parameters
        ----------
        poa_direct : numeric
            The direct irradiance incident upon the module.

        poa_diffuse : numeric
            The diffuse irradiance incident on module.

        airmass_absolute : numeric
            Absolute airmass.

        aoi : numeric
            Angle of incidence in degrees.

        reference_irradiance : numeric, default 1000
            Reference irradiance by which to divide the input irradiance.

        Returns
        -------
        effective_irradiance : numeric
            The SAPM effective irradiance.
        """
        return sapm_effective_irradiance(
            poa_direct, poa_diffuse, airmass_absolute, aoi,
            self.module_parameters, reference_irradiance=reference_irradiance)

    def pvsyst_celltemp(self, poa_global, temp_air, wind_speed=1.0):
        """Uses :py:func:`pvsyst_celltemp` to calculate module temperatures
        based on ``self.racking_model`` and the input parameters.

        Parameters
        ----------
        See pvsystem.pvsyst_celltemp for details

        Returns
        -------
        See pvsystem.pvsyst_celltemp for details
        """
        kwargs = _build_kwargs(['eta_m', 'alpha_absorption'],
                               self.module_parameters)
        return pvsyst_celltemp(poa_global, temp_air, wind_speed,
                               model_params=self.racking_model, **kwargs)

    def first_solar_spectral_loss(self, pw, airmass_absolute):

        """
        Use the :py:func:`first_solar_spectral_correction` function to
        calculate the spectral loss modifier. The model coefficients are
        specific to the module's cell type, and are determined by searching
        for one of the following keys in self.module_parameters (in order):
            'first_solar_spectral_coefficients' (user-supplied coefficients)
            'Technology' - a string describing the cell type, can be read from
            the CEC module parameter database
            'Material' - a string describing the cell type, can be read from
            the Sandia module database.

        Parameters
        ----------
        pw : array-like
            atmospheric precipitable water (cm).

        airmass_absolute : array-like
            absolute (pressure corrected) airmass.

        Returns
        -------
        modifier: array-like
            spectral mismatch factor (unitless) which can be multiplied
            with broadband irradiance reaching a module's cells to estimate
            effective irradiance, i.e., the irradiance that is converted to
            electrical current.
        """

        if 'first_solar_spectral_coefficients' in \
                self.module_parameters.keys():
            coefficients = \
                   self.module_parameters['first_solar_spectral_coefficients']
            module_type = None
        else:
            module_type = self._infer_cell_type()
            coefficients = None

        return atmosphere.first_solar_spectral_correction(pw,
                                                          airmass_absolute,
                                                          module_type,
                                                          coefficients)

    def _infer_cell_type(self):

        """
        Examines module_parameters and maps the Technology key for the CEC
        database and the Material key for the Sandia database to a common
        list of strings for cell type.

        Returns
        -------
        cell_type: str

        """

        _cell_type_dict = {'Multi-c-Si': 'multisi',
                           'Mono-c-Si': 'monosi',
                           'Thin Film': 'cigs',
                           'a-Si/nc': 'asi',
                           'CIS': 'cigs',
                           'CIGS': 'cigs',
                           '1-a-Si': 'asi',
                           'CdTe': 'cdte',
                           'a-Si': 'asi',
                           '2-a-Si': None,
                           '3-a-Si': None,
                           'HIT-Si': 'monosi',
                           'mc-Si': 'multisi',
                           'c-Si': 'multisi',
                           'Si-Film': 'asi',
                           'EFG mc-Si': 'multisi',
                           'GaAs': None,
                           'a-Si / mono-Si': 'monosi'}

        if 'Technology' in self.module_parameters.keys():
            # CEC module parameter set
            cell_type = _cell_type_dict[self.module_parameters['Technology']]
        elif 'Material' in self.module_parameters.keys():
            # Sandia module parameter set
            cell_type = _cell_type_dict[self.module_parameters['Material']]
        else:
            cell_type = None

        return cell_type

    def singlediode(self, photocurrent, saturation_current,
                    resistance_series, resistance_shunt, nNsVth,
                    ivcurve_pnts=None):
        """Wrapper around the :py:func:`singlediode` function.

        Parameters
        ----------
        See pvsystem.singlediode for details

        Returns
        -------
        See pvsystem.singlediode for details
        """
        return singlediode(photocurrent, saturation_current,
                           resistance_series, resistance_shunt, nNsVth,
                           ivcurve_pnts=ivcurve_pnts)

    def i_from_v(self, resistance_shunt, resistance_series, nNsVth, voltage,
                 saturation_current, photocurrent):
        """Wrapper around the :py:func:`i_from_v` function.

        Parameters
        ----------
        See pvsystem.i_from_v for details

        Returns
        -------
        See pvsystem.i_from_v for details
        """
        return i_from_v(resistance_shunt, resistance_series, nNsVth, voltage,
                        saturation_current, photocurrent)

    # inverter now specified by self.inverter_parameters
    def snlinverter(self, v_dc, p_dc):
        """Uses :func:`snlinverter` to calculate AC power based on
        ``self.inverter_parameters`` and the input parameters.

        Parameters
        ----------
        See pvsystem.snlinverter for details

        Returns
        -------
        See pvsystem.snlinverter for details
        """
        return snlinverter(v_dc, p_dc, self.inverter_parameters)

    def adrinverter(self, v_dc, p_dc):
        return adrinverter(v_dc, p_dc, self.inverter_parameters)

    def scale_voltage_current_power(self, data):
        """
        Scales the voltage, current, and power of the DataFrames
        returned by :py:func:`singlediode` and :py:func:`sapm`
        by `self.modules_per_string` and `self.strings_per_inverter`.

        Parameters
        ----------
        data: DataFrame
            Must contain columns `'v_mp', 'v_oc', 'i_mp' ,'i_x', 'i_xx',
            'i_sc', 'p_mp'`.

        Returns
        -------
        scaled_data: DataFrame
            A scaled copy of the input data.
        """

        return scale_voltage_current_power(data,
                                           voltage=self.modules_per_string,
                                           current=self.strings_per_inverter)

    def pvwatts_dc(self, g_poa_effective, temp_cell):
        """
        Calcuates DC power according to the PVWatts model using
        :py:func:`pvwatts_dc`, `self.module_parameters['pdc0']`, and
        `self.module_parameters['gamma_pdc']`.

        See :py:func:`pvwatts_dc` for details.
        """
        kwargs = _build_kwargs(['temp_ref'], self.module_parameters)

        return pvwatts_dc(g_poa_effective, temp_cell,
                          self.module_parameters['pdc0'],
                          self.module_parameters['gamma_pdc'],
                          **kwargs)

    def pvwatts_losses(self):
        """
        Calculates DC power losses according the PVwatts model using
        :py:func:`pvwatts_losses` and ``self.losses_parameters``.`

        See :py:func:`pvwatts_losses` for details.
        """
        kwargs = _build_kwargs(['soiling', 'shading', 'snow', 'mismatch',
                                'wiring', 'connections', 'lid',
                                'nameplate_rating', 'age', 'availability'],
                               self.losses_parameters)
        return pvwatts_losses(**kwargs)

    def pvwatts_ac(self, pdc):
        """
        Calculates AC power according to the PVWatts model using
        :py:func:`pvwatts_ac`, `self.module_parameters['pdc0']`, and
        `eta_inv_nom=self.inverter_parameters['eta_inv_nom']`.

        See :py:func:`pvwatts_ac` for details.
        """
        kwargs = _build_kwargs(['eta_inv_nom', 'eta_inv_ref'],
                               self.inverter_parameters)

        return pvwatts_ac(pdc, self.module_parameters['pdc0'], **kwargs)

    def localize(self, location=None, latitude=None, longitude=None,
                 **kwargs):
        """Creates a LocalizedPVSystem object using this object
        and location data. Must supply either location object or
        latitude, longitude, and any location kwargs

        Parameters
        ----------
        location : None or Location, default None
        latitude : None or float, default None
        longitude : None or float, default None
        **kwargs : see Location

        Returns
        -------
        localized_system : LocalizedPVSystem
        """

        if location is None:
            location = Location(latitude, longitude, **kwargs)

        return LocalizedPVSystem(pvsystem=self, location=location)


class LocalizedPVSystem(PVSystem, Location):
    """
    The LocalizedPVSystem class defines a standard set of installed PV
    system attributes and modeling functions. This class combines the
    attributes and methods of the PVSystem and Location classes.

    The LocalizedPVSystem may have bugs due to the difficulty of
    robustly implementing multiple inheritance. See
    :py:class:`~pvlib.modelchain.ModelChain` for an alternative paradigm
    for modeling PV systems at specific locations.
    """
    def __init__(self, pvsystem=None, location=None, **kwargs):

        # get and combine attributes from the pvsystem and/or location
        # with the rest of the kwargs

        if pvsystem is not None:
            pv_dict = pvsystem.__dict__
        else:
            pv_dict = {}

        if location is not None:
            loc_dict = location.__dict__
        else:
            loc_dict = {}

        new_kwargs = dict(list(pv_dict.items()) +
                          list(loc_dict.items()) +
                          list(kwargs.items()))

        PVSystem.__init__(self, **new_kwargs)
        Location.__init__(self, **new_kwargs)

    def __repr__(self):
        attrs = ['name', 'latitude', 'longitude', 'altitude', 'tz',
                 'surface_tilt', 'surface_azimuth', 'module', 'inverter',
                 'albedo', 'racking_model']
        return ('LocalizedPVSystem: \n  ' + '\n  '.join(
            ('{}: {}'.format(attr, getattr(self, attr)) for attr in attrs)))


def systemdef(meta, surface_tilt, surface_azimuth, albedo, modules_per_string,
              strings_per_inverter):
    '''
    Generates a dict of system parameters used throughout a simulation.

    Parameters
    ----------

    meta : dict
        meta dict either generated from a TMY file using readtmy2 or
        readtmy3, or a dict containing at least the following fields:

            ===============   ======  ====================
            meta field        format  description
            ===============   ======  ====================
            meta.altitude     Float   site elevation
            meta.latitude     Float   site latitude
            meta.longitude    Float   site longitude
            meta.Name         String  site name
            meta.State        String  state
            meta.TZ           Float   timezone
            ===============   ======  ====================

    surface_tilt : float or Series
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : float or Series
        Surface azimuth angles in decimal degrees.
        The azimuth convention is defined
        as degrees east of north
        (North=0, South=180, East=90, West=270).

    albedo : float or Series
        Ground reflectance, typically 0.1-0.4 for surfaces on Earth
        (land), may increase over snow, ice, etc. May also be known as
        the reflection coefficient. Must be >=0 and <=1.

    modules_per_string : int
        Number of modules connected in series in a string.

    strings_per_inverter : int
        Number of strings connected in parallel.

    Returns
    -------
    Result : dict

        A dict with the following fields.

            * 'surface_tilt'
            * 'surface_azimuth'
            * 'albedo'
            * 'modules_per_string'
            * 'strings_per_inverter'
            * 'latitude'
            * 'longitude'
            * 'tz'
            * 'name'
            * 'altitude'

    See also
    --------
    pvlib.tmy.readtmy3
    pvlib.tmy.readtmy2
    '''

    try:
        name = meta['Name']
    except KeyError:
        name = meta['City']

    system = {'surface_tilt': surface_tilt,
              'surface_azimuth': surface_azimuth,
              'albedo': albedo,
              'modules_per_string': modules_per_string,
              'strings_per_inverter': strings_per_inverter,
              'latitude': meta['latitude'],
              'longitude': meta['longitude'],
              'tz': meta['TZ'],
              'name': name,
              'altitude': meta['altitude']}

    return system


def ashraeiam(aoi, b=0.05):
    '''
    Determine the incidence angle modifier using the ASHRAE transmission
    model.

    ashraeiam calculates the incidence angle modifier as developed in
    [1], and adopted by ASHRAE (American Society of Heating,
    Refrigeration, and Air Conditioning Engineers) [2]. The model has
    been used by model programs such as PVSyst [3].

    Note: For incident angles near 90 degrees, this model has a
    discontinuity which has been addressed in this function.

    Parameters
    ----------
    aoi : numeric
        The angle of incidence between the module normal vector and the
        sun-beam vector in degrees. Angles of nan will result in nan.

    b : float, default 0.05
        A parameter to adjust the modifier as a function of angle of
        incidence. Typical values are on the order of 0.05 [3].

    Returns
    -------
    IAM : numeric
        The incident angle modifier calculated as 1-b*(sec(aoi)-1) as
        described in [2,3].

        Returns zeros for all abs(aoi) >= 90 and for all IAM values that
        would be less than 0.

    References
    ----------
    [1] Souka A.F., Safwat H.H., "Determination of the optimum
    orientations for the double exposure flat-plate collector and its
    reflections". Solar Energy vol .10, pp 170-174. 1966.

    [2] ASHRAE standard 93-77

    [3] PVsyst Contextual Help.
    http://files.pvsyst.com/help/index.html?iam_loss.htm retrieved on
    September 10, 2012

    See Also
    --------
    irradiance.aoi
    physicaliam
    '''

    iam = 1 - b * ((1 / np.cos(np.radians(aoi)) - 1))
    aoi_gte_90 = np.full_like(aoi, False, dtype='bool')
    np.greater_equal(np.abs(aoi), 90, where=~np.isnan(aoi), out=aoi_gte_90)
    iam = np.where(aoi_gte_90, 0, iam)
    iam = np.maximum(0, iam)

    if isinstance(iam, pd.Series):
        iam = pd.Series(iam, index=aoi.index)

    return iam


def physicaliam(aoi, n=1.526, K=4., L=0.002):
    '''
    Determine the incidence angle modifier using refractive index,
    extinction coefficient, and glazing thickness.

    physicaliam calculates the incidence angle modifier as described in
    De Soto et al. "Improvement and validation of a model for
    photovoltaic array performance", section 3. The calculation is based
    on a physical model of absorbtion and transmission through a
    cover.

    Note: The authors of this function believe that eqn. 14 in [1] is
    incorrect. This function uses the following equation in its place:
    theta_r = arcsin(1/n * sin(aoi))

    Parameters
    ----------
    aoi : numeric
        The angle of incidence between the module normal vector and the
        sun-beam vector in degrees. Angles of 0 are replaced with 1e-06
        to ensure non-nan results. Angles of nan will result in nan.

    n : numeric, default 1.526
        The effective index of refraction (unitless). Reference [1]
        indicates that a value of 1.526 is acceptable for glass. n must
        be a numeric scalar or vector with all values >=0. If n is a
        vector, it must be the same size as all other input vectors.

    K : numeric, default 4.0
        The glazing extinction coefficient in units of 1/meters.
        Reference [1] indicates that a value of  4 is reasonable for
        "water white" glass. K must be a numeric scalar or vector with
        all values >=0. If K is a vector, it must be the same size as
        all other input vectors.

    L : numeric, default 0.002
        The glazing thickness in units of meters. Reference [1]
        indicates that 0.002 meters (2 mm) is reasonable for most
        glass-covered PV panels. L must be a numeric scalar or vector
        with all values >=0. If L is a vector, it must be the same size
        as all other input vectors.

    Returns
    -------
    iam : numeric
        The incident angle modifier

    References
    ----------
    [1] W. De Soto et al., "Improvement and validation of a model for
    photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
    2006.

    [2] Duffie, John A. & Beckman, William A.. (2006). Solar Engineering
    of Thermal Processes, third edition. [Books24x7 version] Available
    from http://common.books24x7.com/toc.aspx?bookid=17160.

    See Also
    --------
    getaoi
    ephemeris
    spa
    ashraeiam
    '''
    zeroang = 1e-06

    # hold a new reference to the input aoi object since we're going to
    # overwrite the aoi reference below, but we'll need it for the
    # series check at the end of the function
    aoi_input = aoi

    aoi = np.where(aoi == 0, zeroang, aoi)

    # angle of reflection
    thetar_deg = tools.asind(1.0 / n*(tools.sind(aoi)))

    # reflectance and transmittance for normal incidence light
    rho_zero = ((1-n) / (1+n)) ** 2
    tau_zero = np.exp(-K*L)

    # reflectance for parallel and perpendicular polarized light
    rho_para = (tools.tand(thetar_deg - aoi) /
                tools.tand(thetar_deg + aoi)) ** 2
    rho_perp = (tools.sind(thetar_deg - aoi) /
                tools.sind(thetar_deg + aoi)) ** 2

    # transmittance for non-normal light
    tau = np.exp(-K*L / tools.cosd(thetar_deg))

    # iam is ratio of non-normal to normal incidence transmitted light
    # after deducting the reflected portion of each
    iam = ((1 - (rho_para + rho_perp) / 2) / (1 - rho_zero) * tau / tau_zero)

    with np.errstate(invalid='ignore'):
        # angles near zero produce nan, but iam is defined as one
        small_angle = 1e-06
        iam = np.where(np.abs(aoi) < small_angle, 1.0, iam)

        # angles at 90 degrees can produce tiny negative values,
        # which should be zero. this is a result of calculation precision
        # rather than the physical model
        iam = np.where(iam < 0, 0, iam)

        # for light coming from behind the plane, none can enter the module
        iam = np.where(aoi > 90, 0, iam)

    if isinstance(aoi_input, pd.Series):
        iam = pd.Series(iam, index=aoi_input.index)

    return iam


def calcparams_desoto(effective_irradiance, temp_cell,
                      alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s,
                      EgRef=1.121, dEgdT=-0.0002677,
                      irrad_ref=1000, temp_ref=25):
    '''
    Calculates five parameter values for the single diode equation at
    effective irradiance and cell temperature using the De Soto et al.
    model described in [1]. The five values returned by calcparams_desoto
    can be used by singlediode to calculate an IV curve.

    Parameters
    ----------
    effective_irradiance : numeric
        The irradiance (W/m2) that is converted to photocurrent.

    temp_cell : numeric
        The average cell temperature of cells within a module in C.

    alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.

    a_ref : float
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at reference
        conditions, in units of V.

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    EgRef : float
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.  For parameters
        from the SAM CEC module database, EgRef=1.121 is implicit for all
        cell types in the parameter estimation algorithm used by NREL.

    dEgdT : float
        The temperature dependence of the energy bandgap at reference
        conditions in units of 1/K. May be either a scalar value
        (e.g. -0.0002677 as in [1]) or a DataFrame (this may be useful if
        dEgdT is a modeled as a function of temperature). For parameters from
        the SAM CEC module database, dEgdT=-0.0002677 is implicit for all cell
        types in the parameter estimation algorithm used by NREL.

    irrad_ref : float (optional, default=1000)
        Reference irradiance in W/m^2.

    temp_ref : float (optional, default=25)
        Reference cell temperature in C.

    Returns
    -------
    Tuple of the following results:

    photocurrent : numeric
        Light-generated current in amperes

    saturation_current : numeric
        Diode saturation curent in amperes

    resistance_series : float
        Series resistance in ohms

    resistance_shunt : numeric
        Shunt resistance in ohms

    nNsVth : numeric
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at
        specified effective irradiance and cell temperature.

    References
    ----------
    [1] W. De Soto et al., "Improvement and validation of a model for
    photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
    2006.

    [2] System Advisor Model web page. https://sam.nrel.gov.

    [3] A. Dobos, "An Improved Coefficient Calculator for the California
    Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
    Solar Energy Engineering, vol 134, 2012.

    [4] O. Madelung, "Semiconductors: Data Handbook, 3rd ed." ISBN
    3-540-40488-0

    See Also
    --------
    singlediode
    retrieve_sam

    Notes
    -----
    If the reference parameters in the ModuleParameters struct are read
    from a database or library of parameters (e.g. System Advisor
    Model), it is important to use the same EgRef and dEgdT values that
    were used to generate the reference parameters, regardless of the
    actual bandgap characteristics of the semiconductor. For example, in
    the case of the System Advisor Model library, created as described
    in [3], EgRef and dEgdT for all modules were 1.121 and -0.0002677,
    respectively.

    This table of reference bandgap energies (EgRef), bandgap energy
    temperature dependence (dEgdT), and "typical" airmass response (M)
    is provided purely as reference to those who may generate their own
    reference module parameters (a_ref, IL_ref, I0_ref, etc.) based upon
    the various PV semiconductors. Again, we stress the importance of
    using identical EgRef and dEgdT when generation reference parameters
    and modifying the reference parameters (for irradiance, temperature,
    and airmass) per DeSoto's equations.

     Crystalline Silicon (Si):
         * EgRef = 1.121
         * dEgdT = -0.0002677

         >>> M = np.polyval([-1.26E-4, 2.816E-3, -0.024459, 0.086257, 0.9181],
         ...                AMa) # doctest: +SKIP

         Source: [1]

     Cadmium Telluride (CdTe):
         * EgRef = 1.475
         * dEgdT = -0.0003

         >>> M = np.polyval([-2.46E-5, 9.607E-4, -0.0134, 0.0716, 0.9196],
         ...                AMa) # doctest: +SKIP

         Source: [4]

     Copper Indium diSelenide (CIS):
         * EgRef = 1.010
         * dEgdT = -0.00011

         >>> M = np.polyval([-3.74E-5, 0.00125, -0.01462, 0.0718, 0.9210],
         ...                AMa) # doctest: +SKIP

         Source: [4]

     Copper Indium Gallium diSelenide (CIGS):
         * EgRef = 1.15
         * dEgdT = ????

         >>> M = np.polyval([-9.07E-5, 0.0022, -0.0202, 0.0652, 0.9417],
         ...                AMa) # doctest: +SKIP

         Source: Wikipedia

     Gallium Arsenide (GaAs):
         * EgRef = 1.424
         * dEgdT = -0.000433
         * M = unknown

         Source: [4]
    '''

    # test for use of function pre-v0.6.0 API change
    if isinstance(a_ref, dict) or \
       (isinstance(a_ref, pd.Series) and ('a_ref' in a_ref.keys())):
        import warnings
        warnings.warn('module_parameters detected as fourth positional'
                      + ' argument of calcparams_desoto. calcparams_desoto'
                      + ' will require one argument for each module model'
                      + ' parameter in v0.7.0 and later', DeprecationWarning)
        try:
            module_parameters = a_ref
            a_ref = module_parameters['a_ref']
            I_L_ref = module_parameters['I_L_ref']
            I_o_ref = module_parameters['I_o_ref']
            R_sh_ref = module_parameters['R_sh_ref']
            R_s = module_parameters['R_s']
        except Exception as e:
            raise e('Module parameters could not be extracted from fourth'
                    + ' positional argument of calcparams_desoto. Check that'
                    + ' parameters are from the CEC database and/or update'
                    + ' your code for the new API for calcparams_desoto')

    # Boltzmann constant in eV/K
    k = 8.617332478e-05

    # reference temperature
    Tref_K = temp_ref + 273.15
    Tcell_K = temp_cell + 273.15

    E_g = EgRef * (1 + dEgdT*(Tcell_K - Tref_K))

    nNsVth = a_ref * (Tcell_K / Tref_K)

    # In the equation for IL, the single factor effective_irradiance is
    # used, in place of the product S*M in [1]. effective_irradiance is
    # equivalent to the product of S (irradiance reaching a module's cells) *
    # M (spectral adjustment factor) as described in [1].
    IL = effective_irradiance / irrad_ref * \
        (I_L_ref + alpha_sc * (Tcell_K - Tref_K))
    I0 = (I_o_ref * ((Tcell_K / Tref_K) ** 3) *
          (np.exp(EgRef / (k*(Tref_K)) - (E_g / (k*(Tcell_K))))))
    # Note that the equation for Rsh differs from [1]. In [1] Rsh is given as
    # Rsh = Rsh_ref * (S_ref / S) where S is broadband irradiance reaching
    # the module's cells. If desired this model behavior can be duplicated
    # by applying reflection and soiling losses to broadband plane of array
    # irradiance and not applying a spectral loss modifier, i.e.,
    # spectral_modifier = 1.0.
    # use errstate to silence divide by warning
    with np.errstate(divide='ignore'):
        Rsh = R_sh_ref * (irrad_ref / effective_irradiance)
    Rs = R_s

    return IL, I0, Rs, Rsh, nNsVth


def calcparams_cec(effective_irradiance, temp_cell,
                   alpha_sc, a_ref, I_L_ref, I_o_ref, R_sh_ref, R_s,
                   Adjust, EgRef=1.121, dEgdT=-0.0002677,
                   irrad_ref=1000, temp_ref=25):
    '''
    Calculates five parameter values for the single diode equation at
    effective irradiance and cell temperature using the CEC
    model described in [1]. The CEC model differs from the De soto et al.
    model [3] by the parameter Adjust. The five values returned by
    calcparams_cec can be used by singlediode to calculate an IV curve.

    Parameters
    ----------
    effective_irradiance : numeric
        The irradiance (W/m2) that is converted to photocurrent.

    temp_cell : numeric
        The average cell temperature of cells within a module in C.

    alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.

    a_ref : float
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at reference
        conditions, in units of V.

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    Adjust : float
        The adjustment to the temperature coefficient for short circuit
        current, in percent

    EgRef : float
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.  For parameters
        from the SAM CEC module database, EgRef=1.121 is implicit for all
        cell types in the parameter estimation algorithm used by NREL.

    dEgdT : float
        The temperature dependence of the energy bandgap at reference
        conditions in units of 1/K. May be either a scalar value
        (e.g. -0.0002677 as in [3]) or a DataFrame (this may be useful if
        dEgdT is a modeled as a function of temperature). For parameters from
        the SAM CEC module database, dEgdT=-0.0002677 is implicit for all cell
        types in the parameter estimation algorithm used by NREL.

    irrad_ref : float (optional, default=1000)
        Reference irradiance in W/m^2.

    temp_ref : float (optional, default=25)
        Reference cell temperature in C.

    Returns
    -------
    Tuple of the following results:

    photocurrent : numeric
        Light-generated current in amperes

    saturation_current : numeric
        Diode saturation curent in amperes

    resistance_series : float
        Series resistance in ohms

    resistance_shunt : numeric
        Shunt resistance in ohms

    nNsVth : numeric
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at
        specified effective irradiance and cell temperature.

    References
    ----------
    [1] A. Dobos, "An Improved Coefficient Calculator for the California
    Energy Commission 6 Parameter Photovoltaic Module Model", Journal of
    Solar Energy Engineering, vol 134, 2012.

    [2] System Advisor Model web page. https://sam.nrel.gov.

    [3] W. De Soto et al., "Improvement and validation of a model for
    photovoltaic array performance", Solar Energy, vol 80, pp. 78-88,
    2006.

    See Also
    --------
    calcparams_desoto
    singlediode
    retrieve_sam

    '''

    # pass adjusted temperature coefficient to desoto
    return calcparams_desoto(effective_irradiance, temp_cell,
                             alpha_sc*(1.0 - Adjust/100),
                             a_ref, I_L_ref, I_o_ref,
                             R_sh_ref, R_s,
                             EgRef=1.121, dEgdT=-0.0002677,
                             irrad_ref=1000, temp_ref=25)


def calcparams_pvsyst(effective_irradiance, temp_cell,
                      alpha_sc, gamma_ref, mu_gamma,
                      I_L_ref, I_o_ref,
                      R_sh_ref, R_sh_0, R_s,
                      cells_in_series,
                      R_sh_exp=5.5,
                      EgRef=1.121,
                      irrad_ref=1000, temp_ref=25):
    '''
    Calculates five parameter values for the single diode equation at
    effective irradiance and cell temperature using the PVsyst v6
    model described in [1,2,3]. The five values returned by calcparams_pvsyst
    can be used by singlediode to calculate an IV curve.

    Parameters
    ----------
    effective_irradiance : numeric
        The irradiance (W/m2) that is converted to photocurrent.

    temp_cell : numeric
        The average cell temperature of cells within a module in C.

    alpha_sc : float
        The short-circuit current temperature coefficient of the
        module in units of A/C.

    gamma_ref : float
        The diode ideality factor

    mu_gamma : float
        The temperature coefficient for the diode ideality factor, 1/K

    I_L_ref : float
        The light-generated current (or photocurrent) at reference conditions,
        in amperes.

    I_o_ref : float
        The dark or diode reverse saturation current at reference conditions,
        in amperes.

    R_sh_ref : float
        The shunt resistance at reference conditions, in ohms.

    R_sh_0 : float
        The shunt resistance at zero irradiance conditions, in ohms.

    R_s : float
        The series resistance at reference conditions, in ohms.

    cells_in_series : integer
        The number of cells connected in series.

    R_sh_exp : float
        The exponent in the equation for shunt resistance, unitless. Defaults
        to 5.5.

    EgRef : float
        The energy bandgap at reference temperature in units of eV.
        1.121 eV for crystalline silicon. EgRef must be >0.

    irrad_ref : float (optional, default=1000)
        Reference irradiance in W/m^2.

    temp_ref : float (optional, default=25)
        Reference cell temperature in C.

    Returns
    -------
    Tuple of the following results:

    photocurrent : numeric
        Light-generated current in amperes

    saturation_current : numeric
        Diode saturation current in amperes

    resistance_series : float
        Series resistance in ohms

    resistance_shunt : numeric
        Shunt resistance in ohms

    nNsVth : numeric
        The product of the usual diode ideality factor (n, unitless),
        number of cells in series (Ns), and cell thermal voltage at
        specified effective irradiance and cell temperature.

    References
    ----------
    [1] K. Sauer, T. Roessler, C. W. Hansen, Modeling the Irradiance and
     Temperature Dependence of Photovoltaic Modules in PVsyst,
     IEEE Journal of Photovoltaics v5(1), January 2015.

    [2] A. Mermoud, PV modules modelling, Presentation at the 2nd PV
     Performance Modeling Workshop, Santa Clara, CA, May 2013

    [3] A. Mermoud, T. Lejeune, Performance Assessment of a Simulation Model
     for PV modules of any available technology, 25th European Photovoltaic
     Solar Energy Conference, Valencia, Spain, Sept. 2010

    See Also
    --------
    calcparams_desoto
    singlediode

    '''

    # Boltzmann constant in J/K
    k = 1.38064852e-23

    # elementary charge in coulomb
    q = 1.6021766e-19

    # reference temperature
    Tref_K = temp_ref + 273.15
    Tcell_K = temp_cell + 273.15

    gamma = gamma_ref + mu_gamma * (Tcell_K - Tref_K)
    nNsVth = gamma * k / q * cells_in_series * Tcell_K

    IL = effective_irradiance / irrad_ref * \
        (I_L_ref + alpha_sc * (Tcell_K - Tref_K))

    I0 = I_o_ref * ((Tcell_K / Tref_K) ** 3) * \
        (np.exp((q * EgRef) / (k * gamma) * (1 / Tref_K - 1 / Tcell_K)))

    Rsh_tmp = \
        (R_sh_ref - R_sh_0 * np.exp(-R_sh_exp)) / (1.0 - np.exp(-R_sh_exp))
    Rsh_base = np.maximum(0.0, Rsh_tmp)

    Rsh = Rsh_base + (R_sh_0 - Rsh_base) * \
        np.exp(-R_sh_exp * effective_irradiance / irrad_ref)

    Rs = R_s

    return IL, I0, Rs, Rsh, nNsVth


def retrieve_sam(name=None, path=None):
    '''
    Retrieve latest module and inverter info from a local file or the
    SAM website.

    This function will retrieve either:

        * CEC module database
        * Sandia Module database
        * CEC Inverter database
        * Anton Driesse Inverter database

    and return it as a pandas DataFrame.

    Parameters
    ----------
    name : None or string, default None
        Name can be one of:

        * 'CECMod' - returns the CEC module database
        * 'CECInverter' - returns the CEC Inverter database
        * 'SandiaInverter' - returns the CEC Inverter database
          (CEC is only current inverter db available; tag kept for
          backwards compatibility)
        * 'SandiaMod' - returns the Sandia Module database
        * 'ADRInverter' - returns the ADR Inverter database

    path : None or string, default None
        Path to the SAM file. May also be a URL.

    If both name and path are None, a dialogue will open allowing the
    user to select a file.

    Returns
    -------
    samfile : DataFrame
        A DataFrame containing all the elements of the desired database.
        Each column represents a module or inverter, and a specific
        dataset can be retrieved by the command

    Notes
    -----
    Files available at https://sam.nrel.gov/sites/default/files/

    Examples
    --------

    >>> from pvlib import pvsystem
    >>> invdb = pvsystem.retrieve_sam('CECInverter')
    >>> inverter = invdb.AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_
    >>> inverter
    Vac           277.000000
    Paco         6000.000000
    Pdco         6165.670000
    Vdco          361.123000
    Pso            36.792300
    C0             -0.000002
    C1             -0.000047
    C2             -0.001861
    C3              0.000721
    Pnt             0.070000
    Vdcmax        600.000000
    Idcmax         32.000000
    Mppt_low      200.000000
    Mppt_high     500.000000
    Name: AE_Solar_Energy__AE6_0__277V__277V__CEC_2012_, dtype: float64
    '''

    if name is not None:
        name = name.lower()
        data_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), 'data')
        if name == 'cecmod':
            csvdata = os.path.join(
                data_path, 'sam-library-cec-modules-2017-6-5.csv')
        elif name == 'sandiamod':
            csvdata = os.path.join(
                data_path, 'sam-library-sandia-modules-2015-6-30.csv')
        elif name == 'adrinverter':
            csvdata = os.path.join(data_path, 'adr-library-2013-10-01.csv')
        elif name in ['cecinverter', 'sandiainverter']:
            # Allowing either, to provide for old code,
            # while aligning with current expectations
            csvdata = os.path.join(
                data_path, 'sam-library-cec-inverters-2018-3-18.csv')
        else:
            raise ValueError('invalid name {}'.format(name))
    elif path is not None:
        if path.startswith('http'):
            response = urlopen(path)
            csvdata = io.StringIO(response.read().decode(errors='ignore'))
        else:
            csvdata = path
    elif name is None and path is None:
        try:
            # python 2
            import Tkinter as tkinter
            from tkFileDialog import askopenfilename
        except ImportError:
            # python 3
            import tkinter
            from tkinter.filedialog import askopenfilename

        tkinter.Tk().withdraw()
        csvdata = askopenfilename()

    return _parse_raw_sam_df(csvdata)


def _parse_raw_sam_df(csvdata):
    df = pd.read_csv(csvdata, index_col=0, skiprows=[1, 2])
    colnames = df.columns.values.tolist()
    parsedcolnames = []
    for cn in colnames:
        parsedcolnames.append(cn.replace(' ', '_'))

    df.columns = parsedcolnames

    parsedindex = []
    for index in df.index:
        parsedindex.append(index.replace(' ', '_').replace('-', '_')
                                .replace('.', '_').replace('(', '_')
                                .replace(')', '_').replace('[', '_')
                                .replace(']', '_').replace(':', '_')
                                .replace('+', '_').replace('/', '_')
                                .replace('"', '_').replace(',', '_'))

    df.index = parsedindex
    df = df.transpose()
    if 'ADRCoefficients' in df.index:
        ad_ce = 'ADRCoefficients'
        # for each inverter, parses a string of coefficients like
        # ' 1.33, 2.11, 3.12' into a list containing floats:
        # [1.33, 2.11, 3.12]
        df.loc[ad_ce] = df.loc[ad_ce].map(lambda x: list(
            map(float, x.strip(' []').split())))

    return df


def sapm(effective_irradiance, temp_cell, module):
    '''
    The Sandia PV Array Performance Model (SAPM) generates 5 points on a
    PV module's I-V curve (Voc, Isc, Ix, Ixx, Vmp/Imp) according to
    SAND2004-3535. Assumes a reference cell temperature of 25 C.

    Parameters
    ----------
    effective_irradiance : numeric
        Effective irradiance (suns).

    temp_cell : numeric
        The cell temperature (degrees C).

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the notes section for more details.

    Returns
    -------
    A DataFrame with the columns:

        * i_sc : Short-circuit current (A)
        * i_mp : Current at the maximum-power point (A)
        * v_oc : Open-circuit voltage (V)
        * v_mp : Voltage at maximum-power point (V)
        * p_mp : Power at maximum-power point (W)
        * i_x : Current at module V = 0.5Voc, defines 4th point on I-V
          curve for modeling curve shape
        * i_xx : Current at module V = 0.5(Voc+Vmp), defines 5th point on
          I-V curve for modeling curve shape

    Notes
    -----
    The coefficients from SAPM which are required in ``module`` are
    listed in the following table.

    The modules in the Sandia module database contain these
    coefficients, but the modules in the CEC module database do not.
    Both databases can be accessed using :py:func:`retrieve_sam`.

    ================   ========================================================
    Key                Description
    ================   ========================================================
    A0-A4              The airmass coefficients used in calculating
                       effective irradiance
    B0-B5              The angle of incidence coefficients used in calculating
                       effective irradiance
    C0-C7              The empirically determined coefficients relating
                       Imp, Vmp, Ix, and Ixx to effective irradiance
    Isco               Short circuit current at reference condition (amps)
    Impo               Maximum power current at reference condition (amps)
    Aisc               Short circuit current temperature coefficient at
                       reference condition (1/C)
    Aimp               Maximum power current temperature coefficient at
                       reference condition (1/C)
    Bvoco              Open circuit voltage temperature coefficient at
                       reference condition (V/C)
    Mbvoc              Coefficient providing the irradiance dependence for the
                       BetaVoc temperature coefficient at reference irradiance
                       (V/C)
    Bvmpo              Maximum power voltage temperature coefficient at
                       reference condition
    Mbvmp              Coefficient providing the irradiance dependence for the
                       BetaVmp temperature coefficient at reference irradiance
                       (V/C)
    N                  Empirically determined "diode factor" (dimensionless)
    Cells_in_Series    Number of cells in series in a module's cell string(s)
    IXO                Ix at reference conditions
    IXXO               Ixx at reference conditions
    FD                 Fraction of diffuse irradiance used by module
    ================   ========================================================

    References
    ----------
    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
    Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
    NM.

    See Also
    --------
    retrieve_sam
    sapm_celltemp
    '''

    T0 = 25
    q = 1.60218e-19  # Elementary charge in units of coulombs
    kb = 1.38066e-23  # Boltzmann's constant in units of J/K

    # avoid problem with integer input
    Ee = np.array(effective_irradiance, dtype='float64')

    # set up masking for 0, positive, and nan inputs
    Ee_gt_0 = np.full_like(Ee, False, dtype='bool')
    Ee_eq_0 = np.full_like(Ee, False, dtype='bool')
    notnan = ~np.isnan(Ee)
    np.greater(Ee, 0, where=notnan, out=Ee_gt_0)
    np.equal(Ee, 0, where=notnan, out=Ee_eq_0)

    Bvmpo = module['Bvmpo'] + module['Mbvmp']*(1 - Ee)
    Bvoco = module['Bvoco'] + module['Mbvoc']*(1 - Ee)
    delta = module['N'] * kb * (temp_cell + 273.15) / q

    # avoid repeated computation
    logEe = np.full_like(Ee, np.nan)
    np.log(Ee, where=Ee_gt_0, out=logEe)
    logEe = np.where(Ee_eq_0, -np.inf, logEe)
    # avoid repeated __getitem__
    cells_in_series = module['Cells_in_Series']

    out = OrderedDict()

    out['i_sc'] = (
        module['Isco'] * Ee * (1 + module['Aisc']*(temp_cell - T0)))

    out['i_mp'] = (
        module['Impo'] * (module['C0']*Ee + module['C1']*(Ee**2)) *
        (1 + module['Aimp']*(temp_cell - T0)))

    out['v_oc'] = np.maximum(0, (
        module['Voco'] + cells_in_series * delta * logEe +
        Bvoco*(temp_cell - T0)))

    out['v_mp'] = np.maximum(0, (
        module['Vmpo'] +
        module['C2'] * cells_in_series * delta * logEe +
        module['C3'] * cells_in_series * ((delta * logEe) ** 2) +
        Bvmpo*(temp_cell - T0)))

    out['p_mp'] = out['i_mp'] * out['v_mp']

    out['i_x'] = (
        module['IXO'] * (module['C4']*Ee + module['C5']*(Ee**2)) *
        (1 + module['Aisc']*(temp_cell - T0)))

    # the Ixx calculation in King 2004 has a typo (mixes up Aisc and Aimp)
    out['i_xx'] = (
        module['IXXO'] * (module['C6']*Ee + module['C7']*(Ee**2)) *
        (1 + module['Aisc']*(temp_cell - T0)))

    if isinstance(out['i_sc'], pd.Series):
        out = pd.DataFrame(out)

    return out


def sapm_celltemp(poa_global, wind_speed, temp_air,
                  model='open_rack_cell_glassback'):
    '''
    Estimate cell and module temperatures per the Sandia PV Array
    Performance Model (SAPM, SAND2004-3535), from the incident
    irradiance, wind speed, ambient temperature, and SAPM module
    parameters.

    Parameters
    ----------
    poa_global : float or Series
        Total incident irradiance in W/m^2.

    wind_speed : float or Series
        Wind speed in m/s at a height of 10 meters.

    temp_air : float or Series
        Ambient dry bulb temperature in degrees C.

    model : string, list, or dict, default 'open_rack_cell_glassback'
        Model to be used.

        If string, can be:

            * 'open_rack_cell_glassback' (default)
            * 'roof_mount_cell_glassback'
            * 'open_rack_cell_polymerback'
            * 'insulated_back_polymerback'
            * 'open_rack_polymer_thinfilm_steel'
            * '22x_concentrator_tracker'

        If dict, supply the following parameters
        (if list, in the following order):

            * a : float
                SAPM module parameter for establishing the upper
                limit for module temperature at low wind speeds and
                high solar irradiance.

            * b : float
                SAPM module parameter for establishing the rate at
                which the module temperature drops as wind speed increases
                (see SAPM eqn. 11).

            * deltaT : float
                SAPM module parameter giving the temperature difference
                between the cell and module back surface at the
                reference irradiance, E0.

    Returns
    --------
    DataFrame with columns 'temp_cell' and 'temp_module'.
    Values in degrees C.

    References
    ----------
    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
    Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
    NM.

    See Also
    --------
    sapm
    '''

    temp_models = TEMP_MODEL_PARAMS['sapm']

    if isinstance(model, str):
        model = temp_models[model.lower()]

    elif isinstance(model, (dict, pd.Series)):
        model = [model['a'], model['b'], model['deltaT']]

    a = model[0]
    b = model[1]
    deltaT = model[2]

    E0 = 1000.  # Reference irradiance

    temp_module = pd.Series(poa_global * np.exp(a + b * wind_speed) + temp_air)

    temp_cell = temp_module + (poa_global / E0) * (deltaT)

    return pd.DataFrame({'temp_cell': temp_cell, 'temp_module': temp_module})


def pvsyst_celltemp(poa_global, temp_air, wind_speed=1.0, eta_m=0.1,
                    alpha_absorption=0.9, model_params='freestanding'):
    """
    Calculate cell temperature using an emperical heat loss factor model
    as implemented in PVsyst.

    The heat loss factors provided through the 'model_params' argument
    represent the combined effect of convection, radiation and conduction,
    and their values are experimentally determined.

    Parameters
    ----------
    poa_global : numeric
        Total incident irradiance in W/m^2.

    temp_air : numeric
        Ambient dry bulb temperature in degrees C.

    wind_speed : numeric, default 1.0
        Wind speed in m/s measured at the same height for which the wind loss
        factor was determined.  The default value is 1.0, which is the wind
        speed at module height used to determine NOCT.

    eta_m : numeric, default 0.1
        Module external efficiency as a fraction, i.e., DC power / poa_global.

    alpha_absorption : numeric, default 0.9
        Absorption coefficient

    model_params : string, tuple, or list (no dict), default 'freestanding'
        Heat loss factors to be used.

        If string, can be:

            * 'freestanding' (default)
                Modules with rear surfaces exposed to open air (e.g. rack
                mounted).
            * 'insulated'
                Modules with rear surfaces in close proximity to another
                surface (e.g. roof mounted).

        If tuple/list, supply parameters in the following order:

            * constant_loss_factor : float
                Combined heat loss factor coefficient. Freestanding
                default is 29, fully insulated arrays is 15.

            * wind_loss_factor : float
                Combined heat loss factor influenced by wind. Default is 0.

    Returns
    -------
    temp_cell : numeric or Series
        Cell temperature in degrees Celsius

    References
    ----------
    [1]"PVsyst 6 Help", Files.pvsyst.com, 2018. [Online]. Available:
    http://files.pvsyst.com/help/index.html. [Accessed: 10- Dec- 2018].

    [2] Faiman, D. (2008). "Assessing the outdoor operating temperature of
    photovoltaic modules." Progress in Photovoltaics 16(4): 307-315.
    """

    pvsyst_presets = TEMP_MODEL_PARAMS['pvsyst']

    if isinstance(model_params, str):
        model_params = model_params.lower()
        constant_loss_factor, wind_loss_factor = pvsyst_presets[model_params]
    elif isinstance(model_params, (tuple, list)):
        constant_loss_factor, wind_loss_factor = model_params
    else:
        raise TypeError(
            "Please provide model_params as a str, or tuple/list."
        )

    total_loss_factor = wind_loss_factor * wind_speed + constant_loss_factor
    heat_input = poa_global * alpha_absorption * (1 - eta_m)
    temp_difference = heat_input / total_loss_factor
    temp_cell = temp_air + temp_difference

    return temp_cell


def sapm_spectral_loss(airmass_absolute, module):
    """
    Calculates the SAPM spectral loss coefficient, F1.

    Parameters
    ----------
    airmass_absolute : numeric
        Absolute airmass

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the :py:func:`sapm` notes section for more
        details.

    Returns
    -------
    F1 : numeric
        The SAPM spectral loss coefficient.

    Notes
    -----
    nan airmass values will result in 0 output.
    """

    am_coeff = [module['A4'], module['A3'], module['A2'], module['A1'],
                module['A0']]

    spectral_loss = np.polyval(am_coeff, airmass_absolute)

    spectral_loss = np.where(np.isnan(spectral_loss), 0, spectral_loss)

    spectral_loss = np.maximum(0, spectral_loss)

    if isinstance(airmass_absolute, pd.Series):
        spectral_loss = pd.Series(spectral_loss, airmass_absolute.index)

    return spectral_loss


def sapm_aoi_loss(aoi, module, upper=None):
    """
    Calculates the SAPM angle of incidence loss coefficient, F2.

    Parameters
    ----------
    aoi : numeric
        Angle of incidence in degrees. Negative input angles will return
        zeros.

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the :py:func:`sapm` notes section for more
        details.

    upper : None or float, default None
        Upper limit on the results.

    Returns
    -------
    F2 : numeric
        The SAPM angle of incidence loss coefficient.

    Notes
    -----
    The SAPM traditionally does not define an upper limit on the AOI
    loss function and values slightly exceeding 1 may exist for moderate
    angles of incidence (15-40 degrees). However, users may consider
    imposing an upper limit of 1.

    References
    ----------
    [1] King, D. et al, 2004, "Sandia Photovoltaic Array Performance
    Model", SAND Report 3535, Sandia National Laboratories, Albuquerque,
    NM.

    [2] B.H. King et al, "Procedure to Determine Coefficients for the
    Sandia Array Performance Model (SAPM)," SAND2016-5284, Sandia
    National Laboratories (2016).

    [3] B.H. King et al, "Recent Advancements in Outdoor Measurement
    Techniques for Angle of Incidence Effects," 42nd IEEE PVSC (2015).
    DOI: 10.1109/PVSC.2015.7355849
    """

    aoi_coeff = [module['B5'], module['B4'], module['B3'], module['B2'],
                 module['B1'], module['B0']]

    aoi_loss = np.polyval(aoi_coeff, aoi)
    aoi_loss = np.clip(aoi_loss, 0, upper)
    # nan tolerant masking
    aoi_lt_0 = np.full_like(aoi, False, dtype='bool')
    np.less(aoi, 0, where=~np.isnan(aoi), out=aoi_lt_0)
    aoi_loss = np.where(aoi_lt_0, 0, aoi_loss)

    if isinstance(aoi, pd.Series):
        aoi_loss = pd.Series(aoi_loss, aoi.index)

    return aoi_loss


def sapm_effective_irradiance(poa_direct, poa_diffuse, airmass_absolute, aoi,
                              module, reference_irradiance=1000):
    """
    Calculates the SAPM effective irradiance using the SAPM spectral
    loss and SAPM angle of incidence loss functions.

    Parameters
    ----------
    poa_direct : numeric
        The direct irradiance incident upon the module.

    poa_diffuse : numeric
        The diffuse irradiance incident on module.

    airmass_absolute : numeric
        Absolute airmass.

    aoi : numeric
        Angle of incidence in degrees.

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the :py:func:`sapm` notes section for more
        details.

    reference_irradiance : numeric, default 1000
        Reference irradiance by which to divide the input irradiance.

    Returns
    -------
    effective_irradiance : numeric
        The SAPM effective irradiance.
    """

    F1 = sapm_spectral_loss(airmass_absolute, module)
    F2 = sapm_aoi_loss(aoi, module)

    E0 = reference_irradiance

    Ee = F1 * (poa_direct*F2 + module['FD']*poa_diffuse) / E0

    return Ee


def singlediode(photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth, ivcurve_pnts=None,
                method='lambertw'):
    """
    Solve the single-diode model to obtain a photovoltaic IV curve.

    Singlediode solves the single diode equation [1]

    .. math::

        I = IL - I0*[exp((V+I*Rs)/(nNsVth))-1] - (V + I*Rs)/Rsh

    for ``I`` and ``V`` when given ``IL, I0, Rs, Rsh,`` and ``nNsVth
    (nNsVth = n*Ns*Vth)`` which are described later. Returns a DataFrame
    which contains the 5 points on the I-V curve specified in
    SAND2004-3535 [3]. If all IL, I0, Rs, Rsh, and nNsVth are scalar, a
    single curve will be returned, if any are Series (of the same
    length), multiple IV curves will be calculated.

    The input parameters can be calculated using calcparams_desoto from
    meteorological data.

    Parameters
    ----------
    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        IV curve conditions. Often abbreviated ``I_L``.
        0 <= photocurrent

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.
        0 < saturation_current

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.
        0 <= resistance_series < numpy.inf

    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.
        0 < resistance_shunt <= numpy.inf

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).
        0 < nNsVth

    ivcurve_pnts : None or int, default None
        Number of points in the desired IV curve. If None or 0, no
        IV curves will be produced.

    method : str, default 'lambertw'
        Determines the method used to calculate points on the IV curve. The
        options are ``'lambertw'``, ``'newton'``, or ``'brentq'``.

    Returns
    -------
    OrderedDict or DataFrame

    The returned dict-like object always contains the keys/columns:

        * i_sc - short circuit current in amperes.
        * v_oc - open circuit voltage in volts.
        * i_mp - current at maximum power point in amperes.
        * v_mp - voltage at maximum power point in volts.
        * p_mp - power at maximum power point in watts.
        * i_x - current, in amperes, at ``v = 0.5*v_oc``.
        * i_xx - current, in amperes, at ``V = 0.5*(v_oc+v_mp)``.

    If ivcurve_pnts is greater than 0, the output dictionary will also
    include the keys:

        * i - IV curve current in amperes.
        * v - IV curve voltage in volts.

    The output will be an OrderedDict if photocurrent is a scalar,
    array, or ivcurve_pnts is not None.

    The output will be a DataFrame if photocurrent is a Series and
    ivcurve_pnts is None.

    Notes
    -----
    If the method is ``'lambertw'`` then the solution employed to solve the
    implicit diode equation utilizes the Lambert W function to obtain an
    explicit function of :math:`V=f(I)` and :math:`I=f(V)` as shown in [2].

    If the method is ``'newton'`` then the root-finding Newton-Raphson method
    is used. It should be safe for well behaved IV-curves, but the ``'brentq'``
    method is recommended for reliability.

    If the method is ``'brentq'`` then Brent's bisection search method is used
    that guarantees convergence by bounding the voltage between zero and
    open-circuit.

    If the method is either ``'newton'`` or ``'brentq'`` and ``ivcurve_pnts``
    are indicated, then :func:`pvlib.singlediode.bishop88` [4] is used to
    calculate the points on the IV curve points at diode voltages from zero to
    open-circuit voltage with a log spacing that gets closer as voltage
    increases. If the method is ``'lambertw'`` then the calculated points on
    the IV curve are linearly spaced.

    References
    -----------
    [1] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" ISBN
    0 86758 909 4

    [2] A. Jain, A. Kapoor, "Exact analytical solutions of the
    parameters of real solar cells using Lambert W-function", Solar
    Energy Materials and Solar Cells, 81 (2004) 269-277.

    [3] D. King et al, "Sandia Photovoltaic Array Performance Model",
    SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

    [4] "Computer simulation of the effects of electrical mismatches in
    photovoltaic cell interconnection circuits" JW Bishop, Solar Cell (1988)
    https://doi.org/10.1016/0379-6787(88)90059-2

    See also
    --------
    sapm
    calcparams_desoto
    pvlib.singlediode.bishop88
    """
    # Calculate points on the IV curve using the LambertW solution to the
    # single diode equation
    if method.lower() == 'lambertw':
        out = _singlediode._lambertw(
            photocurrent, saturation_current, resistance_series,
            resistance_shunt, nNsVth, ivcurve_pnts
        )
        i_sc, v_oc, i_mp, v_mp, p_mp, i_x, i_xx = out[:7]
        if ivcurve_pnts:
            ivcurve_i, ivcurve_v = out[7:]
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth)  # collect args
        v_oc = _singlediode.bishop88_v_from_i(
            0.0, *args, method=method.lower()
        )
        i_mp, v_mp, p_mp = _singlediode.bishop88_mpp(
            *args, method=method.lower()
        )
        i_sc = _singlediode.bishop88_i_from_v(
            0.0, *args, method=method.lower()
        )
        i_x = _singlediode.bishop88_i_from_v(
            v_oc / 2.0, *args, method=method.lower()
        )
        i_xx = _singlediode.bishop88_i_from_v(
            (v_oc + v_mp) / 2.0, *args, method=method.lower()
        )

        # calculate the IV curve if requested using bishop88
        if ivcurve_pnts:
            vd = v_oc * (
                    (11.0 - np.logspace(np.log10(11.0), 0.0,
                                        ivcurve_pnts)) / 10.0
            )
            ivcurve_i, ivcurve_v, _ = _singlediode.bishop88(vd, *args)

    out = OrderedDict()
    out['i_sc'] = i_sc
    out['v_oc'] = v_oc
    out['i_mp'] = i_mp
    out['v_mp'] = v_mp
    out['p_mp'] = p_mp
    out['i_x'] = i_x
    out['i_xx'] = i_xx

    if ivcurve_pnts:

        out['v'] = ivcurve_v
        out['i'] = ivcurve_i

    if isinstance(photocurrent, pd.Series) and not ivcurve_pnts:
        out = pd.DataFrame(out, index=photocurrent.index)

    return out


def max_power_point(photocurrent, saturation_current, resistance_series,
                    resistance_shunt, nNsVth, method='brentq'):
    """
    Given the single diode equation coefficients, calculates the maximum power
    point (MPP).

    Parameters
    ----------
    photocurrent : numeric
        photo-generated current [A]
    saturation_current : numeric
        diode reverse saturation current [A]
    resistance_series : numeric
        series resitance [ohms]
    resistance_shunt : numeric
        shunt resitance [ohms]
    nNsVth : numeric
        product of thermal voltage ``Vth`` [V], diode ideality factor ``n``,
        and number of serices cells ``Ns``
    method : str
        either ``'newton'`` or ``'brentq'``

    Returns
    -------
    OrderedDict or pandas.Datafrane
        ``(i_mp, v_mp, p_mp)``

    Notes
    -----
    Use this function when you only want to find the maximum power point. Use
    :func:`singlediode` when you need to find additional points on the IV
    curve. This function uses Brent's method by default because it is
    guaranteed to converge.
    """
    i_mp, v_mp, p_mp = _singlediode.bishop88_mpp(
        photocurrent, saturation_current, resistance_series,
        resistance_shunt, nNsVth, method=method.lower()
    )
    if isinstance(photocurrent, pd.Series):
        ivp = {'i_mp': i_mp, 'v_mp': v_mp, 'p_mp': p_mp}
        out = pd.DataFrame(ivp, index=photocurrent.index)
    else:
        out = OrderedDict()
        out['i_mp'] = i_mp
        out['v_mp'] = v_mp
        out['p_mp'] = p_mp
    return out


def v_from_i(resistance_shunt, resistance_series, nNsVth, current,
             saturation_current, photocurrent, method='lambertw'):
    '''
    Device voltage at the given device current for the single diode model.

    Uses the single diode model (SDM) as described in, e.g.,
     Jain and Kapoor 2004 [1].
    The solution is per Eq 3 of [1] except when resistance_shunt=numpy.inf,
     in which case the explict solution for voltage is used.
    Ideal device parameters are specified by resistance_shunt=np.inf and
     resistance_series=0.
    Inputs to this function can include scalars and pandas.Series, but it is
     the caller's responsibility to ensure that the arguments are all float64
     and within the proper ranges.

    Parameters
    ----------
    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.
        0 < resistance_shunt <= numpy.inf

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.
        0 <= resistance_series < numpy.inf

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).
        0 < nNsVth

    current : numeric
        The current in amperes under desired IV curve conditions.

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.
        0 < saturation_current

    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        IV curve conditions. Often abbreviated ``I_L``.
        0 <= photocurrent

    method : str
        Method to use: ``'lambertw'``, ``'newton'``, or ``'brentq'``. *Note*:
        ``'brentq'`` is limited to 1st quadrant only.

    Returns
    -------
    current : np.ndarray or scalar

    References
    ----------
    [1] A. Jain, A. Kapoor, "Exact analytical solutions of the
    parameters of real solar cells using Lambert W-function", Solar
    Energy Materials and Solar Cells, 81 (2004) 269-277.
    '''
    if method.lower() == 'lambertw':
        return _singlediode._lambertw_v_from_i(
            resistance_shunt, resistance_series, nNsVth, current,
            saturation_current, photocurrent
        )
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (current, photocurrent, saturation_current,
                resistance_series, resistance_shunt, nNsVth)
        V = _singlediode.bishop88_v_from_i(*args, method=method.lower())
        # find the right size and shape for returns
        size, shape = _singlediode._get_size_and_shape(args)
        if size <= 1:
            if shape is not None:
                V = np.tile(V, shape)
        if np.isnan(V).any() and size <= 1:
            V = np.repeat(V, size)
            if shape is not None:
                V = V.reshape(shape)
        return V


def i_from_v(resistance_shunt, resistance_series, nNsVth, voltage,
             saturation_current, photocurrent, method='lambertw'):
    '''
    Device current at the given device voltage for the single diode model.

    Uses the single diode model (SDM) as described in, e.g.,
     Jain and Kapoor 2004 [1].
    The solution is per Eq 2 of [1] except when resistance_series=0,
     in which case the explict solution for current is used.
    Ideal device parameters are specified by resistance_shunt=np.inf and
     resistance_series=0.
    Inputs to this function can include scalars and pandas.Series, but it is
     the caller's responsibility to ensure that the arguments are all float64
     and within the proper ranges.

    Parameters
    ----------
    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.
        0 < resistance_shunt <= numpy.inf

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.
        0 <= resistance_series < numpy.inf

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).
        0 < nNsVth

    voltage : numeric
        The voltage in Volts under desired IV curve conditions.

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.
        0 < saturation_current

    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        IV curve conditions. Often abbreviated ``I_L``.
        0 <= photocurrent

    method : str
        Method to use: ``'lambertw'``, ``'newton'``, or ``'brentq'``. *Note*:
        ``'brentq'`` is limited to 1st quadrant only.

    Returns
    -------
    current : np.ndarray or scalar

    References
    ----------
    [1] A. Jain, A. Kapoor, "Exact analytical solutions of the
    parameters of real solar cells using Lambert W-function", Solar
    Energy Materials and Solar Cells, 81 (2004) 269-277.
    '''
    if method.lower() == 'lambertw':
        return _singlediode._lambertw_i_from_v(
            resistance_shunt, resistance_series, nNsVth, voltage,
            saturation_current, photocurrent
        )
    else:
        # Calculate points on the IV curve using either 'newton' or 'brentq'
        # methods. Voltages are determined by first solving the single diode
        # equation for the diode voltage V_d then backing out voltage
        args = (voltage, photocurrent, saturation_current, resistance_series,
                resistance_shunt, nNsVth)
        I = _singlediode.bishop88_i_from_v(*args, method=method.lower())
        # find the right size and shape for returns
        size, shape = _singlediode._get_size_and_shape(args)
        if size <= 1:
            if shape is not None:
                I = np.tile(I, shape)
        if np.isnan(I).any() and size <= 1:
            I = np.repeat(I, size)
            if shape is not None:
                I = I.reshape(shape)
        return I


def snlinverter(v_dc, p_dc, inverter):
    r'''
    Converts DC power and voltage to AC power using Sandia's
    Grid-Connected PV Inverter model.

    Determines the AC power output of an inverter given the DC voltage,
    DC power, and appropriate Sandia Grid-Connected Photovoltaic
    Inverter Model parameters. The output, ac_power, is clipped at the
    maximum power output, and gives a negative power during low-input
    power conditions, but does NOT account for maximum power point
    tracking voltage windows nor maximum current or voltage limits on
    the inverter.

    Parameters
    ----------
    v_dc : numeric
        DC voltages, in volts, which are provided as input to the
        inverter. Vdc must be >= 0.

    p_dc : numeric
        A scalar or DataFrame of DC powers, in watts, which are provided
        as input to the inverter. Pdc must be >= 0.

    inverter : dict-like
        A dict-like object defining the inverter to be used, giving the
        inverter performance parameters according to the Sandia
        Grid-Connected Photovoltaic Inverter Model (SAND 2007-5036) [1].
        A set of inverter performance parameters are provided with
        pvlib, or may be generated from a System Advisor Model (SAM) [2]
        library using retrievesam. See Notes for required keys.

    Returns
    -------
    ac_power : numeric
        Modeled AC power output given the input DC voltage, Vdc, and
        input DC power, Pdc. When ac_power would be greater than Pac0,
        it is set to Pac0 to represent inverter "clipping". When
        ac_power would be less than Ps0 (startup power required), then
        ac_power is set to -1*abs(Pnt) to represent nightly power
        losses. ac_power is not adjusted for maximum power point
        tracking (MPPT) voltage windows or maximum current limits of the
        inverter.

    Notes
    -----

    Required inverter keys are:

    ======   ============================================================
    Column   Description
    ======   ============================================================
    Pac0     AC-power output from inverter based on input power
             and voltage (W)
    Pdc0     DC-power input to inverter, typically assumed to be equal
             to the PV array maximum power (W)
    Vdc0     DC-voltage level at which the AC-power rating is achieved
             at the reference operating condition (V)
    Ps0      DC-power required to start the inversion process, or
             self-consumption by inverter, strongly influences inverter
             efficiency at low power levels (W)
    C0       Parameter defining the curvature (parabolic) of the
             relationship between ac-power and dc-power at the reference
             operating condition, default value of zero gives a
             linear relationship (1/W)
    C1       Empirical coefficient allowing Pdco to vary linearly
             with dc-voltage input, default value is zero (1/V)
    C2       Empirical coefficient allowing Pso to vary linearly with
             dc-voltage input, default value is zero (1/V)
    C3       Empirical coefficient allowing Co to vary linearly with
             dc-voltage input, default value is zero (1/V)
    Pnt      AC-power consumed by inverter at night (night tare) to
             maintain circuitry required to sense PV array voltage (W)
    ======   ============================================================

    References
    ----------
    [1] SAND2007-5036, "Performance Model for Grid-Connected
    Photovoltaic Inverters by D. King, S. Gonzalez, G. Galbraith, W.
    Boyson

    [2] System Advisor Model web page. https://sam.nrel.gov.

    See also
    --------
    sapm
    singlediode
    '''

    Paco = inverter['Paco']
    Pdco = inverter['Pdco']
    Vdco = inverter['Vdco']
    Pso = inverter['Pso']
    C0 = inverter['C0']
    C1 = inverter['C1']
    C2 = inverter['C2']
    C3 = inverter['C3']
    Pnt = inverter['Pnt']

    A = Pdco * (1 + C1*(v_dc - Vdco))
    B = Pso * (1 + C2*(v_dc - Vdco))
    C = C0 * (1 + C3*(v_dc - Vdco))

    ac_power = (Paco/(A-B) - C*(A-B)) * (p_dc-B) + C*((p_dc-B)**2)
    ac_power = np.minimum(Paco, ac_power)
    ac_power = np.where(p_dc < Pso, -1.0 * abs(Pnt), ac_power)

    if isinstance(p_dc, pd.Series):
        ac_power = pd.Series(ac_power, index=p_dc.index)

    return ac_power


def adrinverter(v_dc, p_dc, inverter, vtol=0.10):
    r'''
    Converts DC power and voltage to AC power using Anton Driesse's
    Grid-Connected PV Inverter efficiency model

    Parameters
    ----------
    v_dc : numeric
        A scalar or pandas series of DC voltages, in volts, which are provided
        as input to the inverter. If Vdc and Pdc are vectors, they must be
        of the same size. v_dc must be >= 0. (V)

    p_dc : numeric
        A scalar or pandas series of DC powers, in watts, which are provided
        as input to the inverter. If Vdc and Pdc are vectors, they must be
        of the same size. p_dc must be >= 0. (W)

    inverter : dict-like
        A dict-like object defining the inverter to be used, giving the
        inverter performance parameters according to the model
        developed by Anton Driesse [1].
        A set of inverter performance parameters may be loaded from the
        supplied data table using retrievesam.
        See Notes for required keys.

    vtol : numeric, default 0.1
        A unit-less fraction that determines how far the efficiency model is
        allowed to extrapolate beyond the inverter's normal input voltage
        operating range. 0.0 <= vtol <= 1.0

    Returns
    -------
    ac_power : numeric
        A numpy array or pandas series of modeled AC power output given the
        input DC voltage, v_dc, and input DC power, p_dc. When ac_power would
        be greater than pac_max, it is set to p_max to represent inverter
        "clipping". When ac_power would be less than -p_nt (energy consumed
        rather  than produced) then ac_power is set to -p_nt to represent
        nightly power losses. ac_power is not adjusted for maximum power point
        tracking (MPPT) voltage windows or maximum current limits of the
        inverter.

    Notes
    -----

    Required inverter keys are:

    =======   ============================================================
    Column    Description
    =======   ============================================================
    p_nom     The nominal power value used to normalize all power values,
              typically the DC power needed to produce maximum AC power
              output, (W).

    v_nom     The nominal DC voltage value used to normalize DC voltage
              values, typically the level at which the highest efficiency
              is achieved, (V).

    pac_max   The maximum AC output power value, used to clip the output
              if needed, (W).

    ce_list   This is a list of 9 coefficients that capture the influence
              of input voltage and power on inverter losses, and thereby
              efficiency.

    p_nt      ac-power consumed by inverter at night (night tare) to
              maintain circuitry required to sense PV array voltage, (W).
    =======   ============================================================

    References
    ----------
    [1] Beyond the Curves: Modeling the Electrical Efficiency
        of Photovoltaic Inverters, PVSC 2008, Anton Driesse et. al.

    See also
    --------
    sapm
    singlediode
    '''

    p_nom = inverter['Pnom']
    v_nom = inverter['Vnom']
    pac_max = inverter['Pacmax']
    p_nt = inverter['Pnt']
    ce_list = inverter['ADRCoefficients']
    v_max = inverter['Vmax']
    v_min = inverter['Vmin']
    vdc_max = inverter['Vdcmax']
    mppt_hi = inverter['MPPTHi']
    mppt_low = inverter['MPPTLow']

    v_lim_upper = float(np.nanmax([v_max, vdc_max, mppt_hi]) * (1 + vtol))
    v_lim_lower = float(np.nanmax([v_min, mppt_low]) * (1 - vtol))

    pdc = p_dc / p_nom
    vdc = v_dc / v_nom
    # zero voltage will lead to division by zero, but since power is
    # set to night time value later, these errors can be safely ignored
    with np.errstate(invalid='ignore', divide='ignore'):
        poly = np.array([pdc**0,  # replace with np.ones_like?
                         pdc,
                         pdc**2,
                         vdc - 1,
                         pdc * (vdc - 1),
                         pdc**2 * (vdc - 1),
                         1. / vdc - 1,  # divide by 0
                         pdc * (1. / vdc - 1),  # invalid 0./0. --> nan
                         pdc**2 * (1. / vdc - 1)])  # divide by 0
    p_loss = np.dot(np.array(ce_list), poly)
    ac_power = p_nom * (pdc-p_loss)
    p_nt = -1 * np.absolute(p_nt)

    # set output to nan where input is outside of limits
    # errstate silences case where input is nan
    with np.errstate(invalid='ignore'):
        invalid = (v_lim_upper < v_dc) | (v_dc < v_lim_lower)
    ac_power = np.where(invalid, np.nan, ac_power)

    # set night values
    ac_power = np.where(vdc == 0, p_nt, ac_power)
    ac_power = np.maximum(ac_power, p_nt)

    # set max ac output
    ac_power = np.minimum(ac_power, pac_max)

    if isinstance(p_dc, pd.Series):
        ac_power = pd.Series(ac_power, index=pdc.index)

    return ac_power


def scale_voltage_current_power(data, voltage=1, current=1):
    """
    Scales the voltage, current, and power of the DataFrames
    returned by :py:func:`singlediode` and :py:func:`sapm`.

    Parameters
    ----------
    data: DataFrame
        Must contain columns `'v_mp', 'v_oc', 'i_mp' ,'i_x', 'i_xx',
        'i_sc', 'p_mp'`.
    voltage: numeric, default 1
        The amount by which to multiply the voltages.
    current: numeric, default 1
        The amount by which to multiply the currents.

    Returns
    -------
    scaled_data: DataFrame
        A scaled copy of the input data.
        `'p_mp'` is scaled by `voltage * current`.
    """

    # as written, only works with a DataFrame
    # could make it work with a dict, but it would be more verbose
    data = data.copy()
    voltages = ['v_mp', 'v_oc']
    currents = ['i_mp', 'i_x', 'i_xx', 'i_sc']
    data[voltages] *= voltage
    data[currents] *= current
    data['p_mp'] *= voltage * current

    return data


def pvwatts_dc(g_poa_effective, temp_cell, pdc0, gamma_pdc, temp_ref=25.):
    r"""
    Implements NREL's PVWatts DC power model [1]_:

    .. math::

        P_{dc} = \frac{G_{poa eff}}{1000} P_{dc0} ( 1 + \gamma_{pdc} (T_{cell} - T_{ref}))

    Parameters
    ----------
    g_poa_effective: numeric
        Irradiance transmitted to the PV cells in units of W/m**2. To be
        fully consistent with PVWatts, the user must have already
        applied angle of incidence losses, but not soiling, spectral,
        etc.
    temp_cell: numeric
        Cell temperature in degrees C.
    pdc0: numeric
        Nameplate DC rating.
    gamma_pdc: numeric
        The temperature coefficient in units of 1/C. Typically -0.002 to
        -0.005 per degree C.
    temp_ref: numeric, default 25.0
        Cell reference temperature. PVWatts defines it to be 25 C and
        is included here for flexibility.

    Returns
    -------
    pdc: numeric
        DC power.

    References
    ----------
    .. [1] A. P. Dobos, "PVWatts Version 5 Manual"
           http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
           (2014).
    """

    pdc = (g_poa_effective * 0.001 * pdc0 *
           (1 + gamma_pdc * (temp_cell - temp_ref)))

    return pdc


def pvwatts_losses(soiling=2, shading=3, snow=0, mismatch=2, wiring=2,
                   connections=0.5, lid=1.5, nameplate_rating=1, age=0,
                   availability=3):
    r"""
    Implements NREL's PVWatts system loss model [1]_:

    .. math::

        L_{total}(\%) = 100 [ 1 - \Pi_i ( 1 - \frac{L_i}{100} ) ]

    All parameters must be in units of %. Parameters may be
    array-like, though all array sizes must match.

    Parameters
    ----------
    soiling: numeric, default 2
    shading: numeric, default 3
    snow: numeric, default 0
    mismatch: numeric, default 2
    wiring: numeric, default 2
    connections: numeric, default 0.5
    lid: numeric, default 1.5
        Light induced degradation
    nameplate_rating: numeric, default 1
    age: numeric, default 0
    availability: numeric, default 3

    Returns
    -------
    losses: numeric
        System losses in units of %.

    References
    ----------
    .. [1] A. P. Dobos, "PVWatts Version 5 Manual"
           http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
           (2014).
    """

    params = [soiling, shading, snow, mismatch, wiring, connections, lid,
              nameplate_rating, age, availability]

    # manually looping over params allows for numpy/pandas to handle any
    # array-like broadcasting that might be necessary.
    perf = 1
    for param in params:
        perf *= 1 - param/100

    losses = (1 - perf) * 100.

    return losses


def pvwatts_ac(pdc, pdc0, eta_inv_nom=0.96, eta_inv_ref=0.9637):
    r"""
    Implements NREL's PVWatts inverter model [1]_.

    .. math::

        \eta = \frac{\eta_{nom}}{\eta_{ref}} (-0.0162\zeta - \frac{0.0059}{\zeta} + 0.9858)

    .. math::

        P_{ac} = \min(\eta P_{dc}, P_{ac0})

    where :math:`\zeta=P_{dc}/P_{dc0}` and :math:`P_{dc0}=P_{ac0}/\eta_{nom}`.

    Parameters
    ----------
    pdc: numeric
        DC power.
    pdc0: numeric
        Nameplate DC rating.
    eta_inv_nom: numeric, default 0.96
        Nominal inverter efficiency.
    eta_inv_ref: numeric, default 0.9637
        Reference inverter efficiency. PVWatts defines it to be 0.9637
        and is included here for flexibility.

    Returns
    -------
    pac: numeric
        AC power.

    References
    ----------
    .. [1] A. P. Dobos, "PVWatts Version 5 Manual,"
           http://pvwatts.nrel.gov/downloads/pvwattsv5.pdf
           (2014).
    """

    pac0 = eta_inv_nom * pdc0
    zeta = pdc / pdc0

    # arrays to help avoid divide by 0 for scalar and array
    eta = np.zeros_like(pdc, dtype=float)
    pdc_neq_0 = ~np.equal(pdc, 0)

    # eta < 0 if zeta < 0.006. pac is forced to be >= 0 below. GH 541
    eta = eta_inv_nom / eta_inv_ref * (
        - 0.0162*zeta
        - np.divide(0.0059, zeta, out=eta, where=pdc_neq_0)
        + 0.9858)

    pac = eta * pdc
    pac = np.minimum(pac0, pac)
    pac = np.maximum(0, pac)     # GH 541

    return pac
