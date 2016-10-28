"""
The ``pvsystem`` module contains functions for modeling the output and
performance of PV modules and inverters.
"""

from __future__ import division

from collections import OrderedDict
import os
import io
try:
    from urllib2 import urlopen
except ImportError:
    from urllib.request import urlopen

import numpy as np
import pandas as pd

from pvlib import tools
from pvlib.tools import _build_kwargs
from pvlib.location import Location
from pvlib import irradiance, atmosphere


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
    surface_tilt: float or array-like
        Tilt angle of the module surface.
        Up=0, horizon=90.

    surface_azimuth: float or array-like
        Azimuth angle of the module surface.
        North=0, East=90, South=180, West=270.

    albedo : None, float
        The ground albedo. If ``None``, will attempt to use
        ``surface_type`` and ``irradiance.SURFACE_ALBEDOS``
        to lookup albedo.

    surface_type : None, string
        The ground surface type. See ``irradiance.SURFACE_ALBEDOS``
        for valid values.

    module : None, string
        The model name of the modules.
        May be used to look up the module_parameters dictionary
        via some other method.

    module_parameters : None, dict or Series
        Module parameters as defined by the SAPM, CEC, or other.

    modules_per_string: int or float
        See system topology discussion above.

    strings_per_inverter: int or float
        See system topology discussion above.

    inverter : None, string
        The model name of the inverters.
        May be used to look up the inverter_parameters dictionary
        via some other method.

    inverter_parameters : None, dict or Series
        Inverter parameters as defined by the SAPM, CEC, or other.

    racking_model : None or string
        Used for cell and module temperature calculations.

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
                 name=None,
                 **kwargs):

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
        self.module_parameters = module_parameters

        self.modules_per_string = modules_per_string
        self.strings_per_inverter = strings_per_inverter

        self.inverter = inverter
        self.inverter_parameters = inverter_parameters

        self.racking_model = racking_model

        # needed for tying together Location and PVSystem in LocalizedPVSystem
        super(PVSystem, self).__init__(**kwargs)

    def __repr__(self):
        attrs = ['name', 'surface_tilt', 'surface_azimuth', 'module',
                 'inverter', 'albedo', 'racking_model']
        return ('PVSystem: \n  ' + '\n  '.join(
            (attr + ': ' + str(getattr(self, attr)) for attr in attrs)))

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
        Uses the :py:func:`irradiance.total_irrad` function to calculate
        the plane of array irradiance components on a tilted surface
        defined by ``self.surface_tilt``, ``self.surface_azimuth``, and
        ``self.albedo``.

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
        dni_extra : float or Series
            Extraterrestrial direct normal irradiance
        airmass : float or Series
            Airmass
        model : String
            Irradiance model.

        **kwargs
            Passed to :func:`irradiance.total_irrad`.

        Returns
        -------
        poa_irradiance : DataFrame
            Column names are: ``total, beam, sky, ground``.
        """

        # not needed for all models, but this is easier
        if dni_extra is None:
            dni_extra = irradiance.extraradiation(solar_zenith.index)
            dni_extra = pd.Series(dni_extra, index=solar_zenith.index)

        if airmass is None:
            airmass = atmosphere.relativeairmass(solar_zenith)

        return irradiance.total_irrad(self.surface_tilt,
                                      self.surface_azimuth,
                                      solar_zenith, solar_azimuth,
                                      dni, ghi, dhi,
                                      dni_extra=dni_extra, airmass=airmass,
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

    def calcparams_desoto(self, poa_global, temp_cell, **kwargs):
        """
        Use the :py:func:`calcparams_desoto` function, the input
        parameters and ``self.module_parameters`` to calculate the
        module currents and resistances.

        Parameters
        ----------
        poa_global : float or Series
            The irradiance (in W/m^2) absorbed by the module.

        temp_cell : float or Series
            The average cell temperature of cells within a module in C.

        **kwargs
            See pvsystem.calcparams_desoto for details

        Returns
        -------
        See pvsystem.calcparams_desoto for details
        """
        return calcparams_desoto(poa_global, temp_cell,
                                 self.module_parameters['alpha_sc'],
                                 self.module_parameters,
                                 self.module_parameters['EgRef'],
                                 self.module_parameters['dEgdT'], **kwargs)

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

        **kwargs
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

        reference_irradiance : numeric
            Reference irradiance by which to divide the input irradiance.

        Returns
        -------
        effective_irradiance : numeric
            The SAPM effective irradiance.
        """
        return sapm_effective_irradiance(
            poa_direct, poa_diffuse, airmass_absolute, aoi,
            self.module_parameters, reference_irradiance=reference_irradiance)

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

    def pvwatts_losses(self, **kwargs):
        """
        Calculates DC power losses according the PVwatts model using
        :py:func:`pvwatts_losses`. No attributes are used in this
        calculation, but all keyword arguments will be passed to the
        function.

        See :py:func:`pvwatts_losses` for details.
        """
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
        location : None or Location
        latitude : None or float
        longitude : None or float
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

    See the :py:class:`PVSystem` class for an object model that
    describes an unlocalized PV system.
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

        super(LocalizedPVSystem, self).__init__(**new_kwargs)

    def __repr__(self):
        attrs = [
            'name', 'latitude', 'longitude', 'altitude', 'tz', 'surface_tilt',
            'surface_azimuth', 'module', 'inverter', 'albedo', 'racking_model'
                 ]
        return ('LocalizedPVSystem: \n  ' + '\n  '.join(
            (attr + ': ' + str(getattr(self, attr)) for attr in attrs)))


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
        sun-beam vector in degrees.

    b : float
        A parameter to adjust the modifier as a function of angle of
        incidence. Typical values are on the order of 0.05 [3].

    Returns
    -------
    IAM : numeric
        The incident angle modifier calculated as 1-b*(sec(aoi)-1) as
        described in [2,3].

        Returns nan for all abs(aoi) >= 90 and for all IAM values that
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

    iam = 1 - b*((1/np.cos(np.radians(aoi)) - 1))

    iam = np.where(np.abs(aoi) >= 90, np.nan, iam)
    iam = np.maximum(0, iam)

    if isinstance(iam, pd.Series):
        iam = pd.Series(iam, index=aoi.index)

    return iam


def physicaliam(aoi, n=1.526, K=4., L=0.002):
    '''
    Determine the incidence angle modifier using refractive index,
    glazing thickness, and extinction coefficient

    physicaliam calculates the incidence angle modifier as described in
    De Soto et al. "Improvement and validation of a model for
    photovoltaic array performance", section 3. The calculation is based
    upon a physical model of absorbtion and transmission through a
    cover. Required information includes, incident angle, cover
    extinction coefficient, cover thickness

    Note: The authors of this function believe that eqn. 14 in [1] is
    incorrect. This function uses the following equation in its place:
    theta_r = arcsin(1/n * sin(theta))

    Parameters
    ----------
    aoi : numeric
        The angle of incidence between the module normal vector and the
        sun-beam vector in degrees.

    n : numeric
        The effective index of refraction (unitless). Reference [1]
        indicates that a value of 1.526 is acceptable for glass. n must
        be a numeric scalar or vector with all values >=0. If n is a
        vector, it must be the same size as all other input vectors.

    K : numeric
        The glazing extinction coefficient in units of 1/meters.
        Reference [1] indicates that a value of  4 is reasonable for
        "water white" glass. K must be a numeric scalar or vector with
        all values >=0. If K is a vector, it must be the same size as
        all other input vectors.

    L : numeric
        The glazing thickness in units of meters. Reference [1]
        indicates that 0.002 meters (2 mm) is reasonable for most
        glass-covered PV panels. L must be a numeric scalar or vector
        with all values >=0. If L is a vector, it must be the same size
        as all other input vectors.

    Returns
    -------
    IAM : numeric
        The incident angle modifier as specified in eqns. 14-16 of [1].
        IAM is a column vector with the same number of elements as the
        largest input vector.

        Theta must be a numeric scalar or vector. For any values of
        theta where abs(aoi)>90, IAM is set to 0. For any values of aoi
        where -90 < aoi < 0, theta is set to abs(aoi) and evaluated.

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
    thetar_deg = tools.asind(1.0 / n*(tools.sind(aoi)))

    tau = (np.exp(- 1.0 * (K*L / tools.cosd(thetar_deg))) *
           ((1 - 0.5*((((tools.sind(thetar_deg - aoi)) ** 2) /
            ((tools.sind(thetar_deg + aoi)) ** 2) +
            ((tools.tand(thetar_deg - aoi)) ** 2) /
            ((tools.tand(thetar_deg + aoi)) ** 2))))))

    zeroang = 1e-06

    thetar_deg0 = tools.asind(1.0 / n*(tools.sind(zeroang)))

    tau0 = (np.exp(- 1.0 * (K*L / tools.cosd(thetar_deg0))) *
            ((1 - 0.5*((((tools.sind(thetar_deg0 - zeroang)) ** 2) /
             ((tools.sind(thetar_deg0 + zeroang)) ** 2) +
             ((tools.tand(thetar_deg0 - zeroang)) ** 2) /
             ((tools.tand(thetar_deg0 + zeroang)) ** 2))))))

    iam = tau / tau0

    iam = np.where((np.abs(aoi) >= 90) | (iam < 0), np.nan, iam)

    if isinstance(aoi, pd.Series):
        iam = pd.Series(iam, index=aoi.index)

    return iam


def calcparams_desoto(poa_global, temp_cell, alpha_isc, module_parameters,
                      EgRef, dEgdT, M=1, irrad_ref=1000, temp_ref=25):
    '''
    Applies the temperature and irradiance corrections to inputs for
    singlediode.

    Applies the temperature and irradiance corrections to the IL, I0,
    Rs, Rsh, and a parameters at reference conditions (IL_ref, I0_ref,
    etc.) according to the De Soto et. al description given in [1]. The
    results of this correction procedure may be used in a single diode
    model to determine IV curves at irradiance = S, cell temperature =
    Tcell.

    Parameters
    ----------
    poa_global : numeric
        The irradiance (in W/m^2) absorbed by the module.

    temp_cell : numeric
        The average cell temperature of cells within a module in C.

    alpha_isc : float
        The short-circuit current temperature coefficient of the
        module in units of 1/C.

    module_parameters : dict
        Parameters describing PV module performance at reference
        conditions according to DeSoto's paper. Parameters may be
        generated or found by lookup. For ease of use,
        retrieve_sam can automatically generate a dict based on the
        most recent SAM CEC module
        database. The module_parameters dict must contain the
        following 5 fields:

            * a_ref - modified diode ideality factor parameter at
              reference conditions (units of eV), a_ref can be calculated
              from the usual diode ideality factor (n),
              number of cells in series (Ns),
              and cell temperature (Tcell) per equation (2) in [1].
            * I_L_ref - Light-generated current (or photocurrent)
              in amperes at reference conditions. This value is referred to
              as Iph in some literature.
            * I_o_ref - diode reverse saturation current in amperes,
              under reference conditions.
            * R_sh_ref - shunt resistance under reference conditions (ohms).
            * R_s - series resistance under reference conditions (ohms).

    EgRef : float
        The energy bandgap at reference temperature (in eV).
        1.121 eV for silicon. EgRef must be >0.

    dEgdT : float
        The temperature dependence of the energy bandgap at SRC (in
        1/C). May be either a scalar value (e.g. -0.0002677 as in [1])
        or a DataFrame of dEgdT values corresponding to each input
        condition (this may be useful if dEgdT is a function of
        temperature).

    M : numeric (optional, default=1)
        An optional airmass modifier, if omitted, M is given a value of
        1, which assumes absolute (pressure corrected) airmass = 1.5. In
        this code, M is equal to M/Mref as described in [1] (i.e. Mref
        is assumed to be 1). Source [1] suggests that an appropriate
        value for M as a function absolute airmass (AMa) may be:

        >>> M = np.polyval([-0.000126, 0.002816, -0.024459, 0.086257, 0.918093],
        ...                AMa) # doctest: +SKIP

        M may be a Series.

    irrad_ref : float (optional, default=1000)
        Reference irradiance in W/m^2.

    temp_ref : float (optional, default=25)
        Reference cell temperature in C.

    Returns
    -------
    Tuple of the following results:

    photocurrent : numeric
        Light-generated current in amperes at irradiance=S and
        cell temperature=Tcell.

    saturation_current : numeric
        Diode saturation curent in amperes at irradiance
        S and cell temperature Tcell.

    resistance_series : float
        Series resistance in ohms at irradiance S and cell temperature
        Tcell.

    resistance_shunt : numeric
        Shunt resistance in ohms at irradiance S and cell temperature
        Tcell.

    nNsVth : numeric
        Modified diode ideality factor at irradiance S and cell
        temperature Tcell. Note that in source [1] nNsVth = a (equation
        2). nNsVth is the product of the usual diode ideality factor
        (n), the number of series-connected cells in the module (Ns),
        and the thermal voltage of a cell in the module (Vth) at a cell
        temperature of Tcell.

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
    sapm
    sapm_celltemp
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

     Silicon (Si):
         * EgRef = 1.121
         * dEgdT = -0.0002677

         >>> M = np.polyval([-1.26E-4, 2.816E-3, -0.024459, 0.086257, 0.918093],
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

    M = np.maximum(M, 0)
    a_ref = module_parameters['a_ref']
    IL_ref = module_parameters['I_L_ref']
    I0_ref = module_parameters['I_o_ref']
    Rsh_ref = module_parameters['R_sh_ref']
    Rs_ref = module_parameters['R_s']

    k = 8.617332478e-05
    Tref_K = temp_ref + 273.15
    Tcell_K = temp_cell + 273.15

    E_g = EgRef * (1 + dEgdT*(Tcell_K - Tref_K))

    nNsVth = a_ref * (Tcell_K / Tref_K)

    IL = (poa_global/irrad_ref) * M * (IL_ref + alpha_isc * (Tcell_K - Tref_K))
    I0 = (I0_ref * ((Tcell_K / Tref_K) ** 3) *
          (np.exp(EgRef / (k*(Tref_K)) - (E_g / (k*(Tcell_K))))))
    Rsh = Rsh_ref * (irrad_ref / poa_global)
    Rs = Rs_ref

    return IL, I0, Rs, Rsh, nNsVth


def retrieve_sam(name=None, path=None):
    '''
    Retrieve latest module and inverter info from a local file or the
    SAM website.

    This function will retrieve either:

        * CEC module database
        * Sandia Module database
        * CEC Inverter database

    and return it as a pandas DataFrame.

    Parameters
    ----------
    name : None or string
        Name can be one of:

        * 'CECMod' - returns the CEC module database
        * 'CECInverter' - returns the CEC Inverter database
        * 'SandiaInverter' - returns the CEC Inverter database
          (CEC is only current inverter db available; tag kept for
          backwards compatibility)
        * 'SandiaMod' - returns the Sandia Module database

    path : None or string
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
                data_path, 'sam-library-cec-modules-2015-6-30.csv')
        elif name == 'sandiamod':
            csvdata = os.path.join(
                data_path, 'sam-library-sandia-modules-2015-6-30.csv')
        elif name in ['cecinverter', 'sandiainverter']:
            # Allowing either, to provide for old code,
            # while aligning with current expectations
            csvdata = os.path.join(
                data_path, 'sam-library-cec-inverters-2015-6-30.csv')
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

    Ee = effective_irradiance

    Bvmpo = module['Bvmpo'] + module['Mbvmp']*(1 - Ee)
    Bvoco = module['Bvoco'] + module['Mbvoc']*(1 - Ee)
    delta = module['N'] * kb * (temp_cell + 273.15) / q

    out = OrderedDict()

    out['i_sc'] = (
        module['Isco'] * Ee * (1 + module['Aisc']*(temp_cell - T0)))

    out['i_mp'] = (
        module['Impo'] * (module['C0']*Ee + module['C1']*(Ee**2)) *
        (1 + module['Aimp']*(temp_cell - T0)))

    out['v_oc'] = np.maximum(0, (
        module['Voco'] + module['Cells_in_Series']*delta*np.log(Ee) +
        Bvoco*(temp_cell - T0)))

    out['v_mp'] = np.maximum(0, (
        module['Vmpo'] +
        module['C2']*module['Cells_in_Series']*delta*np.log(Ee) +
        module['C3']*module['Cells_in_Series']*((delta*np.log(Ee)) ** 2) +
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

    model : string, list, or dict
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

    temp_models = {'open_rack_cell_glassback': [-3.47, -.0594, 3],
                   'roof_mount_cell_glassback': [-2.98, -.0471, 1],
                   'open_rack_cell_polymerback': [-3.56, -.0750, 3],
                   'insulated_back_polymerback': [-2.81, -.0455, 0],
                   'open_rack_polymer_thinfilm_steel': [-3.58, -.113, 3],
                   '22x_concentrator_tracker': [-3.23, -.130, 13]
                   }

    if isinstance(model, str):
        model = temp_models[model.lower()]
    elif isinstance(model, list):
        model = model
    elif isinstance(model, (dict, pd.Series)):
        model = [model['a'], model['b'], model['deltaT']]

    a = model[0]
    b = model[1]
    deltaT = model[2]

    E0 = 1000.  # Reference irradiance

    temp_module = pd.Series(poa_global*np.exp(a + b*wind_speed) + temp_air)

    temp_cell = temp_module + (poa_global / E0)*(deltaT)

    return pd.DataFrame({'temp_cell': temp_cell, 'temp_module': temp_module})


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

    spectral_loss = np.maximum(0, np.polyval(am_coeff, airmass_absolute))

    spectral_loss = np.where(np.isnan(spectral_loss), 0, spectral_loss)

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
        nan values.

    module : dict-like
        A dict, Series, or DataFrame defining the SAPM performance
        parameters. See the :py:func:`sapm` notes section for more
        details.

    upper : None or float
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
    aoi_loss = np.where(aoi < 0, np.nan, aoi_loss)

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

    reference_irradiance : numeric
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
                resistance_shunt, nNsVth, ivcurve_pnts=None):
    r'''
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

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.

    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).

    ivcurve_pnts : None or int
        Number of points in the desired IV curve. If None or 0, no
        IV curves will be produced.

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
    The solution employed to solve the implicit diode equation utilizes
    the Lambert W function to obtain an explicit function of V=f(i) and
    I=f(V) as shown in [2].

    References
    -----------
    [1] S.R. Wenham, M.A. Green, M.E. Watt, "Applied Photovoltaics" ISBN
    0 86758 909 4

    [2] A. Jain, A. Kapoor, "Exact analytical solutions of the
    parameters of real solar cells using Lambert W-function", Solar
    Energy Materials and Solar Cells, 81 (2004) 269-277.

    [3] D. King et al, "Sandia Photovoltaic Array Performance Model",
    SAND2004-3535, Sandia National Laboratories, Albuquerque, NM

    See also
    --------
    sapm
    calcparams_desoto
    '''

    # Find short circuit current using Lambert W
    i_sc = i_from_v(resistance_shunt, resistance_series, nNsVth, 0.01,
                    saturation_current, photocurrent)

    # Find open circuit voltage using Lambert W
    v_oc = v_from_i(resistance_shunt, resistance_series, nNsVth, 0.0,
                    saturation_current, photocurrent)

    params = {'r_sh': resistance_shunt,
              'r_s': resistance_series,
              'nNsVth': nNsVth,
              'i_0': saturation_current,
              'i_l': photocurrent}

    p_mp, v_mp = _golden_sect_DataFrame(params, 0, v_oc*1.14, _pwr_optfcn)

    # Invert the Power-Current curve. Find the current where the inverted power
    # is minimized. This is i_mp. Start the optimization at v_oc/2
    i_mp = i_from_v(resistance_shunt, resistance_series, nNsVth, v_mp,
                    saturation_current, photocurrent)

    # Find Ix and Ixx using Lambert W
    i_x = i_from_v(resistance_shunt, resistance_series, nNsVth,
                   0.5*v_oc, saturation_current, photocurrent)

    i_xx = i_from_v(resistance_shunt, resistance_series, nNsVth,
                    0.5*(v_oc+v_mp), saturation_current, photocurrent)

    out = OrderedDict()
    out['i_sc'] = i_sc
    out['v_oc'] = v_oc
    out['i_mp'] = i_mp
    out['v_mp'] = v_mp
    out['p_mp'] = p_mp
    out['i_x'] = i_x
    out['i_xx'] = i_xx

    # create ivcurve
    if ivcurve_pnts:
        ivcurve_v = (np.asarray(v_oc)[..., np.newaxis] *
                     np.linspace(0, 1, ivcurve_pnts))
        ivcurve_i = i_from_v(
            resistance_shunt, resistance_series, nNsVth, ivcurve_v.T,
            saturation_current, photocurrent).T
        out['v'] = ivcurve_v
        out['i'] = ivcurve_i

    if isinstance(photocurrent, pd.Series) and not ivcurve_pnts:
        out = pd.DataFrame(out, index=photocurrent.index)

    return out


# Created April,2014
# Author: Rob Andrews, Calama Consulting

def _golden_sect_DataFrame(params, VL, VH, func):
    '''
    Vectorized golden section search for finding MPPT
    from a dataframe timeseries.

    Parameters
    ----------
    params : dict
        Dictionary containing scalars or arrays
        of inputs to the function to be optimized.
        Each row should represent an independent optimization.

    VL: float
        Lower bound of the optimization

    VH: float
        Upper bound of the optimization

    func: function
        Function to be optimized must be in the form f(array-like, x)

    Returns
    -------
    func(df,'V1') : DataFrame
        function evaluated at the optimal point

    df['V1']: Dataframe
        Dataframe of optimal points

    Notes
    -----
    This funtion will find the MAXIMUM of a function
    '''

    df = params
    df['VH'] = VH
    df['VL'] = VL

    err = df['VH'] - df['VL']
    errflag = True
    iterations = 0

    while errflag:

        phi = (np.sqrt(5)-1)/2*(df['VH']-df['VL'])
        df['V1'] = df['VL'] + phi
        df['V2'] = df['VH'] - phi

        df['f1'] = func(df, 'V1')
        df['f2'] = func(df, 'V2')
        df['SW_Flag'] = df['f1'] > df['f2']

        df['VL'] = df['V2']*df['SW_Flag'] + df['VL']*(~df['SW_Flag'])
        df['VH'] = df['V1']*~df['SW_Flag'] + df['VH']*(df['SW_Flag'])

        err = df['V1'] - df['V2']
        try:
            errflag = (abs(err) > .01).any()
        except ValueError:
            errflag = (abs(err) > .01)

        iterations += 1

        if iterations > 50:
            raise Exception("EXCEPTION:iterations exeeded maximum (50)")

    return func(df, 'V1'), df['V1']


def _pwr_optfcn(df, loc):
    '''
    Function to find power from ``i_from_v``.
    '''

    I = i_from_v(df['r_sh'], df['r_s'], df['nNsVth'],
                 df[loc], df['i_0'], df['i_l'])
    return I*df[loc]


def v_from_i(resistance_shunt, resistance_series, nNsVth, current,
             saturation_current, photocurrent):
    '''
    Calculates voltage from current per Eq 3 Jain and Kapoor 2004 [1].

    Parameters
    ----------
    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).

    current : numeric
        The current in amperes under desired IV curve conditions.

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.

    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        IV curve conditions. Often abbreviated ``I_L``.

    Returns
    -------
    current : np.array

    References
    ----------
    [1] A. Jain, A. Kapoor, "Exact analytical solutions of the
    parameters of real solar cells using Lambert W-function", Solar
    Energy Materials and Solar Cells, 81 (2004) 269-277.
    '''
    try:
        from scipy.special import lambertw
    except ImportError:
        raise ImportError('This function requires scipy')

    Rsh = resistance_shunt
    Rs = resistance_series
    I0 = saturation_current
    IL = photocurrent
    I = current

    argW = I0 * Rsh / nNsVth * np.exp(Rsh * (-I + IL + I0) / nNsVth)
    lambertwterm = lambertw(argW).real

    # Calculate using log(argW) in case argW is really big
    logargW = (np.log(I0) + np.log(Rsh) - np.log(nNsVth) +
               Rsh * (-I + IL + I0) / nNsVth)

    # Three iterations of Newton-Raphson method to solve
    # w+log(w)=logargW. The initial guess is w=logargW. Where direct
    # evaluation (above) results in NaN from overflow, 3 iterations
    # of Newton's method gives approximately 8 digits of precision.
    w = logargW
    for i in range(0, 3):
        w = w * (1 - np.log(w) + logargW) / (1 + w)
    lambertwterm_log = w

    lambertwterm = np.where(np.isfinite(lambertwterm), lambertwterm,
                            lambertwterm_log)

    # Eqn. 3 in Jain and Kapoor, 2004
    V = -I*(Rs + Rsh) + IL*Rsh - nNsVth*lambertwterm + I0*Rsh

    return V


def i_from_v(resistance_shunt, resistance_series, nNsVth, voltage,
             saturation_current, photocurrent):
    '''
    Calculates current from voltage per Eq 2 Jain and Kapoor 2004 [1].

    Parameters
    ----------
    resistance_shunt : numeric
        Shunt resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rsh``.

    resistance_series : numeric
        Series resistance in ohms under desired IV curve conditions.
        Often abbreviated ``Rs``.

    nNsVth : numeric
        The product of three components. 1) The usual diode ideal factor
        (n), 2) the number of cells in series (Ns), and 3) the cell
        thermal voltage under the desired IV curve conditions (Vth). The
        thermal voltage of the cell (in volts) may be calculated as
        ``k*temp_cell/q``, where k is Boltzmann's constant (J/K),
        temp_cell is the temperature of the p-n junction in Kelvin, and
        q is the charge of an electron (coulombs).

    voltage : numeric
        The voltage in Volts under desired IV curve conditions.

    saturation_current : numeric
        Diode saturation current in amperes under desired IV curve
        conditions. Often abbreviated ``I_0``.

    photocurrent : numeric
        Light-generated current (photocurrent) in amperes under desired
        IV curve conditions. Often abbreviated ``I_L``.

    Returns
    -------
    current : np.array

    References
    ----------
    [1] A. Jain, A. Kapoor, "Exact analytical solutions of the
    parameters of real solar cells using Lambert W-function", Solar
    Energy Materials and Solar Cells, 81 (2004) 269-277.
    '''
    try:
        from scipy.special import lambertw
    except ImportError:
        raise ImportError('This function requires scipy')

    # asarray turns Series into arrays so that we don't have to worry
    # about multidimensional broadcasting failing
    Rsh = np.asarray(resistance_shunt)
    Rs = np.asarray(resistance_series)
    I0 = np.asarray(saturation_current)
    IL = np.asarray(photocurrent)
    V = np.asarray(voltage)

    argW = (Rs*I0*Rsh *
            np.exp(Rsh*(Rs*(IL+I0)+V) / (nNsVth*(Rs+Rsh))) /
            (nNsVth*(Rs + Rsh)))
    lambertwterm = lambertw(argW).real

    # Eqn. 4 in Jain and Kapoor, 2004
    I = -V/(Rs + Rsh) - (nNsVth/Rs)*lambertwterm + Rsh*(IL + I0)/(Rs + Rsh)

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


def scale_voltage_current_power(data, voltage=1, current=1):
    """
    Scales the voltage, current, and power of the DataFrames
    returned by :py:func:`singlediode` and :py:func:`sapm`.

    Parameters
    ----------
    data: DataFrame
        Must contain columns `'v_mp', 'v_oc', 'i_mp' ,'i_x', 'i_xx',
        'i_sc', 'p_mp'`.
    voltage: numeric
        The amount by which to multiply the voltages.
    current: numeric
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
    temp_ref: numeric
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
    soiling: numeric
    shading: numeric
    snow: numeric
    mismatch: numeric
    wiring: numeric
    connections: numeric
    lid: numeric
        Light induced degradation
    nameplate_rating: numeric
    age: numeric
    availability: numeric

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
    eta_inv_nom: numeric
        Nominal inverter efficiency.
    eta_inv_ref: numeric
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

    eta = eta_inv_nom / eta_inv_ref * (-0.0162*zeta - 0.0059/zeta + 0.9858)

    pac = eta * pdc
    pac = np.minimum(pac0, pac)

    return pac
