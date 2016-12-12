"""
The ``irradiance`` module contains functions for modeling global
horizontal irradiance, direct normal irradiance, diffuse horizontal
irradiance, and total irradiance under various conditions.
"""

from __future__ import division

import logging

import datetime
from collections import OrderedDict
from functools import partial

import numpy as np
import pandas as pd

from pvlib import tools
from pvlib import solarposition
from pvlib import atmosphere

pvl_logger = logging.getLogger('pvlib')

SURFACE_ALBEDOS = {'urban': 0.18,
                   'grass': 0.20,
                   'fresh grass': 0.26,
                   'soil': 0.17,
                   'sand': 0.40,
                   'snow': 0.65,
                   'fresh snow': 0.75,
                   'asphalt': 0.12,
                   'concrete': 0.30,
                   'aluminum': 0.85,
                   'copper': 0.74,
                   'fresh steel': 0.35,
                   'dirty steel': 0.08}


def extraradiation(datetime_or_doy, solar_constant=1366.1, method='spencer',
                   epoch_year=2014, **kwargs):
    """
    Determine extraterrestrial radiation from day of year.

    Parameters
    ----------
    datetime_or_doy : numeric, array, date, datetime, Timestamp, DatetimeIndex
        Day of year, array of days of year, or datetime-like object

    solar_constant : float
        The solar constant.

    method : string
        The method by which the ET radiation should be calculated.
        Options include ``'pyephem', 'spencer', 'asce', 'nrel'``.

    epoch_year : int
        The year in which a day of year input will be calculated. Only
        applies to day of year input used with the pyephem or nrel
        methods.

    kwargs :
        Passed to solarposition.nrel_earthsun_distance

    Returns
    -------
    dni_extra : float, array, or Series
        The extraterrestrial radiation present in watts per square meter
        on a surface which is normal to the sun. Pandas Timestamp and
        DatetimeIndex inputs will yield a Pandas TimeSeries. All other
        inputs will yield a float or an array of floats.

    References
    ----------
    [1] M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance
    Clear Sky Models: Implementation and Analysis", Sandia National
    Laboratories, SAND2012-2389, 2012.

    [2] <http://solardat.uoregon.edu/SolarRadiationBasics.html>, Eqs.
    SR1 and SR2

    [3] Partridge, G. W. and Platt, C. M. R. 1976. Radiative Processes
    in Meteorology and Climatology.

    [4] Duffie, J. A. and Beckman, W. A. 1991. Solar Engineering of
    Thermal Processes, 2nd edn. J. Wiley and Sons, New York.
    """

    # This block will set the functions that can be used to convert the
    # inputs to either day of year or pandas DatetimeIndex, and the
    # functions that will yield the appropriate output type. It's
    # complicated because there are many day-of-year-like input types,
    # and the different algorithms need different types. Maybe you have
    # a better way to do it.
    if isinstance(datetime_or_doy, pd.DatetimeIndex):
        to_doy = tools._pandas_to_doy  # won't be evaluated unless necessary
        to_datetimeindex = lambda x: datetime_or_doy
        to_output = partial(pd.Series, index=datetime_or_doy)
    elif isinstance(datetime_or_doy, pd.Timestamp):
        to_doy = tools._pandas_to_doy
        to_datetimeindex = \
            tools._datetimelike_scalar_to_datetimeindex
        to_output = tools._scalar_out
    elif isinstance(datetime_or_doy,
                    (datetime.date, datetime.datetime, np.datetime64)):
        to_doy = tools._datetimelike_scalar_to_doy
        to_datetimeindex = \
            tools._datetimelike_scalar_to_datetimeindex
        to_output = tools._scalar_out
    elif np.isscalar(datetime_or_doy):  # ints and floats of various types
        to_doy = lambda x: datetime_or_doy
        to_datetimeindex = partial(tools._doy_to_datetimeindex,
                                   epoch_year=epoch_year)
        to_output = tools._scalar_out
    else:  # assume that we have an array-like object of doy
        to_doy = lambda x: datetime_or_doy
        to_datetimeindex = partial(tools._doy_to_datetimeindex,
                                   epoch_year=epoch_year)
        to_output = tools._array_out

    method = method.lower()
    if method == 'asce':
        B = solarposition._calculate_simple_day_angle(to_doy(datetime_or_doy))
        RoverR0sqrd = 1 + 0.033 * np.cos(B)
    elif method == 'spencer':
        B = solarposition._calculate_simple_day_angle(to_doy(datetime_or_doy))
        RoverR0sqrd = (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) +
                       0.000719 * np.cos(2 * B) + 7.7e-05 * np.sin(2 * B))
    elif method == 'pyephem':
        times = to_datetimeindex(datetime_or_doy)
        RoverR0sqrd = solarposition.pyephem_earthsun_distance(times) ** (-2)
    elif method == 'nrel':
        times = to_datetimeindex(datetime_or_doy)
        RoverR0sqrd = \
            solarposition.nrel_earthsun_distance(times, **kwargs) ** (-2)
    else:
        raise ValueError('Invalid method: %s', method)

    Ea = solar_constant * RoverR0sqrd

    Ea = to_output(Ea)

    return Ea


def aoi_projection(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    """
    Calculates the dot product of the solar vector and the surface
    normal.

    Input all angles in degrees.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal.
    surface_azimuth : numeric
        Panel azimuth from north.
    solar_zenith : numeric
        Solar zenith angle.
    solar_azimuth : numeric
        Solar azimuth angle.

    Returns
    -------
    projection : numeric
        Dot product of panel normal and solar angle.
    """

    projection = (
        tools.cosd(surface_tilt) * tools.cosd(solar_zenith) +
        tools.sind(surface_tilt) * tools.sind(solar_zenith) *
        tools.cosd(solar_azimuth - surface_azimuth))

    try:
        projection.name = 'aoi_projection'
    except AttributeError:
        pass

    return projection


def aoi(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth):
    """
    Calculates the angle of incidence of the solar vector on a surface.
    This is the angle between the solar vector and the surface normal.

    Input all angles in degrees.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal.
    surface_azimuth : numeric
        Panel azimuth from north.
    solar_zenith : numeric
        Solar zenith angle.
    solar_azimuth : numeric
        Solar azimuth angle.

    Returns
    -------
    aoi : numeric
        Angle of incidence in degrees.
    """

    projection = aoi_projection(surface_tilt, surface_azimuth,
                                solar_zenith, solar_azimuth)
    aoi_value = np.rad2deg(np.arccos(projection))

    try:
        aoi_value.name = 'aoi'
    except AttributeError:
        pass

    return aoi_value


def poa_horizontal_ratio(surface_tilt, surface_azimuth,
                         solar_zenith, solar_azimuth):
    """
    Calculates the ratio of the beam components of the plane of array
    irradiance and the horizontal irradiance.

    Input all angles in degrees.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal.
    surface_azimuth : numeric
        Panel azimuth from north.
    solar_zenith : numeric
        Solar zenith angle.
    solar_azimuth : numeric
        Solar azimuth angle.

    Returns
    -------
    ratio : numeric
        Ratio of the plane of array irradiance to the horizontal plane
        irradiance
    """

    cos_poa_zen = aoi_projection(surface_tilt, surface_azimuth,
                                 solar_zenith, solar_azimuth)

    cos_solar_zenith = tools.cosd(solar_zenith)

    # ratio of titled and horizontal beam irradiance
    ratio = cos_poa_zen / cos_solar_zenith

    try:
        ratio.name = 'poa_ratio'
    except AttributeError:
        pass

    return ratio


def beam_component(surface_tilt, surface_azimuth, solar_zenith, solar_azimuth,
                   dni):
    """
    Calculates the beam component of the plane of array irradiance.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal.
    surface_azimuth : numeric
        Panel azimuth from north.
    solar_zenith : numeric
        Solar zenith angle.
    solar_azimuth : numeric
        Solar azimuth angle.
    dni : numeric
        Direct Normal Irradiance

    Returns
    -------
    beam : numeric
        Beam component
    """
    beam = dni * aoi_projection(surface_tilt, surface_azimuth,
                                solar_zenith, solar_azimuth)
    beam = np.maximum(beam, 0)

    return beam


def total_irrad(surface_tilt, surface_azimuth,
                apparent_zenith, azimuth,
                dni, ghi, dhi, dni_extra=None, airmass=None,
                albedo=.25, surface_type=None,
                model='isotropic',
                model_perez='allsitescomposite1990', **kwargs):
    r"""
    Determine diffuse irradiance from the sky on a tilted surface.

    .. math::

       I_{tot} = I_{beam} + I_{sky} + I_{ground}

    See the transposition function documentation for details.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal.
    surface_azimuth : numeric
        Panel azimuth from north.
    solar_zenith : numeric
        Solar zenith angle.
    solar_azimuth : numeric
        Solar azimuth angle.
    dni : numeric
        Direct Normal Irradiance
    ghi : numeric
        Global horizontal irradiance
    dhi : numeric
        Diffuse horizontal irradiance
    dni_extra : numeric
        Extraterrestrial direct normal irradiance
    airmass : numeric
        Airmass
    albedo : numeric
        Surface albedo
    surface_type : String
        Surface type. See grounddiffuse.
    model : String
        Irradiance model.
    model_perez : String
        See perez.

    Returns
    -------
    irradiance : OrderedDict or DataFrame
        Contains keys/columns ``'poa_global', 'poa_direct',
        'poa_sky_diffuse', 'poa_ground_diffuse'``.
    """

    pvl_logger.debug('planeofarray.total_irrad()')

    solar_zenith = apparent_zenith
    solar_azimuth = azimuth

    beam = beam_component(surface_tilt, surface_azimuth,
                          solar_zenith, solar_azimuth, dni)

    model = model.lower()
    if model == 'isotropic':
        sky = isotropic(surface_tilt, dhi)
    elif model in ['klucher', 'klutcher']:
        sky = klucher(surface_tilt, surface_azimuth, dhi, ghi,
                      solar_zenith, solar_azimuth)
    elif model == 'haydavies':
        sky = haydavies(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                        solar_zenith, solar_azimuth)
    elif model == 'reindl':
        sky = reindl(surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
                     solar_zenith, solar_azimuth)
    elif model == 'king':
        sky = king(surface_tilt, dhi, ghi, solar_zenith)
    elif model == 'perez':
        sky = perez(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
                    solar_zenith, solar_azimuth, airmass,
                    model=model_perez)
    else:
        raise ValueError('invalid model selection {}'.format(model))

    ground = grounddiffuse(surface_tilt, ghi, albedo, surface_type)

    diffuse = sky + ground
    total = beam + diffuse

    all_irrad = OrderedDict()
    all_irrad['poa_global'] = total
    all_irrad['poa_direct'] = beam
    all_irrad['poa_diffuse'] = diffuse
    all_irrad['poa_sky_diffuse'] = sky
    all_irrad['poa_ground_diffuse'] = ground

    if isinstance(total, pd.Series):
        all_irrad = pd.DataFrame(all_irrad)

    return all_irrad


def globalinplane(aoi, dni, poa_sky_diffuse, poa_ground_diffuse):
    r'''
    Determine the three components on in-plane irradiance

    Combines in-plane irradaince compoents from the chosen diffuse
    translation, ground reflection and beam irradiance algorithms into
    the total in-plane irradiance.

    Parameters
    ----------
    aoi : numeric
        Angle of incidence of solar rays with respect to the module
        surface, from :func:`aoi`.

    dni : numeric
        Direct normal irradiance (W/m^2), as measured from a TMY file or
        calculated with a clearsky model.

    poa_sky_diffuse : numeric
        Diffuse irradiance (W/m^2) in the plane of the modules, as
        calculated by a diffuse irradiance translation function

    poa_ground_diffuse : numeric
        Ground reflected irradiance (W/m^2) in the plane of the modules,
        as calculated by an albedo model (eg. :func:`grounddiffuse`)

    Returns
    -------
    irrads : OrderedDict or DataFrame
        Contains the following keys:

        * ``poa_global`` : Total in-plane irradiance (W/m^2)
        * ``poa_direct`` : Total in-plane beam irradiance (W/m^2)
        * ``poa_diffuse`` : Total in-plane diffuse irradiance (W/m^2)

    Notes
    ------
    Negative beam irradiation due to aoi :math:`> 90^{\circ}` or AOI
    :math:`< 0^{\circ}` is set to zero.
    '''

    poa_direct = np.maximum(dni * np.cos(np.radians(aoi)), 0)
    poa_global = poa_direct + poa_sky_diffuse + poa_ground_diffuse
    poa_diffuse = poa_sky_diffuse + poa_ground_diffuse

    irrads = OrderedDict()
    irrads['poa_global'] = poa_global
    irrads['poa_direct'] = poa_direct
    irrads['poa_diffuse'] = poa_diffuse

    if isinstance(poa_direct, pd.Series):
        irrads = pd.DataFrame(irrads)

    return irrads


def grounddiffuse(surface_tilt, ghi, albedo=.25, surface_type=None):
    '''
    Estimate diffuse irradiance from ground reflections given
    irradiance, albedo, and surface tilt

    Function to determine the portion of irradiance on a tilted surface
    due to ground reflections. Any of the inputs may be DataFrames or
    scalars.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. Tilt must be >=0 and
        <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90).

    ghi : numeric
        Global horizontal irradiance in W/m^2.

    albedo : numeric
        Ground reflectance, typically 0.1-0.4 for surfaces on Earth
        (land), may increase over snow, ice, etc. May also be known as
        the reflection coefficient. Must be >=0 and <=1. Will be
        overridden if surface_type is supplied.

    surface_type: None or string
        If not None, overrides albedo. String can be one of ``'urban',
        'grass', 'fresh grass', 'snow', 'fresh snow', 'asphalt',
        'concrete', 'aluminum', 'copper', 'fresh steel', 'dirty steel'``.

    Returns
    -------
    grounddiffuse : numeric
        Ground reflected irradiances in W/m^2.


    References
    ----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267.

    The calculation is the last term of equations 3, 4, 7, 8, 10, 11, and 12.

    [2] albedos from:
    http://pvpmc.org/modeling-steps/incident-irradiance/plane-of-array-poa-irradiance/calculating-poa-irradiance/poa-ground-reflected/albedo/
    and
    http://en.wikipedia.org/wiki/Albedo
    '''

    pvl_logger.debug('diffuse_ground.get_diffuse_ground()')

    if surface_type is not None:
        albedo = SURFACE_ALBEDOS[surface_type]
        pvl_logger.info('surface_type=%s mapped to albedo=%s',
                        surface_type, albedo)

    diffuse_irrad = ghi * albedo * (1 - np.cos(np.radians(surface_tilt))) * 0.5

    try:
        diffuse_irrad.name = 'diffuse_ground'
    except AttributeError:
        pass

    return diffuse_irrad


def isotropic(surface_tilt, dhi):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using
    the isotropic sky model.

    .. math::

       I_{d} = DHI \frac{1 + \cos\beta}{2}

    Hottel and Woertz's model treats the sky as a uniform source of
    diffuse irradiance. Thus the diffuse irradiance from the sky (ground
    reflected irradiance is not included in this algorithm) on a tilted
    surface can be found from the diffuse horizontal irradiance and the
    tilt angle of the surface.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in decimal degrees. Tilt must be >=0 and
        <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    dhi : numeric
        Diffuse horizontal irradiance in W/m^2. DHI must be >=0.

    Returns
    -------
    diffuse : numeric
        The sky diffuse component of the solar radiation.

    References
    ----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to
    compute solar irradiance on inclined surfaces for building energy
    simulation" 2007, Solar Energy vol. 81. pp. 254-267

    [2] Hottel, H.C., Woertz, B.B., 1942. Evaluation of flat-plate solar
    heat collector. Trans. ASME 64, 91.
    '''

    pvl_logger.debug('diffuse_sky.isotropic()')

    sky_diffuse = dhi * (1 + tools.cosd(surface_tilt)) * 0.5

    return sky_diffuse


def klucher(surface_tilt, surface_azimuth, dhi, ghi, solar_zenith,
            solar_azimuth):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface
    using Klucher's 1979 model

    .. math::

       I_{d} = DHI \frac{1 + \cos\beta}{2} (1 + F' \sin^3(\beta/2))
       (1 + F' \cos^2\theta\sin^3\theta_z)

    where

    .. math::

       F' = 1 - (I_{d0} / GHI)

    Klucher's 1979 model determines the diffuse irradiance from the sky
    (ground reflected irradiance is not included in this algorithm) on a
    tilted surface using the surface tilt angle, surface azimuth angle,
    diffuse horizontal irradiance, direct normal irradiance, global
    horizontal irradiance, extraterrestrial irradiance, sun zenith
    angle, and sun azimuth angle.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. surface_tilt must be >=0
        and <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. surface_azimuth must
        be >=0 and <=360. The Azimuth convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).

    dhi : numeric
        Diffuse horizontal irradiance in W/m^2. DHI must be >=0.

    ghi : numeric
        Global irradiance in W/m^2. DNI must be >=0.

    solar_zenith : numeric
        Apparent (refraction-corrected) zenith angles in decimal
        degrees. solar_zenith must be >=0 and <=180.

    solar_azimuth : numeric
        Sun azimuth angles in decimal degrees. solar_azimuth must be >=0
        and <=360. The Azimuth convention is defined as degrees east of
        north (e.g. North = 0, East = 90, West = 270).

    Returns
    -------
    diffuse : numeric
        The sky diffuse component of the solar radiation.

    References
    ----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267

    [2] Klucher, T.M., 1979. Evaluation of models to predict insolation on
    tilted surfaces. Solar Energy 23 (2), 111-114.
    '''

    pvl_logger.debug('diffuse_sky.klucher()')

    # zenith angle with respect to panel normal.
    cos_tt = aoi_projection(surface_tilt, surface_azimuth,
                            solar_zenith, solar_azimuth)

    F = 1 - ((dhi / ghi) ** 2)
    try:
        # fails with single point input
        F.fillna(0, inplace=True)
    except AttributeError:
        F = 0

    term1 = 0.5 * (1 + tools.cosd(surface_tilt))
    term2 = 1 + F * (tools.sind(0.5 * surface_tilt) ** 3)
    term3 = 1 + F * (cos_tt ** 2) * (tools.sind(solar_zenith) ** 3)

    sky_diffuse = dhi * term1 * term2 * term3

    return sky_diffuse


def haydavies(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
              solar_zenith=None, solar_azimuth=None, projection_ratio=None):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using
    Hay & Davies' 1980 model

    .. math::
        I_{d} = DHI ( A R_b + (1 - A) (\frac{1 + \cos\beta}{2}) )

    Hay and Davies' 1980 model determines the diffuse irradiance from
    the sky (ground reflected irradiance is not included in this
    algorithm) on a tilted surface using the surface tilt angle, surface
    azimuth angle, diffuse horizontal irradiance, direct normal
    irradiance, extraterrestrial irradiance, sun zenith angle, and sun
    azimuth angle.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. The tilt angle is
        defined as degrees from horizontal (e.g. surface facing up = 0,
        surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. The azimuth
        convention is defined as degrees east of north (e.g. North=0,
        South=180, East=90, West=270).

    dhi : numeric
        Diffuse horizontal irradiance in W/m^2.

    dni : numeric
        Direct normal irradiance in W/m^2.

    dni_extra : numeric
        Extraterrestrial normal irradiance in W/m^2.

    solar_zenith : None or numeric
        Solar apparent (refraction-corrected) zenith angles in decimal
        degrees. Must supply ``solar_zenith`` and ``solar_azimuth`` or
        supply ``projection_ratio``.

    solar_azimuth : None or numeric
        Solar azimuth angles in decimal degrees. Must supply
        ``solar_zenith`` and ``solar_azimuth`` or supply
        ``projection_ratio``.

    projection_ratio : None or numeric
        Ratio of angle of incidence projection to solar zenith angle
        projection. Must supply ``solar_zenith`` and ``solar_azimuth``
        or supply ``projection_ratio``.

    Returns
    --------
    sky_diffuse : numeric
        The sky diffuse component of the solar radiation.

    References
    -----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to
    compute solar irradiance on inclined surfaces for building energy
    simulation" 2007, Solar Energy vol. 81. pp. 254-267

    [2] Hay, J.E., Davies, J.A., 1980. Calculations of the solar
    radiation incident on an inclined surface. In: Hay, J.E., Won, T.K.
    (Eds.), Proc. of First Canadian Solar Radiation Data Workshop, 59.
    Ministry of Supply and Services, Canada.
    '''

    pvl_logger.debug('diffuse_sky.haydavies()')

    # if necessary, calculate ratio of titled and horizontal beam irradiance
    if projection_ratio is None:
        cos_tt = aoi_projection(surface_tilt, surface_azimuth,
                                solar_zenith, solar_azimuth)
        cos_solar_zenith = tools.cosd(solar_zenith)
        Rb = cos_tt / cos_solar_zenith
    else:
        Rb = projection_ratio

    # Anisotropy Index
    AI = dni / dni_extra

    # these are the () and [] sub-terms of the second term of eqn 7
    term1 = 1 - AI
    term2 = 0.5 * (1 + tools.cosd(surface_tilt))

    sky_diffuse = dhi * (AI * Rb + term1 * term2)
    sky_diffuse = np.maximum(sky_diffuse, 0)

    return sky_diffuse


def reindl(surface_tilt, surface_azimuth, dhi, dni, ghi, dni_extra,
           solar_zenith, solar_azimuth):
    r'''
    Determine diffuse irradiance from the sky on a tilted surface using
    Reindl's 1990 model

    .. math::

       I_{d} = DHI (A R_b + (1 - A) (\frac{1 + \cos\beta}{2})
       (1 + \sqrt{\frac{I_{hb}}{I_h}} \sin^3(\beta/2)) )

    Reindl's 1990 model determines the diffuse irradiance from the sky
    (ground reflected irradiance is not included in this algorithm) on a
    tilted surface using the surface tilt angle, surface azimuth angle,
    diffuse horizontal irradiance, direct normal irradiance, global
    horizontal irradiance, extraterrestrial irradiance, sun zenith
    angle, and sun azimuth angle.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. The tilt angle is
        defined as degrees from horizontal (e.g. surface facing up = 0,
        surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. The azimuth
        convention is defined as degrees east of north (e.g. North = 0,
        South=180 East = 90, West = 270).

    dhi : numeric
        diffuse horizontal irradiance in W/m^2.

    dni : numeric
        direct normal irradiance in W/m^2.

    ghi: numeric
        Global irradiance in W/m^2.

    dni_extra : numeric
        Extraterrestrial normal irradiance in W/m^2.

    solar_zenith : numeric
        Apparent (refraction-corrected) zenith angles in decimal degrees.

    solar_azimuth : numeric
        Sun azimuth angles in decimal degrees. The azimuth convention is
        defined as degrees east of north (e.g. North = 0, East = 90,
        West = 270).

    Returns
    -------
    poa_sky_diffuse : numeric
        The sky diffuse component of the solar radiation.

    Notes
    -----
    The poa_sky_diffuse calculation is generated from the Loutzenhiser et al.
    (2007) paper, equation 8. Note that I have removed the beam and ground
    reflectance portion of the equation and this generates ONLY the diffuse
    radiation from the sky and circumsolar, so the form of the equation
    varies slightly from equation 8.

    References
    ----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to
    compute solar irradiance on inclined surfaces for building energy
    simulation" 2007, Solar Energy vol. 81. pp. 254-267

    [2] Reindl, D.T., Beckmann, W.A., Duffie, J.A., 1990a. Diffuse
    fraction correlations. Solar Energy 45(1), 1-7.

    [3] Reindl, D.T., Beckmann, W.A., Duffie, J.A., 1990b. Evaluation of
    hourly tilted surface radiation models. Solar Energy 45(1), 9-17.
    '''

    pvl_logger.debug('diffuse_sky.reindl()')

    cos_tt = aoi_projection(surface_tilt, surface_azimuth,
                            solar_zenith, solar_azimuth)

    cos_solar_zenith = tools.cosd(solar_zenith)

    # ratio of titled and horizontal beam irradiance
    Rb = cos_tt / cos_solar_zenith

    # Anisotropy Index
    AI = dni / dni_extra

    # DNI projected onto horizontal
    HB = dni * cos_solar_zenith
    HB = np.maximum(HB, 0)

    # these are the () and [] sub-terms of the second term of eqn 8
    term1 = 1 - AI
    term2 = 0.5 * (1 + tools.cosd(surface_tilt))
    term3 = 1 + np.sqrt(HB / ghi) * (tools.sind(0.5 * surface_tilt) ** 3)

    sky_diffuse = dhi * (AI * Rb + term1 * term2 * term3)
    sky_diffuse = np.maximum(sky_diffuse, 0)

    return sky_diffuse


def king(surface_tilt, dhi, ghi, solar_zenith):
    '''
    Determine diffuse irradiance from the sky on a tilted surface using
    the King model.

    King's model determines the diffuse irradiance from the sky (ground
    reflected irradiance is not included in this algorithm) on a tilted
    surface using the surface tilt angle, diffuse horizontal irradiance,
    global horizontal irradiance, and sun zenith angle. Note that this
    model is not well documented and has not been published in any
    fashion (as of January 2012).

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. The tilt angle is
        defined as degrees from horizontal (e.g. surface facing up = 0,
        surface facing horizon = 90)

    dhi : numeric
        Diffuse horizontal irradiance in W/m^2.

    ghi : numeric
        Global horizontal irradiance in W/m^2.

    solar_zenith : numeric
        Apparent (refraction-corrected) zenith angles in decimal degrees.

    Returns
    --------
    poa_sky_diffuse : numeric
        The diffuse component of the solar radiation.
    '''

    pvl_logger.debug('diffuse_sky.king()')

    sky_diffuse = (dhi * ((1 + tools.cosd(surface_tilt))) / 2 + ghi *
                   ((0.012 * solar_zenith - 0.04)) *
                   ((1 - tools.cosd(surface_tilt))) / 2)
    sky_diffuse = np.maximum(sky_diffuse, 0)

    return sky_diffuse


def perez(surface_tilt, surface_azimuth, dhi, dni, dni_extra,
          solar_zenith, solar_azimuth, airmass,
          model='allsitescomposite1990', return_components=False):
    '''
    Determine diffuse irradiance from the sky on a tilted surface using
    one of the Perez models.

    Perez models determine the diffuse irradiance from the sky (ground
    reflected irradiance is not included in this algorithm) on a tilted
    surface using the surface tilt angle, surface azimuth angle, diffuse
    horizontal irradiance, direct normal irradiance, extraterrestrial
    irradiance, sun zenith angle, sun azimuth angle, and relative (not
    pressure-corrected) airmass. Optionally a selector may be used to
    use any of Perez's model coefficient sets.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. surface_tilt must be >=0
        and <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)

    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. surface_azimuth must
        be >=0 and <=360. The azimuth convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).

    dhi : numeric
        Diffuse horizontal irradiance in W/m^2. DHI must be >=0.

    dni : numeric
        Direct normal irradiance in W/m^2. DNI must be >=0.

    dni_extra : numeric
        Extraterrestrial normal irradiance in W/m^2.

    solar_zenith : numeric
        apparent (refraction-corrected) zenith angles in decimal
        degrees. solar_zenith must be >=0 and <=180.

    solar_azimuth : numeric
        Sun azimuth angles in decimal degrees. solar_azimuth must be >=0
        and <=360. The azimuth convention is defined as degrees east of
        north (e.g. North = 0, East = 90, West = 270).

    airmass : numeric
        Relative (not pressure-corrected) airmass values. If AM is a
        DataFrame it must be of the same size as all other DataFrame
        inputs. AM must be >=0 (careful using the 1/sec(z) model of AM
        generation)

    model : string (optional, default='allsitescomposite1990')
        A string which selects the desired set of Perez coefficients. If
        model is not provided as an input, the default, '1990' will be
        used. All possible model selections are:

        * '1990'
        * 'allsitescomposite1990' (same as '1990')
        * 'allsitescomposite1988'
        * 'sandiacomposite1988'
        * 'usacomposite1988'
        * 'france1988'
        * 'phoenix1988'
        * 'elmonte1988'
        * 'osage1988'
        * 'albuquerque1988'
        * 'capecanaveral1988'
        * 'albany1988'

    return_components: bool (optional, default=False)
        Flag used to decide whether to return the calculated diffuse components
        or not.

    Returns
    --------
    sky_diffuse : numeric
        The sky diffuse component of the solar radiation on a tilted
        surface. Array input is currently converted to Series output.

    References
    ----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to
    compute solar irradiance on inclined surfaces for building energy
    simulation" 2007, Solar Energy vol. 81. pp. 254-267

    [2] Perez, R., Seals, R., Ineichen, P., Stewart, R., Menicucci, D.,
    1987. A new simplified version of the Perez diffuse irradiance model
    for tilted surfaces. Solar Energy 39(3), 221-232.

    [3] Perez, R., Ineichen, P., Seals, R., Michalsky, J., Stewart, R.,
    1990. Modeling daylight availability and irradiance components from
    direct and global irradiance. Solar Energy 44 (5), 271-289.

    [4] Perez, R. et. al 1988. "The Development and Verification of the
    Perez Diffuse Radiation Model". SAND88-7030
    '''

    kappa = 1.041  # for solar_zenith in radians
    z = np.radians(solar_zenith)  # convert to radians

    # delta is the sky's "brightness"
    delta = dhi * airmass / dni_extra

    # epsilon is the sky's "clearness"
    eps = ((dhi + dni) / dhi + kappa * (z ** 3)) / (1 + kappa * (z ** 3))

    # numpy indexing below will not work with a Series
    if isinstance(eps, pd.Series):
        eps = eps.values

    # Perez et al define clearness bins according to the following
    # rules. 1 = overcast ... 8 = clear (these names really only make
    # sense for small zenith angles, but...) these values will
    # eventually be used as indicies for coeffecient look ups
    ebin = np.zeros_like(eps, dtype=np.int8)
    ebin[eps < 1.065] = 1
    ebin[(eps >= 1.065) & (eps < 1.23)] = 2
    ebin[(eps >= 1.23) & (eps < 1.5)] = 3
    ebin[(eps >= 1.5) & (eps < 1.95)] = 4
    ebin[(eps >= 1.95) & (eps < 2.8)] = 5
    ebin[(eps >= 2.8) & (eps < 4.5)] = 6
    ebin[(eps >= 4.5) & (eps < 6.2)] = 7
    ebin[eps >= 6.2] = 8

    # correct for 0 indexing in coeffecient lookup
    # later, ebin = -1 will yield nan coefficients
    ebin -= 1

    # The various possible sets of Perez coefficients are contained
    # in a subfunction to clean up the code.
    F1c, F2c = _get_perez_coefficients(model)

    # results in invalid eps (ebin = -1) being mapped to nans
    nans = np.array([np.nan, np.nan, np.nan])
    F1c = np.vstack((F1c, nans))
    F2c = np.vstack((F2c, nans))

    F1 = (F1c[ebin, 0] + F1c[ebin, 1] * delta + F1c[ebin, 2] * z)
    F1 = np.maximum(F1, 0)

    F2 = (F2c[ebin, 0] + F2c[ebin, 1] * delta + F2c[ebin, 2] * z)
    F2 = np.maximum(F2, 0)

    A = aoi_projection(surface_tilt, surface_azimuth,
                       solar_zenith, solar_azimuth)
    A = np.maximum(A, 0)

    B = tools.cosd(solar_zenith)
    B = np.maximum(B, tools.cosd(85))

    # Calculate Diffuse POA from sky dome
    term1 = 0.5 * (1 - F1) * (1 + tools.cosd(surface_tilt))
    term2 = F1 * A / B
    term3 = F2 * tools.sind(surface_tilt)

    sky_diffuse = np.maximum(dhi * (term1 + term2 + term3), 0)

    # we've preserved the input type until now, so don't ruin it!
    if isinstance(sky_diffuse, pd.Series):
        sky_diffuse[np.isnan(airmass)] = 0
    else:
        sky_diffuse = np.where(np.isnan(airmass), 0, sky_diffuse)

    if return_components:
        diffuse_components = OrderedDict()

        # Calculate the different components
        diffuse_components['isotropic'] = dhi * term1
        diffuse_components['circumsolar'] = dhi * term2
        diffuse_components['horizon'] = dhi * term3

        # Set values of components to 0 when sky_diffuse is 0
        mask = sky_diffuse == 0
        if isinstance(sky_diffuse, pd.Series):
            diffuse_components = pd.DataFrame(diffuse_components)
            diffuse_components.ix[mask] = 0
        else:
            diffuse_components = {k: np.where(mask, 0, v) for k, v in diffuse_components.items()}

        return sky_diffuse, diffuse_components

    else:
        return sky_diffuse


def disc(ghi, zenith, datetime_or_doy, pressure=101325):
    """
    Estimate Direct Normal Irradiance from Global Horizontal Irradiance
    using the DISC model.

    The DISC algorithm converts global horizontal irradiance to direct
    normal irradiance through empirical relationships between the global
    and direct clearness indices.

    Parameters
    ----------
    ghi : numeric
        Global horizontal irradiance in W/m^2.

    solar_zenith : numeric
        True (not refraction-corrected) solar zenith angles in decimal
        degrees.

    datetime_or_doy : int, float, array, pd.DatetimeIndex
        Day of year or array of days of year e.g.
        pd.DatetimeIndex.dayofyear, or pd.DatetimeIndex.

    pressure : numeric
        Site pressure in Pascal.

    Returns
    -------
    output : OrderedDict or DataFrame
        Contains the following keys:

        * ``dni``: The modeled direct normal irradiance
          in W/m^2 provided by the
          Direct Insolation Simulation Code (DISC) model.
        * ``kt``: Ratio of global to extraterrestrial
          irradiance on a horizontal plane.
        * ``airmass``: Airmass

    References
    ----------
    [1] Maxwell, E. L., "A Quasi-Physical Model for Converting Hourly
    Global Horizontal to Direct Normal Insolation", Technical
    Report No. SERI/TR-215-3087, Golden, CO: Solar Energy Research
    Institute, 1987.

    [2] J.W. "Fourier series representation of the position of the sun".
    Found at:
    http://www.mail-archive.com/sundial@uni-koeln.de/msg01050.html on
    January 12, 2012

    See Also
    --------
    atmosphere.alt2pres
    dirint
    """

    # this is the I0 calculation from the reference
    I0 = extraradiation(datetime_or_doy, 1370, 'spencer')
    I0h = I0 * np.cos(np.radians(zenith))

    am = atmosphere.relativeairmass(zenith, model='kasten1966')
    am = atmosphere.absoluteairmass(am, pressure)

    kt = ghi / I0h
    kt = np.maximum(kt, 0)
    # powers of kt will be used repeatedly, so compute only once
    kt2 = kt * kt  # about the same as kt ** 2
    kt3 = kt2 * kt  # 5-10x faster than kt ** 3

    bools = (kt <= 0.6)
    a = np.where(bools,
                 0.512 - 1.56*kt + 2.286*kt2 - 2.222*kt3,
                 -5.743 + 21.77*kt - 27.49*kt2 + 11.56*kt3)
    b = np.where(bools,
                 0.37 + 0.962*kt,
                 41.4 - 118.5*kt + 66.05*kt2 + 31.9*kt3)
    c = np.where(bools,
                 -0.28 + 0.932*kt - 2.048*kt2,
                 -47.01 + 184.2*kt - 222.0*kt2 + 73.81*kt3)

    delta_kn = a + b * np.exp(c*am)

    Knc = 0.866 - 0.122*am + 0.0121*am**2 - 0.000653*am**3 + 1.4e-05*am**4
    Kn = Knc - delta_kn

    dni = Kn * I0

    dni = np.where((zenith > 87) | (ghi < 0) | (dni < 0), 0, dni)

    output = OrderedDict()
    output['dni'] = dni
    output['kt'] = kt
    output['airmass'] = am

    if isinstance(datetime_or_doy, pd.DatetimeIndex):
        output = pd.DataFrame(output, index=datetime_or_doy)

    return output


def dirint(ghi, zenith, times, pressure=101325., use_delta_kt_prime=True,
           temp_dew=None):
    """
    Determine DNI from GHI using the DIRINT modification of the DISC
    model.

    Implements the modified DISC model known as "DIRINT" introduced in
    [1]. DIRINT predicts direct normal irradiance (DNI) from measured
    global horizontal irradiance (GHI). DIRINT improves upon the DISC
    model by using time-series GHI data and dew point temperature
    information. The effectiveness of the DIRINT model improves with
    each piece of information provided.

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance in W/m^2.

    zenith : array-like
        True (not refraction-corrected) zenith angles in decimal
        degrees. If Z is a vector it must be of the same size as all
        other vector inputs. Z must be >=0 and <=180.

    times : DatetimeIndex

    pressure : float or array-like
        The site pressure in Pascal. Pressure may be measured or an
        average pressure may be calculated from site altitude.

    use_delta_kt_prime : bool
        Indicates if the user would like to utilize the time-series
        nature of the GHI measurements. A value of ``False`` will not
        use the time-series improvements, any other numeric value will
        use time-series improvements. It is recommended that time-series
        data only be used if the time between measured data points is
        less than 1.5 hours. If none of the input arguments are vectors,
        then time-series improvements are not used (because it's not a
        time-series). If True, input data must be Series.

    temp_dew : None, float, or array-like
        Surface dew point temperatures, in degrees C. Values of temp_dew
        may be numeric or NaN. Any single time period point with a
        DewPtTemp=NaN does not have dew point improvements applied. If
        DewPtTemp is not provided, then dew point improvements are not
        applied.

    Returns
    -------
    dni : array-like
        The modeled direct normal irradiance in W/m^2 provided by the
        DIRINT model.

    Notes
    -----
    DIRINT model requires time series data (ie. one of the inputs must
    be a vector of length > 2).

    References
    ----------
    [1] Perez, R., P. Ineichen, E. Maxwell, R. Seals and A. Zelenka,
    (1992). "Dynamic Global-to-Direct Irradiance Conversion Models".
    ASHRAE Transactions-Research Series, pp. 354-369

    [2] Maxwell, E. L., "A Quasi-Physical Model for Converting Hourly
    Global Horizontal to Direct Normal Insolation", Technical Report No.
    SERI/TR-215-3087, Golden, CO: Solar Energy Research Institute, 1987.
    """

    disc_out = disc(ghi, zenith, times, pressure=pressure)
    dni = disc_out['dni']
    kt = disc_out['kt']
    am = disc_out['airmass']

    kt_prime = kt / (1.031 * np.exp(-1.4 / (0.9 + 9.4 / am)) + 0.1)
    kt_prime = np.minimum(kt_prime, 0.82)  # From SRRL code

    # wholmgren:
    # the use_delta_kt_prime statement is a port of the MATLAB code.
    # I am confused by the abs() in the delta_kt_prime calculation.
    # It is not the absolute value of the central difference.
    # current implementation requires that kt_prime is a Series
    if use_delta_kt_prime:
        delta_kt_prime = 0.5*((kt_prime - kt_prime.shift(1)).abs().add(
                              (kt_prime - kt_prime.shift(-1)).abs(),
                              fill_value=0))
    else:
        delta_kt_prime = pd.Series(-1, index=times)

    if temp_dew is not None:
        w = pd.Series(np.exp(0.07 * temp_dew - 0.075), index=times)
    else:
        w = pd.Series(-1, index=times)

    # @wholmgren: the following bin assignments use MATLAB's 1-indexing.
    # Later, we'll subtract 1 to conform to Python's 0-indexing.

    # Create kt_prime bins
    kt_prime_bin = pd.Series(0, index=times, dtype=np.int64)
    kt_prime_bin[(kt_prime >= 0) & (kt_prime < 0.24)] = 1
    kt_prime_bin[(kt_prime >= 0.24) & (kt_prime < 0.4)] = 2
    kt_prime_bin[(kt_prime >= 0.4) & (kt_prime < 0.56)] = 3
    kt_prime_bin[(kt_prime >= 0.56) & (kt_prime < 0.7)] = 4
    kt_prime_bin[(kt_prime >= 0.7) & (kt_prime < 0.8)] = 5
    kt_prime_bin[(kt_prime >= 0.8) & (kt_prime <= 1)] = 6

    # Create zenith angle bins
    zenith_bin = pd.Series(0, index=times, dtype=np.int64)
    zenith_bin[(zenith >= 0) & (zenith < 25)] = 1
    zenith_bin[(zenith >= 25) & (zenith < 40)] = 2
    zenith_bin[(zenith >= 40) & (zenith < 55)] = 3
    zenith_bin[(zenith >= 55) & (zenith < 70)] = 4
    zenith_bin[(zenith >= 70) & (zenith < 80)] = 5
    zenith_bin[(zenith >= 80)] = 6

    # Create the bins for w based on dew point temperature
    w_bin = pd.Series(0, index=times, dtype=np.int64)
    w_bin[(w >= 0) & (w < 1)] = 1
    w_bin[(w >= 1) & (w < 2)] = 2
    w_bin[(w >= 2) & (w < 3)] = 3
    w_bin[(w >= 3)] = 4
    w_bin[(w == -1)] = 5

    # Create delta_kt_prime binning.
    delta_kt_prime_bin = pd.Series(0, index=times, dtype=np.int64)
    delta_kt_prime_bin[(delta_kt_prime >= 0) & (delta_kt_prime < 0.015)] = 1
    delta_kt_prime_bin[(delta_kt_prime >= 0.015) &
                       (delta_kt_prime < 0.035)] = 2
    delta_kt_prime_bin[(delta_kt_prime >= 0.035) & (delta_kt_prime < 0.07)] = 3
    delta_kt_prime_bin[(delta_kt_prime >= 0.07) & (delta_kt_prime < 0.15)] = 4
    delta_kt_prime_bin[(delta_kt_prime >= 0.15) & (delta_kt_prime < 0.3)] = 5
    delta_kt_prime_bin[(delta_kt_prime >= 0.3) & (delta_kt_prime <= 1)] = 6
    delta_kt_prime_bin[delta_kt_prime == -1] = 7

    # get the coefficients
    coeffs = _get_dirint_coeffs()

    # subtract 1 to account for difference between MATLAB-style bin
    # assignment and Python-style array lookup.
    dirint_coeffs = coeffs[kt_prime_bin-1, zenith_bin-1,
                           delta_kt_prime_bin-1, w_bin-1]

    # convert unassigned bins to nan
    dirint_coeffs = np.where((kt_prime_bin == 0) | (zenith_bin == 0) |
                             (w_bin == 0) | (delta_kt_prime_bin == 0),
                             np.nan, dirint_coeffs)

    dni *= dirint_coeffs

    return dni


def dirindex(ghi, ghi_clearsky, dni_clearsky, zenith, times, pressure=101325.,
             use_delta_kt_prime=True, temp_dew=None):
    """
    Determine DNI from GHI using the DIRINDEX model, which is a modification of
    the DIRINT model with information from a clear sky model.

    DIRINDEX [1] improves upon the DIRINT model by taking into account turbidity
    when used with the Ineichen clear sky model results.

    Parameters
    ----------
    ghi : array-like
        Global horizontal irradiance in W/m^2.

    ghi_clearsky : array-like
        Global horizontal irradiance from clear sky model, in W/m^2.

    dni_clearsky : array-like
        Direct normal irradiance from clear sky model, in W/m^2.

    zenith : array-like
        True (not refraction-corrected) zenith angles in decimal
        degrees. If Z is a vector it must be of the same size as all
        other vector inputs. Z must be >=0 and <=180.

    times : DatetimeIndex

    pressure : float or array-like
        The site pressure in Pascal. Pressure may be measured or an
        average pressure may be calculated from site altitude.

    use_delta_kt_prime : bool
        Indicates if the user would like to utilize the time-series
        nature of the GHI measurements. A value of ``False`` will not
        use the time-series improvements, any other numeric value will
        use time-series improvements. It is recommended that time-series
        data only be used if the time between measured data points is
        less than 1.5 hours. If none of the input arguments are vectors,
        then time-series improvements are not used (because it's not a
        time-series). If True, input data must be Series.

    temp_dew : None, float, or array-like
        Surface dew point temperatures, in degrees C. Values of temp_dew
        may be numeric or NaN. Any single time period point with a
        DewPtTemp=NaN does not have dew point improvements applied. If
        DewPtTemp is not provided, then dew point improvements are not
        applied.

    Returns
    -------
    dni : array-like
        The modeled direct normal irradiance in W/m^2.

    Notes
    -----
    DIRINDEX model requires time series data (ie. one of the inputs must
    be a vector of length > 2).

    References
    ----------
    [1] Perez, R., Ineichen, P., Moore, K., Kmiecik, M., Chain, C., George, R.,
    & Vignola, F. (2002). A new operational model for satellite-derived
    irradiances: description and validation. Solar Energy, 73(5), 307-317.
    """

    dni_dirint = dirint(ghi, zenith, times, pressure=pressure,
                        use_delta_kt_prime=use_delta_kt_prime,
                        temp_dew=temp_dew)

    dni_dirint_clearsky = dirint(ghi_clearsky, zenith, times, pressure=pressure,
                                 use_delta_kt_prime=use_delta_kt_prime,
                                 temp_dew=temp_dew)

    dni_dirindex = dni_clearsky * dni_dirint / dni_dirint_clearsky

    dni_dirindex[dni_dirindex < 0] = 0.

    return dni_dirindex


def erbs(ghi, zenith, doy):
    r"""
    Estimate DNI and DHI from GHI using the Erbs model.

    The Erbs model [1]_ estimates the diffuse fraction DF from global
    horizontal irradiance through an empirical relationship between DF
    and the ratio of GHI to extraterrestrial irradiance, Kt. The
    function uses the diffuse fraction to compute DHI as

    .. math::

        DHI = DF \times GHI

    DNI is then estimated as

    .. math::

        DNI = (GHI - DHI)/\cos(Z)

    where Z is the zenith angle.

    Parameters
    ----------
    ghi: numeric
        Global horizontal irradiance in W/m^2.
    zenith: numeric
        True (not refraction-corrected) zenith angles in decimal degrees.
    doy: scalar, array or DatetimeIndex
        The day of the year.

    Returns
    -------
    data : OrderedDict or DataFrame
        Contains the following keys/columns:

            * ``dni``: the modeled direct normal irradiance in W/m^2.
            * ``dhi``: the modeled diffuse horizontal irradiance in
              W/m^2.
            * ``kt``: Ratio of global to extraterrestrial irradiance
              on a horizontal plane.

    References
    ----------
    .. [1] D. G. Erbs, S. A. Klein and J. A. Duffie, Estimation of the
       diffuse radiation fraction for hourly, daily and monthly-average
       global radiation, Solar Energy 28(4), pp 293-302, 1982. Eq. 1

    See also
    --------
    dirint
    disc
    """

    dni_extra = extraradiation(doy)

    # This Z needs to be the true Zenith angle, not apparent,
    # to get extraterrestrial horizontal radiation)
    i0_h = dni_extra * tools.cosd(zenith)

    kt = ghi / i0_h
    kt = np.maximum(kt, 0)

    # For Kt <= 0.22, set the diffuse fraction
    df = 1 - 0.09*kt

    # For Kt > 0.22 and Kt <= 0.8, set the diffuse fraction
    df = np.where((kt > 0.22) & (kt <= 0.8),
                  0.9511 - 0.1604*kt + 4.388*kt**2 -
                  16.638*kt**3 + 12.336*kt**4,
                  df)

    # For Kt > 0.8, set the diffuse fraction
    df = np.where(kt > 0.8, 0.165, df)

    dhi = df * ghi

    dni = (ghi - dhi) / tools.cosd(zenith)

    data = OrderedDict()
    data['dni'] = dni
    data['dhi'] = dhi
    data['kt'] = kt

    if isinstance(dni, pd.Series):
        data = pd.DataFrame(data)

    return data


def liujordan(zenith, transmittance, airmass, pressure=101325.,
              dni_extra=1367.0):
    '''
    Determine DNI, DHI, GHI from extraterrestrial flux, transmittance,
    and optical air mass number.

    Liu and Jordan, 1960, developed a simplified direct radiation model.
    DHI is from an empirical equation for diffuse radiation from Liu and
    Jordan, 1960.

    Parameters
    ----------
    zenith: pd.Series
        True (not refraction-corrected) zenith angles in decimal
        degrees. If Z is a vector it must be of the same size as all
        other vector inputs. Z must be >=0 and <=180.

    transmittance: float
        Atmospheric transmittance between 0 and 1.

    pressure: float
        Air pressure

    dni_extra: float
        Direct irradiance incident at the top of the atmosphere.

    Returns
    -------
    irradiance: DataFrame
        Modeled direct normal irradiance, direct horizontal irradiance,
        and global horizontal irradiance in W/m^2

    References
    ----------
    [1] Campbell, G. S., J. M. Norman (1998) An Introduction to
    Environmental Biophysics. 2nd Ed. New York: Springer.

    [2] Liu, B. Y., R. C. Jordan, (1960). "The interrelationship and
    characteristic distribution of direct, diffuse, and total solar
    radiation".  Solar Energy 4:1-19
    '''

    tao = transmittance

    dni = dni_extra*tao**airmass
    dhi = 0.3 * (1.0 - tao**airmass) * dni_extra * np.cos(np.radians(zenith))
    ghi = dhi + dni * np.cos(np.radians(zenith))

    irrads = OrderedDict()
    irrads['ghi'] = ghi
    irrads['dni'] = dni
    irrads['dhi'] = dhi

    if isinstance(ghi, pd.Series):
        irrads = pd.DataFrame(irrads)

    return irrads


def _get_perez_coefficients(perezmodel):
    '''
    Find coefficients for the Perez model

    Parameters
    ----------

    perezmodel : string (optional, default='allsitescomposite1990')

          a character string which selects the desired set of Perez
          coefficients. If model is not provided as an input, the default,
          '1990' will be used.

    All possible model selections are:

          * '1990'
          * 'allsitescomposite1990' (same as '1990')
          * 'allsitescomposite1988'
          * 'sandiacomposite1988'
          * 'usacomposite1988'
          * 'france1988'
          * 'phoenix1988'
          * 'elmonte1988'
          * 'osage1988'
          * 'albuquerque1988'
          * 'capecanaveral1988'
          * 'albany1988'

    Returns
    --------
    F1coeffs, F2coeffs : (array, array)
          F1 and F2 coefficients for the Perez model

    References
    ----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to
    compute solar irradiance on inclined surfaces for building energy
    simulation" 2007, Solar Energy vol. 81. pp. 254-267

    [2] Perez, R., Seals, R., Ineichen, P., Stewart, R., Menicucci, D.,
    1987. A new simplified version of the Perez diffuse irradiance model
    for tilted surfaces. Solar Energy 39(3), 221-232.

    [3] Perez, R., Ineichen, P., Seals, R., Michalsky, J., Stewart, R.,
    1990. Modeling daylight availability and irradiance components from
    direct and global irradiance. Solar Energy 44 (5), 271-289.

    [4] Perez, R. et. al 1988. "The Development and Verification of the
    Perez Diffuse Radiation Model". SAND88-7030

    '''
    coeffdict = {
        'allsitescomposite1990': [
            [-0.0080,    0.5880,   -0.0620,   -0.0600,    0.0720,   -0.0220],
            [0.1300,    0.6830,   -0.1510,   -0.0190,    0.0660,   -0.0290],
            [0.3300,    0.4870,   -0.2210,    0.0550,   -0.0640,   -0.0260],
            [0.5680,    0.1870,   -0.2950,    0.1090,   -0.1520,   -0.0140],
            [0.8730,   -0.3920,   -0.3620,    0.2260,   -0.4620,    0.0010],
            [1.1320,   -1.2370,   -0.4120,    0.2880,   -0.8230,    0.0560],
            [1.0600,   -1.6000,   -0.3590,    0.2640,   -1.1270,    0.1310],
            [0.6780,   -0.3270,   -0.2500,    0.1560,   -1.3770,    0.2510]],
        'allsitescomposite1988': [
            [-0.0180,    0.7050,   -0.071,   -0.0580,    0.1020,   -0.0260],
            [0.1910,    0.6450,   -0.1710,    0.0120,    0.0090,   -0.0270],
            [0.4400,    0.3780,   -0.2560,    0.0870,   -0.1040,   -0.0250],
            [0.7560,   -0.1210,   -0.3460,    0.1790,   -0.3210,   -0.0080],
            [0.9960,   -0.6450,   -0.4050,    0.2600,   -0.5900,    0.0170],
            [1.0980,   -1.2900,   -0.3930,    0.2690,   -0.8320,    0.0750],
            [0.9730,   -1.1350,   -0.3780,    0.1240,   -0.2580,    0.1490],
            [0.6890,   -0.4120,   -0.2730,    0.1990,   -1.6750,    0.2370]],
        'sandiacomposite1988': [
            [-0.1960,    1.0840,   -0.0060,   -0.1140,    0.1800,   -0.0190],
            [0.2360,    0.5190,   -0.1800,   -0.0110,    0.0200,   -0.0380],
            [0.4540,    0.3210,   -0.2550,    0.0720,   -0.0980,   -0.0460],
            [0.8660,   -0.3810,   -0.3750,    0.2030,   -0.4030,   -0.0490],
            [1.0260,   -0.7110,   -0.4260,    0.2730,   -0.6020,   -0.0610],
            [0.9780,   -0.9860,   -0.3500,    0.2800,   -0.9150,   -0.0240],
            [0.7480,   -0.9130,   -0.2360,    0.1730,   -1.0450,    0.0650],
            [0.3180,   -0.7570,    0.1030,    0.0620,   -1.6980,    0.2360]],
        'usacomposite1988': [
            [-0.0340,    0.6710,   -0.0590,   -0.0590,    0.0860,   -0.0280],
            [0.2550,    0.4740,   -0.1910,    0.0180,   -0.0140,   -0.0330],
            [0.4270,    0.3490,   -0.2450,    0.0930,   -0.1210,   -0.0390],
            [0.7560,   -0.2130,   -0.3280,    0.1750,   -0.3040,   -0.0270],
            [1.0200,   -0.8570,   -0.3850,    0.2800,   -0.6380,   -0.0190],
            [1.0500,   -1.3440,   -0.3480,    0.2800,   -0.8930,    0.0370],
            [0.9740,   -1.5070,   -0.3700,    0.1540,   -0.5680,    0.1090],
            [0.7440,   -1.8170,   -0.2560,    0.2460,   -2.6180,    0.2300]],
        'france1988': [
            [0.0130,    0.7640,   -0.1000,   -0.0580,    0.1270,   -0.0230],
            [0.0950,    0.9200,   -0.1520,         0,    0.0510,   -0.0200],
            [0.4640,    0.4210,   -0.2800,    0.0640,   -0.0510,   -0.0020],
            [0.7590,   -0.0090,   -0.3730,    0.2010,   -0.3820,    0.0100],
            [0.9760,   -0.4000,   -0.4360,    0.2710,   -0.6380,    0.0510],
            [1.1760,   -1.2540,   -0.4620,    0.2950,   -0.9750,    0.1290],
            [1.1060,   -1.5630,   -0.3980,    0.3010,   -1.4420,    0.2120],
            [0.9340,   -1.5010,   -0.2710,    0.4200,   -2.9170,    0.2490]],
        'phoenix1988': [
            [-0.0030,    0.7280,   -0.0970,   -0.0750,    0.1420,   -0.0430],
            [0.2790,    0.3540,   -0.1760,    0.0300,   -0.0550,   -0.0540],
            [0.4690,    0.1680,   -0.2460,    0.0480,   -0.0420,   -0.0570],
            [0.8560,   -0.5190,   -0.3400,    0.1760,   -0.3800,   -0.0310],
            [0.9410,   -0.6250,   -0.3910,    0.1880,   -0.3600,   -0.0490],
            [1.0560,   -1.1340,   -0.4100,    0.2810,   -0.7940,   -0.0650],
            [0.9010,   -2.1390,   -0.2690,    0.1180,   -0.6650,    0.0460],
            [0.1070,    0.4810,    0.1430,   -0.1110,   -0.1370,    0.2340]],
        'elmonte1988': [
            [0.0270,    0.7010,   -0.1190,   -0.0580,    0.1070,  -0.0600],
            [0.1810,    0.6710,   -0.1780,   -0.0790,    0.1940,  -0.0350],
            [0.4760,    0.4070,   -0.2880,    0.0540,   -0.0320,  -0.0550],
            [0.8750,   -0.2180,   -0.4030,    0.1870,   -0.3090,  -0.0610],
            [1.1660,   -1.0140,   -0.4540,    0.2110,   -0.4100,  -0.0440],
            [1.1430,   -2.0640,   -0.2910,    0.0970,   -0.3190,   0.0530],
            [1.0940,   -2.6320,   -0.2590,    0.0290,   -0.4220,   0.1470],
            [0.1550,    1.7230,    0.1630,   -0.1310,   -0.0190,   0.2770]],
        'osage1988': [
            [-0.3530,    1.4740,   0.0570,   -0.1750,    0.3120,   0.0090],
            [0.3630,    0.2180,  -0.2120,    0.0190,   -0.0340,  -0.0590],
            [-0.0310,    1.2620,  -0.0840,   -0.0820,    0.2310,  -0.0170],
            [0.6910,    0.0390,  -0.2950,    0.0910,   -0.1310,  -0.0350],
            [1.1820,   -1.3500,  -0.3210,    0.4080,   -0.9850,  -0.0880],
            [0.7640,    0.0190,  -0.2030,    0.2170,   -0.2940,  -0.1030],
            [0.2190,    1.4120,   0.2440,    0.4710,   -2.9880,   0.0340],
            [3.5780,   22.2310, -10.7450,    2.4260,    4.8920,  -5.6870]],
        'albuquerque1988': [
            [0.0340,    0.5010,  -0.0940,   -0.0630,    0.1060,  -0.0440],
            [0.2290,    0.4670,  -0.1560,   -0.0050,   -0.0190,  -0.0230],
            [0.4860,    0.2410,  -0.2530,    0.0530,   -0.0640,  -0.0220],
            [0.8740,   -0.3930,  -0.3970,    0.1810,   -0.3270,  -0.0370],
            [1.1930,   -1.2960,  -0.5010,    0.2810,   -0.6560,  -0.0450],
            [1.0560,   -1.7580,  -0.3740,    0.2260,   -0.7590,   0.0340],
            [0.9010,   -4.7830,  -0.1090,    0.0630,   -0.9700,   0.1960],
            [0.8510,   -7.0550,  -0.0530,    0.0600,   -2.8330,   0.3300]],
        'capecanaveral1988': [
            [0.0750,    0.5330,   -0.1240,  -0.0670,   0.0420,  -0.0200],
            [0.2950,    0.4970,   -0.2180,  -0.0080,   0.0030,  -0.0290],
            [0.5140,    0.0810,   -0.2610,   0.0750,  -0.1600,  -0.0290],
            [0.7470,   -0.3290,   -0.3250,   0.1810,  -0.4160,  -0.0300],
            [0.9010,   -0.8830,   -0.2970,   0.1780,  -0.4890,   0.0080],
            [0.5910,   -0.0440,   -0.1160,   0.2350,  -0.9990,   0.0980],
            [0.5370,   -2.4020,    0.3200,   0.1690,  -1.9710,   0.3100],
            [-0.8050,    4.5460,    1.0720,  -0.2580,  -0.9500,    0.7530]],
        'albany1988': [
            [0.0120,    0.5540,   -0.0760, -0.0520,   0.0840,  -0.0290],
            [0.2670,    0.4370,   -0.1940,  0.0160,   0.0220,  -0.0360],
            [0.4200,    0.3360,   -0.2370,  0.0740,  -0.0520,  -0.0320],
            [0.6380,   -0.0010,   -0.2810,  0.1380,  -0.1890,  -0.0120],
            [1.0190,   -1.0270,   -0.3420,  0.2710,  -0.6280,   0.0140],
            [1.1490,   -1.9400,   -0.3310,  0.3220,  -1.0970,   0.0800],
            [1.4340,   -3.9940,   -0.4920,  0.4530,  -2.3760,   0.1170],
            [1.0070,   -2.2920,   -0.4820,  0.3900,  -3.3680,   0.2290]], }

    array = np.array(coeffdict[perezmodel])

    F1coeffs = array[:, 0:3]
    F2coeffs = array[:, 3:7]

    return F1coeffs, F2coeffs


def _get_dirint_coeffs():
    """
    A place to stash the dirint coefficients.

    Returns
    -------
    np.array with shape ``(6, 6, 7, 5)``.
    Ordering is ``[kt_prime_bin, zenith_bin, delta_kt_prime_bin, w_bin]``
    """

    # To allow for maximum copy/paste from the MATLAB 1-indexed code,
    # we create and assign values to an oversized array.
    # Then, we return the [1:, 1:, :, :] slice.

    coeffs = np.zeros((7, 7, 7, 5))

    coeffs[1, 1, :, :] = [
        [0.385230, 0.385230, 0.385230, 0.462880, 0.317440],
        [0.338390, 0.338390, 0.221270, 0.316730, 0.503650],
        [0.235680, 0.235680, 0.241280, 0.157830, 0.269440],
        [0.830130, 0.830130, 0.171970, 0.841070, 0.457370],
        [0.548010, 0.548010, 0.478000, 0.966880, 1.036370],
        [0.548010, 0.548010, 1.000000, 3.012370, 1.976540],
        [0.582690, 0.582690, 0.229720, 0.892710, 0.569950]]

    coeffs[1, 2, :, :] = [
        [0.131280, 0.131280, 0.385460, 0.511070, 0.127940],
        [0.223710, 0.223710, 0.193560, 0.304560, 0.193940],
        [0.229970, 0.229970, 0.275020, 0.312730, 0.244610],
        [0.090100, 0.184580, 0.260500, 0.687480, 0.579440],
        [0.131530, 0.131530, 0.370190, 1.380350, 1.052270],
        [1.116250, 1.116250, 0.928030, 3.525490, 2.316920],
        [0.090100, 0.237000, 0.300040, 0.812470, 0.664970]]

    coeffs[1, 3, :, :] = [
        [0.587510, 0.130000, 0.400000, 0.537210, 0.832490],
        [0.306210, 0.129830, 0.204460, 0.500000, 0.681640],
        [0.224020, 0.260620, 0.334080, 0.501040, 0.350470],
        [0.421540, 0.753970, 0.750660, 3.706840, 0.983790],
        [0.706680, 0.373530, 1.245670, 0.864860, 1.992630],
        [4.864400, 0.117390, 0.265180, 0.359180, 3.310820],
        [0.392080, 0.493290, 0.651560, 1.932780, 0.898730]]

    coeffs[1, 4, :, :] = [
        [0.126970, 0.126970, 0.126970, 0.126970, 0.126970],
        [0.810820, 0.810820, 0.810820, 0.810820, 0.810820],
        [3.241680, 2.500000, 2.291440, 2.291440, 2.291440],
        [4.000000, 3.000000, 2.000000, 0.975430, 1.965570],
        [12.494170, 12.494170, 8.000000, 5.083520, 8.792390],
        [21.744240, 21.744240, 21.744240, 21.744240, 21.744240],
        [3.241680, 12.494170, 1.620760, 1.375250, 2.331620]]

    coeffs[1, 5, :, :] = [
        [0.126970, 0.126970, 0.126970, 0.126970, 0.126970],
        [0.810820, 0.810820, 0.810820, 0.810820, 0.810820],
        [3.241680, 2.500000, 2.291440, 2.291440, 2.291440],
        [4.000000, 3.000000, 2.000000, 0.975430, 1.965570],
        [12.494170, 12.494170, 8.000000, 5.083520, 8.792390],
        [21.744240, 21.744240, 21.744240, 21.744240, 21.744240],
        [3.241680, 12.494170, 1.620760, 1.375250, 2.331620]]

    coeffs[1, 6, :, :] = [
        [0.126970, 0.126970, 0.126970, 0.126970, 0.126970],
        [0.810820, 0.810820, 0.810820, 0.810820, 0.810820],
        [3.241680, 2.500000, 2.291440, 2.291440, 2.291440],
        [4.000000, 3.000000, 2.000000, 0.975430, 1.965570],
        [12.494170, 12.494170, 8.000000, 5.083520, 8.792390],
        [21.744240, 21.744240, 21.744240, 21.744240, 21.744240],
        [3.241680, 12.494170, 1.620760, 1.375250, 2.331620]]

    coeffs[2, 1, :, :] = [
        [0.337440, 0.337440, 0.969110, 1.097190, 1.116080],
        [0.337440, 0.337440, 0.969110, 1.116030, 0.623900],
        [0.337440, 0.337440, 1.530590, 1.024420, 0.908480],
        [0.584040, 0.584040, 0.847250, 0.914940, 1.289300],
        [0.337440, 0.337440, 0.310240, 1.435020, 1.852830],
        [0.337440, 0.337440, 1.015010, 1.097190, 2.117230],
        [0.337440, 0.337440, 0.969110, 1.145730, 1.476400]]

    coeffs[2, 2, :, :] = [
        [0.300000, 0.300000, 0.700000, 1.100000, 0.796940],
        [0.219870, 0.219870, 0.526530, 0.809610, 0.649300],
        [0.386650, 0.386650, 0.119320, 0.576120, 0.685460],
        [0.746730, 0.399830, 0.470970, 0.986530, 0.785370],
        [0.575420, 0.936700, 1.649200, 1.495840, 1.335590],
        [1.319670, 4.002570, 1.276390, 2.644550, 2.518670],
        [0.665190, 0.678910, 1.012360, 1.199940, 0.986580]]

    coeffs[2, 3, :, :] = [
        [0.378870, 0.974060, 0.500000, 0.491880, 0.665290],
        [0.105210, 0.263470, 0.407040, 0.553460, 0.582590],
        [0.312900, 0.345240, 1.144180, 0.854790, 0.612280],
        [0.119070, 0.365120, 0.560520, 0.793720, 0.802600],
        [0.781610, 0.837390, 1.270420, 1.537980, 1.292950],
        [1.152290, 1.152290, 1.492080, 1.245370, 2.177100],
        [0.424660, 0.529550, 0.966910, 1.033460, 0.958730]]

    coeffs[2, 4, :, :] = [
        [0.310590, 0.714410, 0.252450, 0.500000, 0.607600],
        [0.975190, 0.363420, 0.500000, 0.400000, 0.502800],
        [0.175580, 0.196250, 0.476360, 1.072470, 0.490510],
        [0.719280, 0.698620, 0.657770, 1.190840, 0.681110],
        [0.426240, 1.464840, 0.678550, 1.157730, 0.978430],
        [2.501120, 1.789130, 1.387090, 2.394180, 2.394180],
        [0.491640, 0.677610, 0.685610, 1.082400, 0.735410]]

    coeffs[2, 5, :, :] = [
        [0.597000, 0.500000, 0.300000, 0.310050, 0.413510],
        [0.314790, 0.336310, 0.400000, 0.400000, 0.442460],
        [0.166510, 0.460440, 0.552570, 1.000000, 0.461610],
        [0.401020, 0.559110, 0.403630, 1.016710, 0.671490],
        [0.400360, 0.750830, 0.842640, 1.802600, 1.023830],
        [3.315300, 1.510380, 2.443650, 1.638820, 2.133990],
        [0.530790, 0.745850, 0.693050, 1.458040, 0.804500]]

    coeffs[2, 6, :, :] = [
        [0.597000, 0.500000, 0.300000, 0.310050, 0.800920],
        [0.314790, 0.336310, 0.400000, 0.400000, 0.237040],
        [0.166510, 0.460440, 0.552570, 1.000000, 0.581990],
        [0.401020, 0.559110, 0.403630, 1.016710, 0.898570],
        [0.400360, 0.750830, 0.842640, 1.802600, 3.400390],
        [3.315300, 1.510380, 2.443650, 1.638820, 2.508780],
        [0.204340, 1.157740, 2.003080, 2.622080, 1.409380]]

    coeffs[3, 1, :, :] = [
        [1.242210, 1.242210, 1.242210, 1.242210, 1.242210],
        [0.056980, 0.056980, 0.656990, 0.656990, 0.925160],
        [0.089090, 0.089090, 1.040430, 1.232480, 1.205300],
        [1.053850, 1.053850, 1.399690, 1.084640, 1.233340],
        [1.151540, 1.151540, 1.118290, 1.531640, 1.411840],
        [1.494980, 1.494980, 1.700000, 1.800810, 1.671600],
        [1.018450, 1.018450, 1.153600, 1.321890, 1.294670]]

    coeffs[3, 2, :, :] = [
        [0.700000, 0.700000, 1.023460, 0.700000, 0.945830],
        [0.886300, 0.886300, 1.333620, 0.800000, 1.066620],
        [0.902180, 0.902180, 0.954330, 1.126690, 1.097310],
        [1.095300, 1.075060, 1.176490, 1.139470, 1.096110],
        [1.201660, 1.201660, 1.438200, 1.256280, 1.198060],
        [1.525850, 1.525850, 1.869160, 1.985410, 1.911590],
        [1.288220, 1.082810, 1.286370, 1.166170, 1.119330]]

    coeffs[3, 3, :, :] = [
        [0.600000, 1.029910, 0.859890, 0.550000, 0.813600],
        [0.604450, 1.029910, 0.859890, 0.656700, 0.928840],
        [0.455850, 0.750580, 0.804930, 0.823000, 0.911000],
        [0.526580, 0.932310, 0.908620, 0.983520, 0.988090],
        [1.036110, 1.100690, 0.848380, 1.035270, 1.042380],
        [1.048440, 1.652720, 0.900000, 2.350410, 1.082950],
        [0.817410, 0.976160, 0.861300, 0.974780, 1.004580]]

    coeffs[3, 4, :, :] = [
        [0.782110, 0.564280, 0.600000, 0.600000, 0.665740],
        [0.894480, 0.680730, 0.541990, 0.800000, 0.669140],
        [0.487460, 0.818950, 0.841830, 0.872540, 0.709040],
        [0.709310, 0.872780, 0.908480, 0.953290, 0.844350],
        [0.863920, 0.947770, 0.876220, 1.078750, 0.936910],
        [1.280350, 0.866720, 0.769790, 1.078750, 0.975130],
        [0.725420, 0.869970, 0.868810, 0.951190, 0.829220]]

    coeffs[3, 5, :, :] = [
        [0.791750, 0.654040, 0.483170, 0.409000, 0.597180],
        [0.566140, 0.948990, 0.971820, 0.653570, 0.718550],
        [0.648710, 0.637730, 0.870510, 0.860600, 0.694300],
        [0.637630, 0.767610, 0.925670, 0.990310, 0.847670],
        [0.736380, 0.946060, 1.117590, 1.029340, 0.947020],
        [1.180970, 0.850000, 1.050000, 0.950000, 0.888580],
        [0.700560, 0.801440, 0.961970, 0.906140, 0.823880]]

    coeffs[3, 6, :, :] = [
        [0.500000, 0.500000, 0.586770, 0.470550, 0.629790],
        [0.500000, 0.500000, 1.056220, 1.260140, 0.658140],
        [0.500000, 0.500000, 0.631830, 0.842620, 0.582780],
        [0.554710, 0.734730, 0.985820, 0.915640, 0.898260],
        [0.712510, 1.205990, 0.909510, 1.078260, 0.885610],
        [1.899260, 1.559710, 1.000000, 1.150000, 1.120390],
        [0.653880, 0.793120, 0.903320, 0.944070, 0.796130]]

    coeffs[4, 1, :, :] = [
        [1.000000, 1.000000, 1.050000, 1.170380, 1.178090],
        [0.960580, 0.960580, 1.059530, 1.179030, 1.131690],
        [0.871470, 0.871470, 0.995860, 1.141910, 1.114600],
        [1.201590, 1.201590, 0.993610, 1.109380, 1.126320],
        [1.065010, 1.065010, 0.828660, 0.939970, 1.017930],
        [1.065010, 1.065010, 0.623690, 1.119620, 1.132260],
        [1.071570, 1.071570, 0.958070, 1.114130, 1.127110]]

    coeffs[4, 2, :, :] = [
        [0.950000, 0.973390, 0.852520, 1.092200, 1.096590],
        [0.804120, 0.913870, 0.980990, 1.094580, 1.042420],
        [0.737540, 0.935970, 0.999940, 1.056490, 1.050060],
        [1.032980, 1.034540, 0.968460, 1.032080, 1.015780],
        [0.900000, 0.977210, 0.945960, 1.008840, 0.969960],
        [0.600000, 0.750000, 0.750000, 0.844710, 0.899100],
        [0.926800, 0.965030, 0.968520, 1.044910, 1.032310]]

    coeffs[4, 3, :, :] = [
        [0.850000, 1.029710, 0.961100, 1.055670, 1.009700],
        [0.818530, 0.960010, 0.996450, 1.081970, 1.036470],
        [0.765380, 0.953500, 0.948260, 1.052110, 1.000140],
        [0.775610, 0.909610, 0.927800, 0.987800, 0.952100],
        [1.000990, 0.881880, 0.875950, 0.949100, 0.893690],
        [0.902370, 0.875960, 0.807990, 0.942410, 0.917920],
        [0.856580, 0.928270, 0.946820, 1.032260, 0.972990]]

    coeffs[4, 4, :, :] = [
        [0.750000, 0.857930, 0.983800, 1.056540, 0.980240],
        [0.750000, 0.987010, 1.013730, 1.133780, 1.038250],
        [0.800000, 0.947380, 1.012380, 1.091270, 0.999840],
        [0.800000, 0.914550, 0.908570, 0.999190, 0.915230],
        [0.778540, 0.800590, 0.799070, 0.902180, 0.851560],
        [0.680190, 0.317410, 0.507680, 0.388910, 0.646710],
        [0.794920, 0.912780, 0.960830, 1.057110, 0.947950]]

    coeffs[4, 5, :, :] = [
        [0.750000, 0.833890, 0.867530, 1.059890, 0.932840],
        [0.979700, 0.971470, 0.995510, 1.068490, 1.030150],
        [0.858850, 0.987920, 1.043220, 1.108700, 1.044900],
        [0.802400, 0.955110, 0.911660, 1.045070, 0.944470],
        [0.884890, 0.766210, 0.885390, 0.859070, 0.818190],
        [0.615680, 0.700000, 0.850000, 0.624620, 0.669300],
        [0.835570, 0.946150, 0.977090, 1.049350, 0.979970]]

    coeffs[4, 6, :, :] = [
        [0.689220, 0.809600, 0.900000, 0.789500, 0.853990],
        [0.854660, 0.852840, 0.938200, 0.923110, 0.955010],
        [0.938600, 0.932980, 1.010390, 1.043950, 1.041640],
        [0.843620, 0.981300, 0.951590, 0.946100, 0.966330],
        [0.694740, 0.814690, 0.572650, 0.400000, 0.726830],
        [0.211370, 0.671780, 0.416340, 0.297290, 0.498050],
        [0.843540, 0.882330, 0.911760, 0.898420, 0.960210]]

    coeffs[5, 1, :, :] = [
        [1.054880, 1.075210, 1.068460, 1.153370, 1.069220],
        [1.000000, 1.062220, 1.013470, 1.088170, 1.046200],
        [0.885090, 0.993530, 0.942590, 1.054990, 1.012740],
        [0.920000, 0.950000, 0.978720, 1.020280, 0.984440],
        [0.850000, 0.908500, 0.839940, 0.985570, 0.962180],
        [0.800000, 0.800000, 0.810080, 0.950000, 0.961550],
        [1.038590, 1.063200, 1.034440, 1.112780, 1.037800]]

    coeffs[5, 2, :, :] = [
        [1.017610, 1.028360, 1.058960, 1.133180, 1.045620],
        [0.920000, 0.998970, 1.033590, 1.089030, 1.022060],
        [0.912370, 0.949930, 0.979770, 1.020420, 0.981770],
        [0.847160, 0.935300, 0.930540, 0.955050, 0.946560],
        [0.880260, 0.867110, 0.874130, 0.972650, 0.883420],
        [0.627150, 0.627150, 0.700000, 0.774070, 0.845130],
        [0.973700, 1.006240, 1.026190, 1.071960, 1.017240]]

    coeffs[5, 3, :, :] = [
        [1.028710, 1.017570, 1.025900, 1.081790, 1.024240],
        [0.924980, 0.985500, 1.014100, 1.092210, 0.999610],
        [0.828570, 0.934920, 0.994950, 1.024590, 0.949710],
        [0.900810, 0.901330, 0.928830, 0.979570, 0.913100],
        [0.761030, 0.845150, 0.805360, 0.936790, 0.853460],
        [0.626400, 0.546750, 0.730500, 0.850000, 0.689050],
        [0.957630, 0.985480, 0.991790, 1.050220, 0.987900]]

    coeffs[5, 4, :, :] = [
        [0.992730, 0.993880, 1.017150, 1.059120, 1.017450],
        [0.975610, 0.987160, 1.026820, 1.075440, 1.007250],
        [0.871090, 0.933190, 0.974690, 0.979840, 0.952730],
        [0.828750, 0.868090, 0.834920, 0.905510, 0.871530],
        [0.781540, 0.782470, 0.767910, 0.764140, 0.795890],
        [0.743460, 0.693390, 0.514870, 0.630150, 0.715660],
        [0.934760, 0.957870, 0.959640, 0.972510, 0.981640]]

    coeffs[5, 5, :, :] = [
        [0.965840, 0.941240, 0.987100, 1.022540, 1.011160],
        [0.988630, 0.994770, 0.976590, 0.950000, 1.034840],
        [0.958200, 1.018080, 0.974480, 0.920000, 0.989870],
        [0.811720, 0.869090, 0.812020, 0.850000, 0.821050],
        [0.682030, 0.679480, 0.632450, 0.746580, 0.738550],
        [0.668290, 0.445860, 0.500000, 0.678920, 0.696510],
        [0.926940, 0.953350, 0.959050, 0.876210, 0.991490]]

    coeffs[5, 6, :, :] = [
        [0.948940, 0.997760, 0.850000, 0.826520, 0.998470],
        [1.017860, 0.970000, 0.850000, 0.700000, 0.988560],
        [1.000000, 0.950000, 0.850000, 0.606240, 0.947260],
        [1.000000, 0.746140, 0.751740, 0.598390, 0.725230],
        [0.922210, 0.500000, 0.376800, 0.517110, 0.548630],
        [0.500000, 0.450000, 0.429970, 0.404490, 0.539940],
        [0.960430, 0.881630, 0.775640, 0.596350, 0.937680]]

    coeffs[6, 1, :, :] = [
        [1.030000, 1.040000, 1.000000, 1.000000, 1.049510],
        [1.050000, 0.990000, 0.990000, 0.950000, 0.996530],
        [1.050000, 0.990000, 0.990000, 0.820000, 0.971940],
        [1.050000, 0.790000, 0.880000, 0.820000, 0.951840],
        [1.000000, 0.530000, 0.440000, 0.710000, 0.928730],
        [0.540000, 0.470000, 0.500000, 0.550000, 0.773950],
        [1.038270, 0.920180, 0.910930, 0.821140, 1.034560]]

    coeffs[6, 2, :, :] = [
        [1.041020, 0.997520, 0.961600, 1.000000, 1.035780],
        [0.948030, 0.980000, 0.900000, 0.950360, 0.977460],
        [0.950000, 0.977250, 0.869270, 0.800000, 0.951680],
        [0.951870, 0.850000, 0.748770, 0.700000, 0.883850],
        [0.900000, 0.823190, 0.727450, 0.600000, 0.839870],
        [0.850000, 0.805020, 0.692310, 0.500000, 0.788410],
        [1.010090, 0.895270, 0.773030, 0.816280, 1.011680]]

    coeffs[6, 3, :, :] = [
        [1.022450, 1.004600, 0.983650, 1.000000, 1.032940],
        [0.943960, 0.999240, 0.983920, 0.905990, 0.978150],
        [0.936240, 0.946480, 0.850000, 0.850000, 0.930320],
        [0.816420, 0.885000, 0.644950, 0.817650, 0.865310],
        [0.742960, 0.765690, 0.561520, 0.700000, 0.827140],
        [0.643870, 0.596710, 0.474460, 0.600000, 0.651200],
        [0.971740, 0.940560, 0.714880, 0.864380, 1.001650]]

    coeffs[6, 4, :, :] = [
        [0.995260, 0.977010, 1.000000, 1.000000, 1.035250],
        [0.939810, 0.975250, 0.939980, 0.950000, 0.982550],
        [0.876870, 0.879440, 0.850000, 0.900000, 0.917810],
        [0.873480, 0.873450, 0.751470, 0.850000, 0.863040],
        [0.761470, 0.702360, 0.638770, 0.750000, 0.783120],
        [0.734080, 0.650000, 0.600000, 0.650000, 0.715660],
        [0.942160, 0.919100, 0.770340, 0.731170, 0.995180]]

    coeffs[6, 5, :, :] = [
        [0.952560, 0.916780, 0.920000, 0.900000, 1.005880],
        [0.928620, 0.994420, 0.900000, 0.900000, 0.983720],
        [0.913070, 0.850000, 0.850000, 0.800000, 0.924280],
        [0.868090, 0.807170, 0.823550, 0.600000, 0.844520],
        [0.769570, 0.719870, 0.650000, 0.550000, 0.733500],
        [0.580250, 0.650000, 0.600000, 0.500000, 0.628850],
        [0.904770, 0.852650, 0.708370, 0.493730, 0.949030]]

    coeffs[6, 6, :, :] = [
        [0.911970, 0.800000, 0.800000, 0.800000, 0.956320],
        [0.912620, 0.682610, 0.750000, 0.700000, 0.950110],
        [0.653450, 0.659330, 0.700000, 0.600000, 0.856110],
        [0.648440, 0.600000, 0.641120, 0.500000, 0.695780],
        [0.570000, 0.550000, 0.598800, 0.400000, 0.560150],
        [0.475230, 0.500000, 0.518640, 0.339970, 0.520230],
        [0.743440, 0.592190, 0.603060, 0.316930, 0.794390]]

    return coeffs[1:, 1:, :, :]
