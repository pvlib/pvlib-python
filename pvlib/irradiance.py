"""
The ``irradiance`` module contains functions for modeling
global horizontal irradiance, direct normal irradiance,
diffuse horizontal irradiance, and total irradiance
under various conditions.
"""

from __future__ import division

import logging
pvl_logger = logging.getLogger('pvlib')

import datetime

import numpy as np
import pandas as pd

from pvlib import tools
from pvlib import solarposition

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

# would be nice if this took pandas index as well.
# Use try:except of isinstance.


def extraradiation(datetime_or_doy, solar_constant=1366.1, method='spencer'):
    """
    Determine extraterrestrial radiation from day of year.

    Parameters
    ----------
    datetime_or_doy : int, float, array, pd.DatetimeIndex
        Day of year, array of days of year e.g. pd.DatetimeIndex.dayofyear,
        or pd.DatetimeIndex.

    solar_constant : float
        The solar constant.

    method : string
        The method by which the ET radiation should be calculated.
        Options include ``'pyephem', 'spencer', 'asce'``.

    Returns
    -------
    float or Series

        The extraterrestrial radiation present in watts per square meter
        on a surface which is normal to the sun. Ea is of the same size as the
        input doy.

        'pyephem' always returns a series.

    Notes
    -----
    The Spencer method contains a minus sign discrepancy between
    equation 12 of [1]. It's unclear what the correct formula is.

    References
    ----------
    [1] M. Reno, C. Hansen, and J. Stein, "Global Horizontal Irradiance Clear
    Sky Models: Implementation and Analysis", Sandia National
    Laboratories, SAND2012-2389, 2012.

    [2] <http://solardat.uoregon.edu/SolarRadiationBasics.html>,
    Eqs. SR1 and SR2

    [3] Partridge, G. W. and Platt, C. M. R. 1976.
    Radiative Processes in Meteorology and Climatology.

    [4] Duffie, J. A. and Beckman, W. A. 1991.
    Solar Engineering of Thermal Processes,
    2nd edn. J. Wiley and Sons, New York.

    See Also
    --------
    pvlib.clearsky.disc
    """

    pvl_logger.debug('irradiance.extraradiation()')

    method = method.lower()

    if isinstance(datetime_or_doy, pd.DatetimeIndex):
        doy = datetime_or_doy.dayofyear
        input_to_datetimeindex = lambda x: datetime_or_doy
    elif isinstance(datetime_or_doy, (int, float)):
        doy = datetime_or_doy
        input_to_datetimeindex = _scalar_to_datetimeindex
    else:  # assume that we have an array-like object of doy. danger?
        doy = datetime_or_doy
        input_to_datetimeindex = _array_to_datetimeindex

    B = (2 * np.pi / 365) * doy

    if method == 'asce':
        pvl_logger.debug('Calculating ET rad using ASCE method')
        RoverR0sqrd = 1 + 0.033 * np.cos(B)
    elif method == 'spencer':
        pvl_logger.debug('Calculating ET rad using Spencer method')
        RoverR0sqrd = (1.00011 + 0.034221 * np.cos(B) + 0.00128 * np.sin(B) +
                       0.000719 * np.cos(2 * B) + 7.7e-05 * np.sin(2 * B))
    elif method == 'pyephem':
        pvl_logger.debug('Calculating ET rad using pyephem method')
        times = input_to_datetimeindex(datetime_or_doy)
        RoverR0sqrd = solarposition.pyephem_earthsun_distance(times) ** (-2)

    Ea = solar_constant * RoverR0sqrd

    return Ea


def _scalar_to_datetimeindex(doy_scalar):
    """
    Convert a scalar day of year number to a pd.DatetimeIndex.

    Parameters
    ----------
    doy_array : int or float
        Contains days of the year

    Returns
    -------
    pd.DatetimeIndex
    """
    return pd.DatetimeIndex([_doy_to_timestamp(doy_scalar)])


def _array_to_datetimeindex(doy_array):
    """
    Convert an array of day of year numbers to a pd.DatetimeIndex.

    Parameters
    ----------
    doy_array : Iterable
        Contains days of the year

    Returns
    -------
    pd.DatetimeIndex
    """
    return pd.DatetimeIndex(list(map(_doy_to_timestamp, doy_array)))


def _doy_to_timestamp(doy, epoch='2013-12-31'):
    """
    Convert a numeric day of the year to a pd.Timestamp.

    Parameters
    ----------
    doy : int or float.
        Numeric day of year.
    epoch : pd.Timestamp compatible object.
        Date to which to add the day of year to.

    Returns
    -------
    pd.Timestamp
    """
    return pd.Timestamp('2013-12-31') + datetime.timedelta(days=float(doy))


def aoi_projection(surf_tilt, surf_az, sun_zen, sun_az):
    """
    Calculates the dot product of the solar vector and the surface normal.

    Input all angles in degrees.

    Parameters
    ==========

    surf_tilt : float or Series.
        Panel tilt from horizontal.
    surf_az : float or Series.
        Panel azimuth from north.
    sun_zen : float or Series.
        Solar zenith angle.
    sun_az : float or Series.
        Solar azimuth angle.

    Returns
    =======
    float or Series. Dot product of panel normal and solar angle.
    """

    projection = (
        tools.cosd(surf_tilt) * tools.cosd(sun_zen) + tools.sind(surf_tilt) *
        tools.sind(sun_zen) * tools.cosd(sun_az - surf_az))

    try:
        projection.name = 'aoi_projection'
    except AttributeError:
        pass

    return projection


def aoi(surf_tilt, surf_az, sun_zen, sun_az):
    """
    Calculates the angle of incidence of the solar vector on a surface.
    This is the angle between the solar vector and the surface normal.

    Input all angles in degrees.

    Parameters
    ==========

    surf_tilt : float or Series.
        Panel tilt from horizontal.
    surf_az : float or Series.
        Panel azimuth from north.
    sun_zen : float or Series.
        Solar zenith angle.
    sun_az : float or Series.
        Solar azimuth angle.

    Returns
    =======
    float or Series. Angle of incidence in degrees.
    """

    projection = aoi_projection(surf_tilt, surf_az, sun_zen, sun_az)
    aoi_value = np.rad2deg(np.arccos(projection))

    try:
        aoi_value.name = 'aoi'
    except AttributeError:
        pass

    return aoi_value


def poa_horizontal_ratio(surf_tilt, surf_az, sun_zen, sun_az):
    """
    Calculates the ratio of the beam components of the
    plane of array irradiance and the horizontal irradiance.

    Input all angles in degrees.

    Parameters
    ==========

    surf_tilt : float or Series.
        Panel tilt from horizontal.
    surf_az : float or Series.
        Panel azimuth from north.
    sun_zen : float or Series.
        Solar zenith angle.
    sun_az : float or Series.
        Solar azimuth angle.

    Returns
    =======
    float or Series. Ratio of the plane of array irradiance to the
    horizontal plane irradiance
    """

    cos_poa_zen = aoi_projection(surf_tilt, surf_az, sun_zen, sun_az)

    cos_sun_zen = tools.cosd(sun_zen)

    # ratio of titled and horizontal beam irradiance
    ratio = cos_poa_zen / cos_sun_zen

    try:
        ratio.name = 'poa_ratio'
    except AttributeError:
        pass

    return ratio


def beam_component(surf_tilt, surf_az, sun_zen, sun_az, DNI):
    """
    Calculates the beam component of the plane of array irradiance.

    Parameters
    ----------
    surf_tilt : float or Series.
        Panel tilt from horizontal.
    surf_az : float or Series.
        Panel azimuth from north.
    sun_zen : float or Series.
        Solar zenith angle.
    sun_az : float or Series.
        Solar azimuth angle.
    DNI : float or Series
        Direct Normal Irradiance

    Returns
    -------
    Series
    """
    beam = DNI * aoi_projection(surf_tilt, surf_az, sun_zen, sun_az)
    beam[beam < 0] = 0

    return beam


# ToDo: how to best structure this function? wholmgren 2014-11-03
def total_irrad(surf_tilt, surf_az,
                sun_zen, sun_az,
                DNI, GHI, DHI, DNI_ET=None, AM=None,
                albedo=.25, surface_type=None,
                model='isotropic',
                model_perez='allsitescomposite1990'):
    '''
    Determine diffuse irradiance from the sky on a
    tilted surface.

    .. math::

       I_{tot} = I_{beam} + I_{sky} + I_{ground}

    Parameters
    ----------
    surf_tilt : float or Series.
        Panel tilt from horizontal.
    surf_az : float or Series.
        Panel azimuth from north.
    sun_zen : float or Series.
        Solar zenith angle.
    sun_az : float or Series.
        Solar azimuth angle.
    DNI : float or Series
        Direct Normal Irradiance
    GHI : float or Series
        Global horizontal irradiance
    DHI : float or Series
        Diffuse horizontal irradiance
    DNI_ET : float or Series
        Extraterrestrial direct normal irradiance
    AM : float or Series
        Airmass
    albedo : float
        Surface albedo
    surface_type : String
        Surface type. See grounddiffuse.
    model : String
        Irradiance model.
    model_perez : String
        See perez.

    Returns
    -------
    DataFrame with columns ``'total', 'beam', 'sky', 'ground'``.

    References
    ----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267
    '''

    pvl_logger.debug('planeofarray.total_irrad()')

    beam = beam_component(surf_tilt, surf_az, sun_zen, sun_az, DNI)

    model = model.lower()
    if model == 'isotropic':
        sky = isotropic(surf_tilt, DHI)
    elif model == 'klutcher':
        sky = klucher(surf_tilt, surf_az, DHI, GHI, sun_zen, sun_az)
    elif model == 'haydavies':
        sky = haydavies(surf_tilt, surf_az, DHI, DNI, DNI_ET, sun_zen, sun_az)
    elif model == 'reindl':
        sky = reindl(surf_tilt, surf_az, DHI, DNI, GHI, DNI_ET, sun_zen,
                     sun_az)
    elif model == 'king':
        sky = king(surf_tilt, DHI, GHI, sun_zen)
    elif model == 'perez':
        sky = perez(surf_tilt, surf_az, DHI, DNI, DNI_ET, sun_zen, sun_az, AM,
                    modelt=model_perez)
    else:
        raise ValueError('invalid model selection {}'.format(model))

    ground = grounddiffuse(surf_tilt, GHI, albedo, surface_type)

    total = beam + sky + ground

    all_irrad = pd.DataFrame({'total': total,
                              'beam': beam,
                              'sky': sky,
                              'ground': ground})

    return all_irrad


# ToDo: keep this or not? wholmgren, 2014-11-03
def globalinplane(AOI, DNI, In_Plane_SkyDiffuse, GR):
    '''
    Determine the three components on in-plane irradiance

    Combines in-plane irradaince compoents from the chosen diffuse translation,
    ground reflection and beam irradiance algorithms into the total in-plane
    irradiance.

    Parameters
    ----------

    AOI : float or Series
          Angle of incidence of solar rays with respect
          to the module surface, from :func:`aoi`.

    DNI : float or Series
          Direct normal irradiance (W/m^2), as measured
          from a TMY file or calculated with a clearsky model.

    In_Plane_SkyDiffuse :  float or Series
          Diffuse irradiance (W/m^2) in the plane of the modules, as
          calculated by a diffuse irradiance translation function

    GR : float or Series
          a scalar or DataFrame of ground reflected irradiance (W/m^2),
          as calculated by a albedo model (eg. :func:`grounddiffuse`)

    Returns
    -------
    DataFrame with the following keys:
        * ``E`` : Total in-plane irradiance (W/m^2)
        * ``Eb`` : Total in-plane beam irradiance (W/m^2)
        * ``Ediff`` : Total in-plane diffuse irradiance (W/m^2)

    Notes
    ------
    Negative beam irradiation due to aoi :math:`> 90^{\circ}` or AOI
    :math:`< 0^{\circ}` is set to zero.
    '''

    Eb = pd.Series(DNI * np.cos(np.radians(AOI))).clip_lower(0)
    E = Eb + In_Plane_SkyDiffuse + GR
    Ediff = In_Plane_SkyDiffuse + GR

    return pd.DataFrame({'E': E, 'Eb': Eb, 'Ediff': Ediff})


def grounddiffuse(surf_tilt, ghi, albedo=.25, surface_type=None):
    '''
    Estimate diffuse irradiance from ground reflections given
    irradiance, albedo, and surface tilt

    Function to determine the portion of irradiance on a tilted surface due
    to ground reflections. Any of the inputs may be DataFrames or scalars.

    Parameters
    ----------
    surf_tilt : float or DataFrame
        Surface tilt angles in decimal degrees.
        SurfTilt must be >=0 and <=180. The tilt angle is defined as
        degrees from horizontal (e.g. surface facing up = 0, surface facing
        horizon = 90).

    ghi : float or DataFrame
        Global horizontal irradiance in W/m^2.

    albedo : float or DataFrame
        Ground reflectance, typically 0.1-0.4 for
        surfaces on Earth (land), may increase over snow, ice, etc. May also
        be known as the reflection coefficient. Must be >=0 and <=1.
        Will be overridden if surface_type is supplied.

    surface_type: None or string in
                  ``'urban', 'grass', 'fresh grass', 'snow', 'fresh snow',
                  'asphalt', 'concrete', 'aluminum', 'copper',
                  'fresh steel', 'dirty steel'``.
                  Overrides albedo.

    Returns
    -------

    float or DataFrame
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
        pvl_logger.info('surface_type={} mapped to albedo={}'
                        .format(surface_type, albedo))

    diffuse_irrad = ghi * albedo * (1 - np.cos(np.radians(surf_tilt))) * 0.5

    try:
        diffuse_irrad.name = 'diffuse_ground'
    except AttributeError:
        pass

    return diffuse_irrad


def isotropic(surf_tilt, DHI):
    r'''
    Determine diffuse irradiance from the sky on a
    tilted surface using the isotropic sky model.

    .. math::

       I_{d} = DHI \frac{1 + \cos\beta}{2}

    Hottel and Woertz's model treats the sky as a uniform source of diffuse
    irradiance. Thus the diffuse irradiance from the sky (ground reflected
    irradiance is not included in this algorithm) on a tilted surface can
    be found from the diffuse horizontal irradiance and the tilt angle of
    the surface.

    Parameters
    ----------

    surf_tilt : float or Series
        Surface tilt angle in decimal degrees.
        surf_tilt must be >=0 and <=180. The tilt angle is defined as
        degrees from horizontal (e.g. surface facing up = 0, surface facing
        horizon = 90)

    DHI : float or Series
        Diffuse horizontal irradiance in W/m^2.
        DHI must be >=0.


    Returns
    -------
    float or Series

    The diffuse component of the solar radiation  on an
    arbitrarily tilted surface defined by the isotropic sky model as
    given in Loutzenhiser et. al (2007) equation 3.
    SkyDiffuse is the diffuse component ONLY and does not include the ground
    reflected irradiance or the irradiance due to the beam.
    SkyDiffuse is a column vector vector with a number of elements equal to
    the input vector(s).


    References
    ----------

    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267

    [2] Hottel, H.C., Woertz, B.B., 1942. Evaluation of flat-plate solar heat
    collector. Trans. ASME 64, 91.
    '''

    pvl_logger.debug('diffuse_sky.isotropic()')

    sky_diffuse = DHI * (1 + tools.cosd(surf_tilt)) * 0.5

    return sky_diffuse


def klucher(surf_tilt, surf_az, DHI, GHI, sun_zen, sun_az):
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
    horizontal irradiance, extraterrestrial irradiance, sun zenith angle,
    and sun azimuth angle.

    Parameters
    ----------

    surf_tilt : float or Series
        Surface tilt angles in decimal degrees.
        surf_tilt must be >=0 and <=180. The tilt angle is defined as
        degrees from horizontal (e.g. surface facing up = 0, surface facing
        horizon = 90)

    surf_az : float or Series
        Surface azimuth angles in decimal degrees.
        surf_az must be >=0 and <=360. The Azimuth convention is defined
        as degrees east of north (e.g. North = 0, South=180 East = 90,
        West = 270).

    DHI : float or Series
        diffuse horizontal irradiance in W/m^2.
        DHI must be >=0.

    GHI : float or Series
        Global  irradiance in W/m^2.
        DNI must be >=0.

    sun_zen : float or Series
        apparent (refraction-corrected) zenith
        angles in decimal degrees.
        sun_zen must be >=0 and <=180.

    sun_az : float or Series
        Sun azimuth angles in decimal degrees.
        sun_az must be >=0 and <=360. The Azimuth convention is defined
        as degrees east of north (e.g. North = 0, East = 90, West = 270).

    Returns
    -------
    float or Series.

    The diffuse component of the solar radiation on an
    arbitrarily tilted surface defined by the Klucher model as given in
    Loutzenhiser et. al (2007) equation 4.
    SkyDiffuse is the diffuse component ONLY and does not include the ground
    reflected irradiance or the irradiance due to the beam.
    SkyDiffuse is a column vector vector with a number of elements equal to
    the input vector(s).

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
    cos_tt = aoi_projection(surf_tilt, surf_az, sun_zen, sun_az)

    F = 1 - ((DHI / GHI) ** 2)
    try:
        # fails with single point input
        F.fillna(0, inplace=True)
    except AttributeError:
        F = 0

    term1 = 0.5 * (1 + tools.cosd(surf_tilt))
    term2 = 1 + F * (tools.sind(0.5 * surf_tilt) ** 3)
    term3 = 1 + F * (cos_tt ** 2) * (tools.sind(sun_zen) ** 3)

    sky_diffuse = DHI * term1 * term2 * term3

    return sky_diffuse


def haydavies(surf_tilt, surf_az, DHI, DNI, DNI_ET, sun_zen, sun_az):
    r'''
    Determine diffuse irradiance from the sky on a
    tilted surface using Hay & Davies' 1980 model

    .. math::
        I_{d} = DHI ( A R_b + (1 - A) (\frac{1 + \cos\beta}{2}) )

    Hay and Davies' 1980 model determines the diffuse irradiance from the sky
    (ground reflected irradiance is not included in this algorithm) on a
    tilted surface using the surface tilt angle, surface azimuth angle,
    diffuse horizontal irradiance, direct normal irradiance,
    extraterrestrial irradiance, sun zenith angle, and sun azimuth angle.

    Parameters
    ----------

    surf_tilt : float or Series
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as
        degrees from horizontal (e.g. surface facing up = 0, surface facing
        horizon = 90)

    surf_az : float or Series
          Surface azimuth angles in decimal degrees.
          The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, South=180 East = 90,
          West = 270).

    DHI : float or Series
          diffuse horizontal irradiance in W/m^2.

    DNI : float or Series
          direct normal irradiance in W/m^2.

    DNI_ET : float or Series
          extraterrestrial normal irradiance in W/m^2.

    sun_zen : float or Series
          apparent (refraction-corrected) zenith
          angles in decimal degrees.

    sun_az : float or Series
          Sun azimuth angles in decimal degrees.
          The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, East = 90, West = 270).

    Returns
    --------

    SkyDiffuse : float or Series

          the diffuse component of the solar radiation  on an
          arbitrarily tilted surface defined by the Perez model as given in
          reference [3].
          SkyDiffuse is the diffuse component ONLY and does not include the
          ground reflected irradiance or the irradiance due to the beam.

    References
    -----------
    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267

    [2] Hay, J.E., Davies, J.A., 1980. Calculations of the solar radiation
    incident on an inclined surface. In: Hay, J.E., Won, T.K. (Eds.), Proc. of
    First Canadian Solar Radiation Data Workshop, 59. Ministry of Supply
    and Services, Canada.
    '''

    pvl_logger.debug('diffuse_sky.haydavies()')

    cos_tt = aoi_projection(surf_tilt, surf_az, sun_zen, sun_az)

    cos_sun_zen = tools.cosd(sun_zen)

    # ratio of titled and horizontal beam irradiance
    Rb = cos_tt / cos_sun_zen

    # Anisotropy Index
    AI = DNI / DNI_ET

    # these are actually the () and [] sub-terms of the second term of eqn 7
    term1 = 1 - AI
    term2 = 0.5 * (1 + tools.cosd(surf_tilt))

    sky_diffuse = DHI * (AI * Rb + term1 * term2)
    sky_diffuse[sky_diffuse < 0] = 0

    return sky_diffuse


def reindl(surf_tilt, surf_az, DHI, DNI, GHI, DNI_ET, sun_zen, sun_az):
    r'''
    Determine diffuse irradiance from the sky on a
    tilted surface using Reindl's 1990 model

    .. math::

       I_{d} = DHI (A R_b + (1 - A) (\frac{1 + \cos\beta}{2})
       (1 + \sqrt{\frac{I_{hb}}{I_h}} \sin^3(\beta/2)) )

    Reindl's 1990 model determines the diffuse irradiance from the sky
    (ground reflected irradiance is not included in this algorithm) on a
    tilted surface using the surface tilt angle, surface azimuth angle,
    diffuse horizontal irradiance, direct normal irradiance, global
    horizontal irradiance, extraterrestrial irradiance, sun zenith angle,
    and sun azimuth angle.

    Parameters
    ----------

    surf_tilt : float or Series.
        Surface tilt angles in decimal degrees.
        The tilt angle is defined as
        degrees from horizontal (e.g. surface facing up = 0, surface facing
        horizon = 90)

    surf_az : float or Series.
        Surface azimuth angles in decimal degrees.
        The Azimuth convention is defined
        as degrees east of north (e.g. North = 0, South=180 East = 90,
        West = 270).

    DHI : float or Series.
        diffuse horizontal irradiance in W/m^2.

    DNI : float or Series.
        direct normal irradiance in W/m^2.

    GHI: float or Series.
        Global irradiance in W/m^2.

    DNI_ET : float or Series.
        extraterrestrial normal irradiance in W/m^2.

    sun_zen : float or Series.
        apparent (refraction-corrected) zenith
        angles in decimal degrees.

    sun_az : float or Series.
        Sun azimuth angles in decimal degrees.
        The Azimuth convention is defined
        as degrees east of north (e.g. North = 0, East = 90, West = 270).

    Returns
    -------

    SkyDiffuse : float or Series.

        The diffuse component of the solar radiation  on an
        arbitrarily tilted surface defined by the Reindl model as given in
        Loutzenhiser et. al (2007) equation 8.
        SkyDiffuse is the diffuse component ONLY and does not include the
        ground reflected irradiance or the irradiance due to the beam.
        SkyDiffuse is a column vector vector with a number of elements equal to
        the input vector(s).


    Notes
    -----

    The POAskydiffuse calculation is generated from the Loutzenhiser et al.
    (2007) paper, equation 8. Note that I have removed the beam and ground
    reflectance portion of the equation and this generates ONLY the diffuse
    radiation from the sky and circumsolar, so the form of the equation
    varies slightly from equation 8.

    References
    ----------

    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267

    [2] Reindl, D.T., Beckmann, W.A., Duffie, J.A., 1990a. Diffuse fraction
    correlations. Solar Energy 45(1), 1-7.

    [3] Reindl, D.T., Beckmann, W.A., Duffie, J.A., 1990b. Evaluation of hourly
    tilted surface radiation models. Solar Energy 45(1), 9-17.
    '''

    pvl_logger.debug('diffuse_sky.reindl()')

    cos_tt = aoi_projection(surf_tilt, surf_az, sun_zen, sun_az)

    cos_sun_zen = tools.cosd(sun_zen)

    # ratio of titled and horizontal beam irradiance
    Rb = cos_tt / cos_sun_zen

    # Anisotropy Index
    AI = DNI / DNI_ET

    # DNI projected onto horizontal
    HB = DNI * cos_sun_zen
    HB[HB < 0] = 0

    # these are actually the () and [] sub-terms of the second term of eqn 8
    term1 = 1 - AI
    term2 = 0.5 * (1 + tools.cosd(surf_tilt))
    term3 = 1 + np.sqrt(HB / GHI) * (tools.sind(0.5 * surf_tilt) ** 3)

    sky_diffuse = DHI * (AI * Rb + term1 * term2 * term3)
    sky_diffuse[sky_diffuse < 0] = 0

    return sky_diffuse


def king(surf_tilt, DHI, GHI, sun_zen):
    '''
    Determine diffuse irradiance from the sky on a tilted surface using the
    King model.

    King's model determines the diffuse irradiance from the sky
    (ground reflected irradiance is not included in this algorithm) on a
    tilted surface using the surface tilt angle, diffuse horizontal
    irradiance, global horizontal irradiance, and sun zenith angle. Note
    that this model is not well documented and has not been published in
    any fashion (as of January 2012).

    Parameters
    ----------

    surf_tilt : float or Series
          Surface tilt angles in decimal degrees.
          The tilt angle is defined as
          degrees from horizontal (e.g. surface facing up = 0, surface facing
          horizon = 90)

    DHI : float or Series
          diffuse horizontal irradiance in W/m^2.

    GHI : float or Series
          global horizontal irradiance in W/m^2.

    sun_zen : float or Series
          apparent (refraction-corrected) zenith
          angles in decimal degrees.

    Returns
    --------

    SkyDiffuse : float or Series

            the diffuse component of the solar radiation  on an
            arbitrarily tilted surface as given by a model developed by
            David L. King at Sandia National Laboratories.
    '''

    pvl_logger.debug('diffuse_sky.king()')

    sky_diffuse = (DHI * ((1 + tools.cosd(surf_tilt))) / 2 + GHI *
                   ((0.012 * sun_zen - 0.04)) *
                   ((1 - tools.cosd(surf_tilt))) / 2)
    sky_diffuse[sky_diffuse < 0] = 0

    return sky_diffuse


def perez(surf_tilt, surf_az, DHI, DNI, DNI_ET, sun_zen, sun_az, AM,
          modelt='allsitescomposite1990'):
    '''
    Determine diffuse irradiance from the sky on a tilted surface using one of
    the Perez models.

    Perez models determine the diffuse irradiance from the sky (ground
    reflected irradiance is not included in this algorithm) on a tilted
    surface using the surface tilt angle, surface azimuth angle, diffuse
    horizontal irradiance, direct normal irradiance, extraterrestrial
    irradiance, sun zenith angle, sun azimuth angle, and relative (not
    pressure-corrected) airmass. Optionally a selector may be used to use
    any of Perez's model coefficient sets.


    Parameters
    ----------

    surf_tilt : float or Series
          Surface tilt angles in decimal degrees.
          surf_tilt must be >=0 and <=180. The tilt angle is defined as
          degrees from horizontal (e.g. surface facing up = 0, surface facing
          horizon = 90)

    surf_az : float or Series
          Surface azimuth angles in decimal degrees.
          surf_az must be >=0 and <=360. The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, South=180 East = 90,
          West = 270).

    DHI : float or Series
          diffuse horizontal irradiance in W/m^2.
          DHI must be >=0.

    DNI : float or Series
          direct normal irradiance in W/m^2.
          DNI must be >=0.

    DNI_ET : float or Series
          extraterrestrial normal irradiance in W/m^2.
           DNI_ET must be >=0.

    sun_zen : float or Series
          apparent (refraction-corrected) zenith
          angles in decimal degrees.
          sun_zen must be >=0 and <=180.

    sun_az : float or Series
          Sun azimuth angles in decimal degrees.
          sun_az must be >=0 and <=360. The Azimuth convention is defined
          as degrees east of north (e.g. North = 0, East = 90, West = 270).

    AM : float or Series
          relative (not pressure-corrected) airmass
          values. If AM is a DataFrame it must be of the same size as all other
          DataFrame inputs. AM must be >=0 (careful using the 1/sec(z) model of
          AM generation)

    Other Parameters
    ----------------

    model : string (optional, default='allsitescomposite1990')

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

    float or Series

          the diffuse component of the solar radiation  on an
          arbitrarily tilted surface defined by the Perez model as given in
          reference [3].
          SkyDiffuse is the diffuse component ONLY and does not include the
          ground reflected irradiance or the irradiance due to the beam.


    References
    ----------

    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267

    [2] Perez, R., Seals, R., Ineichen, P., Stewart, R., Menicucci, D., 1987.
    A new simplified version of the Perez diffuse irradiance model for tilted
    surfaces. Solar Energy 39(3), 221-232.

    [3] Perez, R., Ineichen, P., Seals, R., Michalsky, J., Stewart, R., 1990.
    Modeling daylight availability and irradiance components from direct
    and global irradiance. Solar Energy 44 (5), 271-289.

    [4] Perez, R. et. al 1988. "The Development and Verification of the
    Perez Diffuse Radiation Model". SAND88-7030
    '''

    pvl_logger.debug('diffuse_sky.perez()')

    kappa = 1.041  # for sun_zen in radians
    z = np.radians(sun_zen)  # convert to radians

    # epsilon is the sky's "clearness"
    eps = ((DHI + DNI) / DHI + kappa * (z ** 3)) / (1 + kappa * (z ** 3))

    # Perez et al define clearness bins according to the following rules.
    # 1 = overcast ... 8 = clear
    # (these names really only make sense for small zenith angles, but...)
    # these values will eventually be used as indicies for coeffecient look ups
    ebin = eps.copy()
    ebin[(eps < 1.065)] = 1
    ebin[(eps >= 1.065) & (eps < 1.23)] = 2
    ebin[(eps >= 1.23) & (eps < 1.5)] = 3
    ebin[(eps >= 1.5) & (eps < 1.95)] = 4
    ebin[(eps >= 1.95) & (eps < 2.8)] = 5
    ebin[(eps >= 2.8) & (eps < 4.5)] = 6
    ebin[(eps >= 4.5) & (eps < 6.2)] = 7
    ebin[eps >= 6.2] = 8

    ebin = ebin - 1  # correct for 0 indexing in coeffecient lookup

    # remove night time values
    ebin = ebin.dropna().astype(int)

    # This is added because in cases where the sun is below the horizon
    # (var.sun_zen > 90) but there is still diffuse horizontal light
    # (var.DHI>0), it is possible that the airmass (var.AM) could be NaN, which
    # messes up later calculations. Instead, if the sun is down, and there is
    # still var.DHI, we set the airmass to the airmass value on the horizon
    # (approximately 37-38).
    # var.AM(var.sun_zen >=90 & var.DHI >0) = 37;

    # var.DNI_ET[var.DNI_ET==0] = .00000001 #very hacky, fix this

    # delta is the sky's "brightness"
    delta = DHI * AM / DNI_ET

    # keep only valid times
    delta = delta[ebin.index]
    z = z[ebin.index]

    # The various possible sets of Perez coefficients are contained
    # in a subfunction to clean up the code.
    F1c, F2c = _get_perez_coefficients(modelt)

    F1 = F1c[ebin, 0] + F1c[ebin, 1] * delta + F1c[ebin, 2] * z
    F1[F1 < 0] = 0
    F1 = F1.astype(float)

    F2 = F2c[ebin, 0] + F2c[ebin, 1] * delta + F2c[ebin, 2] * z
    F2[F2 < 0] = 0
    F2 = F2.astype(float)

    A = aoi_projection(surf_tilt, surf_az, sun_zen, sun_az)
    A[A < 0] = 0

    B = tools.cosd(sun_zen)
    B[B < tools.cosd(85)] = tools.cosd(85)

    # Calculate Diffuse POA from sky dome

    term1 = 0.5 * (1 - F1) * (1 + tools.cosd(surf_tilt))
    term2 = F1 * A[ebin.index] / B[ebin.index]
    term3 = F2 * tools.sind(surf_tilt)

    sky_diffuse = DHI[ebin.index] * (term1 + term2 + term3)
    sky_diffuse[sky_diffuse < 0] = 0

    return sky_diffuse


def _get_perez_coefficients(perezmodelt):
    '''
    Find coefficients for the Perez model

    Parameters
    ----------

    perezmodelt : string (optional, default='allsitescomposite1990')

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

          F1coeffs : array
          F1 coefficients for the Perez model
          F2coeffs : array
          F2 coefficients for the Perez model

    References
    ----------

    [1] Loutzenhiser P.G. et. al. "Empirical validation of models to compute
    solar irradiance on inclined surfaces for building energy simulation"
    2007, Solar Energy vol. 81. pp. 254-267

    [2] Perez, R., Seals, R., Ineichen, P., Stewart, R., Menicucci, D., 1987.
    A new simplified version of the Perez diffuse irradiance model for tilted
    surfaces. Solar Energy 39(3), 221-232.

    [3] Perez, R., Ineichen, P., Seals, R., Michalsky, J., Stewart, R., 1990.
    Modeling daylight availability and irradiance components from direct
    and global irradiance. Solar Energy 44 (5), 271-289.

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

    array = np.array(coeffdict[perezmodelt])

    F1coeffs = array.T[0:3].T
    F2coeffs = array.T[3:7].T

    return F1coeffs, F2coeffs
