"""
The ``shading`` module contains functions that model module shading and the
associated effects on PV module output
"""

import numpy as np
import pandas as pd
from pvlib.tools import sind, cosd


def masking_angle(surface_tilt, gcr, slant_height):
    """
    The elevation angle below which diffuse irradiance is blocked.

    The ``height`` parameter determines how far up the module's surface to
    evaluate the masking angle.  The lower the point, the steeper the masking
    angle [1]_.  SAM uses a "worst-case" approach where the masking angle
    is calculated for the bottom of the array (i.e. ``slant_height=0``) [2]_.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal [degrees].

    gcr : float
        The ground coverage ratio of the array [unitless].

    slant_height : numeric
        The distance up the module's slant height to evaluate the masking
        angle, as a fraction [0-1] of the module slant height [unitless].

    Returns
    -------
    mask_angle : numeric
        Angle from horizontal where diffuse light is blocked by the
        preceding row [degrees].

    See Also
    --------
    masking_angle_passias
    sky_diffuse_passias

    References
    ----------
    .. [1] D. Passias and B. Källbäck, "Shading effects in rows of solar cell
       panels", Solar Cells, Volume 11, Pages 281-291.  1984.
       DOI: 10.1016/0379-6787(84)90017-6
    .. [2] Gilman, P. et al., (2018). "SAM Photovoltaic Model Technical
       Reference Update", NREL Technical Report NREL/TP-6A20-67399.
       Available at https://www.nrel.gov/docs/fy18osti/67399.pdf
    """
    # The original equation (8 in [1]) requires pitch and collector width,
    # but it's easy to non-dimensionalize it to make it a function of GCR
    # by factoring out B from the argument to arctan.
    numerator = gcr * (1 - slant_height) * sind(surface_tilt)
    denominator = 1 - gcr * (1 - slant_height) * cosd(surface_tilt)
    phi = np.arctan(numerator / denominator)
    return np.degrees(phi)


def masking_angle_passias(surface_tilt, gcr):
    r"""
    The average masking angle over the slant height of a row.

    The masking angle is the angle from horizontal where the sky dome is
    blocked by the row in front. The masking angle is larger near the lower
    edge of a row than near the upper edge. This function calculates the
    average masking angle as described in [1]_.

    Parameters
    ----------
    surface_tilt : numeric
        Panel tilt from horizontal [degrees].

    gcr : float
        The ground coverage ratio of the array [unitless].

    Returns
    ----------
    mask_angle : numeric
        Average angle from horizontal where diffuse light is blocked by the
        preceding row [degrees].

    See Also
    --------
    masking_angle
    sky_diffuse_passias

    Notes
    -----
    The pvlib-python authors believe that Eqn. 9 in [1]_ is incorrect.
    Here we use an independent equation.  First, Eqn. 8 is non-dimensionalized
    (recasting in terms of GCR):

    .. math::

        \psi(z') = \arctan \left [
            \frac{(1 - z') \sin \beta}
                 {\mathrm{GCR}^{-1} + (z' - 1) \cos \beta}
        \right ]

    Where :math:`GCR = B/C` and :math:`z' = z/B`. The average masking angle
    :math:`\overline{\psi} = \int_0^1 \psi(z') \mathrm{d}z'` is then
    evaluated symbolically using Maxima (using :math:`X = 1/\mathrm{GCR}`):

    .. code-block:: none

        load(scifac)    /* for the gcfac function */
        assume(X>0, cos(beta)>0, cos(beta)-X<0);   /* X is 1/GCR */
        gcfac(integrate(atan((1-z)*sin(beta)/(X+(z-1)*cos(beta))), z, 0, 1))

    This yields the equation implemented by this function:

    .. math::

        \overline{\psi} = \
            &-\frac{X}{2} \sin\beta \log | 2 X \cos\beta - (X^2 + 1)| \\
            &+ (X \cos\beta - 1) \arctan \frac{X \cos\beta - 1}{X \sin\beta} \\
            &+ (1 - X \cos\beta) \arctan \frac{\cos\beta}{\sin\beta} \\
            &+ X \log X \sin\beta

    The pvlib-python authors have validated this equation against numerical
    integration of :math:`\overline{\psi} = \int_0^1 \psi(z') \mathrm{d}z'`.

    References
    ----------
    .. [1] D. Passias and B. Källbäck, "Shading effects in rows of solar cell
       panels", Solar Cells, Volume 11, Pages 281-291.  1984.
       DOI: 10.1016/0379-6787(84)90017-6
    """
    # wrap it in an array so that division by zero is handled well
    beta = np.radians(np.array(surface_tilt))
    sin_b = np.sin(beta)
    cos_b = np.cos(beta)
    X = 1/gcr

    with np.errstate(divide='ignore', invalid='ignore'):  # ignore beta=0
        term1 = -X * sin_b * np.log(np.abs(2 * X * cos_b - (X**2 + 1))) / 2
        term2 = (X * cos_b - 1) * np.arctan((X * cos_b - 1) / (X * sin_b))
        term3 = (1 - X * cos_b) * np.arctan(cos_b / sin_b)
        term4 = X * np.log(X) * sin_b

    psi_avg = term1 + term2 + term3 + term4
    # when beta=0, divide by zero makes psi_avg NaN.  replace with 0:
    psi_avg = np.where(np.isfinite(psi_avg), psi_avg, 0)

    if isinstance(surface_tilt, pd.Series):
        psi_avg = pd.Series(psi_avg, index=surface_tilt.index)

    return np.degrees(psi_avg)


def sky_diffuse_passias(masking_angle):
    r"""
    The diffuse irradiance loss caused by row-to-row sky diffuse shading.

    Even when the sun is high in the sky, a row's view of the sky dome will
    be partially blocked by the row in front. This causes a reduction in the
    diffuse irradiance incident on the module. The reduction depends on the
    masking angle, the elevation angle from a point on the shaded module to
    the top of the shading row. In [1]_ the masking angle is calculated as
    the average across the module height. SAM assumes the "worst-case" loss
    where the masking angle is calculated for the bottom of the array [2]_.

    This function, as in [1]_, makes the assumption that sky diffuse
    irradiance is isotropic.

    Parameters
    ----------
    masking_angle : numeric
        The elevation angle below which diffuse irradiance is blocked
        [degrees].

    Returns
    -------
    derate : numeric
        The fraction [0-1] of blocked sky diffuse irradiance.

    See Also
    --------
    masking_angle
    masking_angle_passias

    References
    ----------
    .. [1] D. Passias and B. Källbäck, "Shading effects in rows of solar cell
       panels", Solar Cells, Volume 11, Pages 281-291.  1984.
       DOI: 10.1016/0379-6787(84)90017-6
    .. [2] Gilman, P. et al., (2018). "SAM Photovoltaic Model Technical
       Reference Update", NREL Technical Report NREL/TP-6A20-67399.
       Available at https://www.nrel.gov/docs/fy18osti/67399.pdf
    """
    return 1 - cosd(masking_angle/2)**2


def calculate_dtf(horizon_azimuths, horizon_angles,
                  surface_tilt, surface_azimuth):
    r"""Author: JPalakapillyKWH
    Calculate the diffuse tilt factor for a tilted plane that is adjusted
    with for horizon profile. The idea for a diffuse tilt factor is explained
    in [1].
    Parameters
    ----------
    horizon_azimuths: numeric
        Azimuth values for points that define the horizon profile. The ith
        element in this array corresponds to the ith element in horizon_angles.
    horizon_angles: numeric
        Elevation angle values for points that define the horizon profile. The
        elevation angle of the horizon is the angle that the horizon makes with
        the horizontal. It is given in degrees above the horizontal. The ith
        element in this array corresponds to the ith element in
        horizon_azimuths.
    surface_tilt : numeric
        Surface tilt angles in decimal degrees. surface_tilt must be >=0
        and <=180. The tilt angle is defined as degrees from horizontal
        (e.g. surface facing up = 0, surface facing horizon = 90)
    surface_azimuth : numeric
        Surface azimuth angles in decimal degrees. surface_azimuth must
        be >=0 and <=360. The azimuth convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).
    Returns
    -------
    dtf: numeric
        The diffuse tilt factor that can be multiplied with the diffuse
        horizontal irradiance (DHI) to get the incident irradiance from
        the sky that is adjusted for the horizon profile and the tilt of
        the plane.

    Notes
    ------
    The dtf in this method is calculated by approximating the surface integral
    over the visible section of the sky dome. The integrand of the surface
    integral is the cosine of the angle between the incoming radiation and the
    vector normal to the surface. The method calculates a sum of integrations
    from the "peak" of the sky dome down to the elevation angle of the horizon.
    A similar method is used in section II of [1] although it is looking at
    both ground and sky diffuse irradiation.
    [2] Wright D. (2019) IEEE Journal of Photovoltaics 9(2), 391-396

    This function was written by @JPalakapillyKWH
    in an uncompleted pvlib-python pull request #758.
    """
    if horizon_azimuths.shape[0] != horizon_angles.shape[0]:
        raise ValueError('azimuths and elevation_angles must be of the same'
                         'length.')
    tilt_rad = np.radians(surface_tilt)
    plane_az_rad = np.radians(surface_azimuth)
    a = np.sin(tilt_rad) * np.cos(plane_az_rad)
    b = np.sin(tilt_rad) * np.sin(plane_az_rad)
    c = np.cos(tilt_rad)

    # this gets either a float or an array of zeros
    dtf = np.multiply(0.0, surface_tilt)
    num_points = horizon_azimuths.shape[0]
    for i in range(horizon_azimuths.shape[0]):
        az = np.radians(horizon_azimuths[i])
        horizon_elev = np.radians(horizon_angles[i])
        temp = np.radians(collection_plane_elev_angle(surface_tilt,
                                                      surface_azimuth,
                                                      horizon_azimuths[i]))
        elev = np.maximum(horizon_elev, temp)

        first_term = .5 * (a*np.cos(az) + b*np.sin(az)) * \
                          (np.pi/2 - elev - np.sin(elev) * np.cos(elev))
        second_term = .5 * c * np.cos(elev)**2
        dtf += 2 * (first_term + second_term) / num_points
    return dtf


def collection_plane_elev_angle(surface_tilt, surface_azimuth, direction):
    r"""
    Determine the elevation angle created by the surface of a tilted plane
    intersecting the plane tangent to the Earth's surface in a given direction.
    The angle is limited to be non-negative. This comes from Equation 10 in [1]
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
    direction : numeric
        The direction along which the elevation angle is to be calculated in
        decimal degrees. The convention is defined as degrees
        east of north (e.g. North = 0, South=180 East = 90, West = 270).
    Returns
    --------
    elevation_angle : numeric
        The angle between the surface of the tilted plane and the horizontal
        when looking in the specified direction. Given in degrees above the
        horizontal and limited to be non-negative.
    [1] doi.org/10.1016/j.solener.2014.09.037

    Notes
    ------
    This function was written by @JPalakapillyKWH
    in an uncompleted pvlib-python pull request #758.
    """
    tilt = np.radians(surface_tilt)
    bearing = np.radians(direction - surface_azimuth - 180.0)

    declination = np.degrees(np.arctan(1.0/np.tan(tilt)/np.cos(bearing)))
    mask = (declination <= 0)
    elevation_angle = 90.0 - declination
    elevation_angle[mask] = 0.0

    return elevation_angle


def dni_horizon_adjustment(horizon_angles, solar_zenith, solar_azimuth):
    r'''
    Calculates an adjustment to direct normal irradiance based on a horizon
    profile. The adjustment is a vector of binary values with the same length
    as the provided solar position values. Where the sun is below the horizon,
    the adjustment vector is 0 and it is 1 elsewhere. The horizon profile must
    be given as a vector with 360 values where the ith value corresponds to the
    ith degree of azimuth (0-359).
    Parameters
    ----------
    horizon_angles: numeric
        Elevation angle values for points that define the horizon profile. The
        elevation angle of the horizon is the angle that the horizon makes with
        the horizontal. It is given in degrees above the horizontal. The ith
        element in this array corresponds to the ith degree of azimuth.
    solar_zenith : numeric
        Solar zenith angle.
    solar_azimuth : numeric
        Solar azimuth angle.
    Returns
    -------
    adjustment : numeric
        A vector of binary values with the same shape as the inputted solar
        position values. 0 when the sun is below the horizon and 1 elsewhere.

    Notes
    ------
    This function was written by @JPalakapillyKWH
    in an uncompleted pvlib-python pull request #758.
    '''
    adjustment = np.ones(solar_zenith.shape)

    if (horizon_angles.shape[0] != 360):
        raise ValueError('horizon_angles must contain exactly 360 values'
                         '(for each degree of azimuth 0-359).')

    rounded_solar_azimuth = np.round(solar_azimuth).astype(int)
    rounded_solar_azimuth[rounded_solar_azimuth == 360] = 0
    horizon_zenith = 90 - horizon_angles[rounded_solar_azimuth]
    mask = solar_zenith > horizon_zenith
    adjustment[mask] = 0
    return adjustment
