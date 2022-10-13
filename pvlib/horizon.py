'''
The 'horizon' module contains function definitions that
retrive & calculate the surrounding horizon using DEM
elevation data and its impact on performance.
Optional dependencies for this module include
gdal, osgeo, geoio, and scikit-image.
'''
import numpy as np
from pathlib import Path

def dni_horizon_adjustment(horizon_angles, solar_zenith, solar_azimuth):
    '''
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

def calculate_dtf(horizon_azimuths, horizon_angles,
                  surface_tilt, surface_azimuth):
    """
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
    _____
    The dtf in this method is calculated by approximating the surface integral
    over the visible section of the sky dome. The integrand of the surface
    integral is the cosine of the angle between the incoming radiation and the
    vector normal to the surface. The method calculates a sum of integrations
    from the "peak" of the sky dome down to the elevation angle of the horizon.
    A similar method is used in section II of [1] although it is looking at
    both ground and sky diffuse irradiation.
    [2] Wright D. (2019) IEEE Journal of Photovoltaics 9(2), 391-396
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
    """
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
    """
    tilt = np.radians(surface_tilt)
    bearing = np.radians(direction - surface_azimuth - 180.0)

    declination = np.degrees(np.arctan(1.0/np.tan(tilt)/np.cos(bearing)))
    mask = (declination <= 0)
    elevation_angle = 90.0 - declination
    elevation_angle[mask] = 0.0

    return elevation_angle
