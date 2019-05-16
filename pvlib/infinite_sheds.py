"""
Modify plane of array irradiance components to account for adjancent rows for
both monofacial and bifacia infinite sheds. Sheds are defined as fixed tilt or
trackers that a fixed GCR on horizontant surface. Future version will also
account for sloped surfaces. The process is divide into 7 steps:
1. transposition
2. ground illumination
3. shade line
4. angle of sky/ground subtended by module
5. view factor of sky/ground from module vs. location of shade line
6. approximate view factors as linear across shade and light regions
7. sum up components

References
----------
[1] Bill Marion, et al. IEEE PVSC 2017
"""

import numpy as np


def solar_projection_tangent(solar_zenith, solar_azimuth, system_azimuth):
    """
    Calculate solar projection on YZ-plane, vertical and perpendicular to rows.

    Parameters
    ----------
    solar_zenith : numeric
        Apparent zenith in degrees.
    solar_azimuth : numeric
        Azimuth in degrees.
    system_azimuth : numeric
        System rotation from north in degrees.

    Returns
    -------
    tanphi : numeric
        tangent of the solar projection
    """
    solar_zenith_rad = np.radians(solar_zenith)
    solar_azimuth_rad = np.radians(solar_azimuth)
    system_azimuth_rad = np.radians(system_azimuth)
    rotation_rad = solar_azimuth_rad - system_azimuth_rad
    tanphi = np.cos(rotation_rad) * np.tan(solar_zenith_rad)
    return tanphi


def solar_projection(solar_zenith, solar_azimuth, system_azimuth):
    """
    Calculate solar projection on YZ-plane, vertical and perpendicular to rows.

    Parameters
    ----------
    solar_zenith : numeric
        Apparent zenith in degrees.
    solar_azimuth : numeric
        Azimuth in degrees.
    system_azimuth : numeric
        System rotation from north in degrees.

    Returns
    -------
    phi : numeric
        4-quadrant arc-tangent of solar projection
    tanphi : numeric
        tangent of the solar projection
    """
    solar_zenith_rad = np.radians(solar_zenith)
    solar_azimuth_rad = np.radians(solar_azimuth)
    system_azimuth_rad = np.radians(system_azimuth)
    rotation_rad = solar_azimuth_rad - system_azimuth_rad
    x1 = np.cos(rotation_rad) * np.sin(solar_zenith_rad)
    x2 = np.cos(solar_zenith_rad)
    tanphi = x1 / x2
    phi = np.arctan2(x1, x2)
    return phi, tanphi


def ground_illumination(GCR, tilt, tanphi):
    """
    Calculate the fraction of the ground visible from the sky.

    Parameters
    ----------
    GCR : numeric
        ratio of module length versus row spacing
    tilt : numeric
        angle of module normal from vertical in degrees, if bifacial use front
    tanphi : numeric
        solar projection tangent

    Returns
    -------
    Fgnd_sky : numeric
        fration of ground illumination
    """
    tilt_rad = np.radians(tilt)
    Fgnd_sky = 1.0 - np.minimum(
        1.0, GCR * np.abs(np.cos(tilt_rad) + np.sin(tilt_rad) * tanphi))
    return Fgnd_sky  # 1 - min(1, abs()) < 1 always


def diffuse_fraction(GHI, DHI):
    """
    ratio of DHI to GHI

    Parameters
    ----------
    GHI : numeric
        global horizontal irradiance in W/m^2
    DHI : numeric
        diffuse horizontal irradiance in W/m^2

    Returns : numeric
        diffuse fraction
    """
    return DHI/GHI


def poa_gnd_sky(poa_ground, f_gnd_sky, diffuse_fraction):
    """
    transposed ground reflected diffuse component adjusted for ground
    illumination, but not accounting for adjacent rows

    Parameters
    ----------
    poa_ground : numeric
        transposed ground reflected diffuse component in W/m^2
    f_gnd_sky : numeric
        ground illumination
    diffuse_fraction : numeric
        ratio of DHI to GHI

    Returns
    -------
    poa_gnd_sky : numeric
        adjusted irradiance on modules reflected from ground
    """
    poa_ground * (f_gnd_sky * (1 - diffuse_fraction) + diffuse_fraction)


def shade_line(GCR, tilt, tanphi):
    """
    calculate fraction of module shaded from the bottom

    Parameters
    ----------
    GCR : numeric
        ratio of module length versus row spacing
    tilt : numeric
        angle of module normal from vertical in degrees, if bifacial use front
    tanphi : numeric
        solar projection tangent

    Returns
    -------
    Fx : numeric
        fraction of module shaded from the bottom
    """
    tilt_rad = np.radians(tilt)
    Fx = 1.0 - 1.0 / GCR / (np.cos(tilt_rad) + np.sin(tilt_rad) * tanphi)
    return np.maximum(0.0, np.minimum(Fx, 1.0))


def sky_angle(GCR, tilt, f_x):
    """
    angle from shade line to top of next row

    .. math::

        \\tan{\\psi_t} &= \\frac{F_y \\text{GCR}
            \\sin{\\beta}}{1 - F_y \\text{GCR} \\cos{\\beta}} \\newline

        F_y &= 1 - F_x

    Parameters
    ----------
    GCR : numeric
        ratio of module length versus row spacing
    tilt : numeric
        angle of module normal from vertical in degrees, if bifacial use front
    f_x : numeric
        fraction of module shaded from bottom

    Returns
    -------
    psi_top : numeric
        angle from shade line to top of next row
    """
    tilt_rad = np.radians(tilt)
    f_y = 1.0 - f_x
    return f_y * np.sin(tilt_rad) / (1/GCR - f_y * np.cos(tilt_rad))



def sky_angle_0(GCR, tilt):
    """
    angle to top of next row with no shade (shade line at bottom)

    .. math::

        \\tan{\\psi_t} &= \\frac{\\text{GCR} \\sin{\\beta}}{1 - \\text{GCR}
            \\cos{\\beta}}

    Parameters
    ----------
    GCR : numeric
        ratio of module length versus row spacing
    tilt : numeric
        angle of module normal from vertical in degrees, if bifacial use front

    Returns
    -------
    psi_top : numeric
        angle from shade line to top of next row
    """
    return sky_angle(GCR, tilt, 0.0)


def get_irradiance():
    """hi"""
    return


class InfiniteSheds():
    """hi"""
    def __init__(self, x, y):
        self.x = x
        self.y = y
