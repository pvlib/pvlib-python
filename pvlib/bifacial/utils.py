"""
The bifacial.utils module contains functions that support bifacial irradiance
modeling.
"""
import numpy as np
from pvlib.tools import sind, cosd, tand


#TODO: make private?
def solar_projection_tangent(solar_zenith, solar_azimuth, system_azimuth):
    """
    Calculate tangent of angle between sun vector projected to the YZ-plane
    (vertical and perpendicular to rows) and zenith vector.

    .. math::
        \\tan \\phi = \\cos\\left(\\text{solar azimuth}-\\text{system azimuth}
        \\right)\\tan\\left(\\text{solar zenith}\\right)

    Parameters
    ----------
    solar_zenith : numeric
        apparent zenith in degrees
    solar_azimuth : numeric
        azimuth in degrees
    system_azimuth : numeric
        system rotation from north in degrees

    Returns
    -------
    tan_phi : numeric
        Tangent of the angle between vertical and the projection of the 
        sun direction onto the YZ plane.
    """
    rotation = solar_azimuth - system_azimuth
    #TODO: I don't think tan_phi should ever be negative, but it could be if
    # rotation > 90 (e.g. sun north of along-row azimuth)
    tan_phi = cosd(rotation) * tand(solar_zenith)
    return tan_phi


def unshaded_ground_fraction(gcr, surface_tilt, surface_azimuth, solar_zenith,
                             solar_azimuth):
    """
    Calculate the fraction of the ground with incident direct irradiance.

    .. math::
        F_{gnd,sky} &= 1 - \\min{\\left(1, \\text{GCR} \\left|\\cos \\beta +
        \\sin \\beta \\tan \\phi \\right|\\right)} \\newline

        \\beta &= \\text{tilt}

    Parameters
    ----------
    gcr : numeric
        Ground coverage ratio, which is the ratio of row slant length to row
        spacing (pitch).
    surface_tilt: numeric
        Surface tilt angle in decimal degrees. The tilt angle is defined as
        degrees from horizontal, e.g., surface facing up = 0, surface facing
        horizon = 90.
    surface_azimuth: numeric
        Azimuth angle of the module surface in degrees.
        North=0, East=90, South=180, West=270.
    solar_zenith : numeric
        Solar zenith angle in degrees.
    solar_azimuth : numeric
        Solar azimuth angle in degrees.

    Returns
    -------
    f_gnd_beam : numeric
        Fraction of row pitch that is illuminated (unshaded).
    """
    #TODO: why np.abs? All angles should be <=90
    tan_phi = solar_projection_tangent(solar_zenith, solar_azimuth,
                                       surface_azimuth)
    f_gnd_beam = 1.0 - np.minimum(
        1.0, gcr * np.abs(sind(surface_tilt) + cosd(surface_tilt) * tan_phi))
    return f_gnd_beam  # 1 - min(1, abs()) < 1 always