"""
The bifacial.utils module contains functions that support bifacial irradiance
modeling.
"""
import numpy as np
from pvlib.tools import sind, cosd, tand


# TODO: make private?
def solar_projection_tangent(solar_zenith, solar_azimuth, system_azimuth):
    """
    Tangent of the angle between the sun vector projected to the YZ-plane
    (vertical and perpendicular to rows) and the zenith vector.

    Tangent is positive when the projection of the sun vector is in the same
    hemisphere as the system azimuth.

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
    # TODO: I don't think tan_phi should ever be negative, but it could be if
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
        Fraction of distance betwen rows (pitch) that has direct irradiance
        (unshaded).
    """
    # TODO: why np.abs? All angles should be <=90
    tan_phi = solar_projection_tangent(solar_zenith, solar_azimuth,
                                       surface_azimuth)
    f_gnd_beam = 1.0 - np.minimum(
        1.0, gcr * np.abs(sind(surface_tilt) + cosd(surface_tilt) * tan_phi))
    return f_gnd_beam  # 1 - min(1, abs()) < 1 always


def vf_ground_sky_2d(x, rotation, gcr, pitch, height, max_rows=10):
    """
    Calculate the fraction of the sky dome visible from pointx on the ground,
    accounting for obstructions by infinitely long rows.

    Parameters
    ----------
    x : numeric
        Position on the ground between two rows, as a fraction of the pitch.
        x = 0 corresponds to the center point of a row.
    rotation : float
        Rotation of left edge relative to row center. [degree]
    gcr : float
        Ratio of row slant length to row spacing (pitch). [unitless]
    height : float
        Height of center point of the row above the ground. Must be in the same
        units as pitch.
    pitch : float
        Distance between two rows. Must be in the same units as height.
    max_rows : int, default 10
        Maximum number of rows to consider in front and behind the current row.

    Returns
    -------
    vf : array-like
        Fraction of sky dome visible from each point on the ground. [unitless]
    wedge_angles : array
        Bounding angles of each wedge of visible sky.
        Shape is (2, len(x), 2*max_rows+1). wedge_angles[0,:,:] is the
        starting angle of each wedge, wedge_angles[1,:,:] is the end angle.
        [degrees]
    """
    x = np.atleast_1d(x)  # handle float
    all_k = np.arange(-max_rows, max_rows + 1)
    width = gcr * pitch / 2.
    # angles from x to left edge of each row
    a1 = height + width * sind(rotation)
    b1 = (all_k - x[:, np.newaxis]) * pitch + width * cosd(rotation)
    phi_1 = np.degrees(np.arctan2(a1, b1))
    # angles from x to right edge of each row
    a2 = height - width * sind(rotation)
    b2 = (all_k - x[:, np.newaxis]) * pitch - width * cosd(rotation)
    phi_2 = np.degrees(np.arctan2(a2, b2))
    phi = np.stack([phi_1, phi_2])
    swap = phi[0, :, :] > phi[1, :, :]
    # swap where phi_1 > phi_2 so that phi_1[0,:,:] is the left edge
    phi = np.where(swap, phi[::-1], phi)
    # right edge of next row - left edge of previous row
    wedge_vfs = 0.5 * (cosd(phi[1,:,1:]) - cosd(phi[0,:,:-1]))
    vf = np.sum(np.where(wedge_vfs > 0, wedge_vfs, 0.), axis=1)
    return vf, phi
