"""
The bifacial.utils module contains functions that support bifacial irradiance
modeling.
"""
import numpy as np
from pvlib.tools import sind, cosd, tand


def _solar_projection_tangent(solar_zenith, solar_azimuth, surface_azimuth):
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
        Solar zenith angle. [degree].
    solar_azimuth : numeric
        Solar azimuth. [degree].
    surface_azimuth: numeric
        Azimuth of the module surface, i.e., North=0, East=90, South=180,
        West=270. [degree]

    Returns
    -------
    tan_phi : numeric
        Tangent of the angle between vertical and the projection of the
        sun direction onto the YZ plane.
    """
    rotation = solar_azimuth - surface_azimuth
    tan_phi = cosd(rotation) * tand(solar_zenith)
    return tan_phi


def unshaded_ground_fraction(gcr, surface_tilt, surface_azimuth, solar_zenith,
                             solar_azimuth, max_zenith=87):
    """
    Calculate the fraction of the ground with incident direct irradiance.

    .. math::
        F_{gnd,sky} &= 1 - \\min{\\left(1, \\text{GCR} \\left|\\cos \\beta +
        \\sin \\beta \\tan \\phi \\right|\\right)} \\newline

    where :math:`\\beta` is the surface tilt.

    Parameters
    ----------
    gcr : numeric
        Ground coverage ratio, which is the ratio of row slant length to row
        spacing (pitch). [unitless]
    surface_tilt: numeric
        Surface tilt angle. The tilt angle is defined as
        degrees from horizontal, e.g., surface facing up = 0, surface facing
        horizon = 90. [degree]
    surface_azimuth: numeric
        Azimuth of the module surface, i.e., North=0, East=90, South=180,
        West=270. [degree]
    solar_zenith : numeric
        Solar zenith angle. [degree].
    solar_azimuth : numeric
        Solar azimuth. [degree].
    max_zenith : numeric, default 87
        Maximum zenith angle. For solar_zenith > max_zenith, unshaded ground
        fraction is set to 0. [degree]

    Returns
    -------
    f_gnd_beam : numeric
        Fraction of distance betwen rows (pitch) with direct irradiance
        (unshaded). [unitless]
    """
    # TODO: why np.abs? All angles should be <=90
    tan_phi = _solar_projection_tangent(solar_zenith, solar_azimuth,
                                        surface_azimuth)
    f_gnd_beam = 1.0 - np.minimum(
        1.0, gcr * np.abs(cosd(surface_tilt) + sind(surface_tilt) * tan_phi))
    np.where(solar_zenith > max_zenith, 0., f_gnd_beam)
    return f_gnd_beam  # 1 - min(1, abs()) < 1 always


def vf_ground_sky_2d(x, rotation, gcr, pitch, height, max_rows=10):
    """
    Calculate the fraction of the sky dome visible from point x on the ground.

    The view factor accounts for the obstruction of the sky by array rows that
    are assumed to be infinitely long.  View factors are thus calculated in
    a 2D geometry. The ground is assumed to be flat and level.

    Parameters
    ----------
    x : numeric
        Position on the ground between two rows, as a fraction of the pitch.
        x = 0 corresponds to the point on the ground directly below the
        center point of a row. Positive x is towards the right.
    rotation : float
        Rotation angle of the row's left edge relative to row center. [degree]
    gcr : float
        Ratio of the row slant length to the row spacing (pitch). [unitless]
    height : float
        Height of the center point of the row above the ground; must be in the
        same units as ``pitch``.
    pitch : float
        Distance between two rows; must be in the same units as ``height``.
    max_rows : int, default 10
        Maximum number of rows to consider on either side of the current
        row. [unitless]

    Returns
    -------
    vf : array-like
        Fraction of sky dome visible from each point on the ground. [unitless]
    wedge_angles : array
        Angles defining each wedge of visible sky. Shape is
        (2, len(x), 2*max_rows+1). ``wedge_angles[0,:,:]`` is the
        starting angle of each wedge, ``wedge_angles[1,:,:]`` is the end angle.
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
    wedge_vfs = 0.5 * (cosd(phi[1, :, 1:]) - cosd(phi[0, :, :-1]))
    vf = np.sum(np.where(wedge_vfs > 0, wedge_vfs, 0.), axis=1)
    return vf, phi
