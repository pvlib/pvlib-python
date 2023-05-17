"""
The bifacial.utils module contains functions that support bifacial irradiance
modeling.
"""
import numpy as np
from pvlib.tools import sind, cosd, tand

def _solar_projection_tangent(solar_zenith, solar_azimuth, surface_azimuth):
    """
    Tangent of the angle between the zenith vector and the sun vector
    projected to the plane defined by the zenith vector and surface_azimuth.

    .. math::
        \\tan \\phi = \\cos\\left(\\text{solar azimuth}-\\text{system azimuth}
        \\right)\\tan\\left(\\text{solar zenith}\\right)

    Parameters
    ----------
    solar_zenith : numeric
        Solar zenith angle. [degree].
    solar_azimuth : numeric
        Solar azimuth. [degree].
    surface_azimuth : numeric
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


def _unshaded_ground_fraction(surface_tilt, surface_azimuth, solar_zenith,
                              solar_azimuth, gcr, max_zenith=87):
    r"""
    Calculate the fraction of the ground with incident direct irradiance.

    .. math::
        F_{gnd,sky} = 1 - \min{\left(1, \text{GCR} \left|\cos \beta +
        \sin \beta \tan \phi \right|\right)}

    where :math:`\beta` is the surface tilt and :math:`\phi` is the angle
    from vertical of the sun vector projected to a vertical plane that
    contains the row azimuth `surface_azimuth`.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle. The tilt angle is defined as
        degrees from horizontal, e.g., surface facing up = 0, surface facing
        horizon = 90. [degree]
    surface_azimuth : numeric
        Azimuth of the module surface, i.e., North=0, East=90, South=180,
        West=270. [degree]
    solar_zenith : numeric
        Solar zenith angle. [degree].
    solar_azimuth : numeric
        Solar azimuth. [degree].
    gcr : float
        Ground coverage ratio, which is the ratio of row slant length to row
        spacing (pitch). [unitless]
    max_zenith : numeric, default 87
        Maximum zenith angle. For solar_zenith > max_zenith, unshaded ground
        fraction is set to 0. [degree]

    Returns
    -------
    f_gnd_beam : numeric
        Fraction of distance betwen rows (pitch) with direct irradiance
        (unshaded). [unitless]

    References
    ----------
    .. [1] Mikofski, M., Darawali, R., Hamer, M., Neubert, A., and Newmiller,
       J. "Bifacial Performance Modeling in Large Arrays". 2019 IEEE 46th
       Photovoltaic Specialists Conference (PVSC), 2019, pp. 1282-1287.
       doi: 10.1109/PVSC40753.2019.8980572.
    """
    tan_phi = _solar_projection_tangent(solar_zenith, solar_azimuth,
                                        surface_azimuth)
    f_gnd_beam = 1.0 - np.minimum(
        1.0, gcr * np.abs(cosd(surface_tilt) + sind(surface_tilt) * tan_phi))
    np.where(solar_zenith > max_zenith, 0., f_gnd_beam)  # [1], Eq. 4
    return f_gnd_beam  # 1 - min(1, abs()) < 1 always


def _vf_ground_sky_2d(x, rotation, gcr, pitch, height, max_rows=10):
    r"""
    Calculate the fraction of the sky dome visible from point x on the ground.

    The view factor accounts for the obstruction of the sky by array rows that
    are assumed to be infinitely long.  View factors are thus calculated in
    a 2D geometry. The ground is assumed to be flat and level.

    Parameters
    ----------
    x : numeric
        Position on the ground between two rows, as a fraction of the pitch.
        x = 0 corresponds to the point on the ground directly below the
        center point of a row. Positive x is towards the right. [unitless]
    rotation : numeric
        Rotation angle of the row's right edge relative to row center.
        [degree]
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
    vf : array
        Fraction of sky dome visible from each point on the ground.
        Shape is (len(x), len(rotation)). [unitless]
    """
    # This function creates large float64 arrays of size
    # (2*len(x)*len(rotation)*len(max_rows)) or ~100 MB for
    # typical time series inputs.  This function makes heavy
    # use of numpy's out parameter to avoid allocating new
    # memory.  Unfortunately that comes at the cost of some
    # readability: because arrays get reused to avoid new allocations,
    # variable names don't always match what they hold.

    # handle floats:
    x = np.atleast_1d(x)[:, np.newaxis, np.newaxis]
    rotation = np.atleast_1d(rotation)[np.newaxis, :, np.newaxis]
    all_k = np.arange(-max_rows, max_rows + 1)
    width = gcr * pitch / 2.
    distance_to_row_centers = (all_k - x) * pitch
    dy = width * sind(rotation)
    dx = width * cosd(rotation)

    phi = np.empty((2, x.shape[0], rotation.shape[1], len(all_k)))

    # angles from x to right edge of each row
    a1 = height + dy
    # temporarily store one leg of the triangle in phi:
    np.add(distance_to_row_centers, dx, out=phi[0])
    np.arctan2(a1, phi[0], out=phi[0])

    # angles from x to left edge of each row
    a2 = height - dy
    np.subtract(distance_to_row_centers, dx, out=phi[1])
    np.arctan2(a2, phi[1], out=phi[1])

    # swap angles so that phi[0,:,:,:] is the lesser angle
    phi.sort(axis=0)

    # now re-use phi's memory again, this time storing cos(phi).
    next_edge = phi[1, :, :, 1:]
    np.cos(next_edge, out=next_edge)
    prev_edge = phi[0, :, :, :-1]
    np.cos(prev_edge, out=prev_edge)
    # right edge of next row - left edge of previous row, again
    # reusing memory so that the difference is stored in next_edge.
    # Note that the 0.5 view factor coefficient is applied after summing
    # as a minor speed optimization.
    np.subtract(next_edge, prev_edge, out=next_edge)
    np.clip(next_edge, a_min=0., a_max=None, out=next_edge)
    vf = np.sum(next_edge, axis=-1) / 2
    return vf
