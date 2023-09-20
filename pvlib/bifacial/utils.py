"""
The bifacial.utils module contains functions that support bifacial irradiance
modeling.
"""
import numpy as np
from pvlib.tools import sind, cosd, tand
import warnings
from pvlib._deprecation import pvlibDeprecationWarning

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


def vf_ground_sky_2d(rotation, gcr, x, pitch, height, max_rows=10):
    r"""
    Calculate the fraction of the sky dome visible from point x on the ground.

    The view factor accounts for the obstruction of the sky by array rows that
    are assumed to be infinitely long.  View factors are thus calculated in
    a 2D geometry. The ground is assumed to be flat and level.

    Parameters
    ----------
    rotation : numeric
        Rotation angle of the row's right edge relative to row center.
        [degree]
    gcr : float
        Ratio of the row slant length to the row spacing (pitch). [unitless]
    x : numeric
        Position on the ground between two rows, as a fraction of the pitch.
        x = 0 corresponds to the point on the ground directly below the
        center point of a row. Positive x is towards the right. [unitless]
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


def vf_ground_sky_2d_integ(surface_tilt, gcr, height, pitch, max_rows=10,
                           npoints=None, vectorize=None):
    """
    Integrated view factor to the sky from the ground underneath
    interior rows of the array.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    gcr : float
        Ratio of row slant length to row spacing (pitch). [unitless]
    height : float
        Height of the center point of the row above the ground; must be in the
        same units as ``pitch``.
    pitch : float
        Distance between two rows. Must be in the same units as ``height``.
    max_rows : int, default 10
        Maximum number of rows to consider in front and behind the current row.
    npoints : int, default 100
        Number of points used to discretize distance along the ground.
    vectorize : bool, default False
        If True, vectorize the view factor calculation across ``surface_tilt``.
        This increases speed with the cost of increased memory usage.

    Returns
    -------
    fgnd_sky : numeric
        Integration of view factor over the length between adjacent, interior
        rows.  Shape matches that of ``surface_tilt``. [unitless]
    """
    if npoints is not None or vectorize is not None:
        warnings.warn("the `npoints` and `vectorize` parameters are deprecated", pvlibDeprecationWarning)

    # Abuse vf_ground_sky_2d by supplying surface_tilt in place
    # of a signed rotation. This is OK because
    # 1) z span the full distance between 2 rows, and
    # 2) max_rows is set to be large upstream, and
    # 3) _vf_ground_sky_2d considers [-max_rows, +max_rows]
    # The VFs to the sky will thus be symmetric around z=0.5


    # TODO: compatibility with scalar inputs
    # TODO: clean this up
    collector_width = pitch * gcr

    base_x1 = 0.5 * collector_width * cosd(surface_tilt)[np.newaxis, :]
    base_y1 = 0.5 * collector_width * sind(surface_tilt)[np.newaxis, :]

    k = np.arange(-max_rows, max_rows+1)[:, np.newaxis]

    x1l = k*pitch - base_x1
    x1r = k*pitch + base_x1
    x2l = x1l + pitch
    x2r = x1r + pitch

    y1l = y2l = height + base_y1
    y1r = y2r = height - base_y1

    dx = x1l
    dy = y1l
    cx = x2l
    cy = y2l
    ax = 0
    ay = 0
    bx = pitch
    by = 0

    o1x = x1r
    o1y = y1r
    o2x = x2r
    o2y = y2r

    theta_ac = np.arctan2(cy - ay, cx - ax)
    theta_ad = np.arctan2(dy - ay, dx - ax)
    theta_ao1 = np.arctan2(o1y - ay, o1x - ax)
    theta_ao2 = np.arctan2(o2y - ay, o2x - ax)

    a_o1 = ((o1x - ax)**2 + (o1y - ay)**2)**0.5
    a_o2 = ((o2x - ax)**2 + (o2y - ay)**2)**0.5
    b_o1 = ((o1x - bx)**2 + (o1y - by)**2)**0.5
    b_o2 = ((o2x - bx)**2 + (o2y - by)**2)**0.5
    c_o1 = ((cx - o1x)**2 + (cy - o1y)**2)**0.5
    c_o2 = collector_width
    d_o1 = collector_width
    d_o2 = ((dx - o2x)**2 + (dy - o2y)**2)**0.5

    ac = ((cx - ax)**2 + (cy - ay)**2)**0.5
    ac = np.where(theta_ac > theta_ao1, a_o1 + c_o1, ac)
    ac = np.where(theta_ac < theta_ao2, a_o2 + c_o2, ac)

    ad = ((dx - ax)**2 + (dy - ay)**2)**0.5
    ad = np.where(theta_ad > theta_ao1, a_o1 + d_o1, ad)
    ad = np.where(theta_ad < theta_ao2, a_o2 + d_o2, ad)

    theta_bc = np.arctan2(cy - by, cx - bx)
    theta_bd = np.arctan2(dy - by, dx - bx)
    theta_ao1 = np.arctan2(o1y - by, o1x - bx)
    theta_ao2 = np.arctan2(o2y - by, o2x - bx)

    bd = ((dx - bx)**2 + (dy - by)**2)**0.5
    bd = np.where(theta_bd < theta_ao2, b_o2 + d_o2, bd)
    bd = np.where(theta_bd > theta_ao1, b_o1 + d_o1, bd)

    bc = ((cx - bx)**2 + (cy - by)**2)**0.5
    bc = np.where(theta_bc < theta_ao2, b_o2 + c_o2, bc)
    bc = np.where(theta_bc > theta_ao1, b_o1 + c_o1, bc)

    vf = np.sum(np.clip(0.5 * (1/pitch) * ((ac + bd) - (bc + ad)), a_min=0, a_max=None), axis=0)

    return vf


def _vf_poly(surface_tilt, gcr, x, delta):
    r'''
    A term common to many 2D view factor calculations

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    gcr : numeric
        Ratio of the row slant length to the row spacing (pitch). [unitless]
    x : numeric
        Position on the row's slant length, as a fraction of the slant length.
        x=0 corresponds to the bottom of the row. [unitless]
    delta : -1 or +1
        A sign indicator for the linear term of the polynomial

    Returns
    -------
    numeric
    '''
    a = 1 / gcr
    c = cosd(surface_tilt)
    return np.sqrt(a*a + 2*delta*a*c*x + x*x)


def vf_row_sky_2d(surface_tilt, gcr, x):
    r'''
    Calculate the view factor to the sky from a point x on a row surface.

    Assumes a PV system of infinitely long rows with uniform pitch on
    horizontal ground. The view to the sky is restricted by the row's surface
    tilt and the top of the adjacent row.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    gcr : numeric
        Ratio of the row slant length to the row spacing (pitch). [unitless]
    x : numeric
        Position on the row's slant length, as a fraction of the slant length.
        x=0 corresponds to the bottom of the row. [unitless]

    Returns
    -------
    vf : numeric
        Fraction of the sky dome visible from the point x. [unitless]

    '''
    p = _vf_poly(surface_tilt, gcr, 1 - x, -1)
    return 0.5*(1 + (1/gcr * cosd(surface_tilt) - (1 - x)) / p)


def vf_row_sky_2d_integ(surface_tilt, gcr, x0=0, x1=1):
    r'''
    Calculate the average view factor to the sky from a segment of the row
    surface between x0 and x1.

    Assumes a PV system of infinitely long rows with uniform pitch on
    horizontal ground. The view to the sky is restricted by the row's surface
    tilt and the top of the adjacent row.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    gcr : numeric
        Ratio of the row slant length to the row spacing (pitch). [unitless]
    x0 : numeric, default 0
        Position on the row's slant length, as a fraction of the slant length.
        x0=0 corresponds to the bottom of the row. x0 should be less than x1.
        [unitless]
    x1 : numeric, default 1
        Position on the row's slant length, as a fraction of the slant length.
        x1 should be greater than x0. [unitless]

    Returns
    -------
    vf : numeric
        Average fraction of the sky dome visible from points in the segment
        from x0 to x1. [unitless]

    '''
    result = 0.5 * (1/gcr + 1 - ((1/gcr)**2 - (2/gcr)*cosd(surface_tilt) + 1)**0.5)
    return result


def vf_row_ground_2d(surface_tilt, gcr, x):
    r'''
    Calculate the view factor to the ground from a point x on a row surface.

    Assumes a PV system of infinitely long rows with uniform pitch on
    horizontal ground. The view to the ground is restricted by the row's
    tilt and the bottom of the facing row.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    gcr : numeric
        Ratio of the row slant length to the row spacing (pitch). [unitless]
    x : numeric
        Position on the row's slant length, as a fraction of the slant length.
        x=0 corresponds to the bottom of the row. [unitless]

    Returns
    -------
    vf : numeric
        View factor to the visible ground from the point x. [unitless]

    '''
    p = _vf_poly(surface_tilt, gcr, x, 1)
    return 0.5 * (1 - (1/gcr * cosd(surface_tilt) + x)/p)


def vf_row_ground_2d_integ(surface_tilt, gcr, x0=0, x1=1):
    r'''
    Calculate the average view factor to the ground from a segment of the row
    surface between x0 and x1.

    Assumes a PV system of infinitely long rows with uniform pitch on
    horizontal ground. The view to the ground is restricted by the row's
    tilt and the bottom of the facing row.

    Parameters
    ----------
    surface_tilt : numeric
        Surface tilt angle in degrees from horizontal, e.g., surface facing up
        = 0, surface facing horizon = 90. [degree]
    gcr : numeric
        Ratio of the row slant length to the row spacing (pitch). [unitless]
    x0 : numeric, default 0.
        Position on the row's slant length, as a fraction of the slant length.
        x0=0 corresponds to the bottom of the row. x0 should be less than x1.
        [unitless]
    x1 : numeric, default 1.
        Position on the row's slant length, as a fraction of the slant length.
        x1 should be greater than x0. [unitless]

    Returns
    -------
    vf : numeric
        Integrated view factor to the visible ground on the interval (x0, x1).
        [unitless]

    '''
    result = 0.5 * (1/gcr + 1 - ((1/gcr)**2 + (2/gcr)*cosd(surface_tilt) + 1)**0.5)
    return result
