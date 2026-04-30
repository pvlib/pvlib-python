"""
The bifacial.utils module contains functions that support bifacial irradiance
modeling.
"""
import numpy as np
from pvlib.tools import sind, cosd, tand
import warnings
from pvlib._deprecation import pvlibDeprecationWarning, renamed_kwarg_warning


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


def _unshaded_ground_fraction(tracker_rotation, phi, gcr, height=None,
                              pitch=None, g0=0, g1=1, max_rows=10,
                              max_zenith=85):
    r"""
    Calculate the fraction of the ground with incident direct irradiance.

    Parameters
    ----------
    tracker_rotation : numeric
        Tracker rotation angle as a right-handed rotation around
        the same axis as ``phi``. [degree]
    phi : numeric
        Projected solar zenith angle, defined around the same axis as
        ``tracker_rotation``. [degree].
    gcr : float
        Ground coverage ratio, which is the ratio of row slant length to row
        spacing (pitch). [unitless]
    height : float, optional
        Height of the center point of the row above the ground; must be in the
        same units as ``pitch``.  Required if ``g0`` is not zero or ``g1`` is
        not one.
    pitch : float, optional
        Distance between two rows; must be in the same units as ``height``.
        Required if ``g0`` is not zero or ``g1`` is not one.
    g0 : numeric, default 0
        Position on the ground surface, as a fraction of the row-to-row
        spacing. ``g0=0`` corresponds to ground underneath the middle of the
        left row. ``g0`` should be less than ``g1``. [unitless]
    g1 : numeric, default 1
        Position on the ground surface, as a fraction of the row-to-row
        spacing. ``g1=1`` corresponds to ground underneath the middle of the
        right row. ``g1`` should be greater than ``g0``. [unitless]
    max_rows : int, default 10
        Maximum number of rows to consider on either side of the current
        row. [unitless]
    max_zenith : numeric, default 85
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
       :doi:`10.1109/PVSC40753.2019.8980572`.
    """
    if np.isscalar(g0) and g0 == 0 and np.isscalar(g1) and g1 == 1:
        # height and pitch have no effect, so set to arbitrary values so
        # that they can be optional parameters
        height = 1
        pitch = 1 / gcr

    # keep track of scalar inputs so that we can have output match at the end
    squeeze = []
    if np.isscalar(g0) and np.isscalar(g1):
        squeeze.append(0)
    if np.isscalar(tracker_rotation) and np.isscalar(phi):
        squeeze.append(1)

    # dimensions: k/max_rows, ground segment, time

    tracker_rotation = np.atleast_1d(tracker_rotation)[np.newaxis, np.newaxis,
                                                       :]
    phi = np.atleast_1d(phi)[np.newaxis, np.newaxis, :]

    g0 = np.atleast_1d(g0)[np.newaxis, :, np.newaxis]
    g1 = np.atleast_1d(g1)[np.newaxis, :, np.newaxis]

    # TODO seems like this should be np.arange(-max_rows, max_rows+1)?
    # see GH #1867
    k = np.arange(-max_rows, max_rows)[:, np.newaxis, np.newaxis]

    collector_width = pitch * gcr
    Lcostheta = collector_width * cosd(tracker_rotation)
    Lsintheta = collector_width * sind(tracker_rotation)
    tanphi = tand(phi)

    # a, b: boundaries of ground segment
    # d, c: left/right shading module edges
    c = (k*pitch + 0.5 * Lcostheta, height + 0.5 * Lsintheta)
    d = (k*pitch - 0.5 * Lcostheta, height - 0.5 * Lsintheta)

    cp = c[0] + c[1] * tanphi
    dp = d[0] + d[1] * tanphi
    swap = dp > cp
    cp, dp = np.where(swap, dp, cp), np.where(swap, cp, dp)

    a = g0*pitch
    b = g1*pitch

    # individual contributions from all k rows
    fs = np.full_like(cp, 1.0)
    # fs = np.where((dp <= a) & (cp >= b), 1.0, fs)  # fill value already 1.0
    fs = np.where((dp <= a) & (a <= cp) & (cp < b), (cp - a) / (b - a), fs)
    fs = np.where((dp < a) & (cp < a), 0.0, fs)
    fs = np.where((a < dp) & (dp <= b) & (cp >= b), (b - dp) / (b - a), fs)
    fs = np.where((a < dp) & (dp < b) & (a < cp) & (cp < b),
                  (cp - dp) / (b - a), fs)
    fs = np.where((dp > b) & (cp > b), 0.0, fs)

    # total shaded fraction is sum of individuals; note that shadows
    # never overlap in this model, except when shaded fraction is 100% anyway
    f_gnd_beam = 1 - np.clip(np.sum(fs, axis=0), 0, 1)  # sum along k dimension

    # using phi is more convenient, and I think better, than using zenith
    phi = phi[0, :, :]  # drop k dimension for the next line
    f_gnd_beam = np.where(np.abs(phi) > max_zenith, 0., f_gnd_beam)

    # dimensions are now ground_segment, time
    f_gnd_beam = f_gnd_beam.squeeze(axis=tuple(squeeze))
    return f_gnd_beam


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
    pitch : float
        Distance between two rows; must be in the same units as ``height``.
    height : float
        Height of the center point of the row above the ground; must be in the
        same units as ``pitch``.
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


@renamed_kwarg_warning("0.15.2", "surface_tilt", "tracker_rotation")
def vf_ground_sky_2d_integ(tracker_rotation, gcr, height, pitch, g0=0, g1=1,
                           max_rows=10, npoints=None, vectorize=None):
    """
    Integrated view factor to the sky from the ground underneath
    interior rows of the array.

    Parameters
    ----------
    tracker_rotation : numeric
        Tracker rotation angle as a right-handed rotation around
        the axis defined by ``axis_tilt`` and ``axis_azimuth``.  For example,
        with ``axis_tilt=0`` and ``axis_azimuth=180``, ``tracker_theta > 0``
        results in ``surface_azimuth`` to the West while ``tracker_theta < 0``
        results in ``surface_azimuth`` to the East. [degree]
    gcr : float
        Ratio of row slant length to row spacing (pitch). [unitless]
    height : float
        Height of the center point of the row above the ground; must be in the
        same units as ``pitch``.
    pitch : float
        Distance between two rows. Must be in the same units as ``height``.
    g0 : numeric, default 0
        Position on the ground surface, as a fraction of the row-to-row
        spacing. ``g0=0`` corresponds to ground underneath the middle of the
        left row. ``g0`` should be less than ``g1``. [unitless]
    g1 : numeric, default 1
        Position on the ground surface, as a fraction of the row-to-row
        spacing. ``g1=1`` corresponds to ground underneath the middle of the
        right row. ``g1`` should be greater than ``g0``. [unitless]
    max_rows : int, default 10
        Maximum number of rows to consider in front and behind the current row.
    npoints : int, optional

        .. deprecated:: 0.15.2

           This parameter has no effect; integrated view factors are now
           calculated exactly instead of with discretized approximations.
           This parameter will be removed in the future.

    vectorize : bool, optional

        .. deprecated:: 0.15.2

           This parameter has no effect; calculations are now vectorized
           with no memory usage penality.
           This parameter will be removed in the future.

    Returns
    -------
    fgnd_sky : numeric
        Integration of view factor over the length between adjacent, interior
        rows.  Shape matches that of ``surface_tilt``. [unitless]
    """
    if npoints is not None or vectorize is not None:
        msg = (
            "The `npoints` and `vectorize` parameters have no effect and will "
            "be removed in a future version."
        )
        warnings.warn(msg, pvlibDeprecationWarning)

    # keep track of scalar inputs so that we can have output match at the end
    squeeze = []
    if np.isscalar(g0) and np.isscalar(g1):
        squeeze.append(0)
    if np.isscalar(tracker_rotation):
        squeeze.append(1)

    # dimensions: k/max_rows, ground segment, time

    tracker_rotation = \
        np.atleast_1d(tracker_rotation)[np.newaxis, np.newaxis, :]

    g0 = np.atleast_1d(g0)[np.newaxis, :, np.newaxis]
    g1 = np.atleast_1d(g1)[np.newaxis, :, np.newaxis]

    # TODO seems like this should be np.arange(-max_rows, max_rows+1)?
    # see GH #1867
    k = np.arange(-max_rows, max_rows)[:, np.newaxis, np.newaxis]

    collector_width = pitch * gcr
    Lcostheta = collector_width * cosd(tracker_rotation)
    Lsintheta = collector_width * sind(tracker_rotation)

    # primary crossed string points:
    # a, b: boundaries of ground segment
    # c, d: upper module edges
    a = (g0*pitch, 0)
    b = (g1*pitch, 0)
    sign = np.sign(tracker_rotation)
    c = ((k+1)*pitch + sign * 0.5 * Lcostheta, height + sign * 0.5 * Lsintheta)
    d = (c[0] - pitch, c[1])

    # view obstruction points (module edges, but need to figure out which ones)

    # first decide whether the left obstruction is the left or right mod edge
    left = (k*pitch - 0.5 * Lcostheta, height - 0.5 * Lsintheta)
    right = (k*pitch + 0.5 * Lcostheta, height + 0.5 * Lsintheta)
    angle_left = _angle(a, left)
    angle_right = _angle(a, right)
    ob_left = (
        np.where(angle_left > angle_right, right[0], left[0]),
        np.where(angle_left > angle_right, right[1], left[1])
    )

    # now for the right obstruction
    left = (left[0] + pitch, left[1])
    right = (right[0] + pitch, right[1])
    angle_left = _angle(b, left)
    angle_right = _angle(b, right)
    ob_right = (
        np.where(angle_left > angle_right, left[0], right[0]),
        np.where(angle_left > angle_right, left[1], right[1])
    )

    # hottel string lengths, considering obstructions
    ac, ad, bc, bd = _obstructed_string_lengths(a, b, c, d, ob_left, ob_right)

    # crossed string formula for VF
    vf_slats = 0.5 * (1/((g1 - g0) * pitch)) * ((ac + bd) - (bc + ad))
    vf_total = np.sum(np.maximum(vf_slats, 0), axis=0)  # sum along k dimension

    # dimensions are now ground_segment, row_segment, time
    vf_total = vf_total.squeeze(axis=tuple(squeeze))

    return vf_total


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
        ``x0=0`` corresponds to the bottom of the row. ``x0`` should be less
        than ``x1``. [unitless]
    x1 : numeric, default 1
        Position on the row's slant length, as a fraction of the slant length.
        ``x1`` should be greater than ``x0``. [unitless]

    Returns
    -------
    vf : numeric
        Average fraction of the sky dome visible from points in the segment
        from x0 to x1. [unitless]

    '''
    # keep track of scalar inputs so that we can have output match at the end
    squeeze = []
    if np.isscalar(x0) and np.isscalar(x1):
        squeeze.append(0)
    if np.isscalar(surface_tilt):
        squeeze.append(1)

    # dimensions: row segment, time

    surface_tilt = np.atleast_1d(surface_tilt)[np.newaxis, :]

    x0 = np.atleast_1d(x0)[:, np.newaxis]
    x1 = np.atleast_1d(x1)[:, np.newaxis]

    swap = surface_tilt < 0
    x0, x1 = np.where(swap, 1 - x1, x0), np.where(swap, 1 - x0, x1)

    u = np.abs(x1 - x0)
    p0 = _vf_poly(surface_tilt, gcr, 1 - x0, -1)
    p1 = _vf_poly(surface_tilt, gcr, 1 - x1, -1)
    with np.errstate(divide='ignore'):
        result = np.where(u < 1e-6,
                          vf_row_sky_2d(surface_tilt, gcr, x0),
                          0.5*(1 + 1/u * (p1 - p0))
                          )
    result = result.squeeze(axis=tuple(squeeze))
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


def vf_row_ground_2d_integ(surface_tilt, gcr, height=None, pitch=None,
                           x0=0, x1=1, g0=0, g1=1, max_rows=20):
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
    height : float, optional
        Height of the center point of the row above the ground; must be in the
        same units as ``pitch``.  Required if ``g0`` or ``x0` is not zero or
        if ``g1`` or ``x1`` is not one.
    pitch : float, optional
        Distance between two rows; must be in the same units as ``height``.
        Required if ``g0`` or ``x0` is not zero or if ``g1`` or ``x1``
        is not one.
    x0 : numeric, default 0.
        Position on the row's slant length, as a fraction of the slant length.
        ``x0=0`` corresponds to the bottom of the row. ``x0`` should be less
        than ``x1``. [unitless]
    x1 : numeric, default 1.
        Position on the row's slant length, as a fraction of the slant length.
        ``x1`` should be greater than ``x0``. [unitless]
    g0 : numeric, default 0
        Position on the ground surface, as a fraction of the row-to-row
        spacing. ``g0=0`` corresponds to ground underneath the middle of the
        left row. ``g0`` should be less than ``g1``. [unitless]
    g1 : numeric, default 1
        Position on the ground surface, as a fraction of the row-to-row
        spacing. ``g1=0`` corresponds to ground underneath the middle of the
        right row. ``g1`` should be greater than ``g0``. [unitless]
    max_rows : int, default 20
        Maximum number of rows to consider in front and behind the current row.

    Returns
    -------
    vf : numeric
        Integrated view factor to the visible ground on the interval (x0, x1).
        [unitless]

    '''
    if all(np.isscalar(x) for x in [x0, x1, g0, g1]) and (
            g0 == 0 and g1 == 1 and x0 == 0 and x1 == 1):
        # height and pitch have no effect, so set to arbitrary values so
        # that they can be optional parameters
        height = 1
        pitch = 1 / gcr

    # keep track of scalar inputs so that we can have output match at the end
    squeeze = []
    if np.isscalar(g0) and np.isscalar(g1):
        squeeze.append(0)
    if np.isscalar(x0) and np.isscalar(x1):
        squeeze.append(1)
    if np.isscalar(surface_tilt):
        squeeze.append(2)

    # dimensions: k/max_rows, ground segment, row segment, time

    # cheat a little to prevent numerical issues with surface_tilt==180, -180
    surface_tilt = np.where(surface_tilt == 180, 179.9999, surface_tilt)
    surface_tilt = np.where(surface_tilt == -180, -179.9999, surface_tilt)

    surface_tilt = \
        np.atleast_1d(surface_tilt)[np.newaxis, np.newaxis, np.newaxis, :]

    x0 = np.atleast_1d(x0)[np.newaxis, np.newaxis, :, np.newaxis]
    x1 = np.atleast_1d(x1)[np.newaxis, np.newaxis, :, np.newaxis]
    g0 = np.atleast_1d(g0)[np.newaxis, :, np.newaxis, np.newaxis]
    g1 = np.atleast_1d(g1)[np.newaxis, :, np.newaxis, np.newaxis]

    # TODO seems like this should be np.arange(-max_rows, max_rows+1)?
    # see GH #1867
    k = np.arange(-max_rows, max_rows)[:, np.newaxis, np.newaxis, np.newaxis]

    collector_width = pitch * gcr
    Lcostheta = collector_width * cosd(surface_tilt)
    Lsintheta = collector_width * sind(surface_tilt)

    # view obstruction points (lower module edges)
    # use a number slightly larger than 0.5 because the obstruction must
    # be a nonzero distance from all points the VF could be calculated from
    ob_right = (-pitch - 0.5001 * Lcostheta, height - 0.5001 * abs(Lsintheta))
    ob_left = (ob_right[0] + pitch, ob_right[1])

    invert = surface_tilt < 0
    temp = ob_right[0]
    ob_right = (np.where(invert, -ob_left[0], ob_right[0]), ob_right[1])
    ob_left = (np.where(invert, -temp, ob_left[0]), ob_left[1])

    # primary crossed string points:
    # a, b: positions on module
    # c, d: boundaries of ground segment

    a = ((x0-0.5) * Lcostheta, height + (x0-0.5) * Lsintheta)
    b = ((x1-0.5) * Lcostheta, height + (x1-0.5) * Lsintheta)
    c = ((k+g0)*pitch, 0)
    d = ((k+g1)*pitch, 0)

    # hottel string lengths, considering obstructions
    ac, ad, bc, bd = _obstructed_string_lengths(a, b, c, d, ob_left, ob_right)

    # crossed string formula for VF
    vf_slats = 1 / (2 * (x1 - x0) * collector_width) * ((ac + bd) - (bc + ad))
    vf_total = np.sum(np.maximum(vf_slats, 0), axis=0)  # sum along k dimension

    # dimensions are now ground_segment, row_segment, time
    vf_total = vf_total.squeeze(axis=tuple(squeeze))

    return vf_total


def _obstructed_string_lengths(a, b, c, d, ob_left, ob_right):
    # string length calculations for Hottel's crossed strings method,
    # considering view obstructions from the left and right.
    # all inputs are (x, y) points.

    # unobstructed (straight-line) distances
    dist_ac = _dist(a, c)
    dist_ad = _dist(a, d)
    dist_bc = _dist(b, c)
    dist_bd = _dist(b, d)
    dist_a_ob_left = _dist(a, ob_left)
    dist_a_ob_right = _dist(a, ob_right)
    dist_b_ob_left = _dist(b, ob_left)
    dist_b_ob_right = _dist(b, ob_right)
    dist_ob_left_c = _dist(ob_left, c)
    dist_ob_right_c = _dist(ob_right, c)
    dist_ob_left_d = _dist(ob_left, d)
    dist_ob_right_d = _dist(ob_right, d)

    # angles
    ang_ac = _angle(a, c)
    ang_ad = _angle(a, d)
    ang_bc = _angle(b, c)
    ang_bd = _angle(b, d)
    ang_a_ob_left = _angle(a, ob_left)
    ang_a_ob_right = _angle(a, ob_right)
    ang_b_ob_left = _angle(b, ob_left)
    ang_b_ob_right = _angle(b, ob_right)

    # obstructed distances
    ac = np.where(ang_ac - ang_a_ob_left > 1e-8,
                  dist_a_ob_left + dist_ob_left_c,
                  dist_ac)
    ac = np.where((ang_a_ob_right - ang_ac > 1e-8),
                  dist_a_ob_right + dist_ob_right_c,
                  ac)

    ad = np.where(ang_ad - ang_a_ob_left > 1e-8,
                  dist_a_ob_left + dist_ob_left_d,
                  dist_ad)
    ad = np.where(ang_a_ob_right - ang_ad > 1e-8,
                  dist_a_ob_right + dist_ob_right_d,
                  ad)

    bc = np.where(ang_bc - ang_b_ob_left > 1e-8,
                  dist_b_ob_left + dist_ob_left_c,
                  dist_bc)
    bc = np.where(ang_b_ob_right - ang_bc > 1e-8,
                  dist_b_ob_right + dist_ob_right_c,
                  bc)

    bd = np.where(ang_bd - ang_b_ob_left > 1e-8,
                  dist_b_ob_left + dist_ob_left_d,
                  dist_bd)
    bd = np.where(ang_b_ob_right - ang_bd > 1e-8,
                  dist_b_ob_right + dist_ob_right_d,
                  bd)

    return ac, ad, bc, bd


def _dist(p1, p2):
    return ((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)**0.5


def _angle(p1, p2):
    return np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
