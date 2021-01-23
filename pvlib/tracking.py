import numpy as np
import pandas as pd

from pvlib.tools import cosd, sind, tand
from pvlib.pvsystem import PVSystem
from pvlib import irradiance, atmosphere


class SingleAxisTracker(PVSystem):
    """
    A class for single-axis trackers that inherits the PV modeling methods from
    :py:class:`~pvlib.pvsystem.PVSystem`. For details on calculating tracker
    rotation see :py:func:`pvlib.tracking.singleaxis`.

    Parameters
    ----------
    axis_tilt : float, default 0
        The tilt of the axis of rotation (i.e, the y-axis defined by
        axis_azimuth) with respect to horizontal, in decimal degrees.

    axis_azimuth : float, default 0
        A value denoting the compass direction along which the axis of
        rotation lies. Measured in decimal degrees east of north.

    max_angle : float, default 90
        A value denoting the maximum rotation angle, in decimal degrees,
        of the one-axis tracker from its horizontal position (horizontal
        if axis_tilt = 0). A max_angle of 90 degrees allows the tracker
        to rotate to a vertical position to point the panel towards a
        horizon. max_angle of 180 degrees allows for full rotation.

    backtrack : bool, default True
        Controls whether the tracker has the capability to "backtrack"
        to avoid row-to-row shading. False denotes no backtrack
        capability. True denotes backtrack capability.

    gcr : float, default 2.0/7.0
        A value denoting the ground coverage ratio of a tracker system
        which utilizes backtracking; i.e. the ratio between the PV array
        surface area to total ground area. A tracker system with modules
        2 meters wide, centered on the tracking axis, with 6 meters
        between the tracking axes has a gcr of 2/6=0.333. If gcr is not
        provided, a gcr of 2/7 is default. gcr must be <=1.

    cross_axis_tilt : float, default 0.0
        The angle, relative to horizontal, of the line formed by the
        intersection between the slope containing the tracker axes and a plane
        perpendicular to the tracker axes. Cross-axis tilt should be specified
        using a right-handed convention. For example, trackers with axis
        azimuth of 180 degrees (heading south) will have a negative cross-axis
        tilt if the tracker axes plane slopes down to the east and positive
        cross-axis tilt if the tracker axes plane slopes up to the east. Use
        :func:`~pvlib.tracking.calc_cross_axis_tilt` to calculate
        `cross_axis_tilt`. [degrees]

    **kwargs
        Passed to :py:class:`~pvlib.pvsystem.PVSystem`. If the `arrays`
        parameter is specified it must have only a single Array. Furthermore
        if a :py:class:`~pvlib.pvsystem.Array` is provided it must have
        ``surface_tilt`` and ``surface_azimuth`` equal to None.

    Raises
    ------
    ValueError
        If more than one Array is specified.
    ValueError
        If an Array is provided with a surface tilt or azimuth not None.

    See also
    --------
    pvlib.tracking.singleaxis
    pvlib.tracking.calc_axis_tilt
    pvlib.tracking.calc_cross_axis_tilt
    """

    def __init__(self, axis_tilt=0, axis_azimuth=0, max_angle=90,
                 backtrack=True, gcr=2.0/7.0, cross_axis_tilt=0.0, **kwargs):

        arrays = kwargs.get('arrays', [])
        if len(arrays) > 1:
            raise ValueError("SingleAxisTracker does not support "
                             "multiple arrays.")
        elif len(arrays) == 1:
            surface_tilt = arrays[0].surface_tilt
            surface_azimuth = arrays[0].surface_azimuth
            if surface_tilt is not None or surface_azimuth is not None:
                raise ValueError(
                    "Array must not have surface_tilt or "
                    "surface_azimuth assigned. You must pass an "
                    "Array with these fields set to None."
                )

        self.axis_tilt = axis_tilt
        self.axis_azimuth = axis_azimuth
        self.max_angle = max_angle
        self.backtrack = backtrack
        self.gcr = gcr
        self.cross_axis_tilt = cross_axis_tilt

        kwargs['surface_tilt'] = None
        kwargs['surface_azimuth'] = None

        super().__init__(**kwargs)

    def __repr__(self):
        attrs = ['axis_tilt', 'axis_azimuth', 'max_angle', 'backtrack', 'gcr',
                 'cross_axis_tilt']
        sat_repr = ('SingleAxisTracker:\n  ' + '\n  '.join(
            f'{attr}: {getattr(self, attr)}' for attr in attrs))
        # get the parent PVSystem info
        pvsystem_repr = super().__repr__()
        # remove the first line (contains 'PVSystem: \n')
        pvsystem_repr = '\n'.join(pvsystem_repr.split('\n')[1:])
        return sat_repr + '\n' + pvsystem_repr

    def singleaxis(self, apparent_zenith, apparent_azimuth):
        """
        Get tracking data. See :py:func:`pvlib.tracking.singleaxis` more
        detail.

        Parameters
        ----------
        apparent_zenith : float, 1d array, or Series
            Solar apparent zenith angles in decimal degrees.

        apparent_azimuth : float, 1d array, or Series
            Solar apparent azimuth angles in decimal degrees.

        Returns
        -------
        tracking data
        """
        tracking_data = singleaxis(apparent_zenith, apparent_azimuth,
                                   self.axis_tilt, self.axis_azimuth,
                                   self.max_angle, self.backtrack,
                                   self.gcr, self.cross_axis_tilt)

        return tracking_data

    def get_aoi(self, surface_tilt, surface_azimuth, solar_zenith,
                solar_azimuth):
        """Get the angle of incidence on the system.

        For a given set of solar zenith and azimuth angles, the
        surface tilt and azimuth parameters are typically determined
        by :py:meth:`~SingleAxisTracker.singleaxis`. The
        :py:meth:`~SingleAxisTracker.singleaxis` method also returns
        the angle of incidence, so this method is only needed
        if using a different tracking algorithm.

        Parameters
        ----------
        surface_tilt : numeric
            Panel tilt from horizontal.
        surface_azimuth : numeric
            Panel azimuth from north
        solar_zenith : float or Series.
            Solar zenith angle.
        solar_azimuth : float or Series.
            Solar azimuth angle.

        Returns
        -------
        aoi : Series
            The angle of incidence in degrees from normal.
        """

        aoi = irradiance.aoi(surface_tilt, surface_azimuth,
                             solar_zenith, solar_azimuth)
        return aoi

    def get_irradiance(self, surface_tilt, surface_azimuth,
                       solar_zenith, solar_azimuth, dni, ghi, dhi,
                       dni_extra=None, airmass=None, model='haydavies',
                       **kwargs):
        """
        Uses the :func:`irradiance.get_total_irradiance` function to
        calculate the plane of array irradiance components on a tilted
        surface defined by the input data and ``self.albedo``.

        For a given set of solar zenith and azimuth angles, the
        surface tilt and azimuth parameters are typically determined
        by :py:meth:`~SingleAxisTracker.singleaxis`.

        Parameters
        ----------
        surface_tilt : numeric
            Panel tilt from horizontal.
        surface_azimuth : numeric
            Panel azimuth from north
        solar_zenith : numeric
            Solar zenith angle.
        solar_azimuth : numeric
            Solar azimuth angle.
        dni : float or Series
            Direct Normal Irradiance
        ghi : float or Series
            Global horizontal irradiance
        dhi : float or Series
            Diffuse horizontal irradiance
        dni_extra : float or Series, default None
            Extraterrestrial direct normal irradiance
        airmass : float or Series, default None
            Airmass
        model : String, default 'haydavies'
            Irradiance model.

        **kwargs
            Passed to :func:`irradiance.get_total_irradiance`.

        Returns
        -------
        poa_irradiance : DataFrame
            Column names are: ``total, beam, sky, ground``.
        """

        # not needed for all models, but this is easier
        if dni_extra is None:
            dni_extra = irradiance.get_extra_radiation(solar_zenith.index)

        if airmass is None:
            airmass = atmosphere.get_relative_airmass(solar_zenith)

        return irradiance.get_total_irradiance(surface_tilt,
                                               surface_azimuth,
                                               solar_zenith,
                                               solar_azimuth,
                                               dni, ghi, dhi,
                                               dni_extra=dni_extra,
                                               airmass=airmass,
                                               model=model,
                                               albedo=self.albedo,
                                               **kwargs)


def singleaxis(apparent_zenith, apparent_azimuth,
               axis_tilt=0, axis_azimuth=0, max_angle=90,
               backtrack=True, gcr=2.0/7.0, cross_axis_tilt=0):
    """
    Determine the rotation angle of a single-axis tracker when given particular
    solar zenith and azimuth angles.

    See [1]_ for details about the equations. Backtracking may be specified,
    and if so, a ground coverage ratio is required.

    Rotation angle is determined in a right-handed coordinate system. The
    tracker `axis_azimuth` defines the positive y-axis, the positive x-axis is
    90 degrees clockwise from the y-axis and parallel to the Earth's surface,
    and the positive z-axis is normal to both x & y-axes and oriented skyward.
    Rotation angle `tracker_theta` is a right-handed rotation around the y-axis
    in the x, y, z coordinate system and indicates tracker position relative to
    horizontal. For example, if tracker `axis_azimuth` is 180 (oriented south)
    and `axis_tilt` is zero, then a `tracker_theta` of zero is horizontal, a
    `tracker_theta` of 30 degrees is a rotation of 30 degrees towards the west,
    and a `tracker_theta` of -90 degrees is a rotation to the vertical plane
    facing east.

    Parameters
    ----------
    apparent_zenith : float, 1d array, or Series
        Solar apparent zenith angles in decimal degrees.

    apparent_azimuth : float, 1d array, or Series
        Solar apparent azimuth angles in decimal degrees.

    axis_tilt : float, default 0
        The tilt of the axis of rotation (i.e, the y-axis defined by
        axis_azimuth) with respect to horizontal, in decimal degrees.

    axis_azimuth : float, default 0
        A value denoting the compass direction along which the axis of
        rotation lies. Measured in decimal degrees east of north.

    max_angle : float, default 90
        A value denoting the maximum rotation angle, in decimal degrees,
        of the one-axis tracker from its horizontal position (horizontal
        if axis_tilt = 0). A max_angle of 90 degrees allows the tracker
        to rotate to a vertical position to point the panel towards a
        horizon. max_angle of 180 degrees allows for full rotation.

    backtrack : bool, default True
        Controls whether the tracker has the capability to "backtrack"
        to avoid row-to-row shading. False denotes no backtrack
        capability. True denotes backtrack capability.

    gcr : float, default 2.0/7.0
        A value denoting the ground coverage ratio of a tracker system
        which utilizes backtracking; i.e. the ratio between the PV array
        surface area to total ground area. A tracker system with modules
        2 meters wide, centered on the tracking axis, with 6 meters
        between the tracking axes has a gcr of 2/6=0.333. If gcr is not
        provided, a gcr of 2/7 is default. gcr must be <=1.

    cross_axis_tilt : float, default 0.0
        The angle, relative to horizontal, of the line formed by the
        intersection between the slope containing the tracker axes and a plane
        perpendicular to the tracker axes. Cross-axis tilt should be specified
        using a right-handed convention. For example, trackers with axis
        azimuth of 180 degrees (heading south) will have a negative cross-axis
        tilt if the tracker axes plane slopes down to the east and positive
        cross-axis tilt if the tracker axes plane slopes up to the east. Use
        :func:`~pvlib.tracking.calc_cross_axis_tilt` to calculate
        `cross_axis_tilt`. [degrees]

    Returns
    -------
    dict or DataFrame with the following columns:
        * `tracker_theta`: The rotation angle of the tracker.
          tracker_theta = 0 is horizontal, and positive rotation angles are
          clockwise. [degrees]
        * `aoi`: The angle-of-incidence of direct irradiance onto the
          rotated panel surface. [degrees]
        * `surface_tilt`: The angle between the panel surface and the earth
          surface, accounting for panel rotation. [degrees]
        * `surface_azimuth`: The azimuth of the rotated panel, determined by
          projecting the vector normal to the panel's surface to the earth's
          surface. [degrees]

    See also
    --------
    pvlib.tracking.calc_axis_tilt
    pvlib.tracking.calc_cross_axis_tilt

    References
    ----------
    .. [1] Kevin Anderson and Mark Mikofski, "Slope-Aware Backtracking for
       Single-Axis Trackers", Technical Report NREL/TP-5K00-76626, July 2020.
       https://www.nrel.gov/docs/fy20osti/76626.pdf
    """

    # MATLAB to Python conversion by
    # Will Holmgren (@wholmgren), U. Arizona. March, 2015.

    if isinstance(apparent_zenith, pd.Series):
        index = apparent_zenith.index
    else:
        index = None

    # convert scalars to arrays
    apparent_azimuth = np.atleast_1d(apparent_azimuth)
    apparent_zenith = np.atleast_1d(apparent_zenith)

    if apparent_azimuth.ndim > 1 or apparent_zenith.ndim > 1:
        raise ValueError('Input dimensions must not exceed 1')

    # Calculate sun position x, y, z using coordinate system as in [1], Eq 1.

    # NOTE: solar elevation = 90 - solar zenith, then use trig identities:
    # sin(90-x) = cos(x) & cos(90-x) = sin(x)
    sin_zenith = sind(apparent_zenith)
    x = sin_zenith * sind(apparent_azimuth)
    y = sin_zenith * cosd(apparent_azimuth)
    z = cosd(apparent_zenith)

    # Assume the tracker reference frame is right-handed. Positive y-axis is
    # oriented along tracking axis; from north, the y-axis is rotated clockwise
    # by the axis azimuth and tilted from horizontal by the axis tilt. The
    # positive x-axis is 90 deg clockwise from the y-axis and parallel to
    # horizontal (e.g., if the y-axis is south, the x-axis is west); the
    # positive z-axis is normal to the x and y axes, pointed upward.

    # Calculate sun position (xp, yp, zp) in tracker coordinate system using
    # [1] Eq 4.

    cos_axis_azimuth = cosd(axis_azimuth)
    sin_axis_azimuth = sind(axis_azimuth)
    cos_axis_tilt = cosd(axis_tilt)
    sin_axis_tilt = sind(axis_tilt)
    xp = x*cos_axis_azimuth - y*sin_axis_azimuth
    yp = (x*cos_axis_tilt*sin_axis_azimuth
          + y*cos_axis_tilt*cos_axis_azimuth
          - z*sin_axis_tilt)
    zp = (x*sin_axis_tilt*sin_axis_azimuth
          + y*sin_axis_tilt*cos_axis_azimuth
          + z*cos_axis_tilt)

    # The ideal tracking angle wid is the rotation to place the sun position
    # vector (xp, yp, zp) in the (y, z) plane, which is normal to the panel and
    # contains the axis of rotation.  wid = 0 indicates that the panel is
    # horizontal. Here, our convention is that a clockwise rotation is
    # positive, to view rotation angles in the same frame of reference as
    # azimuth. For example, for a system with tracking axis oriented south, a
    # rotation toward the east is negative, and a rotation to the west is
    # positive. This is a right-handed rotation around the tracker y-axis.

    # Calculate angle from x-y plane to projection of sun vector onto x-z plane
    # using [1] Eq. 5.

    wid = np.degrees(np.arctan2(xp, zp))

    # filter for sun above panel horizon
    zen_gt_90 = apparent_zenith > 90
    wid[zen_gt_90] = np.nan

    # Account for backtracking
    if backtrack:
        # distance between rows in terms of rack lengths relative to cross-axis
        # tilt
        axes_distance = 1/(gcr * cosd(cross_axis_tilt))

        # NOTE: account for rare angles below array, see GH 824
        temp = np.abs(axes_distance * cosd(wid - cross_axis_tilt))

        # backtrack angle using [1], Eq. 14
        with np.errstate(invalid='ignore'):
            wc = np.degrees(-np.sign(wid)*np.arccos(temp))

        # NOTE: in the middle of the day, arccos(temp) is out of range because
        # there's no row-to-row shade to avoid, & backtracking is unnecessary
        # [1], Eqs. 15-16
        with np.errstate(invalid='ignore'):
            tracker_theta = wid + np.where(temp < 1, wc, 0)
    else:
        tracker_theta = wid

    # NOTE: max_angle defined relative to zero-point rotation, not the
    # system-plane normal
    tracker_theta = np.clip(tracker_theta, -max_angle, max_angle)

    # Calculate panel normal vector in panel-oriented x, y, z coordinates.
    # y-axis is axis of tracker rotation. tracker_theta is a compass angle
    # (clockwise is positive) rather than a trigonometric angle.
    # NOTE: the *0 is a trick to preserve NaN values.
    panel_norm = np.array([sind(tracker_theta),
                           tracker_theta*0,
                           cosd(tracker_theta)])

    # sun position in vector format in panel-oriented x, y, z coordinates
    sun_vec = np.array([xp, yp, zp])

    # calculate angle-of-incidence on panel
    aoi = np.degrees(np.arccos(np.abs(np.sum(sun_vec*panel_norm, axis=0))))

    # Calculate panel tilt and azimuth in a coordinate system where the panel
    # tilt is the angle from horizontal, and the panel azimuth is the compass
    # angle (clockwise from north) to the projection of the panel's normal to
    # the earth's surface. These outputs are provided for convenience and
    # comparison with other PV software which use these angle conventions.

    # Project normal vector to earth surface. First rotate about x-axis by
    # angle -axis_tilt so that y-axis is also parallel to earth surface, then
    # project.

    # Calculate standard rotation matrix
    rot_x = np.array([[1, 0, 0],
                      [0, cosd(-axis_tilt), -sind(-axis_tilt)],
                      [0, sind(-axis_tilt), cosd(-axis_tilt)]])

    # panel_norm_earth contains the normal vector expressed in earth-surface
    # coordinates (z normal to surface, y aligned with tracker axis parallel to
    # earth)
    panel_norm_earth = np.dot(rot_x, panel_norm).T

    # projection to plane tangent to earth surface, in earth surface
    # coordinates
    projected_normal = np.array([panel_norm_earth[:, 0],
                                 panel_norm_earth[:, 1],
                                 panel_norm_earth[:, 2]*0]).T

    # calculate vector magnitudes
    projected_normal_mag = np.sqrt(np.nansum(projected_normal**2, axis=1))

    # renormalize the projected vector, avoid creating nan values.
    non_zeros = projected_normal_mag != 0
    projected_normal[non_zeros] = (projected_normal[non_zeros].T /
                                   projected_normal_mag[non_zeros]).T

    # calculation of surface_azimuth
    surface_azimuth = \
        np.degrees(np.arctan2(projected_normal[:, 1], projected_normal[:, 0]))

    # Rotate 0 reference from panel's x-axis to its y-axis and then back to
    # north.
    surface_azimuth = 90 - surface_azimuth + axis_azimuth

    # Map azimuth into [0,360) domain.
    with np.errstate(invalid='ignore'):
        surface_azimuth = surface_azimuth % 360

    # Calculate surface_tilt
    dotproduct = (panel_norm_earth * projected_normal).sum(axis=1)
    surface_tilt = 90 - np.degrees(np.arccos(dotproduct))

    # Bundle DataFrame for return values and filter for sun below horizon.
    out = {'tracker_theta': tracker_theta, 'aoi': aoi,
           'surface_tilt': surface_tilt, 'surface_azimuth': surface_azimuth}
    if index is not None:
        out = pd.DataFrame(out, index=index)
        out = out[['tracker_theta', 'aoi', 'surface_azimuth', 'surface_tilt']]
        out[zen_gt_90] = np.nan
    else:
        out = {k: np.where(zen_gt_90, np.nan, v) for k, v in out.items()}

    return out


def calc_axis_tilt(slope_azimuth, slope_tilt, axis_azimuth):
    """
    Calculate tracker axis tilt in the global reference frame when on a sloped
    plane.

    Parameters
    ----------
    slope_azimuth : float
        direction of normal to slope on horizontal [degrees]
    slope_tilt : float
        tilt of normal to slope relative to vertical [degrees]
    axis_azimuth : float
        direction of tracker axes on horizontal [degrees]

    Returns
    -------
    axis_tilt : float
        tilt of tracker [degrees]

    See also
    --------
    pvlib.tracking.singleaxis
    pvlib.tracking.calc_cross_axis_tilt

    Notes
    -----
    See [1]_ for derivation of equations.

    References
    ----------
    .. [1] Kevin Anderson and Mark Mikofski, "Slope-Aware Backtracking for
       Single-Axis Trackers", Technical Report NREL/TP-5K00-76626, July 2020.
       https://www.nrel.gov/docs/fy20osti/76626.pdf
    """
    delta_gamma = axis_azimuth - slope_azimuth
    # equations 18-19
    tan_axis_tilt = cosd(delta_gamma) * tand(slope_tilt)
    return np.degrees(np.arctan(tan_axis_tilt))


def _calc_tracker_norm(ba, bg, dg):
    """
    Calculate tracker normal, v, cross product of tracker axis and unit normal,
    N, to the system slope plane.

    Parameters
    ----------
    ba : float
        axis tilt [degrees]
    bg : float
        ground tilt [degrees]
    dg : float
        delta gamma, difference between axis and ground azimuths [degrees]

    Returns
    -------
    vector : tuple
        vx, vy, vz
    """
    cos_ba = cosd(ba)
    cos_bg = cosd(bg)
    sin_bg = sind(bg)
    sin_dg = sind(dg)
    vx = sin_dg * cos_ba * cos_bg
    vy = sind(ba)*sin_bg + cosd(dg)*cos_ba*cos_bg
    vz = -sin_dg*sin_bg*cos_ba
    return vx, vy, vz


def _calc_beta_c(v, dg, ba):
    """
    Calculate the cross-axis tilt angle.

    Parameters
    ----------
    v : tuple
        tracker normal
    dg : float
        delta gamma, difference between axis and ground azimuths [degrees]
    ba : float
        axis tilt [degrees]

    Returns
    -------
    beta_c : float
        cross-axis tilt angle [radians]
    """
    vnorm = np.sqrt(np.dot(v, v))
    beta_c = np.arcsin(
        ((v[0]*cosd(dg) - v[1]*sind(dg)) * sind(ba) + v[2]*cosd(ba)) / vnorm)
    return beta_c


def calc_cross_axis_tilt(
        slope_azimuth, slope_tilt, axis_azimuth, axis_tilt):
    """
    Calculate the angle, relative to horizontal, of the line formed by the
    intersection between the slope containing the tracker axes and a plane
    perpendicular to the tracker axes.

    Use the cross-axis tilt to avoid row-to-row shade when backtracking on a
    slope not parallel with the axis azimuth. Cross-axis tilt should be
    specified using a right-handed convention. For example, trackers with axis
    azimuth of 180 degrees (heading south) will have a negative cross-axis tilt
    if the tracker axes plane slopes down to the east and positive cross-axis
    tilt if the tracker axes plane slopes up to the east.

    Parameters
    ----------
    slope_azimuth : float
        direction of the normal to the slope containing the tracker axes, when
        projected on the horizontal [degrees]
    slope_tilt : float
        angle of the slope containing the tracker axes, relative to horizontal
        [degrees]
    axis_azimuth : float
        direction of tracker axes projected on the horizontal [degrees]
    axis_tilt : float
        tilt of trackers relative to horizontal [degrees]

    Returns
    -------
    cross_axis_tilt : float
        angle, relative to horizontal, of the line formed by the intersection
        between the slope containing the tracker axes and a plane perpendicular
        to the tracker axes [degrees]

    See also
    --------
    pvlib.tracking.singleaxis
    pvlib.tracking.calc_axis_tilt

    Notes
    -----
    See [1]_ for derivation of equations.

    References
    ----------
    .. [1] Kevin Anderson and Mark Mikofski, "Slope-Aware Backtracking for
       Single-Axis Trackers", Technical Report NREL/TP-5K00-76626, July 2020.
       https://www.nrel.gov/docs/fy20osti/76626.pdf
    """
    # delta-gamma, difference between axis and slope azimuths
    delta_gamma = axis_azimuth - slope_azimuth
    # equation 22
    v = _calc_tracker_norm(axis_tilt, slope_tilt, delta_gamma)
    # equation 26
    beta_c = _calc_beta_c(v, delta_gamma, axis_tilt)
    return np.degrees(beta_c)
