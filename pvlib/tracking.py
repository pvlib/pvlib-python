import numpy as np
import pandas as pd

from pvlib.tools import cosd, sind
from pvlib.pvsystem import _combine_localized_attributes
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib import irradiance, atmosphere


class SingleAxisTracker(PVSystem):
    """
    Inherits the PV modeling methods from :py:class:`~pvlib.pvsystem.PVSystem`.


    Parameters
    ----------
    axis_tilt : float, default 0
        The tilt of the axis of rotation (i.e, the y-axis defined by
        axis_azimuth) with respect to horizontal, in decimal degrees.

    axis_azimuth : float, default 0
        A value denoting the compass direction along which the axis of
        rotation lies. Measured in decimal degrees East of North.

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

    """

    def __init__(self, axis_tilt=0, axis_azimuth=0,
                 max_angle=90, backtrack=True, gcr=2.0/7.0, **kwargs):

        self.axis_tilt = axis_tilt
        self.axis_azimuth = axis_azimuth
        self.max_angle = max_angle
        self.backtrack = backtrack
        self.gcr = gcr

        kwargs['surface_tilt'] = None
        kwargs['surface_azimuth'] = None

        super(SingleAxisTracker, self).__init__(**kwargs)

    def __repr__(self):
        attrs = ['axis_tilt', 'axis_azimuth', 'max_angle', 'backtrack', 'gcr']
        sat_repr = ('SingleAxisTracker: \n  ' + '\n  '.join(
            ('{}: {}'.format(attr, getattr(self, attr)) for attr in attrs)))
        # get the parent PVSystem info
        pvsystem_repr = super(SingleAxisTracker, self).__repr__()
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
                                   self.max_angle,
                                   self.backtrack, self.gcr)

        return tracking_data

    def localize(self, location=None, latitude=None, longitude=None,
                 **kwargs):
        """
        Creates a :py:class:`LocalizedSingleAxisTracker` object using
        this object and location data. Must supply either location
        object or latitude, longitude, and any location kwargs

        Parameters
        ----------
        location : None or Location, default None
        latitude : None or float, default None
        longitude : None or float, default None
        **kwargs : see Location

        Returns
        -------
        localized_system : LocalizedSingleAxisTracker
        """

        if location is None:
            location = Location(latitude, longitude, **kwargs)

        return LocalizedSingleAxisTracker(pvsystem=self, location=location)

    def get_aoi(self, surface_tilt, surface_azimuth, solar_zenith,
                solar_azimuth):
        """Get the angle of incidence on the system.

        For a given set of solar zenith and azimuth angles, the
        surface tilt and azimuth parameters are typically determined
        by :py:method:`~SingleAxisTracker.singleaxis`. The
        :py:method:`~SingleAxisTracker.singleaxis` method also returns
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
            Passed to :func:`irradiance.total_irrad`.

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


class LocalizedSingleAxisTracker(SingleAxisTracker, Location):
    """
    The LocalizedSingleAxisTracker class defines a standard set of
    installed PV system attributes and modeling functions. This class
    combines the attributes and methods of the SingleAxisTracker (a
    subclass of PVSystem) and Location classes.

    The LocalizedSingleAxisTracker may have bugs due to the difficulty
    of robustly implementing multiple inheritance. See
    :py:class:`~pvlib.modelchain.ModelChain` for an alternative paradigm
    for modeling PV systems at specific locations.
    """

    def __init__(self, pvsystem=None, location=None, **kwargs):

        new_kwargs = _combine_localized_attributes(
            pvsystem=pvsystem,
            location=location,
            **kwargs,
        )

        SingleAxisTracker.__init__(self, **new_kwargs)
        Location.__init__(self, **new_kwargs)

    def __repr__(self):
        attrs = ['latitude', 'longitude', 'altitude', 'tz']
        return ('Localized' +
                super(LocalizedSingleAxisTracker, self).__repr__() + '\n  ' +
                '\n  '.join(('{}: {}'.format(attr, getattr(self, attr))
                             for attr in attrs)))


def singleaxis(apparent_zenith, apparent_azimuth,
               axis_tilt=0, axis_azimuth=0, max_angle=90,
               backtrack=True, gcr=2.0/7.0, side_slope=0):
    """
    Determine the rotation angle of a single axis tracker using the
    equations in [1]_ when given a particular sun zenith and azimuth
    angle. backtracking may be specified, and if so, a ground coverage
    ratio is required.

    Rotation angle is determined in a panel-oriented coordinate system.
    The tracker azimuth axis_azimuth defines the positive y-axis; the
    positive x-axis is 90 degress clockwise from the y-axis and parallel
    to the earth surface, and the positive z-axis is normal and oriented
    towards the sun. Rotation angle tracker_theta indicates tracker
    position relative to horizontal: tracker_theta = 0 is horizontal,
    and positive tracker_theta is a clockwise rotation around the y axis
    in the x, y, z coordinate system. For example, if tracker azimuth
    axis_azimuth is 180 (oriented south), tracker_theta = 30 is a
    rotation of 30 degrees towards the west, and tracker_theta = -90 is
    a rotation to the vertical plane facing east.

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
        rotation lies. Measured in decimal degrees East of North.

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

    side_slope : float, default 0.0
        The slope in degrees of the "system plane" perpendicular to the
        tracker axes. The "system plane" is defined as the plane that
        contains all of the tracker axes. EG north-south trackers on a
        3-degree eastern slope would have a 3-degree side slope, depending
        on the tracker axis azimuth. Use ``calc_system_tracker_side_slope``
        for more complicated system planes.

    Returns
    -------
    dict or DataFrame with the following columns:
        * `tracker_theta`: The rotation angle of the tracker.
          tracker_theta = 0 is horizontal, and positive rotation angles are
          clockwise.
        * `aoi`: The angle-of-incidence of direct irradiance onto the
          rotated panel surface.
        * `surface_tilt`: The angle between the panel surface and the earth
          surface, accounting for panel rotation.
        * `surface_azimuth`: The azimuth of the rotated panel, determined by
          projecting the vector normal to the panel's surface to the earth's
          surface.

    References
    ----------
    .. [1] Lorenzo, E et al., 2011, "Tracking and back-tracking", Prog. in
       Photovoltaics: Research and Applications, v. 19, pp. 747-753.
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

    # Calculate sun position x, y, z using coordinate system as in [1], Eq 2.

    # Positive y axis is oriented parallel to earth surface along tracking axis
    # (for the purpose of illustration, assume y is oriented to the south);
    # positive x axis is orthogonal, 90 deg clockwise from y-axis, and parallel
    # to the earth's surface (if y axis is south, x axis is west);
    # positive z axis is normal to x, y axes, pointed upward.

    # Equations in [1] assume solar azimuth is relative to reference vector
    # pointed south, with clockwise positive.
    # Here, the input solar azimuth is degrees East of North,
    # i.e., relative to a reference vector pointed
    # north with clockwise positive.
    # Rotate sun azimuth to coordinate system as in [1]
    # to calculate sun position.

    az = apparent_azimuth - 180
    apparent_elevation = 90 - apparent_zenith
    x = cosd(apparent_elevation) * sind(az)
    y = cosd(apparent_elevation) * cosd(az)
    z = sind(apparent_elevation)

    # translate array azimuth from compass bearing to [1] coord system
    # wholmgren: strange to see axis_azimuth calculated differently from az,
    # (not that it matters, or at least it shouldn't...).
    axis_azimuth_south = axis_azimuth - 180

    # translate input array tilt angle axis_tilt to [1] coordinate system.

    # In [1] coordinates, axis_tilt is a rotation about the x-axis.
    # For a system with array azimuth (y-axis) oriented south,
    # the x-axis is oriented west, and a positive axis_tilt is a
    # counterclockwise rotation, i.e, lifting the north edge of the panel.
    # Thus, in [1] coordinate system, in the northern hemisphere a positive
    # axis_tilt indicates a rotation toward the equator,
    # whereas in the southern hemisphere rotation toward the equator is
    # indicated by axis_tilt<0.  Here, the input axis_tilt is
    # always positive and is a rotation toward the equator.

    # Calculate sun position (xp, yp, zp) in panel-oriented coordinate system:
    # positive y-axis is oriented along tracking axis at panel tilt;
    # positive x-axis is orthogonal, clockwise, parallel to earth surface;
    # positive z-axis is normal to x-y axes, pointed upward.
    # Calculate sun position (xp,yp,zp) in panel coordinates using [1] Eq 11
    # note that equation for yp (y' in Eq. 11 of Lorenzo et al 2011) is
    # corrected, after conversation with paper's authors.

    xp = x*cosd(axis_azimuth_south) - y*sind(axis_azimuth_south)
    yp = (x*cosd(axis_tilt)*sind(axis_azimuth_south) +
          y*cosd(axis_tilt)*cosd(axis_azimuth_south) -
          z*sind(axis_tilt))
    zp = (x*sind(axis_tilt)*sind(axis_azimuth_south) +
          y*sind(axis_tilt)*cosd(axis_azimuth_south) +
          z*cosd(axis_tilt))

    # The ideal tracking angle wid is the rotation to place the sun position
    # vector (xp, yp, zp) in the (y, z) plane; i.e., normal to the panel and
    # containing the axis of rotation.  wid = 0 indicates that the panel is
    # horizontal.  Here, our convention is that a clockwise rotation is
    # positive, to view rotation angles in the same frame of reference as
    # azimuth.  For example, for a system with tracking axis oriented south,
    # a rotation toward the east is negative, and a rotation to the west is
    # positive.

    # Use arctan2 and avoid the tmp corrections.

    # angle from x-y plane to projection of sun vector onto x-z plane
#     tmp = np.degrees(np.arctan(zp/xp))

    # Obtain wid by translating tmp to convention for rotation angles.
    # Have to account for which quadrant of the x-z plane in which the sun
    # vector lies.  Complete solution here but probably not necessary to
    # consider QIII and QIV.
#     wid = pd.Series(index=times)
#     wid[(xp>=0) & (zp>=0)] =  90 - tmp[(xp>=0) & (zp>=0)]  # QI
#     wid[(xp<0)  & (zp>=0)] = -90 - tmp[(xp<0)  & (zp>=0)]  # QII
#     wid[(xp<0)  & (zp<0)]  = -90 - tmp[(xp<0)  & (zp<0)]   # QIII
#     wid[(xp>=0) & (zp<0)]  =  90 - tmp[(xp>=0) & (zp<0)]   # QIV

    # Calculate angle from x-y plane to projection of sun vector onto x-z plane
    # and then obtain wid by translating tmp to convention for rotation angles.
    wid = 90 - np.degrees(np.arctan2(zp, xp))

    # filter for sun above panel horizon
    zen_gt_90 = apparent_zenith > 90
    wid[zen_gt_90] = np.nan

    # Account for backtracking; modified from [1] to account for rotation
    # angle convention being used here.
    if backtrack:
        axes_distance = 1/gcr
        # clip needed for low angles. GH 656
        temp = np.clip(axes_distance*cosd(wid + side_slope), -1, 1)

        # backtrack angle
        # (always positive b/c acosd returns values between 0 and 180)
        wc = np.degrees(np.arccos(temp))

        # Eq 4 applied when wid in QIV (wid < 0 evalulates True), QI
        with np.errstate(invalid='ignore'):
            # errstate for GH 622
            tracker_theta = np.where(wid < 0, wid + wc, wid - wc)
    else:
        tracker_theta = wid

    # TODO: use np.clip
    # "Equivalent to but faster than np.maximum(a_min, np.minimum(a, a_max))"
    # NOTE: max_angle defined relative to zero-point rotation, not the
    # system-plane normal
    tracker_theta = np.minimum(tracker_theta, max_angle)
    tracker_theta = np.maximum(tracker_theta, -max_angle)

    # calculate panel normal vector in panel-oriented x, y, z coordinates.
    # y-axis is axis of tracker rotation.  tracker_theta is a compass angle
    # (clockwise is positive) rather than a trigonometric angle.
    # the *0 is a trick to preserve NaN values.
    panel_norm = np.array([sind(tracker_theta),
                           tracker_theta*0,
                           cosd(tracker_theta)])

    # sun position in vector format in panel-oriented x, y, z coordinates
    sun_vec = np.array([xp, yp, zp])

    # calculate angle-of-incidence on panel
    aoi = np.degrees(np.arccos(np.abs(np.sum(sun_vec*panel_norm, axis=0))))

    # calculate panel tilt and azimuth
    # in a coordinate system where the panel tilt is the
    # angle from horizontal, and the panel azimuth is
    # the compass angle (clockwise from north) to the projection
    # of the panel's normal to the earth's surface.
    # These outputs are provided for convenience and comparison
    # with other PV software which use these angle conventions.

    # project normal vector to earth surface.
    # First rotate about x-axis by angle -axis_tilt so that y-axis is
    # also parallel to earth surface, then project.

    # Calculate standard rotation matrix
    rot_x = np.array([[1, 0, 0],
                      [0, cosd(-axis_tilt), -sind(-axis_tilt)],
                      [0, sind(-axis_tilt), cosd(-axis_tilt)]])

    # panel_norm_earth contains the normal vector
    # expressed in earth-surface coordinates
    # (z normal to surface, y aligned with tracker axis parallel to earth)
    panel_norm_earth = np.dot(rot_x, panel_norm).T

    # projection to plane tangent to earth surface,
    # in earth surface coordinates
    projected_normal = np.array([panel_norm_earth[:, 0],
                                 panel_norm_earth[:, 1],
                                 panel_norm_earth[:, 2]*0]).T

    # calculate vector magnitudes
    projected_normal_mag = np.sqrt(np.nansum(projected_normal**2, axis=1))

    # renormalize the projected vector
    # avoid creating nan values.
    non_zeros = projected_normal_mag != 0
    projected_normal[non_zeros] = (projected_normal[non_zeros].T /
                                   projected_normal_mag[non_zeros]).T

    # calculation of surface_azimuth
    # 1. Find the angle.
#     surface_azimuth = pd.Series(
#         np.degrees(np.arctan(projected_normal[:,1]/projected_normal[:,0])),
#                                 index=times)
    surface_azimuth = \
        np.degrees(np.arctan2(projected_normal[:, 1], projected_normal[:, 0]))

    # 2. Clean up atan when x-coord or y-coord is zero
#     surface_azimuth[(projected_normal[:,0]==0) & (projected_normal[:,1]>0)] =  90
#     surface_azimuth[(projected_normal[:,0]==0) & (projected_normal[:,1]<0)] =  -90
#     surface_azimuth[(projected_normal[:,1]==0) & (projected_normal[:,0]>0)] =  0
#     surface_azimuth[(projected_normal[:,1]==0) & (projected_normal[:,0]<0)] = 180

    # 3. Correct atan for QII and QIII
#     surface_azimuth[(projected_normal[:,0]<0) & (projected_normal[:,1]>0)] += 180 # QII
#     surface_azimuth[(projected_normal[:,0]<0) & (projected_normal[:,1]<0)] += 180 # QIII

    # 4. Skip to below

    # at this point surface_azimuth contains angles between -90 and +270,
    # where 0 is along the positive x-axis,
    # the y-axis is in the direction of the tracker azimuth,
    # and positive angles are rotations from the positive x axis towards
    # the positive y-axis.
    # Adjust to compass angles
    # (clockwise rotation from 0 along the positive y-axis)
#    surface_azimuth[surface_azimuth<=90] = 90 - surface_azimuth[surface_azimuth<=90]
#    surface_azimuth[surface_azimuth>90] = 450 - surface_azimuth[surface_azimuth>90]

    # finally rotate to align y-axis with true north
    # PVLIB_MATLAB has this latitude correction,
    # but I don't think it's latitude dependent if you always
    # specify axis_azimuth with respect to North.
#     if latitude > 0 or True:
#         surface_azimuth = surface_azimuth - axis_azimuth
#     else:
#         surface_azimuth = surface_azimuth - axis_azimuth - 180
#     surface_azimuth[surface_azimuth<0] = 360 + surface_azimuth[surface_azimuth<0]

    # the commented code above is mostly part of PVLIB_MATLAB.
    # My (wholmgren) take is that it can be done more simply.
    # Say that we're pointing along the postive x axis (likely west).
    # We just need to rotate 90 degrees to get from the x axis
    # to the y axis (likely south),
    # and then add the axis_azimuth to get back to North.
    # Anything left over is the azimuth that we want,
    # and we can map it into the [0,360) domain.

    # 4. Rotate 0 reference from panel's x axis to it's y axis and
    #    then back to North.
    surface_azimuth = 90 - surface_azimuth + axis_azimuth

    # 5. Map azimuth into [0,360) domain.
    # surface_azimuth[surface_azimuth < 0] += 360
    # surface_azimuth[surface_azimuth >= 360] -= 360
    surface_azimuth = surface_azimuth % 360

    # Calculate surface_tilt
    dotproduct = (panel_norm_earth * projected_normal).sum(axis=1)
    surface_tilt = 90 - np.degrees(np.arccos(dotproduct))

    # Bundle DataFrame for return values and filter for sun below horizon.
    out = {'tracker_theta': tracker_theta, 'aoi': aoi,
           'surface_azimuth': surface_azimuth, 'surface_tilt': surface_tilt}
    if index is not None:
        out = pd.DataFrame(out, index=index)
        out = out[['tracker_theta', 'aoi', 'surface_azimuth', 'surface_tilt']]
        out[zen_gt_90] = np.nan
    else:
        out = {k: np.where(zen_gt_90, np.nan, v) for k, v in out.items()}

    return out


def _get_rotation_matrix(angle, axis=0):
    """
    Return a rotation matrix that when multiplied by a column vector returns
    a new column vector that is rotated clockwise around the given axis by the
    given angle.

    Parameters
    ----------
    angle : float
        Angle of rotation [radians]
    axis : int, default 0
        Axis of rotation, 0=x, 1=y, 2=z

    Returns
    -------
    rotation matrix

    References:
       `Rotation Matrix
       <https://en.wikipedia.org/wiki/Rotation_matrix#Basic_rotations>`_

    """
    r11 = r22 = np.cos(angle)
    r21 = np.sin(angle)
    r12 = -r21
    rot = np.array([
        [1, 0, 0],
        [0, r11, r12],
        [0, r21, r22]])
    rot = np.roll(rot, (axis, axis), (1, 0))
    return rot


def calc_tracker_axis_tilt(system_azimuth, system_zenith, tracker_azimuth):
    """
    Calculate tracker axis tilt in the global reference frame when on a sloped
    plane.

    Parameters
    ----------
    system_azimuth : float
        direction of normal to slope on horizontal [radians]
    system_zenith : float
        tilt of normal to slope relative to vertical [radians]
    tracker_azimuth : float
        direction of tracker axes on horizontal [radians]

    Returns
    -------
    tracker_zenith : float
        tilt of tracker [radians]

    Solving for the tracker tilt on a slope is derived in the following steps:

    1. the trackers axes are in the system plane, so the ``z-coord = 0``

    2. rotate the trackers ``[x_tr_sys, y_tr_sys, 0]`` back to the global, but
       rotated by the tracker global azimuth if there is one, so that the
       tracker axis is constrained to y-z plane so that ``x-coord = 0`` ::

        Rx_sys = [[1,           0,            0],
                  [0, cos(sys_ze), -sin(sys_ze)],
                  [0, sin(sys_ze),  cos(sys_ze)]]

        Rz_sys = [[cos(sys_az-tr_az), -sin(sys_az-tr_az), 0],
                  [sin(sys_az-tr_az),  cos(sys_az-tr_az), 0],
                  [                0,                  0, 1]]

        tr_rot_glo = Rz_sys.T * (Rx_sys.T * [x_tr_sys, y_tr_sys, 0])

        tr_rot_glo = [
          [ x_tr_sys*cos(sys_az-tr_az)+y_tr_sys*sin(sys_az-tr_az)*cos(sys_ze)],
          [-x_tr_sys*sin(sys_az-tr_az)+y_tr_sys*cos(sys_az-tr_az)*cos(sys_ze)],
          [                                             -y_tr_sys*sin(sys_ze)]]

    3. solve for ``x_tr_sys`` ::

        x_tr_sys*cos(sys_az-tr_az)+y_tr_sys*sin(sys_az-tr_az)*cos(sys_ze) = 0
        x_tr_sys = -y_tr_sys*tan(sys_az-tr_az)*cos(sys_ze)

    4. so tracker axis tilt, ``tr_ze = arctan2(tr_rot_glo_z, tr_rot_glo_y)`` ::

        tr_rot_glo_y = y_tr_sys*cos(sys_ze)*(
          tan(sys_az-tr_az)*sin(sys_az-tr_az) + cos(sys_az-tr_az))

        tan(tr_ze) = -y_tr_sys*sin(sys_ze) / tr_rot_glo_y

    The trick is multiply top and bottom by cos(sys_az-tr_az) and remember that
    ``sin^2 + cos^2 = 1`` (or just use sympy.simplify) ::

        tan(tr_ze) = -tan(sys_ze)*cos(sys_az-tr_az)
    """
    sys_az_rel_to_tr_az = system_azimuth - tracker_azimuth
    tan_tr_ze = -np.cos(sys_az_rel_to_tr_az) * np.tan(system_zenith)
    return -np.arctan(tan_tr_ze)


def calc_system_tracker_side_slope(
        tracker_azimuth, tracker_zenith, system_azimuth, system_zenith):
    """
    Calculate the slope perpendicular to the tracker axis relative to the
    system plane containing the axes as well as the rotation of the tracker
    axes relative to the system plane. Note in order for the backtracking
    algorithm to work correctly on a sloped system plane, the side slope must
    be applied to the tracker rotation.

    Parameters
    ----------
    system_azimuth : float
        direction of normal to slope on horizontal [radians]
    system_zenith : float
        tilt of normal to slope relative to vertical [radians]
    tracker_azimuth : float
        direction of tracker axes on horizontal [radians]
    tracker_zenith : float
        tilt of tracker [radians]

    Returns
    -------
    tracker side slope and rotation relative to system plane [radians]
    """
    # find the relative rotation of the trackers in the system plane
    # 1. tracker axis vector
    cos_tr_ze = np.cos(-tracker_zenith)
    sin_tr_az = np.sin(tracker_azimuth)
    cos_tr_az = np.cos(tracker_azimuth)
    tr_ax = np.array([
        [cos_tr_ze*sin_tr_az],
        [cos_tr_ze*cos_tr_az],
        [np.sin(-tracker_zenith)]])
    # 2. rotate tracker axis vector from global to system reference frame
    sys_z_rot = _get_rotation_matrix(system_azimuth, axis=2)
    # first around the z-axis
    tr_ax_sys_z_rot = np.dot(sys_z_rot, tr_ax)
    # then around x-axis so that xy-plane is the plane with slope and trackers
    sys_x_rot = _get_rotation_matrix(system_zenith)
    tr_ax_sys = np.dot(sys_x_rot, tr_ax_sys_z_rot)
    # now that tracker axis is in coordinate system of slope, the relative
    # rotation is the angle from the y axis
    tr_rel_rot = np.arctan2(tr_ax_sys[0, 0], tr_ax_sys[1, 0])
    # find side slope
    # 1. tracker normal vector
    sin_tr_ze = np.sin(tracker_zenith)
    tr_norm = np.array([
        [sin_tr_ze*sin_tr_az],
        [sin_tr_ze*cos_tr_az],
        [cos_tr_ze]])  # note: cos(-x) = cos(x)
    # 2. rotate tracker normal vector from global to system reference frame
    tr_norm_sys_z_rot = np.dot(sys_z_rot, tr_norm)
    tr_norm_sys = np.dot(sys_x_rot, tr_norm_sys_z_rot)
    # 3. side slope is angle between tracker normal and system plane normal
    # np.arccos(tr_norm_sys[2])
    # 4. but we need to know which way the slope is facing, so rotate to
    # tracker use arctan2
    sys_tr_z_rot = _get_rotation_matrix(tr_rel_rot, axis=2)
    tr_norm_sys_tr = np.dot(sys_tr_z_rot, tr_norm_sys)
    side_slope = np.arctan2(tr_norm_sys_tr[0, 0], tr_norm_sys_tr[2, 0])
    return side_slope, tr_rel_rot
