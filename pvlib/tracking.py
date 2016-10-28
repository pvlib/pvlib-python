from __future__ import division

import numpy as np
import pandas as pd

from pvlib.tools import cosd, sind
from pvlib.pvsystem import PVSystem
from pvlib.location import Location
from pvlib import irradiance, atmosphere

import logging
pvl_logger = logging.getLogger('pvlib')


class SingleAxisTracker(PVSystem):
    """
    Inherits all of the PV modeling methods from PVSystem.
    """

    def __init__(self, axis_tilt=0, axis_azimuth=0,
                 max_angle=90, backtrack=True, gcr=2.0/7.0, **kwargs):

        self.axis_tilt = axis_tilt
        self.axis_azimuth = axis_azimuth
        self.max_angle = max_angle
        self.backtrack = backtrack
        self.gcr = gcr

        super(SingleAxisTracker, self).__init__(**kwargs)

    def __repr__(self):
        attrs = ['axis_tilt', 'axis_azimuth', 'max_angle', 'backtrack', 'gcr']
        sat_repr = ('SingleAxisTracker: \n  ' + '\n  '.join(
            (attr + ': ' + str(getattr(self, attr)) for attr in attrs)))
        # get the parent PVSystem info
        pvsystem_repr = super(SingleAxisTracker, self).__repr__()
        # remove the first line (contains 'PVSystem: \n')
        pvsystem_repr = '\n'.join(pvsystem_repr.split('\n')[1:])
        return sat_repr + '\n' + pvsystem_repr


    def singleaxis(self, apparent_zenith, apparent_azimuth):
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
        location : None or Location
        latitude : None or float
        longitude : None or float
        **kwargs : see Location

        Returns
        -------
        localized_system : LocalizedSingleAxisTracker
        """

        if location is None:
            location = Location(latitude, longitude, **kwargs)

        return LocalizedSingleAxisTracker(pvsystem=self, location=location)

    def get_irradiance(self, dni, ghi, dhi,
                       dni_extra=None, airmass=None, model='haydavies',
                       **kwargs):
        """
        Uses the :func:`irradiance.total_irrad` function to calculate
        the plane of array irradiance components on a tilted surface
        defined by ``self.surface_tilt``, ``self.surface_azimuth``, and
        ``self.albedo``.

        Parameters
        ----------
        solar_zenith : float or Series.
            Solar zenith angle.
        solar_azimuth : float or Series.
            Solar azimuth angle.
        dni : float or Series
            Direct Normal Irradiance
        ghi : float or Series
            Global horizontal irradiance
        dhi : float or Series
            Diffuse horizontal irradiance
        dni_extra : float or Series
            Extraterrestrial direct normal irradiance
        airmass : float or Series
            Airmass
        model : String
            Irradiance model.

        **kwargs
            Passed to :func:`irradiance.total_irrad`.

        Returns
        -------
        poa_irradiance : DataFrame
            Column names are: ``total, beam, sky, ground``.
        """

        surface_tilt = kwargs.pop('surface_tilt', self.surface_tilt)
        surface_azimuth = kwargs.pop('surface_azimuth', self.surface_azimuth)

        try:
            solar_zenith = kwargs['solar_zenith']
        except KeyError:
            solar_zenith = self.solar_zenith

        try:
            solar_azimuth = kwargs['solar_azimuth']
        except KeyError:
            solar_azimuth = self.solar_azimuth

        # not needed for all models, but this is easier
        if dni_extra is None:
            dni_extra = irradiance.extraradiation(solar_zenith.index)
            dni_extra = pd.Series(dni_extra, index=solar_zenith.index)

        if airmass is None:
            airmass = atmosphere.relativeairmass(solar_zenith)

        return irradiance.total_irrad(surface_tilt,
                                      surface_azimuth,
                                      solar_zenith,
                                      solar_azimuth,
                                      dni, ghi, dhi,
                                      dni_extra=dni_extra, airmass=airmass,
                                      model=model,
                                      albedo=self.albedo,
                                      **kwargs)


class LocalizedSingleAxisTracker(SingleAxisTracker, Location):
    """Highly experimental."""

    def __init__(self, pvsystem=None, location=None, **kwargs):

        # get and combine attributes from the pvsystem and/or location
        # with the rest of the kwargs

        if pvsystem is not None:
            pv_dict = pvsystem.__dict__
        else:
            pv_dict = {}

        if location is not None:
            loc_dict = location.__dict__
        else:
            loc_dict = {}

        new_kwargs = dict(list(pv_dict.items()) +
                          list(loc_dict.items()) +
                          list(kwargs.items()))

        super(LocalizedSingleAxisTracker, self).__init__(**new_kwargs)

    def __repr__(self):
        attrs = ['latitude', 'longitude', 'altitude', 'tz']
        return ('Localized' +
            super(LocalizedSingleAxisTracker, self).__repr__() + '\n  ' +
            '\n  '.join(
                (attr + ': ' + str(getattr(self, attr)) for attr in attrs)))


def singleaxis(apparent_zenith, apparent_azimuth,
               axis_tilt=0, axis_azimuth=0, max_angle=90,
               backtrack=True, gcr=2.0/7.0):
    """
    Determine the rotation angle of a single axis tracker using the
    equations in [1] when given a particular sun zenith and azimuth
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
    apparent_zenith : Series
        Solar apparent zenith angles in decimal degrees.

    apparent_azimuth : Series
        Solar apparent azimuth angles in decimal degrees.

    axis_tilt : float
        The tilt of the axis of rotation (i.e, the y-axis defined by
        axis_azimuth) with respect to horizontal, in decimal degrees.

    axis_azimuth : float
        A value denoting the compass direction along which the axis of
        rotation lies. Measured in decimal degrees East of North.

    max_angle : float
        A value denoting the maximum rotation angle, in decimal degrees,
        of the one-axis tracker from its horizontal position (horizontal
        if axis_tilt = 0). A max_angle of 90 degrees allows the tracker
        to rotate to a vertical position to point the panel towards a
        horizon. max_angle of 180 degrees allows for full rotation.

    backtrack : bool
        Controls whether the tracker has the capability to "backtrack"
        to avoid row-to-row shading. False denotes no backtrack
        capability. True denotes backtrack capability.

    gcr : float
        A value denoting the ground coverage ratio of a tracker system
        which utilizes backtracking; i.e. the ratio between the PV array
        surface area to total ground area. A tracker system with modules
        2 meters wide, centered on the tracking axis, with 6 meters
        between the tracking axes has a gcr of 2/6=0.333. If gcr is not
        provided, a gcr of 2/7 is default. gcr must be <=1.

    Returns
    -------
    DataFrame with the following columns:

    * tracker_theta: The rotation angle of the tracker.
        tracker_theta = 0 is horizontal, and positive rotation angles are
        clockwise.
    * aoi: The angle-of-incidence of direct irradiance onto the
        rotated panel surface.
    * surface_tilt: The angle between the panel surface and the earth
        surface, accounting for panel rotation.
    * surface_azimuth: The azimuth of the rotated panel, determined by
        projecting the vector normal to the panel's surface to the earth's
        surface.

    References
    ----------
    [1] Lorenzo, E et al., 2011, "Tracking and back-tracking", Prog. in
    Photovoltaics: Research and Applications, v. 19, pp. 747-753.
    """

    pvl_logger.debug('tracking.singleaxis')

    pvl_logger.debug('axis_tilt=%s, axis_azimuth=%s, max_angle=%s, ' +
                     'backtrack=%s, gcr=%.3f',
                     axis_tilt, axis_azimuth, max_angle, backtrack, gcr)

    pvl_logger.debug('\napparent_zenith=\n%s\napparent_azimuth=\n%s',
                     apparent_zenith.head(), apparent_azimuth.head())

    # MATLAB to Python conversion by
    # Will Holmgren (@wholmgren), U. Arizona. March, 2015.

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

    try:
        pd.util.testing.assert_index_equal(apparent_azimuth.index,
                                           apparent_zenith.index)
    except AssertionError:
        raise ValueError('apparent_azimuth.index and ' +
                         'apparent_zenith.index must match.')

    times = apparent_azimuth.index

    az = apparent_azimuth - 180
    apparent_elevation = 90 - apparent_zenith
    x = cosd(apparent_elevation) * sind(az)
    y = cosd(apparent_elevation) * cosd(az)
    z = sind(apparent_elevation)

    # translate array azimuth from compass bearing to [1] coord system
    # wholmgren: strange to see axis_azimuth calculated differently from az,
    # (not that it matters, or at least it shouldn't...).
    axis_azimuth_south = axis_azimuth - 180
    pvl_logger.debug('axis_azimuth_south=%s', axis_azimuth_south)

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
    wid = pd.Series(90 - np.degrees(np.arctan2(zp, xp)), index=times)

    # filter for sun above panel horizon
    wid[zp <= 0] = np.nan

    # Account for backtracking; modified from [1] to account for rotation
    # angle convention being used here.
    if backtrack:
        pvl_logger.debug('applying backtracking')
        axes_distance = 1/gcr
        temp = np.minimum(axes_distance*cosd(wid), 1)

        # backtrack angle
        # (always positive b/c acosd returns values between 0 and 180)
        wc = np.degrees(np.arccos(temp))

        v = wid < 0
        widc = pd.Series(index=times)
        widc[~v] = wid[~v] - wc[~v]  # Eq 4 applied when wid in QI
        widc[v] = wid[v] + wc[v]     # Eq 4 applied when wid in QIV
    else:
        pvl_logger.debug('no backtracking')
        widc = wid

    tracker_theta = widc.copy()
    tracker_theta[tracker_theta > max_angle] = max_angle
    tracker_theta[tracker_theta < -max_angle] = -max_angle

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
    aoi = pd.Series(aoi, index=times)

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
    pvl_logger.debug('rot_x=\n%s', rot_x)

    # panel_norm_earth contains the normal vector
    # expressed in earth-surface coordinates
    # (z normal to surface, y aligned with tracker axis parallel to earth)
    panel_norm_earth = np.dot(rot_x, panel_norm).T
    pvl_logger.debug('panel_norm_earth=%s', panel_norm_earth)

    # projection to plane tangent to earth surface,
    # in earth surface coordinates
    projected_normal = np.array([panel_norm_earth[:, 0],
                                 panel_norm_earth[:, 1],
                                 panel_norm_earth[:, 2]*0]).T
    pvl_logger.debug('projected_normal=%s', projected_normal)

    # calculate vector magnitudes
    panel_norm_earth_mag = np.sqrt(np.nansum(panel_norm_earth**2, axis=1))
    projected_normal_mag = np.sqrt(np.nansum(projected_normal**2, axis=1))
    pvl_logger.debug('panel_norm_earth_mag=%s, projected_normal_mag=%s',
                     panel_norm_earth_mag, projected_normal_mag)

    # renormalize the projected vector
    # avoid creating nan values.
    non_zeros = projected_normal_mag != 0
    projected_normal[non_zeros] = (projected_normal[non_zeros].T /
                                   projected_normal_mag[non_zeros]).T
    pvl_logger.debug('renormalized projected_normal=%s',
                     projected_normal)

    # calculation of surface_azimuth
    # 1. Find the angle.
#     surface_azimuth = pd.Series(
#         np.degrees(np.arctan(projected_normal[:,1]/projected_normal[:,0])),
#                                 index=times)
    surface_azimuth = pd.Series(
        np.degrees(np.arctan2(projected_normal[:, 1], projected_normal[:, 0])),
        index=times)

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
    surface_azimuth[surface_azimuth < 0] += 360
    surface_azimuth[surface_azimuth >= 360] -= 360

    # Calculate surface_tilt
    # Use pandas to calculate the sum because it handles nan values better.
    surface_tilt = (90 - np.degrees(np.arccos(
                            pd.DataFrame(panel_norm_earth * projected_normal,
                                         index=times).sum(axis=1))))

    # Bundle DataFrame for return values and filter for sun below horizon.
    df_out = pd.DataFrame({'tracker_theta': tracker_theta, 'aoi': aoi,
                           'surface_azimuth': surface_azimuth,
                           'surface_tilt': surface_tilt},
                          index=times)

    df_out[apparent_zenith > 90] = np.nan

    return df_out
